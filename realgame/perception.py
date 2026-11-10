"""Screen -> observation for the real game, in exactly the format the simulator policies were trained on.

Input is ONLY the playfield crop (576x672 screen px = 384x448 game px, scale 1.5) of the game window.
  1. bullets = white cores of bullets (min(R,G,B) >= core_thr), dilated a few px to approximate the sprite
     radius; the player's own white parts / yin-yang orbs and giant glows (boss) are removed
  2. player = centroid of the red dress near the last known position (tracked) -> hitbox centre
  3. local 64x64 view (2 game px / cell) centred on the player, global 56x48 view (8 game px / cell),
     4-frame stacks, wall channel, player marker, [x, y, focus] vector
  4. scene check: dialogue / menus have huge white areas (portraits, text) -> not a bullet scene
"""
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

# playfield inside the 1280x720 client area (screen px); the capture itself is the playfield only
PF_X0, PF_Y0, PF_W, PF_H = 354, 24, 576, 672
S = 1.5                                          # screen px per game px
GAME_W, GAME_H = 384.0, 448.0
LOC, LOC_CELL = 64, 2.0
GLB_H, GLB_W = 56, 48
STACK = 4
HITBOX_DY = 13.0     # red-dress centroid is this many screen px below the hitbox centre (calibrated at spawn)
SPAWN = (641.5 - PF_X0, 600.0 - PF_Y0)           # hitbox centre at respawn, playfield screen px
# perception v4: colour bullets without a white core (screen px / 0..255 units) and lasers
COL_TOPHAT, COL_V, COL_SAT = 90, 180, 0.7        # brighter than the local background by, brightness, saturation
                                                 # (stage-5 clock rims reach top-hat ~80 at V ~176; dim yellow rice ~150 at ~195)
COL_MAXLEN, COL_MAXAREA = 48, 500                # a colour-only blob longer / bigger than this is background, not a bullet
LASER_ELONG, LASER_LEN = 3.0, 120                # a huge white area this many times longer than wide and this long = a laser
# perception v5 = the game in AI mode (modding/gamemode.py ai): black background, no Reimu / options / own shots / portraits
V5_LIT, V5_PEAK = 80, 150                        # a pixel is 'something' above this max(R,G,B); a blob counts if its peak reaches V5_PEAK
V5_HIT_R = 8.0                                   # screen px around the hitbox ignored (the focus marker's bright centre dot;
                                                 # memory position vs the drawn marker differ by 1-2 px)
V5_OVER_MEDIAN = 60                              # ... and above the frame's median + this (a whole-screen tint, e.g. at a card capture)
TRACK_AREA, TRACK_SPAN = (12, 500), 34           # what counts as one bullet for tracker.py (screen px, before the 1 px dilation)
# Remilia's big pale translucent bullets (s6c4 Vampire Illusion) peak at only 80-149 and were dropped as 'dim' (~10k px per frame, the player
# was hit inside them; 2026-11-03). The spell-start aura blobs are dim too but small or near the boss.
PALE_AREA = 400                                  # screen px: only big pale blobs count (with Perceiver.pale_peak > 0)
# (Known gap: the big orbs' black centres stay holes in the mask. Filling round blobs did not work: _ccl's 64 steps do not get round
# a 75 px ring, so an orb is split into several blobs, none of them round.)


class Perceiver:
    def __init__(self, device="cuda", core_thr=200, dilate=3, big_blob=11, version=3, stack=STACK, shot_filter=False, lags=None):
        self.lags = sorted({int(x) for x in lags}, reverse=True) if lags else None     # spaced history (decisions back), oldest first
        self.stack = len(self.lags) if self.lags else stack
        self.hist = []                     # recent (loc, glb) maps, newest last, when lags are used
        self.shot_filter = shot_filter      # drop the player's own shots by frame-to-frame motion (see own_shot_filter)
        self.prev_raw = None                # previous frame's mask before that filter
        self.last_removed = None            # pixels the filter removed in the last frame (for debug pictures)
        self.dev = torch.device(device)
        self.thr, self.r, self.big = core_thr, dilate, big_blob
        self.pos = np.array(SPAWN, dtype=float)
        self.loc_stack = self.glb_stack = None
        yy, xx = torch.meshgrid(torch.arange(PF_H, device=self.dev), torch.arange(PF_W, device=self.dev), indexing="ij")
        self.yy, self.xx = yy.float(), xx.float()
        self.ids = torch.arange(1, PF_H * PF_W + 1, device=self.dev, dtype=torch.float32).view(1, 1, PF_H, PF_W)
        self.version = version
        self.orb_rel = []          # tracked yin-yang options: [dx, dy, frames since last seen] relative to the player
        self.core_frac = 0.0
        self.portrait_frac = 0.0
        self.items = []            # [(x, y, is_power)] in game px: power / point items seen in the last v3 mask (they are NOT bullets)
        self.prev_core = None      # last frame's white cores (v4: the music title is the white text that does not move)
        self.text_box = None       # v4: [x0, x1, y centre, frames left] of the music title row last seen
        self.prev_v = None         # v5: last frame's max(R, G, B)
        self.pale_peak = 0         # v5, > 0: a blob of >= PALE_AREA px whose brightest pixel is only this .. V5_PEAK still counts (pale bullets)
        self.tint_q = 0.0          # v5, > 0: the 'whole-screen tint' level is this quantile of the frame instead of the median
        self.tracker, self.lead, self.track_model = None, 0.0, "turn"     # play.py --predict: tracker.py, frames ahead
        self.future = False                                               # the policy also gets the future channels (future_env.py)

    # ------------------------------------------------------------------ player
    def locate_player(self, t):
        """Global search for the largest red blob (Reimu's dress is ~900 px, much bigger than red items or
        bullets) -> hitbox centre (playfield screen px). Keeps the last position if nothing qualifies,
        e.g. while the sprite blinks after a death."""
        red = ((t[..., 0] > 170) & (t[..., 1] < 80) & (t[..., 2] < 80)).float()[None, None]
        kh, kw = 90, 52
        cnt = F.avg_pool2d(red, (kh, kw), stride=4)[0, 0] * (kh * kw)
        best = cnt.max()
        if 300 <= best.item() <= 1900:        # a portrait / big red effect has far more red than Reimu's dress
            i, j = divmod(int(cnt.argmax().item()), cnt.shape[1])
            y0, x0 = i * 4, j * 4
            ys, xs = torch.nonzero(red[0, 0, y0:y0 + kh, x0:x0 + kw], as_tuple=True)
            new = np.array([xs.float().mean().item() + x0, ys.float().mean().item() + y0 - HITBOX_DY])
            self.pos = 0.3 * self.pos + 0.7 * new if np.abs(new - self.pos).max() < 60 else new
        return self.pos

    # ------------------------------------------------------------------ bullets
    def bullet_mask(self, t, pos):
        """bool [PF_H, PF_W] (screen px). Also sets self.core_frac (fraction of white-core pixels)."""
        core = t.amin(dim=2) >= self.thr
        core[:34] = False                      # boss health bar and timer digits along the top edge
        self.core_frac = core.float().mean().item()
        m = core[None, None].float()
        # huge glows (boss / explosions): a blob that survives a big erosion is not a bullet
        k = self.big * 2 + 1
        eroded = -F.max_pool2d(-m, k, 1, self.big)
        big = F.max_pool2d(eroded, k + 6, 1, self.big + 3) > 0
        m = m * (~big).float()
        m = F.max_pool2d(m, 2 * self.r + 1, 1, self.r)                        # approximate sprite radius
        mask = m[0, 0] > 0
        # remove the player's own sprite (white sleeves/skirt/hitbox dot) and the two yin-yang orbs
        dx, dy = self.xx - pos[0], self.yy - pos[1]
        body = (dx.abs() < 27) & (dy > -14) & (dy < 34)
        orbs = torch.zeros_like(body)
        for ox, oy in ((-36.5, 0), (36.5, 0), (-13, -49), (13, -49)):
            orbs |= (dx - ox) ** 2 + (dy - oy) ** 2 < 16 ** 2
        return mask & ~body & ~orbs

    # ------------------------------------------------------------------ bullets, v2: component-based clean-up
    def _ccl(self, mask, iters=40):
        """GPU connected components: every pixel gets the max linear id of its component (0 = background)."""
        m = mask[None, None].float()
        ids = self.ids * m
        for _ in range(iters):
            ids = F.max_pool2d(ids, 3, 1, 1) * m
        return ids[0, 0].long()

    @staticmethod
    def _integral(m):
        return F.pad(m.float().cumsum(0).cumsum(1), (1, 0, 1, 0))

    @staticmethod
    def _box(ii, x0, y0, x1, y1):
        return ii[y1 + 1, x1 + 1] - ii[y0, x1 + 1] - ii[y1 + 1, x0] + ii[y0, x0]

    def bullet_mask_v2(self, t, pos, debug=None):
        """Like bullet_mask, but classifies the white-core blobs and drops the ones that are not bullets:
          * power-up / point items (square sprites with a white border and a red/blue fill, letter inside),
          * the two yin-yang options next to the player (round, red+white halves, black dot),
          * stray bits of the player's own sprite (hair highlight above the head, white shoes below)
        and keeps a smaller blind box around the player than v1."""
        H, W = PF_H, PF_W
        dev = self.dev
        core = t.amin(dim=2) >= self.thr
        core[:34] = False
        self.core_frac = core.float().mean().item()
        r, g, b = t[..., 0].short(), t[..., 1].short(), t[..., 2].short()
        red = (r > 160) & (g < 110) & (b < 110)
        colored = red | ((b > 160) & (r < 110) & (g < 150))
        dark = t.amax(dim=2) < 70
        m = core[None, None].float()
        k = self.big * 2 + 1                                    # huge glows (boss / explosions) are not bullets
        eroded = -F.max_pool2d(-m, k, 1, self.big)
        big = F.max_pool2d(eroded, k + 6, 1, self.big + 3)[0, 0] > 0
        core_s = core & ~big
        labels = self._ccl(core_s)
        ys, xs = torch.nonzero(labels > 0, as_tuple=True)
        drop = torch.zeros_like(core_s)
        info = []
        if ys.numel():
            lab = labels[ys, xs]
            uniq, inv = torch.unique(lab, return_inverse=True)
            n = uniq.numel()
            one = torch.ones(ys.numel(), device=dev)
            area = torch.zeros(n, device=dev).scatter_add_(0, inv, one)
            cx = torch.zeros(n, device=dev).scatter_add_(0, inv, xs.float()) / area
            cy = torch.zeros(n, device=dev).scatter_add_(0, inv, ys.float()) / area
            big_n = torch.full((n,), 10 ** 6, device=dev, dtype=torch.long)
            x0 = big_n.clone().scatter_reduce_(0, inv, xs, "amin", include_self=True)
            y0 = big_n.clone().scatter_reduce_(0, inv, ys, "amin", include_self=True)
            x1 = (-big_n).scatter_reduce_(0, inv, xs, "amax", include_self=True)
            y1 = (-big_n).scatter_reduce_(0, inv, ys, "amax", include_self=True)
            w, h = (x1 - x0 + 1), (y1 - y0 + 1)
            area_box = (w * h).float()
            ii_col, ii_red = self._integral(colored), self._integral(red)
            ii_dark, ii_u = self._integral(dark), self._integral(core_s | colored)
            col_frac = self._box(ii_col, x0, y0, x1, y1) / area_box
            red_frac = self._box(ii_red, x0, y0, x1, y1) / area_box
            dark_cnt = self._box(ii_dark, x0, y0, x1, y1)

            def patch(px_, py_):
                px_ = px_.clamp(0, W - 3); py_ = py_.clamp(0, H - 3)
                return self._box(ii_u, px_, py_, px_ + 2, py_ + 2) / 9.0
            corners = torch.stack([patch(x0 + 1, y0 + 1), patch(x1 - 3, y0 + 1),
                                   patch(x0 + 1, y1 - 3), patch(x1 - 3, y1 - 3)], 1).ge(0.3).sum(1)
            dx, dy = cx - float(pos[0]), cy - float(pos[1])
            wf, hf = w.float(), h.float()
            item = (w >= 10) & (w <= 30) & (h >= 10) & (h <= 30) & ((w - h).abs() <= 5) & (col_frac >= 0.15) & (corners >= 3)
            head = (dx.abs() < 22) & (dy > -30) & (dy < -10) & (hf / wf < 0.5)          # thin highlight above the hair
            shoes = (dx.abs() < 14) & (dy > 26) & (dy < 60) & (w <= 20) & (h <= 16)     # white shoes below the skirt
            # background-music title: many small white strokes in one row along the bottom edge
            band = (y0 >= H - 46) & (x0 >= 90) & (h <= 22)
            text = band if int(band.sum()) >= 6 else torch.zeros_like(band)
            drop_c = item | head | shoes | text
            if drop_c.any():
                drop |= torch.isin(labels, uniq[drop_c])
            for i in torch.nonzero(item).flatten().tolist():                            # also remove the letter inside
                xa, xb, ya, yb = int(x0[i]) - 3, int(x1[i]) + 3, int(y0[i]) - 3, int(y1[i]) + 3
                drop[max(ya, 0):yb + 1, max(xa, 0):xb + 1] = True
            if debug is not None:
                debug.update(items=int(item.sum()), head=int(head.sum()), shoes=int(shoes.sum()), text=int(text.sum()), comps=n)
        # the two yin-yang options: in a small window around the player, white + red together form a filled disc
        ry0, ry1 = max(int(pos[1]) - 110, 0), min(int(pos[1]) + 45, H)
        rx0, rx1 = max(int(pos[0]) - 120, 0), min(int(pos[0]) + 120, W)
        if ry1 - ry0 > 20 and rx1 - rx0 > 20:
            uroi = (core_s | red)[ry0:ry1, rx0:rx1]
            rh, rw = uroi.shape
            ids = torch.arange(1, rh * rw + 1, device=dev, dtype=torch.float32).view(1, 1, rh, rw)
            mm = uroi[None, None].float()
            for _ in range(36):
                ids = F.max_pool2d(ids, 3, 1, 1) * mm
            lab2 = ids[0, 0].long()
            yy2, xx2 = torch.nonzero(lab2 > 0, as_tuple=True)
            if yy2.numel():
                uq, iv = torch.unique(lab2[yy2, xx2], return_inverse=True)
                m_ = uq.numel()
                ar = torch.zeros(m_, device=dev).scatter_add_(0, iv, torch.ones(yy2.numel(), device=dev))
                bigv = torch.full((m_,), 10 ** 6, device=dev, dtype=torch.long)
                ax0 = bigv.clone().scatter_reduce_(0, iv, xx2, "amin", include_self=True)
                ay0 = bigv.clone().scatter_reduce_(0, iv, yy2, "amin", include_self=True)
                ax1 = (-bigv).scatter_reduce_(0, iv, xx2, "amax", include_self=True)
                ay1 = (-bigv).scatter_reduce_(0, iv, yy2, "amax", include_self=True)
                ww, hh = ax1 - ax0 + 1, ay1 - ay0 + 1
                red_n = torch.zeros(m_, device=dev).scatter_add_(0, iv, red[ry0:ry1, rx0:rx1][yy2, xx2].float())
                white_n = torch.zeros(m_, device=dev).scatter_add_(0, iv, core_s[ry0:ry1, rx0:rx1][yy2, xx2].float())
                fill = ar / (ww * hh).float()
                ocx = rx0 + (ax0 + ax1).float() / 2 - float(pos[0])
                ocy = ry0 + (ay0 + ay1).float() / 2 - float(pos[1])
                orb = ((ww >= 17) & (ww <= 34) & (hh >= 17) & (hh <= 34) & ((ww - hh).abs() <= 5) & (fill >= 0.6)
                       & (red_n / ar >= 0.18) & (white_n / ar >= 0.18) & (ocx.abs() < 80) & (ocy > -80) & (ocy < 30))
                for i in torch.nonzero(orb).flatten().tolist():
                    ccx, ccy = rx0 + (int(ax0[i]) + int(ax1[i])) / 2, ry0 + (int(ay0[i]) + int(ay1[i])) / 2
                    rad = 0.55 * max(int(ww[i]), int(hh[i])) + 4
                    drop |= (self.xx - ccx) ** 2 + (self.yy - ccy) ** 2 < rad ** 2
                if debug is not None:
                    debug["orbs"] = int(orb.sum())
        keep = core_s & ~drop
        m = F.max_pool2d(keep[None, None].float(), 2 * self.r + 1, 1, self.r)[0, 0] > 0
        dxp, dyp = self.xx - float(pos[0]), self.yy - float(pos[1])
        body = (dxp.abs() < 25) & (dyp > -12) & (dyp < 30)                                # smaller blind box than v1
        return m & ~body

    def bullet_mask_v3(self, t, pos, debug=None):
        """One connected-component pass over (white core | red | blue) pixels. Component statistics (size, how red/blue
        vs how white) decide what is NOT a bullet:
          item   : ~14-24 px square, mostly red/blue with a little white (the letter) -> power-up / point items
          orb    : ~14-24 px disc, red AND white halves, next to the player           -> the two yin-yang options
          head   : thin white highlight above the hair; shoes: small white blobs below the skirt
          text   : many small white strokes in one row along the bottom edge          -> background-music title
        Everything else keeps its white core (dilated). A small blind box around the player remains.
        Version 4 adds (a) bullets WITHOUT a white core (e.g. plain yellow rice): small, bright, saturated pixels much brighter than
        their surroundings (white top-hat of max(R,G,B)); the player's own shots are pale (low saturation) and stay out; and
        (b) lasers: a huge white area that is long and thin is kept as bullets, only roundish ones (boss glow, explosions) go."""
        H, W = PF_H, PF_W
        dev = self.dev
        core = t.amin(dim=2) >= self.thr
        core[:34] = False
        self.core_frac = core.float().mean().item()
        r, g, b = t[..., 0].short(), t[..., 1].short(), t[..., 2].short()
        red = (r > 160) & (g < 110) & (b < 110)
        blue = (b > 160) & (r < 110) & (g < 150)
        v4 = self.version >= 4
        if v4:
            V = t.amax(dim=2).float()
            sat = (V - t.amin(dim=2).float()) / V.clamp(min=1)
            op = F.max_pool2d(-F.max_pool2d(-V[None, None], 15, 1, 7), 15, 1, 7)[0, 0]       # opening: the local background
            colour = ((V - op) >= COL_TOPHAT) & (V >= COL_V) & (sat >= COL_SAT) & ~core
            colour[:34] = False
            c_ = colour[None, None].float()                       # 3x3 opening: thin background lines and specks go
            colour = (F.max_pool2d(-F.max_pool2d(-c_, 3, 1, 1), 3, 1, 1)[0, 0] > 0) & colour
        else:
            colour = torch.zeros_like(core)
        m = core[None, None].float()
        k = self.big * 2 + 1                                    # huge glows (boss / explosions) are not bullets
        eroded = -F.max_pool2d(-m, k, 1, self.big)
        big = F.max_pool2d(eroded, k + 6, 1, self.big + 3)[0, 0] > 0
        if v4 and big.any():
            big = big & ~self._laser_region(big)                # ... but lasers are
        core_s = core & ~big
        u = core_s | red | blue | colour
        labels = self._ccl(u, 36)
        ys, xs = torch.nonzero(labels > 0, as_tuple=True)
        drop = torch.zeros_like(core_s)
        px, py = float(pos[0]), float(pos[1])
        self.items = []
        if ys.numel():
            lab = labels[ys, xs]
            uniq, inv = torch.unique(lab, return_inverse=True)
            n = uniq.numel()
            one = torch.ones(ys.numel(), device=dev)
            area = torch.zeros(n, device=dev).scatter_add_(0, inv, one)
            white_n = torch.zeros(n, device=dev).scatter_add_(0, inv, core_s[ys, xs].float())
            red_n = torch.zeros(n, device=dev).scatter_add_(0, inv, red[ys, xs].float())
            blue_n = torch.zeros(n, device=dev).scatter_add_(0, inv, blue[ys, xs].float())
            col_n = torch.zeros(n, device=dev).scatter_add_(0, inv, colour[ys, xs].float())
            big_n = torch.full((n,), 10 ** 6, device=dev, dtype=torch.long)
            x0 = big_n.clone().scatter_reduce_(0, inv, xs, "amin", include_self=True)
            y0 = big_n.clone().scatter_reduce_(0, inv, ys, "amin", include_self=True)
            x1 = (-big_n).scatter_reduce_(0, inv, xs, "amax", include_self=True)
            y1 = (-big_n).scatter_reduce_(0, inv, ys, "amax", include_self=True)
            w, h = x1 - x0 + 1, y1 - y0 + 1
            wf_, hf_ = w.float(), h.float()
            fill = area / (wf_ * hf_)
            wh, rd, bl = white_n / area, red_n / area, blue_n / area
            ccx, ccy = (x0 + x1).float() / 2 - px, (y0 + y1).float() / 2 - py
            sq = lambda lo, hi, tol: (w >= lo) & (w <= hi) & (h >= lo) & (h <= hi) & ((w - h).abs() <= tol)
            item = sq(13, 26, 4) & (fill >= 0.5) & ((rd + bl) >= 0.42) & (wh <= 0.2)
            # a yin-yang option is half red / half white, so its red pixels sit off-centre; a red-rimmed white RING BULLET has its red
            # centred. (The old rule deleted every red-and-white disc within 85 px of the player, i.e. ring bullets right before they hit.)
            redf = red[ys, xs].float()
            rcx = torch.zeros(n, device=dev).scatter_add_(0, inv, redf * xs.float()) / red_n.clamp(min=1)
            rcy = torch.zeros(n, device=dev).scatter_add_(0, inv, redf * ys.float()) / red_n.clamp(min=1)
            asym = torch.sqrt((rcx - (x0 + x1).float() / 2) ** 2 + (rcy - (y0 + y1).float() / 2) ** 2) / torch.maximum(wf_, hf_) > 0.12
            # ... or it sits on the options' own short path between (+-36.5, 0) (normal) and (+-13, -49) (focused)
            tt_ = (((ccx.abs() - 36.5) * -23.5 + ccy * -49.0) / (23.5 ** 2 + 49.0 ** 2)).clamp(0, 1)
            on_path = ((ccx.abs() - 36.5 + tt_ * 23.5) ** 2 + (ccy + tt_ * 49.0) ** 2) < 14.0 ** 2
            orb = (sq(14, 26, 5) & (fill >= 0.55) & (rd >= 0.2) & (rd <= 0.75) & (wh >= 0.25) & (wh <= 0.8) & (asym | on_path)
                   & (ccx.abs() < 85) & (ccy > -85) & (ccy < 35))
            # track the two options relative to the player: they drift slowly, so a frame where one is fused with the
            # red bow (no clean disc) can still be masked from its last known place
            seen = [(float(ccx[i]), float(ccy[i])) for i in torch.nonzero(orb).flatten().tolist()]
            for sx, sy in seen:
                best = min(self.orb_rel, key=lambda o: (o[0] - sx) ** 2 + (o[1] - sy) ** 2, default=None)
                if best is not None and (best[0] - sx) ** 2 + (best[1] - sy) ** 2 < 28 ** 2:
                    best[0], best[1], best[2] = sx, sy, 0
                elif len(self.orb_rel) < 2:
                    self.orb_rel.append([sx, sy, 0])
            for o in self.orb_rel:
                o[2] += 1
            self.orb_rel = [o for o in self.orb_rel if o[2] <= 25]
            for o in self.orb_rel:
                drop |= (self.xx - (px + o[0])) ** 2 + (self.yy - (py + o[1])) ** 2 < 16 ** 2
            head = (ccx.abs() < 22) & (ccy > -30) & (ccy < -10) & (hf_ / wf_ < 0.5) & (wh >= 0.6)
            shoes = (ccx.abs() < 20) & (ccy > 24) & (ccy < 56) & (w <= 14) & (h <= 14) & (wh >= 0.6)
            band = (y0 >= H - 46) & (x0 >= 90) & (h <= 22)
            if v4:
                # v3 dropped EVERY white blob in the bottom band once there were 6 of them, i.e. also rows of bullets right where
                # the player usually is. The music title is different: it does not move (same pixels as in the last frame) and
                # its strokes sit in one row; only such blobs (and the colour outline next to them) count as text.
                prev = self.prev_core if self.prev_core is not None else torch.zeros_like(core)
                still = torch.zeros(n, device=dev).scatter_add_(0, inv, (core_s & prev)[ys, xs].float()) / white_n.clamp(min=1)
                cand = band & (wh >= 0.6) & (still >= 0.85)
                text = torch.zeros_like(band)
                if int(cand.sum()) >= 6:
                    yc = (y0 + y1).float() / 2
                    row = cand & ((yc - yc[cand].median()).abs() <= 12)
                    if int(row.sum()) >= 6:            # the whole text row (strokes that are still fading in included)
                        self.text_box = [int(x0[row].min()), int(x1[row].max()), float(yc[cand].median()), 45]
                if self.text_box is not None:          # ... and for 1.5 s after it was last seen (it fades out unevenly)
                    xr0, xr1, ym, ttl = self.text_box
                    yc = (y0 + y1).float() / 2
                    text = band & (x0 >= xr0 - 8) & (x1 <= xr1 + 8) & ((yc - ym).abs() <= 14)
                    self.text_box[3] -= 1
                    if self.text_box[3] <= 0:
                        self.text_box = None
                if debug is not None:
                    debug.update(text_still=int(cand.sum()))
            else:
                text = band & ((wh >= 0.6) | (white_n == 0)) if int((band & (wh >= 0.6)).sum()) >= 6 else torch.zeros_like(band)
            drop_c = item | orb | head | shoes | text
            if v4:
                # colour-only blobs that are long or huge: bright background lines / glows, not bullets
                drop_c = drop_c | ((white_n == 0) & (col_n > 0) & ((torch.maximum(w, h) > COL_MAXLEN) | (area > COL_MAXAREA)))
                # Reimu's red bow just above the blind box
                drop_c = drop_c | ((ccx.abs() < 22) & (ccy > -44) & (ccy < -8) & (rd >= 0.4) & (wh <= 0.2) & (w <= 36) & (h <= 28))
            if drop_c.any():
                drop |= torch.isin(labels, uniq[drop_c])
            for i in torch.nonzero(item).flatten().tolist():
                self.items.append((float((x0[i] + x1[i]) / 2) / S, float((y0[i] + y1[i]) / 2) / S, bool(rd[i] > bl[i])))
            if debug is not None:
                debug.update(items=int(item.sum()), orbs=int(orb.sum()), head=int(head.sum()), shoes=int(shoes.sum()),
                             text=int(text.sum()), comps=n)
                near = (ccx.abs() < 100) & (ccy > -95) & (ccy < 70)
                debug["near"] = [dict(w=int(w[i]), h=int(h[i]), fill=round(float(fill[i]), 2), white=round(float(wh[i]), 2),
                                      red=round(float(rd[i]), 2), blue=round(float(bl[i]), 2), dx=round(float(ccx[i])),
                                      dy=round(float(ccy[i])), item=bool(item[i]), orb=bool(orb[i]))
                                 for i in torch.nonzero(near).flatten().tolist()]
        self.prev_core = core_s
        keep = core_s & ~drop
        mm = F.max_pool2d(keep[None, None].float(), 2 * self.r + 1, 1, self.r)[0, 0] > 0
        if v4:                         # a colour body is the sprite itself: grow it by 1 px only (a white core is grown to the rim)
            mm |= F.max_pool2d((colour & ~drop)[None, None].float(), 3, 1, 1)[0, 0] > 0
        dxp, dyp = self.xx - px, self.yy - py
        body = (dxp.abs() < 29) & (dyp > -16) & (dyp < 34)        # blind box (the dilation spills ~3 px past the sprite)
        return mm & ~body

    def bullet_mask_v5(self, t, pos, debug=None):
        """For the game in AI mode (black background, Reimu and her options / shots invisible): everything clearly brighter than
        the black background is an obstacle -- bullets of any colour, big balls, lasers, the boss. No blind zone. Removed:
        dim blobs (spell-start aura, the dark spikes of the focus marker), power / point items, the static music-title text,
        the HUD rows at the top, and a 6 px disc on the hitbox (the focus marker's bright centre dot)."""
        H, W = PF_H, PF_W
        dev = self.dev
        V = t.amax(dim=2)
        if self.tint_q > 0:
            # a whole-screen tint lifts the darkest pixels too; a screen full of pale bullets still has black gaps. (The median rose to
            # 80-137 under Remilia's pale bullets and the threshold with it -> they vanished; at 255 the old uint8 compare wraps round.)
            base = float(torch.quantile(V[34::4, ::4].float().flatten(), self.tint_q).item())
            lit = V.short() >= min(256, int(max(V5_LIT, base + V5_OVER_MEDIAN)))
        else:
            med = int(V[34::4, ::4].float().median().item())          # ~0 on the black background
            lit = V >= max(V5_LIT, med + V5_OVER_MEDIAN)
        lit[:34] = False
        core = t.amin(dim=2) >= self.thr
        core[:34] = False
        self.core_frac = core.float().mean().item()
        r, g, b = t[..., 0].short(), t[..., 1].short(), t[..., 2].short()
        red = (r > 160) & (g < 110) & (b < 110)
        blue = (b > 160) & (r < 110) & (g < 150)
        labels = self._ccl(lit, 64)
        ys, xs = torch.nonzero(labels > 0, as_tuple=True)
        drop = torch.zeros_like(lit)
        px, py = float(pos[0]), float(pos[1])
        self.items = []
        if ys.numel():
            lab = labels[ys, xs]
            uniq, inv = torch.unique(lab, return_inverse=True)
            n = uniq.numel()
            acc = lambda v: torch.zeros(n, device=dev).scatter_add_(0, inv, v)
            area = acc(torch.ones(ys.numel(), device=dev))
            peak = torch.zeros(n, device=dev).scatter_reduce_(0, inv, V[ys, xs].float(), "amax", include_self=True)
            wh, rd, bl = acc(core[ys, xs].float()) / area, acc(red[ys, xs].float()) / area, acc(blue[ys, xs].float()) / area
            big_n = torch.full((n,), 10 ** 6, device=dev, dtype=torch.long)
            x0 = big_n.clone().scatter_reduce_(0, inv, xs, "amin", include_self=True)
            y0 = big_n.clone().scatter_reduce_(0, inv, ys, "amin", include_self=True)
            x1 = (-big_n).scatter_reduce_(0, inv, xs, "amax", include_self=True)
            y1 = (-big_n).scatter_reduce_(0, inv, ys, "amax", include_self=True)
            w, h = x1 - x0 + 1, y1 - y0 + 1
            fill = area / (w.float() * h.float())
            sq = (w >= 13) & (w <= 26) & (h >= 13) & (h <= 26) & ((w - h).abs() <= 4)
            item = sq & (fill >= 0.5) & ((rd + bl) >= 0.42) & (wh <= 0.2)
            dim = peak < V5_PEAK
            if self.pale_peak > 0:
                dim = dim & ~((peak >= self.pale_peak) & (area >= PALE_AREA))
            # music title: blobs in the bottom band whose pixels did not change at all since the last frame. On black the
            # strokes and their outline merge into a few wide blobs, so (unlike v4) 2 such blobs in one row are enough.
            band = (y0 >= H - 46) & (x0 >= 90) & (h <= 24)
            pv = self.prev_v if self.prev_v is not None else torch.full_like(V, -100)
            same = lit & ((V.short() - pv.short()).abs() <= 3)
            still = acc(same[ys, xs].float()) / area
            cand = band & (still >= 0.8) & (area >= 12)
            text = torch.zeros_like(band)
            yc = (y0 + y1).float() / 2
            if int(cand.sum()) >= 2:
                row = cand & ((yc - yc[cand].median()).abs() <= 12)
                if int(row.sum()) >= 2 and int((x1[row].max() - x0[row].min()).item()) >= 60:
                    self.text_box = [int(x0[row].min()), int(x1[row].max()), float(yc[cand].median()), 45]
            if self.text_box is not None:
                xr0, xr1, ym, ttl = self.text_box
                text = band & (x0 >= xr0 - 8) & (x1 <= xr1 + 8) & ((yc - ym).abs() <= 14)
                self.text_box[3] -= 1
                if self.text_box[3] <= 0:
                    self.text_box = None
            drop_c = item | dim | text
            if drop_c.any():
                drop = torch.isin(labels, uniq[drop_c])
            for i in torch.nonzero(item).flatten().tolist():
                self.items.append((float((x0[i] + x1[i]) / 2) / S, float((y0[i] + y1[i]) / 2) / S, bool(rd[i] > bl[i])))
            if debug is not None:
                debug.update(comps=n, items=int(item.sum()), dim=int(dim.sum()), text=int(text.sum()))
        self.prev_core = core
        self.prev_v = V
        keep = lit & ~drop
        mm = F.max_pool2d(keep[None, None].float(), 3, 1, 1)[0, 0] > 0
        far = (self.xx - px) ** 2 + (self.yy - py) ** 2 > V5_HIT_R ** 2
        if self.tracker is not None:                                      # --predict: reuse these blobs, without the focus marker
            self.track_keep = keep & far
            self.track_area = area if ys.numel() else None
            if ys.numel():
                sel = ~drop_c[inv] & far[ys, xs]
                dot_ = ~drop_c & (area >= TRACK_AREA[0]) & (area <= TRACK_AREA[1]) & (w <= TRACK_SPAN) & (h <= TRACK_SPAN)
                sel = sel.nonzero()[:, 0]
                self.track_blobs = torch.stack([acc(xs.float()) / area, acc(ys.float()) / area], 1) / 2, dot_, (ys[sel], xs[sel], inv[sel])
            else:
                self.track_blobs = torch.zeros(0, 2, device=dev), torch.zeros(0, dtype=torch.bool, device=dev), (ys, xs, ys)
        return mm & far

    def future_channels(self, seen, pos):
        """Where the tracked bullets will go during the next second, drawn as in the simulator (future_env.future_maps). seen: the player
        in the picture (screen px), pos: the centre of the local view. Counted from the moment the policy sees, `lead` frames ahead."""
        from future_env import future_maps, TAU
        T = self.tracker
        p = torch.tensor([[pos[0] / S, pos[1] / S]], device=self.dev, dtype=torch.float32)
        tau = torch.tensor(TAU).long() + int(round(self.lead)) - 1
        det = T.det.to(self.dev)
        d = T.path(int(tau.max()) + 1, self.track_model)[:, tau].to(self.dev)
        xy = (det[:, None] + d) * 2 / S                                   # half-size px -> game px
        near = (det[:, 0] * 2 - float(seen[0])) ** 2 + (det[:, 1] * 2 - float(seen[1])) ** 2 <= V5_HIT_R ** 2     # the focus marker's dot
        e = (T.known.to(self.dev) & ~near).nonzero()[:, 0]
        rad = torch.sqrt(self.track_area[T.dot_idx][e] / math.pi) / S if e.numel() else torch.zeros(0, device=self.dev)
        return future_maps(1, torch.zeros_like(e), xy[e, :, 0], xy[e, :, 1], rad, p[:, 0], p[:, 1])

    def tracked_paths(self, frames):
        """For shield.py -> the tracked bullets' positions at the next frames [M, frames, 2] and their radii [M], game px. The focus marker's
        dot is not a bullet."""
        T = self.tracker
        if T.det is None or T.det.shape[0] == 0 or not bool(T.known.any()):
            return None, None
        det = T.det.to(self.dev)
        xy = (det[:, None] + T.path(frames, self.track_model).to(self.dev)) * 2 / S
        near = (det[:, 0] * 2 - float(self.pos[0])) ** 2 + (det[:, 1] * 2 - float(self.pos[1])) ** 2 <= V5_HIT_R ** 2
        e = (T.known.to(self.dev) & ~near).nonzero()[:, 0]
        return xy[e], torch.sqrt(self.track_area[T.dot_idx][e] / math.pi) / S

    def _laser_region(self, big):
        """big: bool [H, W] huge white areas. Returns the part of it that belongs to long thin shapes (lasers), judged on an 8x
        coarser grid (cheap connected components that span the whole field) by the ratio of the two principal axes."""
        g = F.max_pool2d(big[None, None].float(), 8, 8)[0, 0]                 # 84 x 72
        gh, gw = g.shape
        ids = (torch.arange(1, gh * gw + 1, device=self.dev, dtype=torch.float32).view(gh, gw) * g)[None, None]
        for _ in range(100):
            nxt = F.max_pool2d(ids, 3, 1, 1) * g
            if torch.equal(nxt, ids):
                break
            ids = nxt
        lab = ids[0, 0].long()
        ys, xs = torch.nonzero(lab > 0, as_tuple=True)
        out = torch.zeros_like(big)
        if not ys.numel():
            return out
        uniq, inv = torch.unique(lab[ys, xs], return_inverse=True)
        n = uniq.numel()
        acc = lambda v: torch.zeros(n, device=self.dev).scatter_add_(0, inv, v)
        cnt = acc(torch.ones_like(xs, dtype=torch.float32))
        xf, yf = xs.float(), ys.float()
        mx, my = acc(xf) / cnt, acc(yf) / cnt
        sxx, syy, sxy = acc(xf * xf) / cnt - mx ** 2, acc(yf * yf) / cnt - my ** 2, acc(xf * yf) / cnt - mx * my
        tr, det = sxx + syy, sxx * syy - sxy ** 2
        disc = (tr ** 2 / 4 - det).clamp(min=0).sqrt()
        l1, l2 = tr / 2 + disc, (tr / 2 - disc).clamp(min=0)
        # long axis of a uniform bar = 12^0.5 * sqrt(l1) cells; laser = at least LASER_ELONG times longer than wide and >= LASER_LEN px
        laser = ((l1 + 1 / 12) / (l2 + 1 / 12) >= LASER_ELONG ** 2) & ((12 * (l1 + 1 / 12)).sqrt() * 8 >= LASER_LEN)
        if laser.any():
            keep = torch.isin(lab, uniq[laser]).float()
            out = F.interpolate(keep[None, None], scale_factor=8, mode="nearest")[0, 0, :big.shape[0], :big.shape[1]] > 0
        return out & big

    def own_shot_filter(self, m, pos):
        """Remove the player's own amulet shots by motion. They start at the player and fly straight up ~45 screen px per
        decision (15 game px per frame); enemy bullets move at most ~14 px per decision. Above the player a blob is an own shot if
        nothing occupied its neighbourhood (<= 14 px) in the previous frame AND either something was 30..75 px directly below it
        in the previous frame (it moved up fast) or it sits in the spawn zone right above the player (a shot that just appeared).
        A real bullet always has its own previous position nearby, so it is kept. m: bool mask [H, W]."""
        prev, self.prev_raw = self.prev_raw, m
        if prev is None:
            self.last_removed = torch.zeros_like(m)
            return m
        pf = prev[None, None].float()
        near = F.max_pool2d(pf, 29, 1, 14)[0, 0] > 0                    # something within 14 px a decision ago
        up = torch.zeros_like(m)
        for dy in range(30, 76, 3):                                    # was dy px lower a decision ago
            up[:-dy] |= prev[dy:]
        up = F.max_pool2d(up[None, None].float(), (1, 25), 1, (0, 12))[0, 0] > 0      # homing shots drift sideways a little
        dxp, dyp = self.xx - float(pos[0]), self.yy - float(pos[1])
        above = dyp < -10
        spawn = (dxp.abs() < 30) & (dyp > -95) & (dyp < -10)
        own = m & ~near & above & (up | spawn)
        self.last_removed = own
        return m & ~own

    def in_dialogue(self, thr=0.04, portrait_thr=0.08, white_rule=False):
        """Story scene = Reimu's big red-and-black portrait in the bottom-left corner (red fraction of that region ~0.19 in every
        dialogue frame vs ~0.001 normally). The old extra rule 'white pixels > 4% of the screen' is OFF by default: it also fired
        when a spell turns the whole screen of bullets white (Cirno's Perfect Freeze) and at every spell-card cut-in, and the
        loop then released all keys and stood still (25% of the stage-2 boss deaths happened inside such a freeze)."""
        return self.portrait_frac > portrait_thr or (white_rule and self.core_frac > thr)

    # ------------------------------------------------------------------ observation
    def observe(self, pf, focus, reset=False, mem_xy=None, extra_vec=(), grab_t=None, view_xy=None, mask_override=None):
        """mem_xy: exact player (x, y) in game px read from the game's memory; if None, fall back to vision.
        grab_t, view_xy (--predict): the time of the grab in seconds; where the player will be `lead` frames ahead (the focus marker is
        drawn where it is now, so the mask is cleaned around mem_xy first and around view_xy again after the bullets have moved).
        mask_override (play.py --memperc): the bullet mask drawn from the game's memory (memperception.py); pf is then only used for the
        dialogue check and may be a few steps old."""
        t = torch.from_numpy(np.ascontiguousarray(pf)).to(self.dev)
        reg = t[440:672, 0:300].short()
        self.portrait_frac = ((reg[..., 0] > 150) & (reg[..., 1] < 90) & (reg[..., 2] < 90)).float().mean().item()
        if mem_xy is not None:
            self.pos = np.array([mem_xy[0] * S, mem_xy[1] * S])        # hitbox centre in playfield screen px
            pos = self.pos
        else:
            pos = self.locate_player(t)
        mask = (mask_override if mask_override is not None else
                self.bullet_mask_v5(t, pos) if self.version == 5 else
                self.bullet_mask_v3(t, pos) if self.version >= 3 else
                self.bullet_mask_v2(t, pos) if self.version == 2 else self.bullet_mask(t, pos))
        if self.shot_filter:
            if reset:
                self.prev_raw = None
            mask = self.own_shot_filter(mask, pos)
        if self.tracker is not None:
            if reset:
                self.tracker.reset()
            t_ = time.perf_counter()
            self.tracker.update(None, grab_t, self.track_blobs)
            self.t_update = time.perf_counter() - t_
            if self.lead > 0:
                mask = F.max_pool2d(self.tracker.shift(self.track_keep, self.lead, self.track_model, 2)[None, None].float(), 3, 1, 1)[0, 0] > 0
            self.t_shift = time.perf_counter() - t_ - self.t_update
            seen = pos
            if view_xy is not None:
                self.pos = np.array([view_xy[0] * S, view_xy[1] * S])
                pos = self.pos
            if self.lead > 0:
                mask = mask & ((self.xx - float(pos[0])) ** 2 + (self.yy - float(pos[1])) ** 2 > V5_HIT_R ** 2)
            fut = self.future_channels(seen, pos) if self.future else None
            self.t_future = time.perf_counter() - t_ - self.t_update - self.t_shift
        px_g, py_g = pos[0] / S, pos[1] / S
        m = mask[None, None].float()
        glb = F.adaptive_max_pool2d(m, (GLB_H, GLB_W))[0, 0]                 # whole field, 8 game px / cell
        half = 96                                                            # 128 game px window = 192 screen px
        padded = F.pad(m, (half, half, half, half))
        cx, cy = int(round(pos[0])) + half, int(round(pos[1])) + half
        win = padded[:, :, cy - half:cy + half, cx - half:cx + half]
        loc = F.adaptive_max_pool2d(win, (LOC, LOC))[0, 0]
        loc, glb = (loc > 0.5).to(torch.uint8), (glb > 0.5).to(torch.uint8)
        if self.lags:
            if reset or not self.hist:
                self.hist = [(loc, glb)] * (self.lags[0] + 1)
            else:
                self.hist = self.hist[1:] + [(loc, glb)]
            self.loc_stack = torch.stack([self.hist[-1 - L][0] for L in self.lags], 0)
            self.glb_stack = torch.stack([self.hist[-1 - L][1] for L in self.lags], 0)
        elif reset or self.loc_stack is None:
            self.loc_stack = loc[None].repeat(self.stack, 1, 1)
            self.glb_stack = glb[None].repeat(self.stack, 1, 1)
        else:
            self.loc_stack = torch.cat([self.loc_stack[1:], loc[None]], 0)
            self.glb_stack = torch.cat([self.glb_stack[1:], glb[None]], 0)
        ax = torch.arange(LOC, device=self.dev).float() + 0.5 - LOC / 2
        wx, wy = px_g + ax * LOC_CELL, py_g + ax * LOC_CELL
        wall = (((wy < 0) | (wy > GAME_H))[:, None] | ((wx < 0) | (wx > GAME_W))[None, :]).to(torch.uint8)
        marker = torch.zeros(GLB_H, GLB_W, dtype=torch.uint8, device=self.dev)
        marker[int(min(max(py_g // 8, 0), GLB_H - 1)), int(min(max(px_g // 8, 0), GLB_W - 1))] = 1
        vec = torch.tensor([px_g / GAME_W * 2 - 1, py_g / GAME_H * 2 - 1, float(focus), *extra_vec],
                           device=self.dev, dtype=torch.float32)
        loc_, glb_ = [self.loc_stack, wall[None]], [self.glb_stack, marker[None]]
        if self.future:
            loc_.append(fut[0][0]); glb_.append(fut[1][0])
        obs = dict(loc=torch.cat(loc_, 0)[None], glob=torch.cat(glb_, 0)[None], vec=vec[None])
        return obs, (px_g, py_g, mask)


def overlay(pf, mask, pos, removed=None):
    """Debug picture: playfield with detected bullets painted green (and pixels dropped as own shots magenta), player marked."""
    img = np.ascontiguousarray(pf).copy()
    m = mask.cpu().numpy()
    img[m] = (0.4 * img[m] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
    if removed is not None:
        r = removed.cpu().numpy()
        img[r] = (0.4 * img[r] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
    px, py = int(pos[0]), int(pos[1])
    img[max(py - 1, 0):py + 2, max(px - 8, 0):px + 9] = (255, 255, 0)
    img[max(py - 8, 0):py + 9, max(px - 1, 0):px + 2] = (255, 255, 0)
    return img
