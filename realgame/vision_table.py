"""Vision-only bullet table for the reflex layer (2026-11-06; user: "can the pure-vision scheme + the reflex layer clear the game?"). From the picture alone (AI-mode
textures: black background, Reimu and her shots invisible) -> the table reflex.Reflex wants: bullets x y vx vy + hitbox. Nothing from
the game's memory except the player's position (the standing rule).
 * blobs: v5's lit pixels (max(R,G,B) >= 80 and above the frame's tint level) on the HALF-size picture; HUD rows, dim blobs (peak < 150),
   power / point items and the focus marker's dot (a disc on the player) dropped
 * motion: tracker.BulletTracker on the dot-sized blobs (straight line; its own clock from the bullets, no frame counter needed)
 * hitbox: from the blob's lit area (hb_from_area, calibrated against the memory tables by vision_eval.py)
 * everything else lit (big blobs = boss / lasers / merged clusters, and dots whose motion is not known yet): every lit pixel a still
   obstacle (1 half-size px = 1.33 game px)
Units: half-size px (h) -> game px: * 2 / 1.5."""
import numpy as np
import torch
import torch.nn.functional as F
from tracker import BulletTracker

G_PER_H = 2.0 / 1.5                       # game px per half-size px
HUD_ROWS = 17                             # (34 screen rows)
LIT, PEAK, OVER = 80, 150, 60
HIT_R_H = 4.0                             # half-size px around the player: the focus marker's bright dot


def blobs(img, tint_q=0.1, ids=None, device="cpu"):
    """img uint8 [H, W, 3] half-size -> (lit-and-kept bool [H, W], per blob: centroid xy [n, 2] (half px), area [n], w, h, peak, label map)
    (the pixels on `device`; the per-blob values come back on the CPU)"""
    T = torch.from_numpy(np.ascontiguousarray(img)).to(device)
    V = T.amax(2)
    base = float(torch.quantile(V[HUD_ROWS::2, ::2].float().flatten(), tint_q))
    lit = V.short() >= min(256, int(max(LIT, base + OVER)))
    lit[:HUD_ROWS] = False
    H, W = lit.shape
    if ids is None:
        ids = torch.arange(1, H * W + 1, dtype=torch.float32, device=device).view(1, 1, H, W)
    g = lit[None, None].float()
    lab = ids * g
    for i in range(20):
        lab = F.max_pool2d(lab, 3, 1, 1) * g
        if i % 2:
            f = lab.view(-1)
            lab = (f[(f.long() - 1).clamp(min=0)] * (f > 0)).view_as(lab)
    ys, xs = torch.nonzero(lit, as_tuple=True)
    if ys.numel() == 0:
        e = torch.zeros(0)
        return lit.cpu(), torch.zeros(0, 2), e, e, e, e, (ys, xs, ys, e.bool())
    uniq, inv = torch.unique(lab[0, 0, ys, xs].long(), return_inverse=True)
    n = uniq.numel()
    dv = lit.device
    add = lambda v: torch.zeros(n, device=dv).scatter_add_(0, inv, v)
    area = add(torch.ones(ys.numel(), device=dv))
    xf, yf = xs.float(), ys.float()
    c = torch.stack([add(xf) / area, add(yf) / area], 1)
    mx = lambda v: torch.full((n,), -1e6, device=dv).scatter_reduce_(0, inv, v, "amax", include_self=True)
    mn = lambda v: torch.full((n,), 1e6, device=dv).scatter_reduce_(0, inv, v, "amin", include_self=True)
    w, h = mx(xf) - mn(xf) + 1, mx(yf) - mn(yf) + 1
    peak = mx(V[ys, xs].float())
    rgb = T[ys, xs].short()
    red = ((rgb[:, 0] > 160) & (rgb[:, 1] < 110) & (rgb[:, 2] < 110)).float()
    blue = ((rgb[:, 2] > 160) & (rgb[:, 0] < 110) & (rgb[:, 1] < 150)).float()
    wh = (rgb.amin(1) >= 200).float()
    rd, bl, white = add(red) / area, add(blue) / area, add(wh) / area
    sq = (w >= 6) & (w <= 13) & (h >= 6) & (h <= 13) & ((w - h).abs() <= 2)
    item = sq & (area / (w * h) >= 0.5) & ((rd + bl) >= 0.42) & (white <= 0.2)
    drop = item | (peak < PEAK)
    keep_blob = ~drop
    keep = lit.clone()
    keep[ys[drop[inv]], xs[drop[inv]]] = False
    cpu = lambda v: v[keep_blob].cpu()
    return keep.cpu(), cpu(c), cpu(area), cpu(w), cpu(h), cpu(peak), (ys, xs, inv, keep_blob)


class VisionTable:
    """per picture: update(img, t, player_xy) -> table for reflex.Reflex (bullets with 'hb' = full hitbox width in game px)."""

    def __init__(self, hb_fn=None, max_area=170, max_span=18, still_pixels=True, device="cpu"):
        self.tr = BulletTracker(device="cpu", max_area=max_area, max_span=max_span)
        self.hb_fn = hb_fn or hb_from_area
        self.still_pixels = still_pixels
        self.ids = None
        self.last = None
        self.dev = device
        self.mov_area = np.zeros(0, np.float32)

    def reset(self):
        self.tr.reset()

    def update(self, img, t, player_xy):
        H, W = img.shape[:2]
        if self.ids is None:
            self.ids = torch.arange(1, H * W + 1, dtype=torch.float32, device=self.dev).view(1, 1, H, W)
        keep, c, area, w, h, peak, pix = blobs(img, ids=self.ids, device=self.dev)
        px, py = player_xy[0] / G_PER_H, player_xy[1] / G_PER_H
        if len(c):
            near = (c[:, 0] - px) ** 2 + (c[:, 1] - py) ** 2 <= HIT_R_H ** 2
            small = area <= 12
            ok = ~(near & small)                                             # the focus marker's dot
            c, area, w, h, peak = c[ok], area[ok], w[ok], h[ok], peak[ok]
        dot = (area >= self.tr.min_area) & (area <= self.tr.max_area) & (w <= self.tr.max_span) & (h <= self.tr.max_span)
        self.tr.update(None, t, (c, dot, (None, None, None)))
        di = self.tr.dot_idx
        known = self.tr.known
        x = c[di][known] * G_PER_H
        v = self.tr.v_line[known] * G_PER_H
        a_ = area[di][known]
        self.mov_area = a_.numpy()                                            # (the first len(mov_area) rows of the table are the moving dots)
        bx =[x[:, 0].numpy()]; by = [x[:, 1].numpy()]; bvx = [v[:, 0].numpy()]; bvy = [v[:, 1].numpy()]
        bhb = [self.hb_fn(a_.numpy())]
        if self.still_pixels:                                                 # the rest of the lit pixels: still 1-px obstacles
            ysx = torch.nonzero(keep, as_tuple=True)
            m = torch.zeros(H, W, dtype=torch.bool)
            dc = c[di][known]                                                 # remove the moving dots' own pixels (a disc of their size)
            if len(dc):
                r = torch.sqrt(a_ / np.pi) + 1.0
                yy, xx = ysx[0].float(), ysx[1].float()
                d2 = (xx[:, None] - dc[None, :, 0]) ** 2 + (yy[:, None] - dc[None, :, 1]) ** 2
                own = (d2 <= (r[None] ** 2)).any(1)
            else:
                own = torch.zeros(len(ysx[0]), dtype=torch.bool)
            yy, xx = ysx[0][~own].float(), ysx[1][~own].float()
            far = (xx - px) ** 2 + (yy - py) ** 2 > HIT_R_H ** 2
            yy, xx = yy[far], xx[far]
            bx.append(((xx + 0.5) * G_PER_H).numpy()); by.append(((yy + 0.5) * G_PER_H).numpy())
            bvx.append(np.zeros(len(xx), np.float32)); bvy.append(np.zeros(len(xx), np.float32))
            bhb.append(np.full(len(xx), G_PER_H, np.float32))
        tab = dict(bullets=dict(x=np.concatenate(bx).astype(np.float32), y=np.concatenate(by).astype(np.float32),
                                vx=np.concatenate(bvx).astype(np.float32), vy=np.concatenate(bvy).astype(np.float32),
                                hb=np.concatenate(bhb).astype(np.float32)),
                   enemies=dict(x=np.zeros(0, np.float32), y=np.zeros(0, np.float32), w=np.zeros(0, np.float32), h=np.zeros(0, np.float32)),
                   lasers=dict(x=np.zeros(0, np.float32)))
        self.last = dict(n_moving=int(known.sum()), n_dots=int(dot.sum()), n_blobs=len(c), frames=self.tr.step_frames)
        return tab


HB_AREA = None                            # (area upper bounds, hitbox) from vision_eval.py --calib; None -> the rule below


def hb_from_area(area):
    """full hitbox width (game px) of a moving dot from its lit area (half-size px). Default (before calibration): the drawn disc's
    diameter times 0.5, at least 4 -- the templates' hitbox is about half the sprite (16 px sprite: hitbox 6)."""
    area = np.asarray(area, np.float32)
    if HB_AREA is not None:
        ub, hb = HB_AREA
        return np.asarray(hb, np.float32)[np.minimum(np.searchsorted(ub, area), len(hb) - 1)]
    d = 2 * np.sqrt(area / np.pi) * G_PER_H
    return np.maximum(4.0, 0.5 * d).astype(np.float32)


def _ccl(mask, ids, iters=20):
    """connected components of a bool [H, W] mask (8-neighbourhood, max-pool + pointer jumping) -> label map [H, W] (0 = off)."""
    g = mask[None, None].float()
    lab = ids * g
    for i in range(iters):
        lab = F.max_pool2d(lab, 3, 1, 1) * g
        if i % 2:
            f = lab.view(-1)
            lab = (f[(f.long() - 1).clamp(min=0)] * (f > 0)).view_as(lab)
    return lab[0, 0]


def _stats(ys, xs, lab):
    """per component of the pixels (ys, xs) with labels lab[ys, xs] -> inv, n, area, centroid [n, 2], w, h, minor axis (px)."""
    uniq, inv = torch.unique(lab[ys, xs].long(), return_inverse=True)
    n = uniq.numel()
    dv = ys.device
    add = lambda v: torch.zeros(n, device=dv).scatter_add_(0, inv, v)
    xf, yf = xs.float(), ys.float()
    area = add(torch.ones_like(xf))
    cx, cy = add(xf) / area, add(yf) / area
    mx = lambda v: torch.full((n,), -1e6, device=dv).scatter_reduce_(0, inv, v, "amax", include_self=True)
    mn = lambda v: torch.full((n,), 1e6, device=dv).scatter_reduce_(0, inv, v, "amin", include_self=True)
    w, h = mx(xf) - mn(xf) + 1, mx(yf) - mn(yf) + 1
    dx, dy = xf - cx[inv], yf - cy[inv]
    sxx, syy, sxy = add(dx * dx) / area, add(dy * dy) / area, add(dx * dy) / area
    lmin = (sxx + syy) / 2 - torch.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
    minor = 4 * torch.sqrt(lmin.clamp(min=0) + 1 / 12)                        # (uniform ellipse: axis = 4 sigma; + a pixel's own spread)
    return inv, n, area, torch.stack([cx, cy], 1), w, h, minor


ITEM_FILL = 0.85                          # items are filled squares (dfix1: fill p50 0.99); a round bullet fills 0.6-0.8 of its box and was dropped as an item
                                          # with the v5 rule (>= 0.5): 5% of the "items" in dfix1, and the leading bullets of Rumia's stream in vis_smoke
CORE_HB = None                            # (minor-axis upper bounds (half px), hitbox) from a calibration; None -> the rule in hb_core


def hb_core(minor):
    """full hitbox width (game px) of a bullet from its white core's minor axis (half-size px). dfix1, isolated bullets: the core is about
    1.5x the template hitbox across (513: core 7 half px, hitbox 6 game px; 512: 5 -> 4; 516: 6 -> 5)."""
    minor = np.asarray(minor, np.float32)
    if CORE_HB is not None:
        ub, hb = CORE_HB
        return np.asarray(hb, np.float32)[np.minimum(np.searchsorted(ub, minor), len(hb) - 1)]
    return np.clip(0.68 * minor * G_PER_H, 4.0, 32.0).astype(np.float32)


class VisionTable2:
    """v2 (2026-11-06): the bullets' WHITE CORES (min(R,G,B) >= core) as the things to track -- the glows of nearby bullets merge into
    one blob (54% of the bullets near the player in dfix1), their cores mostly do not. Per picture:
     * lit pixels as in v5 (max(R,G,B) >= max(80, q10 + 60), no HUD), glow blobs; dim blobs (peak < 150) and items dropped
     * cores = lit pixels with min(R,G,B) >= core; every core blob is a detection (big ones = merged cores / lasers / big bullets:
       tracked as a rigid group, `big`); a glow blob without any core pixel (a coloured bullet without a white centre) is one
       detection itself if it is dot-sized
     * BulletTracker on the detections; a track seen once may jump reach_pf half px per frame that passed (fast bullets)
     * velocity known after 3 sightings; with use2 also after 2 (then its pixels stay a still obstacle as well: if the 2-point
       match was wrong, it is still where it is)
     * the picture is older than the moment the reflex starts from: each moving bullet is put in at every lag of `lags` (frames)
       -- the 0..1 frame the picture is behind the memory read is not known per picture, so all of it is covered
     * hitbox from the core's minor axis (hb_core) or the glow's area; big groups: every core pixel a 1-px box moving with the group
     * still 1-px obstacles: core pixels not confidently tracked, and every pixel of core-less glow blobs that are not tracked (big
       coloured things: boss, lasers without a white line, ...). The glow around a core is NOT an obstacle (the hitbox is inside it)."""

    def __init__(self, core=200, lags=(0.0, 0.6, 1.2), max_core_span=14, still_pixels=True, device="cpu", tint_q=0.1,
                 reach_pf=0.0, use2=False, big=False, hb_min=4.0):
        self.tr = BulletTracker(device="cpu", min_area=1, max_area=10 ** 6, max_span=10 ** 6, reach_per_frame=reach_pf,
                                min_pts=10 ** 6)                         # (straight lines only: no parabola fits, ~2 ms saved)
        self.core, self.lags, self.max_core_span, self.still_pixels, self.dev, self.tint_q = core, tuple(lags), max_core_span, still_pixels, device, tint_q
        self.use2, self.big, self.hb_min = use2, big, hb_min
        self.ids = None
        self.last = None
        self.mov_area = np.zeros(0, np.float32)
        self.extra_lag = 0.0                     # frames added to every lag this picture (play.py: the time from the grab to the position read)

    def reset(self):
        self.tr.reset()

    @torch.no_grad()
    def update(self, img, t, player_xy):
        H, W = img.shape[:2]
        dv = self.dev
        if self.ids is None:
            self.ids = torch.arange(1, H * W + 1, dtype=torch.float32, device=dv).view(1, 1, H, W)
        T = torch.from_numpy(np.ascontiguousarray(img)).to(dv)
        V, Mn = T.amax(2), T.amin(2)
        base = float(torch.quantile(V[HUD_ROWS::2, ::2].float().flatten(), self.tint_q))
        lit = V.short() >= min(256, int(max(LIT, base + OVER)))
        lit[:HUD_ROWS] = False
        px, py = player_xy[0] / G_PER_H, player_xy[1] / G_PER_H
        e2 = np.zeros(0, np.float32)
        empty = dict(bullets=dict(x=e2, y=e2, vx=e2, vy=e2, hb=e2), enemies=dict(x=e2, y=e2, w=e2, h=e2), lasers=dict(x=[]))
        ys, xs = torch.nonzero(lit, as_tuple=True)
        if ys.numel() == 0:
            self.tr.update(None, t, (torch.zeros(0, 2), torch.zeros(0, dtype=torch.bool), (None, None, None)))
            self.mov_area = e2
            self.last = dict(n_moving=0, n_dots=0, n_blobs=0, n_cores=0, frames=self.tr.step_frames)
            return empty
        # glow blobs: drop the dim ones and the items (as v5)
        labG = _ccl(lit, self.ids)
        invG, nG, areaG, cG, wG, hG, minorG = _stats(ys, xs, labG)
        addG = lambda v: torch.zeros(nG, device=dv).scatter_add_(0, invG, v)
        mxG = lambda v: torch.full((nG,), -1e6, device=dv).scatter_reduce_(0, invG, v, "amax", include_self=True)
        peak = mxG(V[ys, xs].float())
        rgb = T[ys, xs].short()
        red = ((rgb[:, 0] > 160) & (rgb[:, 1] < 110) & (rgb[:, 2] < 110)).float()
        blue = ((rgb[:, 2] > 160) & (rgb[:, 0] < 110) & (rgb[:, 1] < 150)).float()
        isc = Mn[ys, xs] >= self.core
        white = (rgb.amin(1) >= 200).float()
        rd, bl, wh = addG(red) / areaG, addG(blue) / areaG, addG(white) / areaG
        sq = (wG >= 6) & (wG <= 13) & (hG >= 6) & (hG <= 13) & ((wG - hG).abs() <= 2)
        item = sq & (areaG / (wG * hG) >= ITEM_FILL) & ((rd + bl) >= 0.42) & (wh <= 0.2)
        keepG = ~(item | (peak < PEAK))
        ncoreG = addG(isc.float())
        pk = keepG[invG]                                                         # per lit pixel: its blob is kept
        # cores
        cm = isc & pk
        cys, cxs = ys[cm], xs[cm]
        dets, hbs, kind = [], [], []                                             # kind: 0 core dot, 1 core-less glow dot, 2 big core group
        own_c = None
        nC = 0
        if cys.numel():
            core_mask = torch.zeros(H, W, dtype=torch.bool, device=dv)
            core_mask[cys, cxs] = True
            labC = _ccl(core_mask, self.ids, 12)
            invC, nC, areaC, cC, wC, hC, minorC = _stats(cys, cxs, labC)
            dotC = (wC <= self.max_core_span) & (hC <= self.max_core_span)
            near = ((cC[:, 0] - px) ** 2 + (cC[:, 1] - py) ** 2 <= HIT_R_H ** 2) & (areaC <= 12)
            useC = ~near                                                         # (the focus marker's dot)
            if not self.big:
                useC = useC & dotC
            dets.append(cC[useC]); hbs.append(minorC[useC])
            kind.append(torch.where(dotC[useC], 0, 2))
            ci = torch.full((nC,), -1, dtype=torch.long, device=dv)
            ci[useC] = torch.arange(int(useC.sum()), device=dv)
            own_c = (cys, cxs, ci[invC])                                         # core pixel -> detection index (-1: not a detection)
        # core-less glow blobs of dot size
        dotG = keepG & (ncoreG == 0) & (areaG >= 4) & (areaG <= 170) & (wG <= 18) & (hG <= 18)
        nearG = ((cG[:, 0] - px) ** 2 + (cG[:, 1] - py) ** 2 <= HIT_R_H ** 2) & (areaG <= 12)
        dotG = dotG & ~nearG
        nd0 = sum(int(d.shape[0]) for d in dets)
        dets.append(cG[dotG]); hbs.append(areaG[dotG]); kind.append(torch.ones(int(dotG.sum()), dtype=torch.long, device=dv))
        gi = torch.full((nG,), -1, dtype=torch.long, device=dv)
        gi[dotG] = torch.arange(nd0, nd0 + int(dotG.sum()), device=dv)
        Dd, Hs, Kd = torch.cat(dets).cpu(), torch.cat(hbs).cpu(), torch.cat(kind).cpu()
        nD = Dd.shape[0]
        self.tr.update(None, t, (Dd, torch.ones(nD, dtype=torch.bool), (None, None, None)))
        # (dot_idx = all detections, in order)
        known = self.tr.known.clone()
        vel = torch.where(known[:, None], self.tr.v_line, self.tr.v2)
        moving = known | ((self.tr.n_pts >= 2) & self.use2)
        lags = [l + self.extra_lag for l in self.lags]
        bx, by, bvx, bvy, bhb = [], [], [], [], []
        def put(x_, y_, vx_, vy_, hb_):
            for l in lags:
                bx.append(x_ + vx_ * l); by.append(y_ + vy_ * l); bvx.append(vx_); bvy.append(vy_); bhb.append(hb_)
        sm = moving & (Kd != 2)
        x = (Dd[sm] * G_PER_H).numpy(); v = (vel[sm] * G_PER_H).numpy()
        hk, kk = Hs[sm].numpy(), Kd[sm].numpy()
        hb = np.where(kk == 1, hb_from_area(hk), np.maximum(hb_core(hk), self.hb_min)).astype(np.float32)
        put(x[:, 0], x[:, 1], v[:, 0], v[:, 1], hb)
        self.mov_area = hk
        n_big = 0
        if own_c is not None and self.big and nD:                                # big core groups that move: all their core pixels move
            cy_, cx_, ix = own_c
            bm = moving & (Kd == 2)
            ok = (ix >= 0)
            ixc = ix.clamp(min=0).cpu()
            sel = (ok.cpu() & bm[ixc])
            if sel.any():
                yy, xx = cy_.cpu()[sel].float(), cx_.cpu()[sel].float()
                vv = vel[ixc[sel]] * G_PER_H
                n_big = int(sel.sum())
                put((xx * G_PER_H).numpy(), (yy * G_PER_H).numpy(), vv[:, 0].numpy(), vv[:, 1].numpy(), np.full(n_big, G_PER_H, np.float32))
        if self.still_pixels:
            conf = torch.zeros(nD + 1, dtype=torch.bool)
            conf[:-1] = known                                                    # (index -1 -> the last slot: False)
            conf = conf.to(dv)
            parts = []
            if own_c is not None:                                                # core pixels of cores not confidently tracked
                cy_, cx_, ix = own_c
                st = ~conf[ix]
                parts.append((cy_[st], cx_[st]))
            cl = pk & (ncoreG[invG] == 0) & ~conf[gi[invG]]                      # pixels of core-less glow blobs not confidently tracked
            parts.append((ys[cl], xs[cl]))
            yy = torch.cat([p[0] for p in parts]).float()
            xx = torch.cat([p[1] for p in parts]).float()
            far = (xx - px) ** 2 + (yy - py) ** 2 > HIT_R_H ** 2
            yy, xx = yy[far].cpu(), xx[far].cpu()
            bx.append((xx * G_PER_H).numpy()); by.append((yy * G_PER_H).numpy())
            bvx.append(np.zeros(len(xx), np.float32)); bvy.append(np.zeros(len(xx), np.float32))
            bhb.append(np.full(len(xx), G_PER_H, np.float32))
        cat = lambda L: np.concatenate(L).astype(np.float32) if L else e2
        tab = dict(bullets=dict(x=cat(bx), y=cat(by), vx=cat(bvx), vy=cat(bvy), hb=cat(bhb)), enemies=empty["enemies"], lasers=empty["lasers"])
        self.n_mov_rows = int(sm.sum()) * len(lags)                              # (the first rows: moving dots, lag by lag)
        self.mov_xy = (x, v)
        self.last = dict(n_moving=int(sm.sum()), n_big_px=n_big, n_dots=nD, n_blobs=nG, n_cores=nC, frames=self.tr.step_frames)
        return tab


def _seg(lab):
    """lab: int labels per pixel (numpy) -> (inv: component index per pixel, starts, order) with components in sorted label order."""
    order = np.argsort(lab, kind="stable")
    l = lab[order]
    newc = np.r_[True, l[1:] != l[:-1]]
    starts = np.flatnonzero(newc)
    inv_s = np.cumsum(newc) - 1
    inv = np.empty_like(inv_s)
    inv[order] = inv_s
    return inv, starts, order


def _seg_stats(inv, starts, order, xs, ys):
    """per component: area, centroid (x, y), w, h, minor axis (as _stats)."""
    n = len(starts)
    area = np.bincount(inv, minlength=n).astype(np.float32)
    cx = np.bincount(inv, xs, n) / area
    cy = np.bincount(inv, ys, n) / area
    xo, yo = xs[order], ys[order]
    w = np.maximum.reduceat(xo, starts) - np.minimum.reduceat(xo, starts) + 1
    h = np.maximum.reduceat(yo, starts) - np.minimum.reduceat(yo, starts) + 1
    dx, dy = xs - cx[inv], ys - cy[inv]
    sxx, syy, sxy = np.bincount(inv, dx * dx, n) / area, np.bincount(inv, dy * dy, n) / area, np.bincount(inv, dx * dy, n) / area
    lmin = (sxx + syy) / 2 - np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
    minor = 4 * np.sqrt(np.maximum(lmin, 0) + 1 / 12)
    return area, np.stack([cx, cy], 1).astype(np.float32), w.astype(np.float32), h.astype(np.float32), minor.astype(np.float32)


class VisionTable3(VisionTable2):
    """VisionTable2, faster (2026-11-06): v2 spent ~18 ms per picture launching ~500 tiny GPU kernels (the GPU itself worked ~1 ms).
    Here the GPU only thresholds and labels the pixels (glow and core components in one batched loop); the lit pixels come back to the
    CPU in one copy and everything per blob is numpy. Same output as v2."""

    def _thr(self, img):
        sub = img[HUD_ROWS::2, ::2].max(2).ravel()
        kq = int(self.tint_q * (len(sub) - 1))
        base = float(np.partition(sub, kq)[kq])
        return min(256, int(max(LIT, base + OVER)))

    @torch.no_grad()
    def _pixels(self, img):
        """-> int [n, 9] per lit pixel (row-major): y, x, glow label, core label (0 = not core), min(RGB), max(RGB), R, G, B."""
        H, W = img.shape[:2]
        dv = self.dev
        if self.ids is None:
            self.ids = torch.arange(1, H * W + 1, dtype=torch.float32, device=dv).view(1, 1, H, W).repeat(1, 2, 1, 1)
        thr = self._thr(img)
        T = torch.from_numpy(img).to(dv)
        V, Mn = T.amax(2), T.amin(2)
        lit = V.short() >= thr
        lit[:HUD_ROWS] = False
        g = torch.stack([lit, lit & (Mn >= self.core)])[None].float()        # [1, 2, H, W]: glow, core
        lab = self.ids * g
        for i in range(20):
            lab = F.max_pool2d(lab, 3, 1, 1) * g
            if i % 2:
                f = lab.view(2, -1)
                lab = (torch.gather(f, 1, (f.long() - 1).clamp(min=0)) * (f > 0)).view_as(lab)
        ys_, xs_ = torch.nonzero(lit, as_tuple=True)
        return torch.stack([ys_, xs_, lab[0, 0, ys_, xs_].long(), lab[0, 1, ys_, xs_].long(), Mn[ys_, xs_].long(), V[ys_, xs_].long(),
                            T[ys_, xs_, 0].long(), T[ys_, xs_, 1].long(), T[ys_, xs_, 2].long()], 1).cpu().numpy()

    @torch.no_grad()
    def update(self, img, t, player_xy):
        img = np.ascontiguousarray(img)
        pk_ = self._pixels(img)
        px, py = player_xy[0] / G_PER_H, player_xy[1] / G_PER_H
        e2 = np.zeros(0, np.float32)
        empty = dict(bullets=dict(x=e2, y=e2, vx=e2, vy=e2, hb=e2), enemies=dict(x=e2, y=e2, w=e2, h=e2), lasers=dict(x=[]))
        if len(pk_) == 0:
            self._D0, self._Kd, self._Hs, self._vel_used = np.zeros((0, 2), np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros((0, 2), np.float32)
            self.tr.update(None, t, (torch.zeros(0, 2), torch.zeros(0, dtype=torch.bool), (None, None, None)))
            self.mov_area = e2
            self.n_mov_rows = 0
            self.last = dict(n_moving=0, n_big_px=0, n_dots=0, n_blobs=0, n_cores=0, frames=self.tr.step_frames)
            return empty
        ys, xs, lg, lc, mn_, v_ = pk_[:, 0], pk_[:, 1], pk_[:, 2], pk_[:, 3], pk_[:, 4], pk_[:, 5]
        rgb = pk_[:, 6:9]
        xf, yf = xs.astype(np.float64), ys.astype(np.float64)
        # glow blobs: drop the dim ones and the items (as v5)
        invG, stG, orG = _seg(lg)
        nG = len(stG)
        areaG, cG, wG, hG, _ = _seg_stats(invG, stG, orG, xf, yf)
        peak = np.maximum.reduceat(v_[orG], stG)
        red = (rgb[:, 0] > 160) & (rgb[:, 1] < 110) & (rgb[:, 2] < 110)
        blue = (rgb[:, 2] > 160) & (rgb[:, 0] < 110) & (rgb[:, 1] < 150)
        isc = lc > 0
        white = mn_ >= 200
        rd, bl, wh = (np.bincount(invG, m.astype(np.float64), nG) / areaG for m in (red, blue, white))
        sq = (wG >= 6) & (wG <= 13) & (hG >= 6) & (hG <= 13) & (np.abs(wG - hG) <= 2)
        item = sq & (areaG / (wG * hG) >= ITEM_FILL) & ((rd + bl) >= 0.42) & (wh <= 0.2)
        keepG = ~(item | (peak < PEAK))
        ncoreG = np.bincount(invG, isc.astype(np.float64), nG)
        pk = keepG[invG]
        # cores
        cm = isc & pk
        dets, hbs, kind = [], [], []                                             # kind: 0 core dot, 1 core-less dot, 2 big core group,
        own_c = None                                                             #       4 big bullet (orb: one round glow blob)
        nC = 0
        ncompG = np.zeros(nG, np.int64)
        if cm.any():
            cxs, cys = xf[cm], yf[cm]
            invC, stC, orC = _seg(lc[cm])
            nC = len(stC)
            compG = invG[cm][orC[stC]]                                           # the glow blob of each core blob
            ncompG = np.bincount(compG, minlength=nG)
        # big bullets (2026-11-06, std_lunatic_vis2 death at 1000 s: a 64 px black-centred orb, hitbox 32, at 2.4 / 3.1 px per frame):
        # its coloured ring has no white core -> it was a STILL ring of pixels with an empty middle, and the hitbox is inside the ring.
        # A big, round-ish glow blob with at most one core blob = one bullet: tracked by its centre, hitbox = half its diameter
        # (the templates: 64 px orb -> 32, 32 -> 16).
        mnwh, mxwh = np.minimum(wG, hG), np.maximum(wG, hG)
        bigG = keepG & (ncompG <= 1) & (mxwh > 18) & (mnwh >= 16) & (mxwh <= 64) & (mxwh <= 1.5 * mnwh) & (areaG >= 0.25 * wG * hG)
        if cm.any():
            areaC, cC, wC, hC, minorC = _seg_stats(invC, stC, orC, cxs, cys)
            dotC = (wC <= self.max_core_span) & (hC <= self.max_core_span)
            near = ((cC[:, 0] - px) ** 2 + (cC[:, 1] - py) ** 2 <= HIT_R_H ** 2) & (areaC <= 12)
            useC = (~near if self.big else (~near & dotC)) & ~bigG[compG]
            dets.append(cC[useC]); hbs.append(minorC[useC]); kind.append(np.where(dotC[useC], 0, 2))
            ci = np.full(nC, -1, np.int64)
            ci[useC] = np.arange(int(useC.sum()))
            own_c = (cys, cxs, ci[invC])
        dotG = keepG & (ncoreG == 0) & (areaG >= 4) & (areaG <= 170) & (wG <= 18) & (hG <= 18)
        nearG = ((cG[:, 0] - px) ** 2 + (cG[:, 1] - py) ** 2 <= HIT_R_H ** 2) & (areaG <= 12)
        dotG = dotG & ~nearG
        nd0 = sum(len(d) for d in dets)
        dets.append(cG[dotG]); hbs.append(areaG[dotG]); kind.append(np.ones(int(dotG.sum()), np.int64))
        gi = np.full(nG, -1, np.int64)
        gi[dotG] = np.arange(nd0, nd0 + int(dotG.sum()))
        nd1 = nd0 + int(dotG.sum())
        dets.append(cG[bigG]); hbs.append(mnwh[bigG]); kind.append(np.full(int(bigG.sum()), 4, np.int64))
        gi[bigG] = np.arange(nd1, nd1 + int(bigG.sum()))
        Dn = np.concatenate(dets).astype(np.float32)
        Hs, Kd = np.concatenate(hbs).astype(np.float32), np.concatenate(kind)
        nD = len(Dn)
        self.tr.update(None, t, (torch.from_numpy(Dn), torch.ones(nD, dtype=torch.bool), (None, None, None)))
        known = self.tr.known.numpy().copy()
        vel = np.where(known[:, None], self.tr.v_line.numpy(), self.tr.v2.numpy())
        moving = known | ((self.tr.n_pts.numpy() >= 2) & self.use2)
        self._D0, self._Kd, self._Hs = Dn, Kd, Hs                                # (for VisionTable4)
        self._vel_used = np.where(moving[:, None], vel, np.nan).astype(np.float32)
        lags = [l + self.extra_lag for l in self.lags]
        bx, by, bvx, bvy, bhb = [], [], [], [], []

        def put(x_, y_, vx_, vy_, hb_):
            for l in lags:
                bx.append(x_ + vx_ * l); by.append(y_ + vy_ * l); bvx.append(vx_); bvy.append(vy_); bhb.append(hb_)
        sm = moving & (Kd != 2)
        x, v = Dn[sm] * G_PER_H, vel[sm] * G_PER_H
        hk, kk = Hs[sm], Kd[sm]
        hb = np.where(kk == 1, hb_from_area(hk), np.where(kk == 4, 0.5 * hk * G_PER_H, np.maximum(hb_core(hk), self.hb_min))).astype(np.float32)
        put(x[:, 0], x[:, 1], v[:, 0], v[:, 1], hb)
        self.mov_area = hk
        n_big = 0
        if own_c is not None and self.big and nD:
            cy_, cx_, ix = own_c
            sel = (ix >= 0) & (moving & (Kd == 2))[np.maximum(ix, 0)]
            if sel.any():
                vv = vel[ix[sel]] * G_PER_H
                n_big = int(sel.sum())
                put(cx_[sel] * G_PER_H, cy_[sel] * G_PER_H, vv[:, 0], vv[:, 1], np.full(n_big, G_PER_H, np.float32))
        if self.still_pixels:
            conf = np.r_[known, False]                                           # (index -1 -> the last slot: False)
            sx, sy = [], []
            if own_c is not None:
                cy_, cx_, ix = own_c
                st = ~conf[ix]
                sx.append(cx_[st]); sy.append(cy_[st])
            cl = pk & (ncoreG[invG] == 0) & ~conf[gi[invG]] & ~bigG[invG]
            sx.append(xf[cl]); sy.append(yf[cl])
            xx, yy = np.concatenate(sx), np.concatenate(sy)
            far = (xx - px) ** 2 + (yy - py) ** 2 > HIT_R_H ** 2
            xx, yy = xx[far], yy[far]
            bx.append(xx * G_PER_H); by.append(yy * G_PER_H)
            bvx.append(np.zeros(len(xx), np.float32)); bvy.append(np.zeros(len(xx), np.float32))
            bhb.append(np.full(len(xx), G_PER_H, np.float32))
            ub = (Kd == 4) & ~known                                              # big bullets without a motion yet: their hitbox, still
            if ub.any():
                bx.append(Dn[ub, 0] * G_PER_H); by.append(Dn[ub, 1] * G_PER_H)
                bvx.append(np.zeros(int(ub.sum()), np.float32)); bvy.append(np.zeros(int(ub.sum()), np.float32))
                bhb.append((0.5 * Hs[ub] * G_PER_H).astype(np.float32))
        cat = lambda L: np.concatenate(L).astype(np.float32) if L else e2
        tab = dict(bullets=dict(x=cat(bx), y=cat(by), vx=cat(bvx), vy=cat(bvy), hb=cat(bhb)), enemies=empty["enemies"], lasers=empty["lasers"])
        self.n_mov_rows = int(sm.sum()) * len(lags)
        self.mov_xy = (x, v)
        self.last = dict(n_moving=int(sm.sum()), n_big_px=n_big, n_dots=nD, n_blobs=nG, n_cores=nC, n_orbs=int(bigG.sum()), frames=self.tr.step_frames)
        return tab


class VisionTable4(VisionTable3):
    """VisionTable3 + motion hypotheses from three pictures (2026-11-06, std_lunatic_vis_smoke death at 88 s: Rumia's stream of identical
    bullets ~25 px apart at 6.5 px/frame, 2-3 frames between pictures -> the tracker matched every bullet to the one behind it and
    measured the column as crawling UP; the reflex saw nothing coming; its frame clock, recovered from those tracks, was off too).
     * frame gaps: for one bullet at constant velocity the two displacements over the last two gaps are in the ratio of the gaps
       (g1 / g2). The ratio that makes the most triplets consistent wins (pairs that are not moving do not vote); the scale from the
       wall clock (the gap pair with that ratio nearest to it). Matching a regular stream the wrong way round is consistent only
       when g1 == g2.
     * for every detection near the player: each detection of the previous picture within reach gives a velocity, kept if the
       picture before has a detection where that velocity puts it. Every consistent velocity goes in (at most max_hyp), and the
       tracker's own; a detection with none stays a still obstacle (its core pixels)."""

    RATIOS = sorted({(a, b) for a in range(1, 6) for b in range(1, 6)}, key=lambda ab: (ab[0] + ab[1], ab))

    def __init__(self, *a, near=160.0, vmax=7.5, tol=1.6, max_hyp=3, **k):
        super().__init__(*a, **k)
        self.near, self.vmax, self.tol, self.max_hyp = near / G_PER_H, vmax, tol, max_hyp
        self.hist = []                                          # [(detections [n, 2] half px, wall time s)] newest last, at most 2
        self.gaps = (0, 0)

    def reset(self):
        super().reset()
        self.hist = []

    def _raster(self, D2):
        H, W = 336, 288
        R_ = np.zeros((H + 8, W + 8), bool)
        t = int(np.ceil(self.tol))
        yy = np.clip(np.round(D2[:, 1]).astype(int) + 4, 0, H + 7)
        xx = np.clip(np.round(D2[:, 0]).astype(int) + 4, 0, W + 7)
        for dy in range(-t, t + 1):
            for dx in range(-t, t + 1):
                if dx * dx + dy * dy <= self.tol ** 2 + 0.5:
                    R_[np.clip(yy + dy, 0, H + 7), np.clip(xx + dx, 0, W + 7)] = True
        return R_

    @staticmethod
    def _look(R_, q):
        qy, qx = np.round(q[:, 1]).astype(int) + 4, np.round(q[:, 0]).astype(int) + 4
        ok = (qy >= 0) & (qy < R_.shape[0]) & (qx >= 0) & (qx < R_.shape[1])
        out = np.zeros(len(q), bool)
        out[ok] = R_[qy[ok], qx[ok]]
        return out

    def _hyp(self, D0, t0, px, py):
        z_ = (np.zeros(0, np.int64), np.zeros((0, 2), np.float32))
        if len(self.hist) < 2 or not len(D0):
            return z_
        (D2, t2), (D1, t1) = self.hist
        w1, w2 = (t0 - t1) * 60.0, (t1 - t2) * 60.0
        if not (0.3 <= w1 <= 9 and 0.3 <= w2 <= 9) or not len(D1) or not len(D2):
            return z_
        ci = np.nonzero((D0[:, 0] - px) ** 2 + (D0[:, 1] - py) ** 2 <= self.near ** 2)[0]
        if not len(ci):
            return z_
        C = D0[ci]
        r = self.vmax * (w1 + 1.5) + 1.0
        d = C[:, None, :] - D1[None, :, :]
        ok = (np.abs(d[..., 0]) <= r) & (np.abs(d[..., 1]) <= r)
        a, b = np.nonzero(ok)
        if not len(a):
            return z_
        dd = d[a, b]
        R_ = self._raster(D2)
        mv = np.hypot(dd[:, 0], dd[:, 1]) >= 2.0                # (still things fit every ratio: they do not vote)
        best, best_n = None, -1
        if mv.any():
            Dm, ddm = D1[b[mv]], dd[mv]
            for g1, g2 in self.RATIOS:
                n_ = int(self._look(R_, Dm - ddm * (g2 / g1)).sum())
                if best is None or n_ > best_n + 2:              # (ties: the simpler ratio, listed first)
                    best, best_n = (g1, g2), n_
        else:
            best = (1, 1)
        rho = best[1] / best[0]
        cands = [(m_ * best[0], m_ * best[1]) for m_ in range(1, 9) if m_ * best[0] <= 8 and m_ * best[1] <= 8]
        g1, g2 = min(cands, key=lambda g: (g[0] - w1) ** 2 + (g[1] - w2) ** 2)
        self.gaps = (g1, g2)
        v = dd / g1
        sp = np.hypot(v[:, 0], v[:, 1])
        good = (sp <= self.vmax) & self._look(R_, D1[b] - dd * rho)
        a, v, sp = a[good], v[good], sp[good]
        if self.max_hyp and len(a):                             # at most max_hyp per detection: the slowest ones
            o = np.lexsort((sp, a))
            a, v = a[o], v[o]
            first = np.r_[True, a[1:] != a[:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(len(a)), 0))
            keep = (np.arange(len(a)) - start) < self.max_hyp
            a, v = a[keep], v[keep]
        return ci[a], v.astype(np.float32)

    @torch.no_grad()
    def update(self, img, t, player_xy):
        tab = super().update(img, t, player_xy)
        D0 = self._D0
        px, py = player_xy[0] / G_PER_H, player_xy[1] / G_PER_H
        di, v = self._hyp(D0, t, px, py)
        self.hist = (self.hist + [(D0, t)])[-2:]
        self.last["n_hyp"] = len(di)
        self.last["gaps"] = self.gaps
        if not len(di):
            return tab
        trk_v = self._vel_used                                  # tracker's velocity in the table (nan: none)
        same = np.hypot(*(v - trk_v[di]).T) < 0.35
        di, v = di[~same], v[~same]
        self.last["n_hyp_rows"] = len(di)
        if not len(di):
            return tab
        x, vv = D0[di] * G_PER_H, v * G_PER_H
        hb = np.where(self._Kd[di] == 1, hb_from_area(self._Hs[di]), np.maximum(hb_core(self._Hs[di]), self.hb_min)).astype(np.float32)
        lags = [l + self.extra_lag for l in self.lags]
        b = tab["bullets"]
        cat = lambda L: np.concatenate(L).astype(np.float32)
        tab = dict(tab, bullets=dict(x=cat([b["x"]] + [x[:, 0] + vv[:, 0] * l for l in lags]),
                                     y=cat([b["y"]] + [x[:, 1] + vv[:, 1] * l for l in lags]),
                                     vx=cat([b["vx"]] + [vv[:, 0]] * len(lags)), vy=cat([b["vy"]] + [vv[:, 1]] * len(lags)),
                                     hb=cat([b["hb"]] + [hb] * len(lags))))
        return tab


def ccl_runs(mask):
    """8-connected components of a bool [H, W] mask on the CPU, through its row runs (numpy only; ~1-2 ms for a playfield).
    -> (ys, xs, labels) of the set pixels in row-major order (as np.nonzero), labels >= 1."""
    H, W = mask.shape
    d = np.diff(mask.astype(np.int8), axis=1, prepend=np.int8(0), append=np.int8(0))
    ry, rx0 = np.nonzero(d == 1)                                # run starts, row-major
    _, rx1 = np.nonzero(d == -1)                                # run ends (exclusive), same order
    n = len(ry)
    if n == 0:
        z = np.zeros(0, np.int64)
        return z, z, z
    lab = np.arange(n)
    # runs of row y + 1 against the runs of row y: 8-connected when x0_a <= x1_b and x0_b <= x1_a (x1 exclusive)
    key0 = ry * (W + 2) + rx0                                   # sorted (row-major)
    key1 = ry * (W + 2) + rx1                                   # sorted too (runs in a row do not overlap)
    lo = np.searchsorted(key1, (ry - 1) * (W + 2) + rx0, side="left")      # first run of the row above with end >= my start
    hi = np.searchsorted(key0, (ry - 1) * (W + 2) + rx1, side="right")     # past the last run of the row above with start <= my end
    lo = np.maximum(lo, np.searchsorted(ry, ry - 1, side="left"))
    hi = np.minimum(hi, np.searchsorted(ry, ry - 1, side="right"))
    cnt = np.maximum(hi - lo, 0)
    if cnt.sum():
        b = np.repeat(np.arange(n), cnt)
        a = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        for _ in range(200):
            m = np.minimum(lab[a], lab[b])
            new = lab.copy()
            np.minimum.at(new, a, m)
            np.minimum.at(new, b, m)
            new = new[new]
            new = new[new]
            if np.array_equal(new, lab):
                break
            lab = new
    ln = rx1 - rx0
    ys = np.repeat(ry, ln)
    xs = np.repeat(rx0, ln) + (np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln))
    return ys, xs, np.repeat(lab + 1, ln)


class VisionTableCPU(VisionTable3):
    """VisionTable3 with the pixel stage in numpy (2026-11-06): for the vision worker process (vision_proc.py), which grabs every game
    frame -- a second CUDA context next to play.py's would time-slice the GPU with the policy; torch's CPU max-pool loop took ~190 ms."""

    def __init__(self, *a, **k):
        k["device"] = "cpu"
        super().__init__(*a, **k)

    def _pixels(self, img):
        thr = self._thr(img)
        V, Mn = img.max(2), img.min(2)
        lit = V >= thr
        lit[:HUD_ROWS] = False
        ys, xs, lg = ccl_runs(lit)
        if not len(ys):
            return np.zeros((0, 9), np.int64)
        core = lit & (Mn >= self.core)
        cy, cx, lc_ = ccl_runs(core)
        lc = np.zeros(len(ys), np.int64)
        if len(cy):                                             # both lists are row-major: map core pixels onto the lit list
            W = img.shape[1]
            pos = np.searchsorted(ys * W + xs, cy * W + cx)
            lc[pos] = lc_
        rgb = img[ys, xs].astype(np.int64)
        return np.column_stack([ys, xs, lg, lc, Mn[ys, xs], V[ys, xs], rgb[:, 0], rgb[:, 1], rgb[:, 2]]).astype(np.int64)
