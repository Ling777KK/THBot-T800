"""The bullet mask from th06nc's memory instead of the screen (READ-ONLY; user's rule since 2026-11-04: bullet positions / velocities
may be read during play, nothing is ever written). Same output as perception.Perceiver.bullet_mask_v5: bool [PF_H, PF_W] in playfield
screen px (1.5 per game px), so observe(), the policy and the traj network take it unchanged.

What is read every step (addresses are rvas = offsets from th06nc.exe's base, the same on every launch; found 2026-11-04, see NOTES):
  bullets  640 slots x 1568 B from 0x436c40; x at +744 (float index X=186): x, y = X, X+1; vx, vy = X-10, X-9; alive = the frame timer
           at X-1 moved since the previous read (the int at X-7 is NOT an alive flag: 0 on moving bullets in s4c11); sprite id = int at
           X+22, minus 512 = the original game's bullet type 0..9 -> templates (10 x 0x5B0 B from 0x536508): sprite px (int +0x5a8),
           hitbox px (float +0x5ac); int at X+3 = colour
  lasers   64 slots x 0x298 B from 0x52bf08: angle +0x260, in use +0x274 (1), origin x, y +0x278 / +0x27c, max length +0x288, width +0x004,
           state +0x290 (0 warning line, 1 beam, 2 fading); the beam = origin + [start +0x010, end +0x014] along the angle, end grows by
           speed +0x26c per frame; frames in the state +0x00c, warning frames +0x270 / +0x28c
  enemies  256 slots x 0x10B0 B, slot k's x at 0xaee168 + k * 0x10B0: x, y; hitbox w, h at x+0x210 / +0x214; alive = the timer at
           x-0xac moved since the previous read (dead slots keep their last position)
How each thing is drawn (sizes in screen px) is in DRAW, fitted to the v5 masks of the same frames (memcalib.py).
"""
import ctypes
import math
import numpy as np
import torch
import torch.nn.functional as F
import gamemem as G
from perception import PF_W, PF_H, S, V5_HIT_R

k32 = ctypes.windll.kernel32

B_RVA, B_SIZE, B_N, BX = 0x436C40, 1568, 640, 744 // 4
T_RVA, T_SIZE, T_N = 0x536508, 0x5B0, 10
L_RVA, L_SIZE, L_N = 0x52BF08, 0x298, 64
E_X0, E_SIZE, E_N, E_OFF = 0xAEE168, 0x10B0, 256, 0xC0           # read each enemy slot from x - E_OFF (the timer is at x - 0xac)
E_TIMER = (E_OFF - 0xAC) // 4
# items (2026-11-09, item_scan / item_fields / item_snap.py on replay No.13): slots of 0x160 B at rva multiples of 0x160; x, y, z at +0x148,
# vx, vy, vz at +0x154, type int at +0x0C = the original EoSD enum (0 small P, 1 point, 2 big P, 3 bomb, 4 full power, 5 1UP, 6 star;
# 0/1/2/6 checked on screen, the sprite pointer at +0x100 steps 0x10 per type). Free slots are all zero. Items were seen in slots
# 303..1338 from 0xBE00A0 (a 15-min replay watch: 974 distinct slots held stars); not items from ~1430 on (odd values, still positions).
# In-game check (std_lunatic_seek1, AI-mode textures hide bomb / 1UP items): a type-3 item at 971.7 s, and the HUD stars went back from
# 2 to 3 right after a bomb; a type-5 item at 1430.7 s, and the hearts went from 4 to 5.
I_RVA, I_SIZE, I_N = 0xBE00A0 + 280 * 0x160, 0x160, 1200      # (290..1350 saw no bomb / 1UP item in seekB1: back to the range of seek1,
                                                               #  where a bomb item was picked up; the non-item slots there never move)
ITEM_BOMB, ITEM_LIFE = 3, 5

# drawing: per bullet kind (half length along the flight direction, half width across) as a multiple of the template's sprite size
# (screen px = factor * sprite px * 1.5 / 2); spawning bullets (state 2) get SPAWN_F of that; lasers: half widths for each state;
# enemies: half sizes = factor * hitbox * 1.5 / 2. Initial guesses, replaced by memcalib.py's fit.
DRAW = dict(
    along={k: 1.0 for k in range(T_N)}, across={k: 1.0 for k in range(T_N)},
    spawn_f=1.0, laser_half={0: 1.5, 1: 0.5, 2: 0.3},          # laser (old): state 0 absolute px, states 1 / 2 times the width field * 1.5
    laser_rel={0: 0.31, 1: 0.52, 2: 0.29},                    # laser: half width / (width field x 1.5) per state, measured on s1c1 (width 32)
    enemy_f=1.0, hole=True,
)


class MemReader:
    """Reads the three tables into reusable numpy buffers (no copies)."""
    def __init__(self, gm=None):
        self.gm = gm or G.GameMem()
        self.h, self.base = self.gm.h, self.gm.base
        self.bbuf = np.zeros(B_N * B_SIZE // 4, np.float32)
        self.lbuf = np.zeros(L_N * L_SIZE // 4, np.float32)
        self.ebuf = np.zeros(E_N * E_SIZE // 4, np.float32)
        self.tbuf = np.zeros(T_N * T_SIZE // 4, np.float32)
        self.prev_timer = None              # enemies' timers at the previous read
        self.prev_btimer = None             # bullets' timers at the previous read
        self.prev_ex = np.zeros(E_N, np.float32)    # enemies' positions and alive flags at the previous read (per slot): their velocity
        self.prev_ey = np.zeros(E_N, np.float32)    # = position change / game frames between the reads (2026-11-08: Remilia's bats in
        self.prev_ealive = np.zeros(E_N, bool)      # Vampire Illusion are enemies flying ~4.5 px / frame; the reflex took them as still)
        self.frames = 0                             # game frames between the last two reads (0 = unknown)
        self.prev_evx = np.zeros(E_N, np.float32)   # enemies' velocities at the previous read (kept over a same-frame read)
        self.prev_evy = np.zeros(E_N, np.float32)
        self.kept_once = False                      # the previous read was a same-frame read (keep the old enemies only once)
        self.ibuf = None                            # item slots (read_items, only when asked for)
        self.prev_ixy = None
        self.l_ang = np.zeros(L_N, np.float32)      # per laser slot at the previous read: angle, timer, state, in use; estimated
        self.l_tim = np.zeros(L_N, np.int64)        # angular velocity (rad / frame): the game turns a laser by changing its angle
        self.l_st = np.full(L_N, -1, np.int64)      # every frame from the pattern script -- no struct field holds the speed
        self.l_om = np.zeros(L_N, np.float32)       # (2026-11-06, std_lunatic_beam3: Patchouli's cross 0.0088 rad / frame; one death)

        self.got = ctypes.c_size_t()
        self.read_templates()

    def _rd(self, rva, buf):
        ok = k32.ReadProcessMemory(ctypes.c_void_p(self.h), ctypes.c_void_p(self.base + rva), buf.ctypes.data_as(ctypes.c_void_p),
                                   buf.nbytes, ctypes.byref(self.got))
        return bool(ok) and self.got.value == buf.nbytes

    def read_items(self):
        """-> dict of numpy arrays (game px): items on screen x y vx vy type slot. An item counts when its slot has a position that moved
        since the previous call (items always fall or fly to the player; free slots are zero)."""
        if self.ibuf is None:
            self.ibuf = np.zeros(I_N * I_SIZE // 4, np.float32)
        ok = self._rd(I_RVA, self.ibuf)
        A = self.ibuf.reshape(I_N, I_SIZE // 4)
        I = A.view(np.int32)
        x, y = A[:, 0x148 // 4], A[:, 0x14C // 4]
        ty = I[:, 0x0C // 4]
        xy = np.stack([x, y], 1).copy()
        moved = np.any(xy != self.prev_ixy, 1) if self.prev_ixy is not None else np.zeros(I_N, bool)
        self.prev_ixy = xy
        on = ok & moved & (np.abs(x) > 1e-6) & (x > -32) & (x < 416) & (y > -48) & (y < 480) & (ty >= 0) & (ty <= 6)
        return dict(x=x[on].copy(), y=y[on].copy(), vx=A[on, 0x154 // 4].copy(), vy=A[on, 0x158 // 4].copy(), type=ty[on].copy(),
                    slot=np.nonzero(on)[0])

    def read_templates(self):
        ok = self._rd(T_RVA, self.tbuf)
        t = self.tbuf.reshape(T_N, T_SIZE // 4)
        self.sprite = t.view(np.int32)[:, 0x5A8 // 4].astype(np.float32).copy()
        self.hitbox = t[:, 0x5AC // 4].copy()
        if not ok or not np.all((self.sprite > 0) & (self.sprite <= 128)):
            self.sprite = np.array([8, 16, 16, 16, 16, 16, 32, 30, 32, 64], np.float32)      # as read on 2026-11-04
            self.hitbox = np.array([4, 6, 4, 6, 5, 4, 16, 11, 9, 32], np.float32)
        return ok

    def read(self):
        """-> dict of numpy arrays (game px): bullets x y vx vy kind state; lasers x y ang len width state; enemies x y w h."""
        ok = self._rd(B_RVA, self.bbuf) & self._rd(L_RVA, self.lbuf) & self._rd(E_X0 - E_OFF, self.ebuf)
        A = self.bbuf.reshape(B_N, B_SIZE // 4)
        I = A.view(np.int32)
        st = I[:, BX - 7]
        timer = I[:, BX - 1].copy()                # frame counter of the bullet: runs every frame while it lives, stops in a free slot
        # (s4c11: many moving bullets have int[x-7] == 0, so that field cannot tell; the timer did, 7105 of 7105 moving slots ran and
        # 0 of 22975 still ones did). First read: no previous timer yet -> any non-zero state.
        # ... but in a time stop (s5c2 Jack the Ludo Bile) the frozen knives' timers stop too, and they have a non-zero state:
        # alive = timer moved OR state != 0 (a free slot always had state 0 and a still timer in every probe: s2c2, s4c11, s5c4)
        live = st != 0
        frames = 0
        if self.prev_btimer is not None:
            live = live | (timer != self.prev_btimer)
            d_ = (timer - self.prev_btimer)[(st != 0) & (timer > self.prev_btimer)]     # bullets alive in both reads: their frame
            d_ = d_[d_ <= 8]                                                             # counters ran exactly the frames between
            if len(d_):
                frames = int(np.median(d_))
        self.prev_btimer = timer
        kind = np.clip(I[live, BX + 3], 0, T_N - 1)
        b = dict(x=A[live, BX].copy(), y=A[live, BX + 1].copy(), vx=A[live, BX - 10].copy(), vy=A[live, BX - 9].copy(),
                 kind=kind.copy(), state=np.where((st[live] & 0x4) != 0, 4, 2).astype(np.int32), slot=np.nonzero(live)[0],
                 key=I[live, BX + 22].copy(),       # sprite id (515, 516, 518, ...; x+3 is the colour) -> drawn size
                 age=timer[live].copy(),            # frames since it was fired (the timer above)
                 hbf=A[live, BX - 23].copy())       # 2026-11-05: float x-23 (= x-22) is the template hitbox on ~2/3 of the bullets and 0 on
        # the rest (more often on old ones: 2% at age < 10, 40% at age > 100); whether 0 means harmless is not known yet -- recorded for that
        L = self.lbuf.reshape(L_N, L_SIZE // 4)
        LI = L.view(np.int32)
        use = LI[:, 0x274 // 4] == 1
        ang_, tim_, st_ = L[:, 0x260 // 4], LI[:, 0x00C // 4].astype(np.int64), LI[:, 0x290 // 4].astype(np.int64)
        dt_ = tim_ - self.l_tim
        da_ = (ang_ - self.l_ang + np.pi) % (2 * np.pi) - np.pi
        same = use & (st_ == self.l_st) & (dt_ >= 1) & (dt_ <= 8)            # same laser, same state, 1-8 frames on
        nxt = use & (st_ == self.l_st + 1)                                   # same laser, next state (timer restarted): keep its speed
        keep = use & (st_ == self.l_st) & (dt_ == 0)                         # no frame passed since the last read
        om_ = np.where(same, da_ / np.maximum(dt_, 1), np.where(nxt | keep, self.l_om, 0.0)).astype(np.float32)
        om_[np.abs(om_) > 0.2] = 0.0
        self.l_om, self.l_ang, self.l_tim = om_, ang_.copy(), tim_
        self.l_st = np.where(use, st_, -1)
        las = dict(x=L[use, 0x278 // 4].copy(), y=L[use, 0x27C // 4].copy(), ang=L[use, 0x260 // 4].copy(),
                   len=L[use, 0x288 // 4].copy(), width=L[use, 0x004 // 4].copy(), state=LI[use, 0x290 // 4].copy(),
                   start=L[use, 0x010 // 4].copy(), end=L[use, 0x014 // 4].copy(), speed=L[use, 0x26C // 4].copy(),
                   timer=LI[use, 0x00C // 4].copy(), warn=np.maximum(LI[use, 0x270 // 4], LI[use, 0x28C // 4]),
                   dur=LI[use, 0x268 // 4].copy(),     # frames in state 1 (9999 = until it leaves the screen)
                   omega=om_[use].copy(), slot=np.nonzero(use)[0],
                   raw=L[use].copy())
        # 2026-11-06 (std_lunatic_laser1 laser_raw): the beam is the segment origin + [start, end] x (cos, sin) -- 'len' (+0x288) is only
        # its longest length. Cirno's / Meiling's bolts: len 192, speed 4 px / frame, end += speed each frame, start = max(0, end - len),
        # state 1 from the first frame. Rumia's / Patchouli's beams: len 500, start 0, end 500, speed 0; state 0 (warning line) for
        # 'warn' = 120 frames (timer +0x00C counts the frames of the current state), then state 1 for 60, then state 2 (fading).
        E = self.ebuf.reshape(E_N, E_SIZE // 4)
        EI = E.view(np.int32)
        timer = EI[:, E_TIMER].copy()
        x0 = E_OFF // 4
        ex, ey = E[:, x0], E[:, x0 + 1]
        inb = (ex > -64) & (ex < 448) & (ey > -64) & (ey < 512)
        # 2026-11-09: the frames between two reads come from each enemy's OWN timer, read together
        # with its position -- the bullets' counters (read ~1 ms earlier, another ReadProcessMemory) were a frame short in 2-3% of the reads
        # (a bat at 8.4 px / frame taken as 12.5, pushed ~40 px too far -> a wall that was not there -> one wasted bomb in 5 games).
        # And a read in the SAME game frame as the previous one (~2% of reads: no timer ran) used to drop every enemy from the table
        # (alive = timer changed) and zero their velocities on the next read: now it keeps the previous enemies and velocities (once).
        dte = np.zeros(E_N, np.int64)
        same_frame = False
        if self.prev_timer is None:
            alive = np.zeros(E_N, bool)
        else:
            dte = timer.astype(np.int64) - self.prev_timer
            ran = (timer != self.prev_timer) & self.prev_ealive
            # no enemy timer ran: the same frame for the enemy table, unless the bullets (read just before it, so at most a frame
            # behind it) ran 2+ frames -- then they all died
            same_frame = bool(self.prev_ealive.any()) and not ran.any() and frames <= 1 and not self.kept_once
            alive = (self.prev_ealive.copy() if same_frame else timer != self.prev_timer) & inb
        self.kept_once = same_frame
        ok_d = (dte >= 1) & (dte <= 8)
        good = alive & self.prev_ealive & ok_d
        fe = int(np.median(dte[good])) if good.any() else frames              # the enemies' frames (median), else the bullets'
        if frames == 0:
            frames = fe
        self.frames = frames
        dslot = np.where(ok_d & (np.abs(dte - fe) <= 1), dte, fe)              # own timer, unless it jumped (a script loop resets it)
        self.prev_timer = timer.astype(np.int64)
        # velocity (game px / frame) of the enemies alive in both reads; 0 for new ones, when the frames are unknown, and for jumps
        # faster than 16 px / frame (a slot reused by a new enemy, a teleport)
        if same_frame:
            evx, evy = self.prev_evx.copy(), self.prev_evy.copy()
            dslot = np.zeros(E_N, np.int64)
        else:
            both = alive & self.prev_ealive
            evx = np.where(both, ex - self.prev_ex, 0.0) / np.maximum(dslot, 1)
            evy = np.where(both, ey - self.prev_ey, 0.0) / np.maximum(dslot, 1)
            bad = (dslot <= 0) | (np.hypot(evx, evy) > 16.0)
            evx, evy = np.where(bad, 0.0, evx).astype(np.float32), np.where(bad, 0.0, evy).astype(np.float32)
        self.prev_evx, self.prev_evy = evx, evy
        self.prev_ex, self.prev_ey, self.prev_ealive = ex.copy(), ey.copy(), alive.copy()
        en = dict(x=ex[alive].copy(), y=ey[alive].copy(), w=E[alive, x0 + 0x210 // 4].copy(), h=E[alive, x0 + 0x214 // 4].copy(),
                  vx=evx[alive], vy=evy[alive], slot=np.nonzero(alive)[0], dt=dslot[alive].astype(np.float32))
        return dict(ok=ok, bullets=b, lasers=las, enemies=en, sprite=self.sprite)


class MemMask:
    """Draws the tables into a bool [PF_H, PF_W] mask on the GPU."""
    def __init__(self, device="cuda", draw=None):
        self.dev = torch.device(device)
        self.draw = dict(DRAW if draw is None else draw)
        yy, xx = torch.meshgrid(torch.arange(PF_H, device=self.dev), torch.arange(PF_W, device=self.dev), indexing="ij")
        self.yy, self.xx = yy.float(), xx.float()
        self._grids = {}
        self._fa = np.array([float(self.draw["along"].get(k, 1.0)) for k in range(T_N)], np.float32)
        self._fc = np.array([float(self.draw["across"].get(k, 1.0)) for k in range(T_N)], np.float32)
        self._lut = None
        if self.draw.get("keys"):                 # memcalib_fit2.py: half sizes (screen px) per sprite id and state, measured on v5
            dflt = self.draw.get("default", [9.0, 9.0])
            lut = np.empty((2, 4096, 2), np.float32)
            lut[0], lut[1] = dflt[0], dflt[1]
            for ks, (al, ac) in self.draw["keys"].items():
                k = int(str(ks).split(",")[0])                            # (older fits had "id,state"; the size does not depend on it)
                if 0 <= k < 4096:
                    lut[0, k, :], lut[1, k, :] = al, ac
            self._lut = lut

    BUCKETS = (14, 26, 50, 100)              # stamp half sizes; small shapes go into 14, all bigger ones into the one the biggest fits

    def _grid(self, r):
        g = self._grids.get(r)
        if g is None:
            o = torch.arange(-r, r + 1, device=self.dev, dtype=torch.float32)
            oy, ox = torch.meshgrid(o, o, indexing="ij")
            g = self._grids[r] = (ox.reshape(1, -1), oy.reshape(1, -1))
        return g

    def _stamp(self, flat, cx, cy, ax, ay, ang):
        """OR filled rotated ellipses into flat (bool [PF_H * PF_W + 1], the last cell is a dump for pixels outside the field).
        cx, cy, ax, ay, ang: numpy float32 [N] (screen px, half axes, radians). One upload per size bucket, no GPU -> CPU sync."""
        if len(cx) == 0:
            return
        need = np.ceil(np.maximum(ax, ay)) + 1
        big = need > self.BUCKETS[0]
        bk = np.zeros(len(cx), int)
        if big.any():                                                     # (each pass costs ~0.6 ms of launches: at most two)
            bk[big] = min(int(np.searchsorted(np.array(self.BUCKETS), need[big].max())), len(self.BUCKETS) - 1)
        for b in np.unique(bk):
            sel = bk == b
            r = self.BUCKETS[b]
            q = torch.from_numpy(np.stack([cx[sel], cy[sel], np.maximum(ax[sel], 0.5), np.maximum(ay[sel], 0.5),
                                           np.cos(ang[sel]), np.sin(ang[sel])], 1).astype(np.float32)).to(self.dev)
            ox, oy = self._grid(r)
            rx, ry = torch.round(q[:, :1]), torch.round(q[:, 1:2])
            px, py = rx + ox, ry + oy                                     # [n, B*B]
            dx, dy = px - q[:, :1], py - q[:, 1:2]
            u = (dx * q[:, 4:5] + dy * q[:, 5:6]) / q[:, 2:3]
            v = (dy * q[:, 4:5] - dx * q[:, 5:6]) / q[:, 3:4]
            ok = (u * u + v * v <= 1.0) & (px >= 0) & (px < PF_W) & (py >= 0) & (py < PF_H)
            idx = torch.where(ok, py * PF_W + px, torch.full_like(px, PF_H * PF_W)).long()
            flat[idx.view(-1)] = True

    def mask(self, tab, player_xy=None):
        d = self.draw
        flat = torch.zeros(PF_H * PF_W + 1, dtype=torch.bool, device=self.dev)
        b, en = tab["bullets"], tab["enemies"]
        k = b["kind"]
        if self._lut is not None and "key" in b:
            kk, si = np.clip(b["key"], 0, 4095), (b["state"] == 4).astype(int)
            al_, ac_ = self._lut[0, kk, si], self._lut[1, kk, si]
        else:
            half = tab["sprite"][k] * (S / 2) * np.where(b["state"] == 2, d["spawn_f"], 1.0).astype(np.float32)
            al_, ac_ = half * self._fa[k], half * self._fc[k]
        f = d["enemy_f"] * S / 2
        ew, eh = np.asarray(en["w"], np.float32) * f, np.asarray(en["h"], np.float32) * f
        cat = lambda a, e: np.concatenate([np.asarray(a, np.float32), np.asarray(e, np.float32)])
        self._stamp(flat, cat(b["x"] * S, np.asarray(en["x"], np.float32) * S), cat(b["y"] * S, np.asarray(en["y"], np.float32) * S),
                    cat(al_, ew), cat(ac_, eh), cat(np.arctan2(b["vy"], b["vx"]), np.zeros_like(ew)))
        out = flat[:-1].view(PF_H, PF_W)
        la = tab["lasers"]
        for i in range(len(la["x"])):                          # a few at most: one long thin rectangle each
            st = int(la["state"][i])
            if "laser_rel" in d:                                # half width = factor x the laser's width field x 1.5, every state
                hw = max(float(d["laser_rel"].get(st, 0.5)) * float(la["width"][i]) * S, 1.0)
            else:                                               # (old fits: state 0 a fixed px width -- far too wide for thin lasers)
                hw = d["laser_half"][0] if st == 0 else d["laser_half"].get(st, 0.5) * float(la["width"][i]) * S
            s0, s1 = (float(la["start"][i]), float(la["end"][i])) if "start" in la else (0.0, float(la["len"][i]))
            L = max(s1 - s0, 1.0) * S / 2
            ang = float(la["ang"][i])
            cx = float(la["x"][i]) * S + (s0 * S + L) * math.cos(ang)
            cy = float(la["y"][i]) * S + (s0 * S + L) * math.sin(ang)
            self._laser(out, cx, cy, L, hw, ang)
        if d.get("hole", True) and player_xy is not None:      # like v5: the focus marker's dot is not a bullet (keeps the input the same)
            px, py = player_xy[0] * S, player_xy[1] * S
            r = int(V5_HIT_R) + 2
            x0, x1 = max(int(px) - r, 0), min(int(px) + r + 1, PF_W)
            y0, y1 = max(int(py) - r, 0), min(int(py) + r + 1, PF_H)
            if x0 < x1 and y0 < y1:
                out[y0:y1, x0:x1] &= (self.xx[y0:y1, x0:x1] - px) ** 2 + (self.yy[y0:y1, x0:x1] - py) ** 2 > V5_HIT_R ** 2
        return out

    def _laser(self, out, cx, cy, half_len, half_w, ang):
        """Long rectangle: test every pixel of its bounding box (cheaper than a huge stamp)."""
        c, s = math.cos(ang), math.sin(ang)
        ex, ey = abs(half_len * c) + abs(half_w * s), abs(half_len * s) + abs(half_w * c)
        x0, x1 = max(int(cx - ex) - 1, 0), min(int(cx + ex) + 2, PF_W)
        y0, y1 = max(int(cy - ey) - 1, 0), min(int(cy + ey) + 2, PF_H)
        if x0 >= x1 or y0 >= y1:
            return
        dx, dy = self.xx[y0:y1, x0:x1] - cx, self.yy[y0:y1, x0:x1] - cy
        u, v = dx * c + dy * s, -dx * s + dy * c
        out[y0:y1, x0:x1] |= (u.abs() <= half_len) & (v.abs() <= max(half_w, 0.5))
