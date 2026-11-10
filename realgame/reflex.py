"""Reflex layer (expert's step 4, 2026-11-05): from the game's memory AT the decision (bullets x y vx vy moved on in straight lines, lasers and
enemies where they are, the TRUE hitboxes: bullet box = template hitbox, player half size 1.5 px -- replay_check.py explained 93% of the
recorded hits with it and 0.01% of survived steps), search the next frames on a 1 px grid around the player: which of the 18 actions,
played for 2 frames from where the player will be when they start to act, are not hit and end where some escape still exists?
If the policy's action is not one of them and another is, take the safe one with the most room around it (then the one nearest the policy's).
Offline (reflex_check.py, 69 memory recordings): at the decisive decision before the hits the action was truly safe 50% -> 73%; on 544
random survived decisions it never changed the action. Numpy on a local window, ~1-2 ms. Read-only; never presses X.
   play.py --memperc ... --reflex 8   (frames ahead)"""
import numpy as np

HB = np.array([4, 6, 4, 6, 5, 4, 16, 11, 9, 32], np.float32)
DX = np.array([0, 0, 0, -1, 1, -1, 1, -1, 1], float)
DY = np.array([0, -1, 1, 0, 0, -1, -1, 1, 1], float)
VX = np.array([DX[a % 9] * (2.0 if a >= 9 else 4.0) * (0.70710678 if DX[a % 9] and DY[a % 9] else 1.0) for a in range(18)])
VY = np.array([DY[a % 9] * (2.0 if a >= 9 else 4.0) * (0.70710678 if DX[a % 9] and DY[a % 9] else 1.0) for a in range(18)])
MOVES = [(0, 0)] + [(dx * s, dy * s) for s in (4, 2) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))] + \
        [(dx * g, dy * g) for g in (3, 1) for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
X_LO, X_HI, Y_LO, Y_HI = 8, 376, 16, 432                       # where the player can be (game px)


def laser_span(la, i, fr):
    """laser i's lethal stretch along its axis at frames fr (0 = the read): (lo [F], hi [F]), lo > hi where harmless; None if never.
    2026-11-06 (memperception.py): the beam is [start, end]; end grows by speed per frame and start = max(start, end - len) (bolts);
    a warning line (state 0) turns lethal when its timer reaches 'warn'; a beam (state 1) with a duration < 9999 ends after it.
    Tables without 'start' (older recordings / --laser_extend): state 1 only, [0, len], as before."""
    st = int(la["state"][i])
    if "start" not in la:
        return (np.zeros(len(fr), np.float32), np.full(len(fr), float(la["len"][i]), np.float32)) if st == 1 else None
    s0, e0, ln, sp = float(la["start"][i]), float(la["end"][i]), float(la["len"][i]), float(la["speed"][i])
    tm, warn = int(la["timer"][i]), int(la["warn"][i])
    dur = int(la["dur"][i]) if "dur" in la else 9999
    if st == 1:
        on = (tm + fr <= dur + 1) if 0 < dur < 9999 else np.ones(len(fr), bool)
    elif st == 0 and warn > 0:
        on = tm + fr >= warn - 1                                   # (a frame early: the hitbox may come with the switch)
    else:
        return None
    if not on.any():
        return None
    hi = e0 + sp * fr
    lo = np.maximum(s0, hi - ln) if sp > 0 else np.full(len(fr), s0, np.float32)
    return np.where(on, lo, 1e9).astype(np.float32), np.where(on, hi, -1e9).astype(np.float32)


def laser_dirs(la, i, fr):
    """laser i's axis at frames fr: (ux [F], uy [F]); turning at its estimated angular velocity 'omega' (rad / frame, from two reads
    of the same slot -- memperception.MemReader; 0 when unknown)."""
    a = float(la["ang"][i]) + (float(la["omega"][i]) if "omega" in la else 0.0) * fr
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)


def hb_of(key):
    h = np.full(len(key), np.nan, np.float32)
    s = (key >= 512) & (key <= 521)
    h[s] = HB[key[s] - 512]
    h[key == 698] = 32.0
    return h


def bullet_hb(b):
    """full hitbox width per bullet: the table's own 'hb' (vision_table.py: from the blob) or the template of its sprite id 'key'."""
    return np.asarray(b["hb"], np.float32) if "hb" in b else hb_of(np.asarray(b["key"]))


class Reflex:
    def __init__(self, h=8, delay=2, pr=1.5):
        self.h, self.delay, self.pr = h, delay, pr
        self.move_enemies = True                                # enemies with a velocity in the table move on (2026-11-08)
        self.F = delay + h + 1                                  # frames 0 .. delay + h after the decision
        self.R = (delay + h) * 4 + 6                            # window half size (game px): as far as the player can get
        self.G = 2 * self.R + 1
        self.n_seen = self.n_changed = self.n_none = self.n_quiet = 0
        G = self.G
        self.yy, self.xx = np.mgrid[0:G, 0:G].astype(np.float32)

    def _enemy_v(self, e):
        """the enemies' velocities (game px / frame) when the table has them (memperception: per slot from two reads) and
        self.move_enemies; None = treat them as still (older tables, the vision tables, --reflex_enemy_static)."""
        if not self.move_enemies or "vx" not in e:
            return None, None
        vx, vy = np.asarray(e["vx"], np.float32), np.asarray(e["vy"], np.float32)
        return (vx, vy) if (vx.any() or vy.any()) else (None, None)

    def _boxes(self, x, y, hx, hy, out):
        """stamp boxes (window coords, per frame) into out [F, G, G] via corner differences + cumsums."""
        F, G = out.shape[0], self.G
        x0 = np.clip(np.ceil(x - hx), 0, G).astype(np.int64); x1 = np.clip(np.floor(x + hx) + 1, 0, G).astype(np.int64)
        y0 = np.clip(np.ceil(y - hy), 0, G).astype(np.int64); y1 = np.clip(np.floor(y + hy) + 1, 0, G).astype(np.int64)
        f = np.broadcast_to(np.arange(F)[:, None], x0.shape) * ((G + 1) * (G + 1))
        idx = np.concatenate([(f + yy * (G + 1) + xx).ravel() for yy, xx in ((y0, x0), (y0, x1), (y1, x0), (y1, x1))])
        w = np.repeat(np.array([1, -1, -1, 1], np.int32), x0.size)
        d = np.bincount(idx, weights=w, minlength=F * (G + 1) * (G + 1)).reshape(F, G + 1, G + 1)
        out |= d.cumsum(1).cumsum(2)[:, :G, :G] > 0.5

    def lethal(self, tab, cx, cy):
        """[F, G, G] bool: hit at frame f (0 = the state read) in the window centred on (cx, cy) (integer game px)."""
        F, G, R = self.F, self.G, self.R
        out = np.zeros((F, G, G), bool)
        b = tab["bullets"]
        if len(b["x"]):
            hb = bullet_hb(b) / 2 + self.pr
            x, y, vx, vy = (np.asarray(b[k], np.float32) for k in ("x", "y", "vx", "vy"))
            reach = R + np.hypot(vx, vy) * F + np.nan_to_num(hb, nan=0) + 2
            m = np.isfinite(hb) & (np.abs(x - cx) < reach) & (np.abs(y - cy) < reach)
            if m.any():
                st = m & (vx == 0) & (vy == 0)                  # still ones (vision tables: thousands of lit pixels): stamped once
                if st.sum() > 32:
                    one = np.zeros((1, G, G), bool)
                    self._boxes((x[st] - cx + R)[None], (y[st] - cy + R)[None], hb[st][None], hb[st][None], one)
                    out |= one
                    m = m & ~st
                if m.any():
                    fr = np.arange(F, dtype=np.float32)[:, None]
                    xs = x[m][None] + vx[m][None] * fr - cx + R
                    ys = y[m][None] + vy[m][None] * fr - cy + R
                    hh = np.broadcast_to(hb[m][None], xs.shape)
                    self._boxes(xs, ys, hh, hh, out)
        e = tab["enemies"]
        if len(e["x"]):
            ex, ey = np.asarray(e["x"], np.float32) - cx + R, np.asarray(e["y"], np.float32) - cy + R
            ew, eh = np.asarray(e["w"], np.float32) / 2 + self.pr, np.asarray(e["h"], np.float32) / 2 + self.pr
            evx, evy = self._enemy_v(e)
            if evx is None:                                     # still: one box for every frame
                m = (np.abs(ex - R) < R + ew + 2) & (np.abs(ey - R) < R + eh + 2)
                if m.any():
                    st = np.zeros((1, G, G), bool)
                    self._boxes(ex[m][None], ey[m][None], ew[m][None], eh[m][None], st)
                    out |= st
            else:                                               # moving on in straight lines, like the bullets (2026-11-08)
                sp = np.hypot(evx, evy) * F
                m = (np.abs(ex - R) < R + ew + sp + 2) & (np.abs(ey - R) < R + eh + sp + 2)
                if m.any():
                    fr = np.arange(F, dtype=np.float32)[:, None]
                    xs, ys = ex[m][None] + evx[m][None] * fr, ey[m][None] + evy[m][None] * fr
                    self._boxes(xs, ys, np.broadcast_to(ew[m][None], xs.shape), np.broadcast_to(eh[m][None], xs.shape), out)
        la = tab["lasers"]
        if len(la["x"]):
            gx = gy = None
            fr = np.arange(F, dtype=np.float32)
            for i in range(len(la["x"])):
                sp_ = laser_span(la, i, fr)                     # the beam kills (warning lines from when they turn into one)
                if sp_ is None:
                    continue
                lo, hi = sp_
                ox, oy, hw = float(la["x"][i]), float(la["y"][i]), float(la["width"][i]) / 2 + self.pr
                reach = R * 1.42 + 2
                om = float(la["omega"][i]) if "omega" in la else 0.0
                if om == 0.0:
                    ux, uy = np.cos(float(la["ang"][i])), np.sin(float(la["ang"][i]))
                    a0, p0 = (cx - ox) * ux + (cy - oy) * uy, abs(-(cx - ox) * uy + (cy - oy) * ux)
                    if p0 > reach + hw or a0 < lo.min() - reach or a0 > hi.max() + reach:
                        continue                                # nowhere near the window
                    if gx is None:
                        gx, gy = self.xx + cx - R, self.yy + cy - R
                    rx, ry = gx - ox, gy - oy
                    al = rx * ux + ry * uy
                    near = np.abs(-rx * uy + ry * ux) <= hw
                    out |= near[None] & (al[None] >= lo[:, None, None]) & (al[None] <= hi[:, None, None])
                    continue
                ux, uy = laser_dirs(la, i, fr)                  # turning laser (2026-11-06): its axis frame by frame
                a0 = (cx - ox) * ux + (cy - oy) * uy
                p0 = np.abs(-(cx - ox) * uy + (cy - oy) * ux)
                fs = np.nonzero((p0 <= reach + hw) & (a0 >= lo - reach) & (a0 <= hi + reach))[0]
                if not len(fs):
                    continue
                if gx is None:
                    gx, gy = self.xx + cx - R, self.yy + cy - R
                rx, ry = (gx - ox)[None], (gy - oy)[None]
                u_, v_ = ux[fs][:, None, None], uy[fs][:, None, None]
                al = rx * u_ + ry * v_
                out[fs] |= (np.abs(-rx * v_ + ry * u_) <= hw) & (al >= lo[fs][:, None, None]) & (al <= hi[fs][:, None, None])
        return out

    def quiet(self, tab, px, py):
        """True if nothing that kills can come anywhere near where the player could be in the next F frames (then every action is safe)."""
        reach = 4.0 * self.F + 2
        b = tab["bullets"]
        if len(b["x"]):
            x, y, vx, vy = (np.asarray(b[k], np.float32) for k in ("x", "y", "vx", "vy"))
            hb = np.nan_to_num(bullet_hb(b), nan=0.0) / 2 + self.pr
            fr = np.arange(self.F, dtype=np.float32)[:, None]
            dx = np.abs(x[None] + vx[None] * fr - px) - hb[None]
            dy = np.abs(y[None] + vy[None] * fr - py) - hb[None]
            if np.any((dx < reach) & (dy < reach)):
                return False
        e = tab["enemies"]
        if len(e["x"]):
            evx, evy = self._enemy_v(e)
            ex, ey = np.asarray(e["x"], np.float32)[None], np.asarray(e["y"], np.float32)[None]
            if evx is not None:
                fr = np.arange(self.F, dtype=np.float32)[:, None]
                ex, ey = ex + evx[None] * fr, ey + evy[None] * fr
            if np.any((np.abs(ex - px) - np.asarray(e["w"])[None] / 2 < reach + 2) &
                      (np.abs(ey - py) - np.asarray(e["h"])[None] / 2 < reach + 2)):
                return False
        la = tab["lasers"]
        fr = np.arange(self.F, dtype=np.float32)
        for i in range(len(la["x"])):
            sp_ = laser_span(la, i, fr)
            if sp_ is None:
                continue
            ux, uy = laser_dirs(la, i, fr)
            rx, ry = px - float(la["x"][i]), py - float(la["y"][i])
            a0, p0 = rx * ux + ry * uy, np.abs(-rx * uy + ry * ux)
            if np.any((p0 < reach + float(la["width"][i]) / 2 + self.pr) & (a0 > sp_[0] - reach) & (a0 < sp_[1] + reach)):
                return False
        return True

    def safe(self, tab, px, py):
        """-> bool [18] (px, py: where the player is when the action starts to act = `delay` frames after the read), and the alive maps."""
        cx, cy = int(round(px)), int(round(py))
        R, G, F = self.R, self.G, self.F
        if self.quiet(tab, px, py):
            self.n_quiet += 1
            return np.ones(18, bool), None, (cx, cy)
        L = self.lethal(tab, cx, cy)
        yy, xx = np.mgrid[0:G, 0:G]
        legal = (xx + cx - R >= X_LO) & (xx + cx - R <= X_HI) & (yy + cy - R >= Y_LO) & (yy + cy - R <= Y_HI)
        A = np.zeros((F, G, G), bool)
        A[F - 1] = ~L[F - 1] & legal
        for f in range(F - 2, -1, -1):
            nxt, acc = A[f + 1], np.zeros((G, G), bool)
            for dx, dy in MOVES:                                 # acc[y, x] |= nxt[y + dy, x + dx]
                ys, yd = (slice(dy, None), slice(0, G - dy)) if dy >= 0 else (slice(0, G + dy), slice(-dy, None))
                xs, xd = (slice(dx, None), slice(0, G - dx)) if dx >= 0 else (slice(0, G + dx), slice(-dx, None))
                acc[yd, xd] |= nxt[ys, xs]
            A[f] = acc & ~L[f] & legal
        S = np.zeros(18, bool)
        f0 = self.delay
        for a in range(18):
            ok = True
            for j in (1, 2):
                x = int(round(np.clip(px + VX[a] * j, X_LO, X_HI))) - cx + R
                y = int(round(np.clip(py + VY[a] * j, Y_LO, Y_HI))) - cy + R
                if j == 1 and L[f0 + 1, y, x]:
                    ok = False
                    break
                if j == 2:
                    ok = bool(A[f0 + 2, max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].any())
            S[a] = ok
        return S, A, (cx, cy)

    def _alive(self, L, cx, cy):
        R, G, F = self.R, self.G, L.shape[0]
        yy, xx = np.mgrid[0:G, 0:G]
        legal = (xx + cx - R >= X_LO) & (xx + cx - R <= X_HI) & (yy + cy - R >= Y_LO) & (yy + cy - R <= Y_HI)
        A = np.zeros((F, G, G), bool)
        A[F - 1] = ~L[F - 1] & legal
        for f in range(F - 2, -1, -1):
            nxt, acc = A[f + 1], np.zeros((G, G), bool)
            for dx, dy in MOVES:
                ys, yd = (slice(dy, None), slice(0, G - dy)) if dy >= 0 else (slice(0, G + dy), slice(-dy, None))
                xs, xd = (slice(dx, None), slice(0, G - dx)) if dx >= 0 else (slice(0, G + dx), slice(-dx, None))
                acc[yd, xd] |= nxt[ys, xs]
            A[f] = acc & ~L[f] & legal
        return A

    def safe_robust(self, tab, x0, y0, vl, delays=(1, 2, 3), hold=4):
        """Timing-robust check (2026-11-05, std_lunatic_diag1: between two 33 ms reads 1 / 2 / 3 / 4 game frames passed 21 / 46 / 24 / 6%,
        so the key may act 1..3 frames after the read and the action may stay on for up to 4 frames). (x0, y0): the player at the read,
        vl: its per-frame velocity under the previous action (on until the new key acts). For each delay d: the player drifts on with vl
        for d frames (a hit there = no action helps for that d), then action a is held `hold` frames: not hit, and the end cell has an
        escape. -> (n_ok [18]: for how many of the delays a passes, A, (cx, cy)); n_ok = len(delays) everywhere when quiet."""
        dm = max(delays)
        cx, cy = int(round(x0 + vl[0] * dm / 2)), int(round(y0 + vl[1] * dm / 2))
        if self.quiet(tab, cx, cy):
            self.n_quiet += 1
            return np.full(18, len(delays)), None, (cx, cy)
        L = self.lethal(tab, cx, cy)
        A = self._alive(L, cx, cy)
        R, F = self.R, self.F
        assert dm + hold < F, "reflex horizon too short for these delays / hold"
        G = self.G
        ix = lambda x: np.round(np.clip(x, X_LO, X_HI)).astype(np.int64) - cx + R      # (vectorised 2026-11-06: same cells as the
        iy = lambda y: np.round(np.clip(y, Y_LO, Y_HI)).astype(np.int64) - cy + R      # scalar loop it replaced, ~3x faster)
        n_ok = np.zeros(18, int)
        jj = np.arange(1, hold + 1)
        for d in delays:
            if d > 0:
                f = np.arange(1, d + 1)
                if L[f, iy(y0 + vl[1] * f), ix(x0 + vl[0] * f)].any():
                    continue                                      # hit while the old action is still on: nothing helps
            pdx, pdy = x0 + vl[0] * d, y0 + vl[1] * d
            X = ix(pdx + VX[:, None] * jj[None])                  # [18, hold]
            Y = iy(pdy + VY[:, None] * jj[None])
            hit = L[d + jj[None], Y, X].any(1)
            E = A[d + hold]
            Ep = np.zeros((G + 2, G + 2), bool)                   # escape: any alive cell in the 3x3 around the end cell
            Ep[1:-1, 1:-1] = E
            D3 = np.zeros((G, G), bool)
            for ddy in (0, 1, 2):
                for ddx in (0, 1, 2):
                    D3 |= Ep[ddy:ddy + G, ddx:ddx + G]
            n_ok += (~hit & D3[Y[:, -1], X[:, -1]]).astype(int)
        return n_ok, A, (cx, cy)

    def choose_robust(self, tab, x0, y0, vl, a, delays=(1, 2, 3), hold=4):
        """the policy's action a if it passes for as many delays as the best action does, else the best one (most delays, then the
        most room at the end, then the nearest to a). -> (action, n_ok)"""
        n_ok, A, (cx, cy) = self.safe_robust(tab, x0, y0, vl, delays, hold)
        self.n_seen += 1
        best = n_ok.max()
        if best == 0:
            self.n_none += 1
            return a, n_ok
        if n_ok[a] == best:
            return a, n_ok
        R, d2 = self.R, int(np.median(delays))
        pdx, pdy = x0 + vl[0] * d2, y0 + vl[1] * d2
        cand = None
        for b in np.nonzero(n_ok == best)[0]:
            x = int(round(np.clip(pdx + VX[b] * hold, X_LO, X_HI))) - cx + R
            y = int(round(np.clip(pdy + VY[b] * hold, Y_LO, Y_HI))) - cy + R
            room = int(A[d2 + hold, max(y - 4, 0):y + 5, max(x - 4, 0):x + 5].sum()) if A is not None else 0
            c_ = (room, -abs(int(b) - a), int(b))
            cand = c_ if cand is None or c_ > cand else cand
        self.n_changed += 1
        return cand[2], n_ok

    def choose(self, tab, px, py, a):
        """the policy's action a, or a safe replacement if a is not safe and another is."""
        S, A, (cx, cy) = self.safe(tab, px, py)
        self.n_seen += 1
        if S[a]:
            return a, S
        if not S.any():
            self.n_none += 1
            return a, S
        R, f2 = self.R, self.delay + 2
        best = None
        for b in np.nonzero(S)[0]:
            x = int(round(np.clip(px + VX[b] * 2, X_LO, X_HI))) - cx + R
            y = int(round(np.clip(py + VY[b] * 2, Y_LO, Y_HI))) - cy + R
            room = int(A[f2, max(y - 4, 0):y + 5, max(x - 4, 0):x + 5].sum())
            cand = (room, -abs(int(b) - a), int(b))
            best = cand if best is None or cand > best else best
        self.n_changed += 1
        return best[2], S
