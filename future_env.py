"""The final form (user, 2026-11-02): the policy estimates from the past frames where every bullet will go during the next second and
dodges with that. Four more input channels in each view, one per stretch of the future: the cells the bullets pass through during it.

future_maps draws them from positions; the simulator (SimPredictor) and the real game (realgame/perception.py, from tracker.py) use the
same drawing. SimPredictor gives the policy only what the tracker can know, never the true future: a bullet has no motion until it was
seen three times, a track can be lost and must be rebuilt, some bullets are never tracked (merged, too big), a new track's velocity is
noisy, the velocity comes from the positions of the last 20 decisions, so it lags behind a bullet that accelerates or turns and stays
wrong for a while after a sudden turn (a fast one breaks the track, as the tracker's gate does), curving is noticed only on a long track,
and later retargeting and bullets not fired yet are unknown. The bullets' own random numbers are untouched (own generator), so validation
stays comparable."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from danmaku_env import LOC, LOC_CELL, GLB_H, GLB_W, GLB_CELL

BANDS = ((2, 4, 6, 8), (10, 12, 14, 16), (18, 21, 24, 27, 30), (35, 40, 45, 50, 55, 60))        # frames from now
N_BANDS = len(BANDS)
TAU = [t for b in BANDS for t in b]
BAND_OF = [i for i, b in enumerate(BANDS) for _ in b]
LOC_K = (1, 2, 3, 4)                                                        # half-size of a bullet in local cells, by radius class
MIN_SEEN, CURVE_SEEN = 2, 9                                                 # decisions since a track began (tracker.py: 3 and 10 points)
WINDOW = 19                                                                 # decisions a velocity looks back (tracker.py: 20 slots)
HALF = 0.75                                                                 # half-size px per game px
GATE, GATE_V, GATE_NEW = 5.0, 0.35, 14.0                                    # tracker.py's matching gate, half-size px


@torch.no_grad()
def future_maps(n, env_i, x, y, rad, px, py):
    """n views. env_i: long [M], the view of each bullet; x, y: [M, len(TAU)] game px; rad: [M] radius in game px; px, py: [n] the centre
    of the local view. -> loc [n, N_BANDS, LOC, LOC], glob [n, N_BANDS, GLB_H, GLB_W], uint8."""
    d = x.device
    band = torch.tensor(BAND_OF, device=d)[None]
    env_ = env_i[:, None]
    # local view: the bullet's cell, then grown by its size (four sizes, so four pictures before the growing)
    ii = torch.floor((y - py[env_]) / LOC_CELL + LOC / 2).long()
    jj = torch.floor((x - px[env_]) / LOC_CELL + LOC / 2).long()
    ok = (ii >= 0) & (ii < LOC) & (jj >= 0) & (jj < LOC)
    cl = torch.ceil((rad + 1.0) / LOC_CELL - 0.5).long().clamp(1, len(LOC_K))[:, None] - 1
    one = N_BANDS * LOC * LOC
    size = n * len(LOC_K) * one
    code = (env_ * len(LOC_K) + cl) * one + band * LOC * LOC + ii * LOC + jj
    g = torch.zeros(size + 1, dtype=torch.bool, device=d)
    g[torch.where(ok, code, size)] = True
    g = g[:-1].view(n, len(LOC_K), N_BANDS, LOC, LOC)
    loc = torch.zeros(n, N_BANDS, LOC, LOC, device=d, dtype=torch.float16)
    for c, k in enumerate(LOC_K):
        loc = torch.maximum(loc, F.max_pool2d(g[:, c].to(torch.float16), 2 * k + 1, 1, k))
    # global view: the bullet's cell; big bullets cover the neighbours too
    gi, gj = torch.floor(y / GLB_CELL).long(), torch.floor(x / GLB_CELL).long()
    ok = (gi >= 0) & (gi < GLB_H) & (gj >= 0) & (gj < GLB_W)
    big = (rad > 6.0).long()[:, None]
    one = N_BANDS * GLB_H * GLB_W
    size = n * 2 * one
    code = (env_ * 2 + big) * one + band * GLB_H * GLB_W + gi * GLB_W + gj
    g = torch.zeros(size + 1, dtype=torch.bool, device=d)
    g[torch.where(ok, code, size)] = True
    g = g[:-1].view(n, 2, N_BANDS, GLB_H, GLB_W)
    glb = torch.maximum(g[:, 0].to(torch.float16), F.max_pool2d(g[:, 1].to(torch.float16), 3, 1, 1))
    return (loc > 0).to(torch.uint8), (glb > 0).to(torch.uint8)


class SimPredictor:
    def __init__(self, env, seed=0, fixed=False, p_track=(0.25, 0.7), q_lost=(0.02, 0.14)):
        """fixed: validation, the middle of the ranges for every episode instead of a new draw per episode. The ranges are centred on the
        real tracker's recall (future_calib.py: real 46% / 35% at 30 / 60 frames) and reach above and below it."""
        self.env, self.seed, self.fixed = env, seed, fixed
        self.p_range, self.q_range = p_track, q_lost
        d, N, B1 = env.dev, env.N, env.balive.shape[1]
        self.gen = torch.Generator(device=d)
        self.gen.manual_seed(seed)
        self.prev_alive = torch.zeros(N, B1, dtype=torch.bool, device=d)
        self.prev_x, self.prev_y = torch.zeros(N, B1, device=d), torch.zeros(N, B1, device=d)
        self.age = torch.zeros(N, B1, device=d)
        self.tracked = torch.zeros(N, B1, dtype=torch.bool, device=d)
        self.ex, self.ey = torch.zeros(N, B1, device=d), torch.zeros(N, B1, device=d)
        self.vx, self.vy = torch.zeros(N, B1, device=d), torch.zeros(N, B1, device=d)
        self.hx, self.hy = torch.zeros(WINDOW + 1, N, B1, device=d), torch.zeros(WINDOW + 1, N, B1, device=d)       # the last positions
        self.t = 0
        self.p = torch.full((N,), sum(p_track) / 2, device=d)
        self.q = torch.full((N,), sum(q_lost) / 2, device=d)

    def _rand(self, *s):
        return torch.rand(*s, generator=self.gen, device=self.env.dev)

    def _randn(self, *s):
        return torch.randn(*s, generator=self.gen, device=self.env.dev)

    @torch.no_grad()
    def __call__(self, reset_mask):
        e, b, X, Y = self.estimate(reset_mask)
        return future_maps(self.env.N, e, X, Y, self.env.brad[e, b], self.env.px, self.env.py)

    @torch.no_grad()
    def estimate(self, reset_mask):
        """-> env, bullet slot [M] of the bullets with a known motion, their x, y [M, len(TAU)]. Call once per decision."""
        self.update(reset_mask)
        e, b = self.known()
        X, Y = self.positions(e, b, torch.tensor(TAU, device=self.env.dev, dtype=torch.float32))
        return e, b, X, Y

    @torch.no_grad()
    def update(self, reset_mask):
        env, N = self.env, self.env.N
        if self.fixed:
            if bool(reset_mask.all()):
                self.gen.manual_seed(self.seed)
        elif bool(reset_mask.any()):
            lo, hi = self.p_range
            self.p = torch.where(reset_mask, lo + (hi - lo) * self._rand(N), self.p)
            lo, hi = self.q_range
            self.q = torch.where(reset_mask, lo + (hi - lo) * self._rand(N), self.q)
        s = env.balive.shape
        fs = float(env.fs)
        jump = torch.hypot(env.bx - self.prev_x, env.by - self.prev_y) > fs * (env.bs + 3.0) + 8.0
        new = env.balive & (~self.prev_alive | jump)
        # where the track expected the bullet: too far off and the tracker does not match it any more (a sudden turn of a fast bullet)
        off = torch.hypot(env.bx - self.prev_x - self.vx * fs, env.by - self.prev_y - self.vy * fs) * HALF
        gate = torch.where(self.age >= 1, GATE + GATE_V * torch.hypot(self.vx, self.vy) * HALF * fs, torch.full_like(self.vx, GATE_NEW))
        restart = new | (off > gate) | (self._rand(*s) < self.q[:, None])
        self.age = torch.where(restart, torch.zeros_like(self.age), self.age + 1)
        self.tracked = torch.where(new, self._rand(*s) < self.p[:, None], self.tracked)
        self.ex = torch.where(restart, self._randn(*s), self.ex)
        self.ey = torch.where(restart, self._randn(*s), self.ey)
        self.prev_alive, self.prev_x, self.prev_y = env.balive.clone(), env.bx, env.by
        # the velocity is that of the last positions of the track, so about half a window old
        self.t = (self.t + 1) % (WINDOW + 1)
        self.hx[self.t], self.hy[self.t] = env.bx, env.by
        w = self.age.clamp(max=WINDOW).long()
        back = ((self.t - w) % (WINDOW + 1))[None]
        span = (w * fs).clamp(min=1.0)
        self.vx = (env.bx - self.hx.gather(0, back)[0]) / span
        self.vy = (env.by - self.hy.gather(0, back)[0]) / span

    def known(self):
        """-> env, bullet slot [M] of the bullets with a known motion."""
        return (self.env.balive & self.tracked & (self.age >= MIN_SEEN)).nonzero(as_tuple=True)

    def positions(self, e, b, tau):
        """Where those bullets are expected tau [T] frames from now -> x, y [M, T]."""
        env, fs = self.env, float(self.env.fs)
        x, y = env.bx[e, b], env.by[e, b]
        sp, th, acc, om, age = env.bs[e, b], env.bth[e, b], env.bacc[e, b], env.bom[e, b], self.age[e, b]
        sig = 0.3 / (age * fs + 1.0) + 0.01
        vx, vy = self.vx[e, b] + self.ex[e, b] * sig, self.vy[e, b] + self.ey[e, b] * sig
        X, Y = x[:, None] + vx[:, None] * tau, y[:, None] + vy[:, None] * tau
        cv = ((age >= CURVE_SEEN) & ((om.abs() > 0.004) | (acc.abs() > 0.01))).nonzero()[:, 0]
        if cv.numel():                                                      # a long track that curves: tracker.path's 'turn' model
            prev = torch.cat([tau[:1] * 0, tau[:-1]])
            mid, dt = (tau + prev)[None] / 2, (tau - prev)[None]
            ss = (sp[cv][:, None] + acc[cv].clamp(-0.2, 0.2)[:, None] * mid).clamp(min=0)
            hh = th[cv][:, None] + om[cv].clamp(-0.12, 0.12)[:, None] * mid
            X[cv] = x[cv][:, None] + torch.cumsum(ss * torch.cos(hh) * dt, 1)
            Y[cv] = y[cv][:, None] + torch.cumsum(ss * torch.sin(hh) * dt, 1)
        return X, Y


def track_only(env, seed=0, fixed=False):
    """The same knowledge for something other than the policy's input (foresight.shield with model 'track'): no new channels."""
    pred = SimPredictor(env, seed, fixed)
    orig = env._observe

    def observe(reset_mask):
        pred.update(reset_mask)
        return orig(reset_mask)
    env._observe = observe
    env.future = pred
    return pred


def attach(env, seed=0, fixed=False):
    """Add the future channels to env's observations (after the wall / marker channel)."""
    pred = SimPredictor(env, seed, fixed)
    orig = env._observe

    def observe(reset_mask):
        o = orig(reset_mask)
        loc, glb = pred(reset_mask)
        return dict(o, loc=torch.cat([o["loc"], loc], 1), glob=torch.cat([o["glob"], glb], 1))
    env._observe = observe
    env.future = pred
    return pred


def widen(net, extra=N_BANDS):
    """More input channels for an ActorCritic, the new ones at zero weight: at first it acts exactly as it did."""
    for seq in (net.loc, net.glb):
        old = seq[0]
        new = nn.Conv2d(old.in_channels + extra, old.out_channels, old.kernel_size, old.stride, old.padding).to(old.weight.device)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_channels] = old.weight
            new.bias.copy_(old.bias)
        seq[0] = new.to(memory_format=torch.channels_last)
    return net
