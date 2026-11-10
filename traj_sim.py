"""TrajNet in the simulator's closed loop (eval_policy.py --foresight 30 --fs_model traj --traj ckpt): the simulator's bullets drawn as perception v5's
half-size mask (no streaks, no boss sprite), the network sees only that, the search vetoes actions as on the real game. traj_sim_record.py records the
same masks for training."""
import torch
import torch.nn.functional as F

from foresight import pending_position, future_occupancy
from traj_net import build, SoftSearch, player_marker, H_HALF, W_HALF, H_CELL, W_CELL, HALF

HOLE = 8.0 / 2                                                              # half-size px: perception v5's disc on the hitbox (8 screen px)


@torch.no_grad()
def raster(env, idx=None):
    """-> [N, 336, 288] uint8, every bullet a square of its size (radius in game px x 0.75), nothing near the player's centre."""
    N, d = env.N, env.dev
    e, b = env.balive.nonzero(as_tuple=True)
    i, j = (env.by[e, b] * HALF).long(), (env.bx[e, b] * HALF).long()
    k = (env.brad[e, b] * HALF).round().long().clamp(1, 14)
    on = (i >= 0) & (i < H_HALF) & (j >= 0) & (j < W_HALF)
    img = torch.zeros(N, H_HALF, W_HALF, dtype=torch.bool, device=d)
    for kk in k[on].unique().tolist():
        m = on & (k == kk)
        g = torch.zeros(N, 1, H_HALF, W_HALF, device=d, dtype=torch.float16)
        g[e[m], 0, i[m], j[m]] = 1
        img |= F.max_pool2d(g, 2 * kk + 1, 1, kk)[:, 0] > 0
    yy, xx = torch.meshgrid(torch.arange(H_HALF, device=d).float(), torch.arange(W_HALF, device=d).float(), indexing="ij")
    img &= (xx[None] - env.px[:, None, None] * HALF) ** 2 + (yy[None] - env.py[:, None, None] * HALF) ** 2 > HOLE ** 2
    return img.to(torch.uint8)


class SimTraj:
    def __init__(self, env, ckpt, horizon=30, margin=0.2, delay=0, chunk=96, dilate=0, oracle=False):
        self.env, self.dev = env, env.dev
        ck = torch.load(ckpt, map_location=env.dev)
        self.net = build(ck)
        self.search = SoftSearch(max(1, horizon // env.fs - 1), env.dev, env.ymin_play)
        self.margin, self.delay, self.chunk, self.dilate, self.oracle = margin, delay, chunk, dilate, oracle
        self.L = ck["lags"][0] + 1
        self.lags = torch.tensor(ck["lags"], device=env.dev)
        self.hist, self.ptr = None, 0
        self.first = (delay + 2) // 2                                       # as on the game: the maps are one decision old when they are used
        self.net_steps = (self.first + self.search.steps + self.net.every - 1) // self.net.every     # only as far as the search looks
        self.V = torch.ones(env.N, 1, H_CELL, W_CELL, device=env.dev)

    @torch.no_grad()
    def push(self, reset_mask=None):
        """Once per decision."""
        m = raster(self.env)
        if self.hist is None:
            self.hist = m[:, None].repeat(1, self.L, 1, 1)
            return
        self.ptr = (self.ptr + 1) % self.L
        self.hist[:, self.ptr] = m
        if reset_mask is not None and bool(reset_mask.any()):
            self.hist[reset_mask] = m[reset_mask][:, None]

    @torch.no_grad()
    def predict(self, idx):
        """idx: [n] -> chance of a bullet [n, T, 112, 96]"""
        sel = (self.ptr - self.lags) % self.L
        out = []
        for c in range(0, len(idx), self.chunk):
            ii = idx[c:c + self.chunk]
            x = torch.cat([self.hist[ii][:, sel], player_marker(self.env.px[ii], self.env.py[ii])[:, None]], 1).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.sigmoid(self.net(x, self.net_steps).float()))
        return torch.cat(out)

    @torch.no_grad()
    def choose(self, idx, logits):
        """idx: [n], logits: [n, 18] -> the policy's choice among the actions not clearly less survivable than the best, by the maps of the last decision;
        then the maps for the next one."""
        px, py = pending_position(self.env, idx, self.delay)
        s = self.search.lookup(self.V[idx], px, py)
        veto = s < s.max(1, keepdim=True).values * (1 - self.margin)
        if self.oracle:                                                     # the search's upper bound: the true paths of the bullets alive now, as near-sure things
            p = 0.9 * future_occupancy(self.env, idx, self.first + self.search.steps, "kin").float().transpose(0, 1)
        else:
            p = self.predict(idx)
        if self.dilate:
            p = F.max_pool2d(p, 2 * self.dilate + 1, 1, self.dilate)
        self.V[idx] = self.search.values(p, self.first)
        return logits.masked_fill(veto, float("-inf")).argmax(1), s
