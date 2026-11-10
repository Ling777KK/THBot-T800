"""Two-stream CNN actor-critic: a fine egocentric view around the player + a coarse whole-playfield view."""
import torch
import torch.nn as nn
from danmaku_env import N_ACTIONS, LOC, GLB_H, GLB_W, STACK


def _init(m, gain=2 ** 0.5):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


class ActorCritic(nn.Module):
    def __init__(self, n_actions=N_ACTIONS, vec_dim=3, stack=STACK, lags=None):
        super().__init__()
        # lags: which past decisions the stacked channels show (oldest first, newest last); default = the last `stack` steps
        self.lags = sorted({int(x) for x in lags}, reverse=True) if lags else list(range(stack - 1, -1, -1))
        stack = len(self.lags)
        self.n_actions, self.vec_dim, self.stack = n_actions, vec_dim, stack
        c = stack + 1
        self.loc = nn.Sequential(
            nn.Conv2d(c, 32, 5, 2, 2), nn.ReLU(),          # 64 -> 32
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),         # 32 -> 16
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),         # 16 -> 8
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.Flatten())
        self.glb = nn.Sequential(
            nn.Conv2d(c, 32, 3, 1, 1), nn.ReLU(),          # 56x48
            nn.Conv2d(32, 48, 3, 2, 1), nn.ReLU(),         # 28x24
            nn.Conv2d(48, 64, 3, 2, 1), nn.ReLU(),         # 14x12
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Flatten())  # 7x6
        d_loc = 64 * (LOC // 8) ** 2
        d_glb = 64 * 7 * 6
        self.fc = nn.Sequential(nn.Linear(d_loc + d_glb + vec_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)
        self.apply(_init)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def warm_start_from(self, state_dict, src_lags=(3, 2, 1, 0)):
        """Load an older checkpoint whose vec input / action head / frame history are smaller: copy what fits, new inputs start
        at zero weight (no effect at first) and new actions start with a small random row. For the first conv layers the
        history channels are matched by lag (a source lag missing in the new set goes to the nearest new lag)."""
        own = self.state_dict()
        for k, v in state_dict.items():
            if k not in own:
                continue
            if own[k].shape == v.shape:
                own[k] = v.clone()
            elif k in ("loc.0.weight", "glb.0.weight"):     # [out, lags + 1 extra, kh, kw]
                w = torch.zeros_like(own[k])
                used = set()
                for si, L in enumerate(src_lags):
                    ti = min(range(len(self.lags)), key=lambda j: (abs(self.lags[j] - L), j in used))
                    if ti not in used:
                        w[:, ti] = v[:, si]; used.add(ti)
                w[:, -1] = v[:, -1]                          # wall map / player marker
                own[k] = w
            elif k == "fc.0.weight":                   # [512, d_in_old] <-> [512, d_in_new]: keep the shared columns
                n = min(v.shape[1], own[k].shape[1])   # (the vector input is last: x, y, focus, [bombs, invuln])
                own[k][:, :n] = v[:, :n]
                own[k][:, n:] = 0
            elif k in ("pi.weight", "pi.bias"):        # [A_old, ...] <-> [A_new, ...]: keep the shared actions
                n = min(v.shape[0], own[k].shape[0])
                own[k][:n] = v[:n]
        self.load_state_dict(own)

    def hidden(self, obs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = torch.cat([self.loc(obs["loc"].float()), self.glb(obs["glob"].float()), obs["vec"]], 1)
            h = self.fc(h)
        return h.float()

    def forward(self, obs):
        h = self.hidden(obs)
        return self.pi(h), self.v(h).squeeze(-1)
