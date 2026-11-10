"""The final form as the user put it (2026-11-02): from the past frames and the current one, predict where the bullets will be during the next
second, then dodge with that. Fourth way, after the tracker (recall too low), PPO on future channels (did not learn to use them) and the safety
network (one number per action, caught 30-40%): a network that only predicts, trained by plain supervised learning on the real game's own
masks, where the future frame is the label. It cannot read anything but the masks, the player's position marks the aimed shots.

TrajNet: half-size masks at LAGS decisions back + player marker -> [T, 112, 96] logits, one map per decision ahead (1 decision = 2 frames, cell =
4 game px): is a bullet in this cell then? Fully convolutional, so it knows no card and no time.
SoftSearch: the search on these maps, with probabilities instead of yes / no: the best chance to stay alive for each of the 18 actions."""
import torch
import torch.nn as nn
import torch.nn.functional as F

LAGS = (32, 24, 16, 12, 8, 6, 4, 3, 2, 1, 0)                                # decisions back, oldest first: fine steps match the bullets, long ones give speed
T_MAX = 30                                                                  # decisions ahead = 1 s
H_HALF, W_HALF, H_CELL, W_CELL = 336, 288, 112, 96
CELL, HALF = 4.0, 0.75                                                      # game px per cell, half-size px per game px
W, H = 384.0, 448.0


def block(a, b, stride=1, dil=1):
    return nn.Sequential(nn.Conv2d(a, b, 3, stride, dil, dil), nn.ReLU(inplace=True))


class TrajNet(nn.Module):
    def __init__(self, lags=LAGS, T=T_MAX, base=32):
        super().__init__()
        self.lags, self.T, self.base, self.every = tuple(lags), T, base, 1
        c = base
        self.e1 = nn.Sequential(nn.Conv2d(len(lags) + 1, c, 5, 3, 2), nn.ReLU(inplace=True), block(c, c))            # 96 x 112
        self.e2 = nn.Sequential(block(c, 2 * c, 2), block(2 * c, 2 * c))                                            # 48 x 56
        self.e3 = nn.Sequential(block(2 * c, 3 * c, 2), block(3 * c, 3 * c))                                        # 24 x 28
        self.e4 = nn.Sequential(block(3 * c, 4 * c, 2), block(4 * c, 4 * c), block(4 * c, 4 * c, 1, 2))             # 12 x 14, about 100 game px each way
        self.d3 = nn.Sequential(block(7 * c, 3 * c), block(3 * c, 3 * c))
        self.d2 = nn.Sequential(block(5 * c, 2 * c), block(2 * c, 2 * c))
        self.d1 = nn.Sequential(block(3 * c, 2 * c), block(2 * c, 2 * c))
        self.out = nn.Conv2d(2 * c, T, 1)
        nn.init.constant_(self.out.bias, -3.0)                              # most cells are empty

    def forward(self, x, steps=None, pause=None):
        """x: [n, len(lags) + 1, 336, 288] float 0..1 -> logits [n, T, 112, 96]"""
        a = self.e1(x)
        b = self.e2(a)
        c = self.e3(b)
        d = self.e4(c)
        u = lambda v: F.interpolate(v, scale_factor=2, mode="nearest")
        y = self.d3(torch.cat([u(d), c], 1))
        y = self.d2(torch.cat([u(y), b], 1))
        y = self.d1(torch.cat([u(y), a], 1))
        return self.out(y)


class TrajRNN(nn.Module):
    """The same input, but the future is rolled out step by step: a U-Net has to carry a bullet's cell across dozens of cells in one go (it beats the
    tracker at 8 and 16 frames, not at 30 and 60), a recurrent step only has to move it a little. Fine state on the 4 px cells, coarse state on 8 px
    cells for the fast ones. One step = `every` decisions and gives `every` maps; forward(x, steps) stops early, so looking less far costs less."""

    def __init__(self, lags=LAGS, T=T_MAX, base=24, every=2):
        super().__init__()
        self.lags, self.T, self.base, self.every = tuple(lags), T, base, every
        c = base
        self.e1 = nn.Sequential(nn.Conv2d(len(lags) + 1, c, 5, 3, 2), nn.ReLU(inplace=True), block(c, c))            # 96 x 112
        self.e2 = nn.Sequential(block(c, 2 * c, 2), block(2 * c, 2 * c))                                            # 48 x 56
        self.e3 = nn.Sequential(block(2 * c, 3 * c, 2), block(3 * c, 3 * c))                                        # 24 x 28
        self.e4 = nn.Sequential(block(3 * c, 4 * c, 2), block(4 * c, 4 * c), block(4 * c, 4 * c, 1, 2))             # 12 x 14
        self.d3 = block(7 * c, 3 * c)
        self.d2 = block(5 * c, 2 * c)                                       # coarse state: 2c on 48 x 56
        self.d1 = block(3 * c, c)                                           # fine state: c on 96 x 112
        self.fine = nn.Sequential(block(3 * c, c), nn.Conv2d(c, c, 3, 1, 1))
        self.coarse = nn.Sequential(block(3 * c, 2 * c), nn.Conv2d(2 * c, 2 * c, 3, 1, 1))
        self.out = nn.Conv2d(c, every, 1)
        nn.init.constant_(self.out.bias, -3.0)

    def forward(self, x, steps=None, pause=None):
        """-> logits [n, every * steps, 112, 96]. pause: called between the parts (on the game: let its own frames use the GPU)."""
        a = self.e1(x)
        b = self.e2(a)
        c = self.e3(b)
        d = self.e4(c)
        u = lambda v: F.interpolate(v, scale_factor=2, mode="nearest")
        g = self.d2(torch.cat([u(self.d3(torch.cat([u(d), c], 1))), b], 1))
        h = self.d1(torch.cat([u(g), a], 1))
        maps = []
        for _ in range(steps or (self.T + self.every - 1) // self.every):
            if pause is not None:
                pause()
            g = g + self.coarse(torch.cat([g, F.avg_pool2d(h, 2)], 1))
            h = h + self.fine(torch.cat([h, u(g)], 1))
            maps.append(self.out(h))
        return torch.cat(maps, 1)


def build(ck):
    """A network from its checkpoint, in eval mode, on the checkpoint's device."""
    dev = next(iter(ck["net"].values())).device
    net = (TrajRNN(ck["lags"], ck["T"], ck["base"], ck["every"]) if ck.get("arch") == "rnn" else TrajNet(ck["lags"], ck["T"], ck["base"]))
    net = net.to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(ck["net"])
    return net.eval()


def player_marker(px, py):
    """px, py: [n] game px -> [n, 336, 288] uint8, a 7 x 7 half-size px block where the player is."""
    n, d = len(px), px.device
    m = torch.zeros(n, 1, H_HALF, W_HALF, device=d, dtype=torch.float16)
    i = (py * HALF).long().clamp(0, H_HALF - 1)
    j = (px * HALF).long().clamp(0, W_HALF - 1)
    m[torch.arange(n, device=d), 0, i, j] = 1
    return (F.max_pool2d(m, 7, 1, 3)[:, 0] > 0).to(torch.uint8)


class SoftSearch:
    """The search of foresight.py / realgame/shield.py with probabilities: V[t, cell] = the best chance to be alive to the horizon, being in this cell
    at decision t and moving as well as possible afterwards. One decision: up to 8 px straight (2 cells), less diagonally. Cells count as independent."""

    def __init__(self, steps=15, device="cuda", ymin=16.0):
        d = self.dev = torch.device(device)
        self.steps = steps
        xs = (torch.arange(W_CELL, device=d).float() + 0.5) * CELL
        ys = (torch.arange(H_CELL, device=d).float() + 0.5) * CELL
        self.legal = (((ys >= ymin - CELL) & (ys <= H - 16 + CELL))[:, None] & ((xs >= 8 - CELL) & (xs <= W - 8 + CELL))[None, :])[None, None]
        self.ymin = ymin
        DX = torch.tensor([0, 0, 0, -1, 1, -1, 1, -1, 1], device=d).float()
        DY = torch.tensor([0, -1, 1, 0, 0, -1, -1, 1, 1], device=d).float()
        a = torch.arange(18, device=d)
        move, foc = a % 9, a // 9
        norm = torch.where((DX[move] != 0) & (DY[move] != 0), 0.70710678, 1.0)
        spd = torch.where(foc > 0, 2.0, 4.0) * norm * 2
        self.ax, self.ay = (DX[move] * spd)[None], (DY[move] * spd)[None]

    @torch.no_grad()
    def values(self, p, first=0, pause=None):
        """p: [n, T, 112, 96] chance of a bullet per cell, one map per decision ahead; first: the map of the decision an action plays out in. -> V at that
        decision, [n, 1, 112, 96]. The heavy part: on the game it runs after the keys are sent, in the time left over, so that the next decision only
        has to look up (lookup) and is not delayed."""
        V = None
        for t in range(min(first + self.steps, p.shape[1]) - 1, first - 1, -1):
            if V is None:
                V = torch.ones(p.shape[0], 1, H_CELL, W_CELL, device=self.dev)
            else:
                V = torch.maximum(torch.maximum(F.max_pool2d(V, (5, 1), 1, (2, 0)), F.max_pool2d(V, (1, 5), 1, (0, 2))), F.max_pool2d(V, 3, 1, 1))
            V = V * (1 - p[:, t:t + 1]) * self.legal
            if pause is not None and t % 4 == 0:
                pause()
        return V

    @torch.no_grad()
    def lookup(self, V, px, py):
        """px, py: [n], where the player is when the action starts to act -> [n, 18], chance to survive after each action."""
        ci = ((px[:, None] + self.ax).clamp(8, W - 8) / CELL).long().clamp(0, W_CELL - 1)
        cj = ((py[:, None] + self.ay).clamp(self.ymin, H - 16) / CELL).long().clamp(0, H_CELL - 1)
        return V[torch.arange(V.shape[0], device=self.dev)[:, None], 0, cj, ci]
