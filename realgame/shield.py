"""The search on top of the policy, on the real game (foresight.py in the simulator, --fs_model track): the CNN picks an action, and keeps it
unless that shortens how long the player can foreseeably stay alive, judged from the paths of the tracked bullets over the next frames.
Only bullets with a known motion are in it; nothing here reads the game's memory except the player's position, which perception also uses."""
import torch
import torch.nn.functional as F

W, H, CELL = 384.0, 448.0, 4.0
WC, HC = int(W // CELL), int(H // CELL)
FS = 2                                                                      # game frames per decision
PAD = 6
RP, HIT_SCALE = 2.4, 0.8                                                    # as the simulator: a hit is closer than RP + HIT_SCALE * radius


class Shield:
    def __init__(self, horizon=30, delay=3, device="cuda"):
        self.steps, self.delay, self.dev = max(1, horizon // FS), delay, torch.device(device)
        d = self.dev
        self.xs = (torch.arange(WC, device=d).float() + 0.5) * CELL
        self.ys = (torch.arange(HC, device=d).float() + 0.5) * CELL
        self.legal = ((self.ys >= 16 - CELL) & (self.ys <= H - 16 + CELL))[:, None] & ((self.xs >= 8 - CELL) & (self.xs <= W - 8 + CELL))[None, :]
        DX = torch.tensor([0, 0, 0, -1, 1, -1, 1, -1, 1], device=d).float()
        DY = torch.tensor([0, -1, 1, 0, 0, -1, -1, 1, 1], device=d).float()
        a = torch.arange(18, device=d)
        move, foc = a % 9, a // 9
        norm = torch.where((DX[move] != 0) & (DY[move] != 0), 0.70710678, 1.0)
        spd = torch.where(foc > 0, 2.0, 4.0) * norm * FS
        self.ax, self.ay = DX[move] * spd, DY[move] * spd
        self.vetoes, self.calls, self.seen = 0, 0, 0

    @torch.no_grad()
    def occupancy(self, xy, rad):
        """xy: [M, frames, 2] game px, frame delay + 1 first; rad: [M] game px. -> [T, HC, WC] bool: where a player centre would be hit
        during each of the next decisions."""
        T, d = self.steps, self.dev
        k = torch.floor((RP + HIT_SCALE * rad) / CELL + 0.5).clamp(1, 6).long()
        ci, cj = torch.floor(xy[..., 0] / CELL).long(), torch.floor(xy[..., 1] / CELL).long()
        on = (ci >= -PAD) & (ci < WC + PAD) & (cj >= -PAD) & (cj < HC + PAD)
        step = (torch.arange(xy.shape[1], device=d) // FS)[None].expand_as(ci)
        occ = torch.zeros(T, HC, WC, device=d, dtype=torch.bool)
        for kk in k.unique().tolist():
            m = on & (k == kk)[:, None]
            g = torch.zeros(T, 1, HC + 2 * PAD, WC + 2 * PAD, device=d, dtype=torch.float16)
            g[step[m], 0, cj[m] + PAD, ci[m] + PAD] = 1
            occ |= F.max_pool2d(g, 2 * kk + 1, 1, kk)[:, 0, PAD:PAD + HC, PAD:PAD + WC] > 0
        return occ

    @torch.no_grad()
    def scores(self, occ, px, py):
        """-> [18]: decisions survived after taking each action (0 = hit during this very move, T = alive to the horizon)."""
        T = occ.shape[0]
        V = torch.full((1, 1, HC, WC), float(T), device=self.dev)
        V = torch.where(self.legal[None, None], V, torch.full_like(V, -1.0))
        for t in range(T - 1, 0, -1):                                       # one decision: up to 8 px straight (2 cells), less diagonally
            reach = torch.maximum(torch.maximum(F.max_pool2d(V, (5, 1), 1, (2, 0)), F.max_pool2d(V, (1, 5), 1, (0, 2))), F.max_pool2d(V, 3, 1, 1))
            V = torch.where(occ[t][None, None], torch.full_like(V, float(t)), reach)
            V = torch.where(self.legal[None, None], V, torch.full_like(V, -1.0))
        ci = ((px + self.ax).clamp(8, W - 8) / CELL).long().clamp(0, WC - 1)
        cj = ((py + self.ay).clamp(16, H - 16) / CELL).long().clamp(0, HC - 1)
        return torch.where(occ[0][cj, ci], torch.zeros(18, device=self.dev), V[0, 0, cj, ci])

    @torch.no_grad()
    def veto(self, xy, rad, px, py):
        """xy: [M, delay + steps * FS, 2] game px, the tracked bullets over the next frames (or None); px, py: where the player is when an
        action chosen now starts to act. -> [18] bool: the actions that would shorten its foreseeable life, or None if nothing is at stake."""
        self.calls += 1
        if xy is None or xy.shape[0] == 0:
            return None
        s = self.scores(self.occupancy(xy[:, self.delay:], rad), px, py)
        v = s < s.max()
        return v if bool(v.any()) else None
