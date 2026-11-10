"""Bullet-by-bullet tracking from the mask alone (user, 2026-11-02: a straight line cannot follow bullets that curl, and the future is
every moment on the way, not one picture). Nothing here reads the game's memory.

Every dot-sized blob of the half-size mask is a detection. Tracks are matched to the detections next to where they should be now.
The clock is recovered from the bullets: between two grabs the game advances a whole number of frames (1, 2 or 3 for 33 ms), and that
number is the same for every bullet, so the median progress of the moving tracks gives it, which removes most of the position noise.
The last ~0.5 s of a track is fitted with a straight line and with a parabola; the curve is used only if it explains the track
significantly better (F test), since fitting an acceleration to a straight bullet only adds noise. From that fit a track rolls forward
frame by frame as the simulator's bullets do: speed changes by a constant, the heading turns by a constant. Specks, big blobs (boss,
lasers, merged clusters) and bullets seen for the first time stay where they are.

path(frames) -> [n, T, 2]: how far each current detection will have moved at every future frame; shift(lead) -> the mask at one of them."""
import torch
import torch.nn.functional as F


class BulletTracker:
    def __init__(self, shape=(336, 288), device="cuda", window=20, min_area=4, max_area=170, max_span=18, reach=14.0, misses=2,
                 min_pts=10, f_thr=20.0, max_r2=0.25, reach_per_frame=0.0):
        self.dev, self.cpu = torch.device(device), torch.device("cpu")          # pixels on the GPU; the few hundred tracks are faster on the CPU
        self.shape, self.K, self.min_area, self.max_area, self.max_span = shape, window, min_area, max_area, max_span
        self.reach, self.misses, self.min_pts, self.f_thr, self.max_r2 = reach, misses, min_pts, f_thr, max_r2
        self.reach_pf = reach_per_frame          # > 0: a track seen once may jump this many px per frame that passed (fast bullets, 2-4 frame gaps)
        H, W = shape
        self.ids = torch.arange(1, H * W + 1, device=self.dev, dtype=torch.float32).view(1, 1, H, W)
        self.reset()

    def reset(self):
        z = lambda *s, **k: torch.zeros(*s, device=self.cpu, **k)
        self.tp, self.tv = z(0, self.K, 2), z(0, self.K, dtype=torch.bool)      # track positions (x, y) half-size px, newest last; slot used
        self.tt = z(self.K)                                                     # game frame of each slot (recovered clock)
        self.miss = z(0, dtype=torch.long)
        self.tid, self.next_id = z(0, dtype=torch.long), 0                      # (a stable number per track, for the offline checks)
        self.t0 = None
        self.x0, self.tr_v, self.tr_vc, self.tr_a, self.tr_c = z(0, 2), z(0, 2), z(0, 2), z(0, 2), z(0, dtype=torch.bool)
        self.det, self.pix, self.dot_idx, self.n_blobs = None, None, None, 0
        self.v_line = self.v_curve = self.a = self.curved = self.known = None
        self.step_frames = 0.0

    @torch.no_grad()
    def _blobs(self, m):
        """-> centroids [n, 2], is-dot [n], and for every lit pixel (ys, xs, its blob)."""
        g = m[None, None].float()
        ids = self.ids * g
        for i in range(14):
            ids = F.max_pool2d(ids, 3, 1, 1) * g
            if i % 2:                                                           # pointer jumping: take the label that the pixel your
                f = ids.view(-1)                                                # label names has by now (far fewer iterations)
                ids = (f[(f.long() - 1).clamp(min=0)] * (f > 0)).view_as(ids)
        ys, xs = torch.nonzero(m, as_tuple=True)
        if ys.numel() == 0:
            return torch.zeros(0, 2, device=self.dev), torch.zeros(0, dtype=torch.bool, device=self.dev), (ys, xs, ys)
        uniq, inv = torch.unique(ids[0, 0, ys, xs].long(), return_inverse=True)
        n = uniq.numel()
        add = lambda v: torch.zeros(n, device=self.dev).scatter_add_(0, inv, v)
        cnt = add(torch.ones_like(xs, dtype=torch.float32))
        xf, yf = xs.float(), ys.float()
        c = torch.stack([add(xf) / cnt, add(yf) / cnt], 1)
        span = lambda v, how, init: torch.full((n,), init, device=self.dev).scatter_reduce_(0, inv, v, how, include_self=False)
        w = span(xf, "amax", -1.0) - span(xf, "amin", 1e6)
        h = span(yf, "amax", -1.0) - span(yf, "amin", 1e6)
        return c, (cnt >= self.min_area) & (cnt <= self.max_area) & (w <= self.max_span) & (h <= self.max_span), (ys, xs, inv)

    @torch.no_grad()
    def _fit(self):
        """Line and parabola through each track's window, tau = 0 at the newest slot."""
        w = self.tv.float()
        n = w.sum(1)
        tau = (self.tt - self.tt[-1])[None] * w
        # straight line: x = x0 + v tau
        st, stt = tau.sum(1), (tau ** 2).sum(1)
        sx, stx = (self.tp * w[..., None]).sum(1), (self.tp * (w * tau)[..., None]).sum(1)
        det = (n * stt - st ** 2).clamp(min=1e-6)
        has = n >= 2
        v1 = torch.where(has[:, None], (n[:, None] * stx - st[:, None] * sx) / det[:, None], torch.zeros_like(sx))
        x1 = (sx - v1 * st[:, None]) / n[:, None].clamp(min=1)
        self.x0, self.tr_v, self.tr_vc, self.tr_a, self.tr_c = x1, v1, torch.zeros_like(v1), torch.zeros_like(v1), torch.zeros_like(n, dtype=torch.bool)
        lng = (n >= self.min_pts).nonzero()[:, 0]                               # (shorter tracks are never called curved)
        if lng.numel() == 0:
            return
        # parabola: x = x0 + v tau + a tau^2 / 2
        w, n, tau, tp = w[lng], n[lng], tau[lng], self.tp[lng]
        A = torch.stack([w, tau, tau ** 2 / 2], 2)
        AtA = A.transpose(1, 2) @ A + torch.diag(torch.tensor([1e-4, 1e-4, 1e-4], device=self.cpu))
        s = torch.linalg.solve(AtA, A.transpose(1, 2) @ (tp * w[..., None]))
        x2, v2, a2 = s[:, 0], s[:, 1], s[:, 2]
        r1 = (((x1[lng][:, None] + v1[lng][:, None] * tau[..., None] - tp) ** 2).sum(2) * w).sum(1)
        r2 = (((x2[:, None] + v2[:, None] * tau[..., None] + a2[:, None] * (tau[..., None] ** 2) / 2 - tp) ** 2).sum(2) * w).sum(1)
        dof = (2 * n - 6).clamp(min=1)
        F_ = (r1 - r2) / 2 / (r2 / dof).clamp(min=0.02)                         # (0.02 px^2: the centroid cannot do better than that)
        # (a track that jumped to another bullet also looks curved, but no smooth curve fits it)
        curved = (F_ >= self.f_thr) & (r2 / dof <= self.max_r2)
        self.x0[lng], self.tr_vc[lng], self.tr_a[lng], self.tr_c[lng] = torch.where(curved[:, None], x2, x1[lng]), v2, a2, curved

    @torch.no_grad()
    def update(self, m, t, blobs=None):
        """m: bool half-size mask, t: seconds. blobs: (centroids in half-size px, is-dot, (ys, xs, blob of each pixel)) if the caller
        has found them already; m is not used then (perception v5 labels every blob anyway, on the mask before its dilation)."""
        now = t * 60.0
        det, dot, self.pix = blobs if blobs is not None else self._blobs(m)
        self.dot_idx, self.n_blobs = dot.nonzero()[:, 0], det.shape[0]          # (only these are tracked; specks and big blobs stay on the GPU)
        det = det[self.dot_idx].cpu()
        self.det = det
        n, m_old = det.shape[0], self.tp.shape[0]
        dev = self.cpu
        dt = 0.0 if self.t0 is None else now - self.t0
        step = max(float(round(dt)), 1.0)
        hit = torch.full((n,), -1, dtype=torch.long, device=dev)
        ok = torch.zeros(m_old, dtype=torch.bool, device=dev)
        if m_old and n and 0 < dt < 15:
            two = self.tv.sum(1) >= 2
            vel = torch.where(self.tr_c[:, None], self.tr_vc, self.tr_v)
            acc = torch.where(self.tr_c[:, None], self.tr_a, torch.zeros_like(self.tr_a))
            guess = self.x0 + vel * dt + acc * dt ** 2 / 2
            D = torch.cdist(guess, det)
            speed = vel.norm(dim=1)
            gate = torch.where(two, 5.0 + speed * dt * 0.35, torch.full_like(speed, max(self.reach, self.reach_pf * dt)))
            near_det, near_trk = D.argmin(1), D.argmin(0)
            all_t = torch.arange(m_old, device=dev)
            ok = (near_trk[near_det] == all_t) & (D[all_t, near_det] <= gate)
            hit[near_det[ok]] = all_t[ok]
            mv = ok & self.tv[:, -1] & (self.tv.sum(1) >= 4) & (speed >= 0.75)
            if int(mv.sum()) >= 8:                                              # the frames that really passed, from the bullets themselves
                dirs = vel[mv] / speed[mv][:, None]
                prog = (((det[near_det[mv]] - self.tp[mv, -1]) * dirs).sum(1) / speed[mv]).median().item()
                if abs(prog - dt) < 2.5:
                    step = max(float(round(prog)), 1.0)
        if dt > 0:                                                              # never drift more than a frame from the wall clock (a fast
            off = float(self.tt[-1]) + step - now                               # estimate of the speed would otherwise feed itself)
            if abs(off) > 1.0:
                step = max(step - round(off), 1.0)
        else:
            self.tt = self.tt.clone()
            self.tt[-1] = now
        self.step_frames = step
        self.t0 = now
        if dt > 0:
            self.tt = torch.cat([self.tt[1:], (self.tt[-1] + step)[None]])
            if m_old:
                self.tp = torch.cat([self.tp[:, 1:], torch.zeros(m_old, 1, 2, device=dev)], 1)
                self.tv = torch.cat([self.tv[:, 1:], torch.zeros(m_old, 1, dtype=torch.bool, device=dev)], 1)
        di = (hit >= 0).nonzero()[:, 0]
        if di.numel():
            self.tp[hit[di], -1] = det[di]
            self.tv[hit[di], -1] = True
        miss = torch.where(ok, torch.zeros_like(self.miss), self.miss + 1) if m_old else self.miss
        keep = miss <= self.misses
        new = (hit < 0).nonzero()[:, 0]
        ntp = torch.zeros(new.numel(), self.K, 2, device=dev)
        ntv = torch.zeros(new.numel(), self.K, dtype=torch.bool, device=dev)
        ntp[:, -1] = det[new]
        ntv[:, -1] = True
        idx = torch.full((m_old,), -1, dtype=torch.long, device=dev)
        idx[keep] = torch.arange(int(keep.sum()), device=dev)
        self.tp, self.tv = torch.cat([self.tp[keep], ntp]), torch.cat([self.tv[keep], ntv])
        self.miss = torch.cat([miss[keep], torch.zeros(new.numel(), dtype=torch.long, device=dev)])
        self.tid = torch.cat([self.tid[keep], torch.arange(self.next_id, self.next_id + new.numel(), device=dev)])
        self.next_id += new.numel()
        self._fit()
        # the motion of the current detections, from their refreshed tracks
        self.v_line, self.v_curve, self.a = torch.zeros(n, 2, device=dev), torch.zeros(n, 2, device=dev), torch.zeros(n, 2, device=dev)
        self.curved = torch.zeros(n, dtype=torch.bool, device=dev)
        self.known = torch.zeros(n, dtype=torch.bool, device=dev)
        self.det_tid = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.n_pts = torch.ones(n, dtype=torch.long, device=dev)                 # sightings in the window of each detection's track
        self.v2 = torch.zeros(n, 2, device=dev)                                 # its velocity from as few as 2 sightings
        if di.numel():
            seen = idx[hit[di]]
            good = (seen >= 0)
            s2 = seen.clamp(min=0)
            self.n_pts[di] = torch.where(good, self.tv[s2].sum(1), torch.ones_like(s2))
            self.v2[di] = torch.where(good[:, None], self.tr_v[s2], torch.zeros_like(self.tr_v[s2]))
            good = good & (self.tv[seen.clamp(min=0)].sum(1) >= 3)
            d_, s_ = di[good], seen[good]
            self.v_line[d_], self.v_curve[d_], self.a[d_], self.curved[d_] = self.tr_v[s_], self.tr_vc[s_], self.tr_a[s_], self.tr_c[s_]
            self.known[d_] = True
            self.det_tid[d_] = self.tid[s_]

    @torch.no_grad()
    def path(self, frames, model="turn"):
        """-> [n, frames, 2]: displacement of each current detection after 1, 2, ... frames. line: constant velocity for all; quad:
        constant acceleration where the track is significantly curved; turn: the same tracks with constant change of speed and constant
        turning, as the simulator's bullets move."""
        c = self.curved[:, None] if model != "line" else torch.zeros_like(self.curved)[:, None]
        v, a = torch.where(c, self.v_curve, self.v_line), torch.where(c, self.a, torch.zeros_like(self.a))
        tau = torch.arange(1, frames + 1, device=self.cpu).float()[None]
        if model in ("line", "quad"):
            return v[:, None] * tau[..., None] + a[:, None] * tau[..., None] ** 2 / 2
        s = v.norm(dim=1).clamp(min=1e-3)
        th = torch.atan2(v[:, 1], v[:, 0])
        moving = s > 0.3
        om = torch.where(moving, ((v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]) / s ** 2).clamp(-0.12, 0.12), torch.zeros_like(s))     # rad per frame
        acc = torch.where(moving, ((v * a).sum(1) / s).clamp(-0.2, 0.2), torch.zeros_like(s))
        ss = (s[:, None] + acc[:, None] * tau).clamp(min=0)
        hh = th[:, None] + om[:, None] * tau
        return torch.stack([torch.cumsum(ss * torch.cos(hh), 1), torch.cumsum(ss * torch.sin(hh), 1)], 2)

    @torch.no_grad()
    def shift(self, m, lead, model="turn", scale=1):
        """The mask `lead` frames ahead (scale: its pixels per half-size px). Pixels of bullets without a known motion stay in place."""
        if lead <= 0 or self.det is None or self.det.shape[0] == 0:
            return m
        d = torch.zeros(self.n_blobs, 2, device=self.dev)
        d[self.dot_idx] = self.path(max(int(round(lead)), 1), model)[:, -1].to(self.dev) * scale
        ys, xs, blob = self.pix
        ny, nx = ys + torch.round(d[blob, 1]).long(), xs + torch.round(d[blob, 0]).long()
        H, W = m.shape
        keep = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        out = torch.zeros(H * W + 1, dtype=torch.bool, device=self.dev)         # (the last cell takes what left the picture: no boolean
        out[torch.where(keep, ny * W + nx, H * W)] = True                       # indexing, which would wait for the GPU)
        return out[:-1].view(H, W)
