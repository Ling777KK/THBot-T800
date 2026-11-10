"""Simulator-only upper bound for "let the policy see the future" (user's idea, 2026-11-02: the CNN walks into doomed areas).
Privileged: uses the simulator's true bullet state, so it cannot run on the real game as it is -- it answers how much a predictor
could gain at best before one is built. Only bullets alive NOW are rolled forward; future shots are unknown, as on the real game.

future_occupancy: [T, N, HC, WC] bool, cells where a player centre would be hit during each of the next T decisions.
  model 'kin'  = true kinematics (speed limits, acceleration, turning); 'line' = constant current velocity (what a velocity estimate
  from the picture could give). Retarget timers are not applied (where those bullets go next depends on the player).
action_scores: for each of the 18 actions, for how many decisions the player can stay alive after taking it and then moving as well
  as possible (a backward pass over the time-expanded grid). shield: the policy's choice among the actions with the best score."""
import torch
import torch.nn.functional as F

from danmaku_env import W, H, N_ACTIONS

CELL = 4.0
WC, HC = int(W // CELL), int(H // CELL)


@torch.no_grad()
def pending_position(env, idx, delay):
    """Where the player will be when an action chosen now starts to act, `delay` frames from now: the actions already sent but not
    yet effective keep moving it (the same frame -> decision mapping as DanmakuEnv.step with lagf)."""
    x, y = env.px[idx].clone(), env.py[idx].clone()
    for f in range(delay):
        vx, vy, _ = env._vel(env.prev_act if (delay - f + env.fs - 1) // env.fs <= 1 else env.prev_act2)
        x = (x + vx[idx]).clamp(8, W - 8)
        y = (y + vy[idx]).clamp(env.ymin_play, H - 16)
    return x, y


def wrap_observe(env, lead):
    """Upper bound for cancelling the delay by prediction: the policy sees the world `lead` frames ahead (true kinematics of the
    bullets alive now, the player moved by its pending actions), so it acts on what will be there when its action takes effect."""
    orig = env._observe
    every = torch.arange(env.N, device=env.dev)

    def ahead(reset_mask):
        saved = {n: getattr(env, n) for n in ("bx", "by", "bs", "bth", "px", "py")}
        try:
            s, th, x, y = env.bs, env.bth, env.bx, env.by
            for _ in range(lead):
                s = torch.minimum(torch.maximum(s + env.bacc, env.bsmin), env.bsmax)
                th = th + env.bom
                x, y = x + s * torch.cos(th), y + s * torch.sin(th)
            px, py = pending_position(env, every, lead)
            env.bs, env.bth, env.bx, env.by, env.px, env.py = s, th, x, y, px, py
            return orig(reset_mask)
        finally:
            for n, v in saved.items():
                setattr(env, n, v)
    env._observe = ahead


@torch.no_grad()
def future_occupancy(env, idx, steps, model="kin", skip=0):
    """idx: long [n], the environments to plan for. skip: frames to roll the bullets forward before the first decision counts."""
    x, y = env.bx[idx].clone(), env.by[idx].clone()
    s, th = env.bs[idx].clone(), env.bth[idx].clone()
    acc, om, smin, smax = env.bacc[idx], env.bom[idx], env.bsmin[idx], env.bsmax[idx]
    alive = env.balive[idx]
    thr = env.rp[idx][:, None] + env.hit_scale * env.brad[idx]
    k = torch.floor(thr / CELL + 0.5).clamp(1, 6).long()                # dilation of a bullet's cell, in cells
    kinds = [int(v) for v in k[alive].unique().tolist()] if bool(alive.any()) else []
    n = len(idx)
    occ = torch.zeros(steps, n, HC, WC, dtype=torch.bool, device=env.dev)
    row = torch.arange(n, device=env.dev)[:, None].expand_as(x)
    if model == "line":                                                 # velocity as it is now, kept constant
        vx, vy = s * torch.cos(th), s * torch.sin(th)
    for _ in range(skip):
        if model == "kin":
            s = torch.minimum(torch.maximum(s + acc, smin), smax)
            th = th + om
            x, y = x + s * torch.cos(th), y + s * torch.sin(th)
        else:
            x, y = x + vx, y + vy
    for t in range(steps):
        for _ in range(env.fs):
            if model == "kin":
                s = torch.minimum(torch.maximum(s + acc, smin), smax)
                th = th + om
                x, y = x + s * torch.cos(th), y + s * torch.sin(th)
            else:
                x, y = x + vx, y + vy
            ci, cj = torch.floor(x / CELL).long(), torch.floor(y / CELL).long()
            on = alive & (ci >= -6) & (ci < WC + 6) & (cj >= -6) & (cj < HC + 6)
            for kk in kinds:
                m = on & (k == kk)
                if not bool(m.any()):
                    continue
                pad = 6                                                 # bullets just outside the field still reach into it
                g = torch.zeros(n, 1, HC + 2 * pad, WC + 2 * pad, device=env.dev, dtype=torch.float16)
                g[row[m], 0, cj[m] + pad, ci[m] + pad] = 1
                d = F.max_pool2d(g, 2 * kk + 1, 1, kk)[:, 0, pad:pad + HC, pad:pad + WC]
                occ[t] |= d > 0
    return occ


@torch.no_grad()
def tracked_occupancy(env, idx, steps, skip=0):
    """future_occupancy from what the image tracker can know (env.future, future_env.SimPredictor): the bullets without a track are not in it."""
    n, d = len(idx), env.dev
    occ = torch.zeros(steps, n, HC, WC, dtype=torch.bool, device=d)
    where = torch.full((env.N,), -1, device=d)
    where[idx] = torch.arange(n, device=d)
    e, b = env.future.known()
    keep = where[e] >= 0
    e, b = e[keep], b[keep]
    if e.numel() == 0:
        return occ
    X, Y = env.future.positions(e, b, torch.arange(skip + 1, skip + steps * env.fs + 1, device=d, dtype=torch.float32))
    k = torch.floor((env.rp[e] + env.hit_scale * env.brad[e, b]) / CELL + 0.5).clamp(1, 6).long()
    ci, cj = torch.floor(X / CELL).long(), torch.floor(Y / CELL).long()
    on = (ci >= -6) & (ci < WC + 6) & (cj >= -6) & (cj < HC + 6)
    row = where[e][:, None].expand_as(ci)
    pad = 6
    for t in range(steps):
        fr = slice(t * env.fs, (t + 1) * env.fs)
        for kk in k.unique().tolist():
            m = on[:, fr] & (k == kk)[:, None]
            if not bool(m.any()):
                continue
            g = torch.zeros(n, 1, HC + 2 * pad, WC + 2 * pad, device=d, dtype=torch.float16)
            g[row[:, fr][m], 0, cj[:, fr][m] + pad, ci[:, fr][m] + pad] = 1
            occ[t] |= F.max_pool2d(g, 2 * kk + 1, 1, kk)[:, 0, pad:pad + HC, pad:pad + WC] > 0
    return occ


@torch.no_grad()
def action_scores(env, idx, occ, px=None, py=None):
    """-> [n, 18] float: decisions survived after taking each action (0 = hit during this very move, T = alive to the horizon).
    px, py: where the player is when the action starts to act (default: now)."""
    px = env.px[idx] if px is None else px
    py = env.py[idx] if py is None else py
    T, n = occ.shape[0], occ.shape[1]
    d = env.dev
    xs = (torch.arange(WC, device=d).float() + 0.5) * CELL
    ys = (torch.arange(HC, device=d).float() + 0.5) * CELL
    legal = ((ys >= env.ymin_play - CELL) & (ys <= H - 16 + CELL))[:, None] & ((xs >= 8 - CELL) & (xs <= W - 8 + CELL))[None, :]
    V = torch.full((n, 1, HC, WC), float(T), device=d)
    V = torch.where(legal[None, None], V, torch.full_like(V, -1.0))
    ax, dg = env.fs, int(0.7071 * env.fs + 0.5)                         # one decision = fs frames at up to 4 px: +-fs cells, less diagonally
    for t in range(T - 1, 0, -1):
        reach = torch.maximum(torch.maximum(F.max_pool2d(V, (2 * ax + 1, 1), 1, (ax, 0)), F.max_pool2d(V, (1, 2 * ax + 1), 1, (0, ax))),
                              F.max_pool2d(V, 2 * dg + 1, 1, dg))
        V = torch.where(occ[t][:, None], torch.full_like(V, float(t)), reach)
        V = torch.where(legal[None, None], V, torch.full_like(V, -1.0))
    acts = torch.arange(N_ACTIONS, device=d)
    move, foc = acts % 9, (acts // 9).float()
    ddx, ddy = env.DX[move].float(), env.DY[move].float()
    norm = torch.where((ddx != 0) & (ddy != 0), 0.70710678, 1.0)
    spd = torch.where(foc > 0, 2.0, 4.0)[None] * env.pspeed[idx][:, None] * norm[None]
    nx = (px[:, None] + ddx[None] * spd * env.fs).clamp(8, W - 8)
    ny = (py[:, None] + ddy[None] * spd * env.fs).clamp(env.ymin_play, H - 16)
    ci, cj = (nx / CELL).long().clamp(0, WC - 1), (ny / CELL).long().clamp(0, HC - 1)
    row = torch.arange(n, device=d)[:, None].expand_as(ci)
    score = V[row, 0, cj, ci]
    return torch.where(occ[0][row, cj, ci], torch.zeros_like(score), score)


@torch.no_grad()
def doomed_actions(env, idx, horizon_frames, delay=0):
    """Labels for training (ppo_train.py --teach): [n, 18] bool, the actions that shorten how long the player can stay alive, by the true
    paths of the bullets alive now. The network sees only its own input, so it has to work out from the past frames where they are going."""
    steps = max(1, horizon_frames // env.fs)
    px, py = pending_position(env, idx, delay)
    score = action_scores(env, idx, future_occupancy(env, idx, steps, "kin", skip=delay), px, py)
    return score < score.max(1, keepdim=True).values


@torch.no_grad()
def shield(env, idx, logits, horizon_frames, model="kin", pure=False, delay=0):
    """The policy keeps its choice unless that shortens how long the player can foreseeably stay alive. logits: [n, 18].
    pure: no policy at all, best score, ties broken towards the lower middle of the field and fewer moves.
    delay: frames until an action chosen now starts to act: plan from that moment and that position."""
    steps = max(1, horizon_frames // env.fs)
    px, py = pending_position(env, idx, delay)
    occ = tracked_occupancy(env, idx, steps, delay) if model == "track" else future_occupancy(env, idx, steps, model, skip=delay)
    score = action_scores(env, idx, occ, px, py)
    best = score.max(1, keepdim=True).values
    if pure:
        acts = torch.arange(N_ACTIONS, device=env.dev)
        move, foc = acts % 9, (acts // 9).float()
        nx = env.px[idx][:, None] + env.DX[move][None] * 4.0
        ny = env.py[idx][:, None] + env.DY[move][None] * 4.0
        tie = -0.002 * (nx - W / 2).abs() - 0.002 * (ny - (H - 70)).abs() - 0.01 * (move != 0).float()[None] + 0.005 * foc[None]
        return (score + tie).argmax(1), (score < best)
    vetoed = score < best
    return logits.masked_fill(vetoed, float("-inf")).argmax(1), vetoed
