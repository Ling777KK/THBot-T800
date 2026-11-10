"""Evaluate a policy on fixed (family-group, difficulty) conditions, all in one batched env.

Score = survival time of each env's FIRST episode, capped at --cap_frames (cap counts as 'survived').
Policies: ckpt path, or one of: random | stay | greedy (privileged heuristic reference).
"""
import argparse, sys, time, json
import torch
from danmaku_env import *
from policy import ActorCritic

ap = argparse.ArgumentParser()
ap.add_argument("policy", help="path to ckpt.pt, or random | stay | greedy")
ap.add_argument("--per_cond", type=int, default=48)
ap.add_argument("--cap_frames", type=int, default=2400)
ap.add_argument("--diffs", default="0.3,0.6,0.9")
ap.add_argument("--argmax", action="store_true", help="deterministic policy instead of sampling")
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--json", default="")
ap.add_argument("--bomb", type=int, default=-1, help="-1: use what the checkpoint was trained with; 0/1 override")
ap.add_argument("--n_bombs", type=int, default=3, help="bombs available per episode (0 = a bomb policy must dodge only)")
ap.add_argument("--blind", default="", help="x_half,up,down game px blind zone around the player (as the real screen reader)")
ap.add_argument("--freeze_history", default="none", choices=["none", "loc", "glob", "both"],
                help="probe: replace the stacked past frames by copies of the newest one (no motion information) in the local / global view")
ap.add_argument("--focus_mode", default="policy", choices=["policy", "never", "always"],
                help="never: strip Shift (slow movement) from every action; always: force Shift (18-action policies only)")
ap.add_argument("--ymin_play", type=float, default=-1.0, help="-1: use the checkpoint's value; else force this limit")
ap.add_argument("--big_draw", default="24,64", help="big bullets drawn at full size (local nearest, global biggest)")
ap.add_argument("--suite", default="v1", choices=["v1", "v2", "v2train", "v5", "v5c", "v6", "v9", "v11"],
                help="v1: 5 train + 4 held-out families. v2: v1-train, v2-new-train, v1-held-out, strict-OOD. "
                     "v2train: every v2 training family separately (+ wall_gap for comparison). "
                     "v5: v2-train, boss-spell-like families, strict-OOD")
ap.add_argument("--art_p", type=float, default=0.0, help="fraction of episodes with self-sprite / item / enemy-blob artifacts (1 = always)")
ap.add_argument("--shot_p", type=float, default=0.0, help="fraction of episodes with the player's own amulet shots drawn into the observation")
ap.add_argument("--art_lib", default="", help="optional .pt with key 'lib': real sprite-residue masks")
ap.add_argument("--lat_p", type=float, default=0.0, help="fraction of episodes where the action acts one decision late")
ap.add_argument("--lat2_frac", type=float, default=0.0, help="among late episodes, the share that is two decisions late")
ap.add_argument("--fs_jitter", type=float, default=0.0, help="probability per step of a 3-frame decision interval")
ap.add_argument("--lagf", default="", help="probabilities of 0,1,2,.. game frames of lag, drawn every step (jittery lag)")
ap.add_argument("--lag_profile", action="store_true",
                help="print corr(action at decision k, movement between obs k+L and k+L+1) for L=-1..4 (compare with the real logs)")
ap.add_argument('--timing_profile',default='',help='joint timing JSON; by default inherit it from a timing-trained checkpoint')
ap.add_argument('--timing_trace',default='',help='timing replay JSON; by default inherit from checkpoint')
ap.add_argument("--act_repeat", type=int, default=1, help=">1: the policy may change its action only every k-th decision (k = 2: 15 changes per second)")
ap.add_argument("--sticky", type=float, default=0.0, help="add this to the logit of the previous action (fewer direction changes, same decision rate)")
ap.add_argument("--foresight", type=int, default=0, help=">0: simulator-only upper bound (foresight.py): veto moves that shorten how long the player can stay alive, planning this many frames ahead with the true state of the bullets alive now")
ap.add_argument("--fs_model", default="kin", choices=["kin", "line", "track", "traj"], help="--foresight: true kinematics, constant current velocity, only what the image tracker can know (future_env.SimPredictor), or what the prediction network says (traj_sim.py, --traj)")
ap.add_argument("--traj", default="", help="--fs_model traj: checkpoint of the prediction network")
ap.add_argument("--traj_margin", type=float, default=0.2, help="--fs_model traj: an action is out when its chance to survive is this share below the best action's")
ap.add_argument("--traj_dilate", type=int, default=0, help="--fs_model traj: grow the predicted maps by this many cells before the search (a bullet in the next cell can hit too)")
ap.add_argument("--traj_oracle", action="store_true", help="--fs_model traj: the true paths of the bullets alive now instead of the network, to test the search alone")
ap.add_argument("--fs_delay", type=int, default=0, help="--foresight: plan from the moment an action chosen now takes effect, this many frames ahead")
ap.add_argument("--obs_lead", type=int, default=0, help="simulator-only upper bound: the policy sees the world this many frames ahead (foresight.wrap_observe)")
ap.add_argument("--fs_pure", action="store_true", help="--foresight: no policy, follow the plan alone")
ap.add_argument("--safety", default="", help="safety network (safety_train.py): overrules the policy's choice when it calls that action doomed")
ap.add_argument("--safety_thr", type=float, default=0.7, help="--safety: the probability of being doomed from which an action is out")
ap.add_argument("--future_p", type=float, default=-1.0, help="models with future channels: share of the bullets tracked (0 = the channels stay empty, 1 with --future_q 0 = perfect tracking); default the middle of SimPredictor's range")
ap.add_argument("--future_q", type=float, default=0.0, help="--future_p: chance per decision to lose a track")
ap.add_argument("--bomber", default="", help="two-model setup: this bomb model decides bomb / no bomb, the policy (dodger) moves")
ap.add_argument("--bomb_thr", type=float, default=0.5, help="the bomb model bombs when P(bomb) exceeds this")
ap.add_argument("--bomb_gate", type=float, default=0.0, help=">0: the bomb model may bomb only when a bullet's edge is within this "
                "many game px of the player (as play.py --bomb_gate on the real game)")
args = ap.parse_args()
torch.manual_seed(args.seed)
dev = torch.device("cuda")

if args.suite == "v1":
    groups = [("TRAIN(all 5)", TRAIN_FAMILIES)] + [(f"  {FAMILY_NAMES[i]}", [i]) for i in TRAIN_FAMILIES] \
             + [("HELD-OUT(all 4)", HELDOUT_FAMILIES)] + [(f"  {FAMILY_NAMES[i]}", [i]) for i in HELDOUT_FAMILIES]
elif args.suite == "v2train":
    groups = [(f"  {FAMILY_NAMES[i]}", [i]) for i in list(range(9, 16)) + [7]]
elif args.suite in ("v5", "v5c", "v6", "v9", "v11"):
    groups = [("v2-TRAIN (12 fams)", TRAIN_V2), ("BOSS-like (3 fams)", BOSS_V5)] \
             + [(f"  {FAMILY_NAMES[i]}", [i]) for i in BOSS_V5 + (LATTICE if args.suite in ("v5c", "v6", "v9", "v11") else []) + (BOSS_V6 if args.suite in ("v6", "v9", "v11") else []) + (BOSS_V9 if args.suite in ("v9", "v11") else [])] + [("STRICT-OOD (5 fams)", OOD_V2)]
    if args.suite == "v11":
        groups += [("LATE-STAGE (6 fams)", BOSS_V11)] + [(f"  {FAMILY_NAMES[i]}", [i]) for i in BOSS_V11]
else:
    groups = [("v1-TRAIN (5 fams)", TRAIN_FAMILIES), ("v2-TRAIN new (7 fams)", [9, 10, 11, 12, 13, 14, 15]),
              ("v1-HELDOUT (4 fams)", HELDOUT_FAMILIES)] \
             + [(f"  {FAMILY_NAMES[i]}", [i]) for i in HELDOUT_FAMILIES] \
             + [("STRICT-OOD (5 fams)", OOD_V2), ("  diag_rain", [16, 17])] \
             + [(f"  {FAMILY_NAMES[i]}", [i]) for i in (18, 19, 20)]
diffs = [float(x) for x in args.diffs.split(",")]
conds = [(g, d) for d in diffs for g in groups]
fam_lists, d_vals, cond_id = [], [], []
for ci, (g, d) in enumerate(conds):
    for _ in range(args.per_cond):
        fam_lists.append(g[1]); d_vals.append(d); cond_id.append(ci)
N = len(fam_lists)
ck, ck_args = None, {}
if args.policy not in ("random", "stay", "greedy"):
    ck = torch.load(args.policy, map_location=dev)
    ck_args = ck.get("args", {})
bomb = bool(ck_args.get("bomb", 0)) if args.bomb < 0 else bool(args.bomb)
bomb = bomb or bool(args.bomber)
ymin = float(ck_args.get("ymin_play", 16.0)) if args.ymin_play < 0 else args.ymin_play
env = DanmakuEnv(n_envs=N, frame_skip=int(ck_args.get("frame_skip", 2)), max_frames=args.cap_frames, bomb_mode=bomb, ymin_play=ymin, n_bombs=args.n_bombs,
                 art_p=args.art_p, art_lib=torch.load(args.art_lib)["lib"] if args.art_lib else None, lat_p=args.lat_p,
                 fs_jitter=args.fs_jitter, d_max=max(1.0, max(diffs)), stack=(ck or {}).get("stack", 4), lags=(ck or {}).get("lags"), blind=[float(x) for x in args.blind.split(",")] if args.blind else None,
                 shot_p=args.shot_p, lat2_frac=args.lat2_frac,
                 prev_act_obs=(ck or {}).get("vec_dim", 3) >= 21, lagf=args.lagf.split(",") if args.lagf else None,
                 big_draw=[int(x) for x in args.big_draw.split(",")],
                 timing_profile=args.timing_profile or ((ck_args.get('timing_profile') or None) if not args.lagf else None),
                 timing_trace=args.timing_trace or ((ck_args.get('timing_trace') or None) if not args.lagf else None))
if (ck or {}).get("future"):
    import future_env
    fp = future_env.attach(env, args.seed, fixed=True)
    if args.future_p >= 0:
        fp.p[:], fp.q[:] = args.future_p, args.future_q
elif args.fs_model == "track":
    import future_env
    fp = future_env.track_only(env, args.seed, fixed=True)
    if args.future_p >= 0:
        fp.p[:], fp.q[:] = args.future_p, args.future_q
if args.obs_lead > 0:               # (after the future channels: those are then counted from the same moment ahead)
    import foresight
    foresight.wrap_observe(env, args.obs_lead)
env.set_conditions(fam_lists, d_vals)
if args.lag_profile and env.timing_clock is not None:
    raise ValueError('lag_profile assumes fixed intervals; use timing_eval.py for replay timing')
obs = env.reset()
print(f"env: bombs={'on' if bomb else 'off'}  player zone y>={ymin:g}")

net = None
if ck is not None:
    n_act, vdim = ck.get("n_actions", 18), ck.get("vec_dim", 3)
    if ck.get("arch", "cnn") == "fly":
        from flybrain.flypolicy import FlyActorCritic
        fl = ck["fly"]
        net = FlyActorCritic(n_act, vdim, ck["lags"], graph=fl["graph"], th=fl["th"], micro=fl["micro"], seed=fl["seed"]).to(dev)
    elif ck.get("arch") == "cnn_gru":
        from memory_policy import build_policy
        net = build_policy(ck, dev)
    else:
        net = ActorCritic(n_act, vdim, ck.get("stack", 4), ck.get("lags")).to(dev).to(memory_format=torch.channels_last)
        if ck.get("future"):
            future_env.widen(net)
    net.load_state_dict(ck["net"]); net.eval()
    print(f"loaded {args.policy} (update {ck['update']}, {ck['steps']/1e6:.1f}M steps, {n_act} actions, vec {vdim})")
    if (vdim != env.vec_dim or n_act != env.n_actions) and not args.bomber:
        sys.exit("checkpoint and env disagree on actions / vector size (a bomb policy needs --bomb 1, an old one --bomb 0)")
bomber = None
if args.bomber:
    bk = torch.load(args.bomber, map_location=dev)
    bomber = ActorCritic(2, bk["vec_dim"], bk["stack"], bk["lags"]).to(dev).to(memory_format=torch.channels_last)
    bomber.load_state_dict(bk["net"]); bomber.eval()
    print(f"bomb model {args.bomber} (update {bk['update']}, {bk['steps']/1e6:.1f}M steps), bombs when P > {args.bomb_thr}")
safety = None
if args.safety:
    sk = torch.load(args.safety, map_location=dev)
    safety = ActorCritic(sk["n_actions"], sk["vec_dim"], sk["stack"], sk["lags"]).to(dev).to(memory_format=torch.channels_last)
    safety.load_state_dict(sk["net"]); safety.eval()
    print(f"safety network {args.safety} ({sk['update']} cycles), action out when P(doomed) > {args.safety_thr}")
sim_traj = None
if args.fs_model == "traj":
    from traj_sim import SimTraj
    sim_traj = SimTraj(env, args.traj, args.foresight, args.traj_margin, args.fs_delay, dilate=args.traj_dilate, oracle=args.traj_oracle)


recurrent_state = None
recurrent_starts = None
fs_stats = [0, 0]                   # --foresight: decisions changed, decisions planned
hold = dict(a=None, n=0, changes=0.0, seen=0.0)     # --act_repeat / --sticky: previous action; direction changes counted for every run


def act(obs):
    global recurrent_state
    if args.policy == "random": return random_action(env)
    if args.policy == "stay": return stay_action(env)
    if args.policy == "greedy": return greedy_action(env)
    if args.freeze_history != "none":
        obs = dict(obs)
        k = obs["loc"].shape[1] - 1                    # stacked frames; the last channel is the wall / marker map
        for key in (("loc",) if args.freeze_history == "loc" else ("glob",) if args.freeze_history == "glob" else ("loc", "glob")):
            x = obs[key].clone()
            x[:, :k] = x[:, k - 1:k]
            obs[key] = x
    with torch.no_grad():
        policy_obs = dict(obs, vec=obs["vec"][:, :net.vec_dim]) if bomber is not None else obs
        if getattr(net, "is_recurrent", False):
            logits, _, recurrent_state = net.forward_step(policy_obs, recurrent_state, recurrent_starts)
        else:
            logits, _ = net(policy_obs)
        if args.sticky > 0 and hold["a"] is not None:
            logits = logits.clone(); logits[torch.arange(len(logits), device=logits.device), hold["a"]] += args.sticky
        a = logits.argmax(1) if args.argmax else torch.distributions.Categorical(logits=logits).sample()
        if safety is not None:
            veto = torch.sigmoid(safety(obs)[0].float()) > args.safety_thr
            veto[veto.all(1)] = False
            a_s = logits.masked_fill(veto, -1e6).argmax(1)
            fs_stats[0] += int((a_s != a)[~first_done].sum()); fs_stats[1] += int((~first_done).sum())
            a = a_s
        if args.foresight > 0:
            idx_ = (~first_done).nonzero()[:, 0]
            if sim_traj is not None:
                sim_traj.push(recurrent_starts)
            if len(idx_):
                import foresight
                if sim_traj is not None:
                    a_fs = sim_traj.choose(idx_, logits[idx_])[0]
                else:
                    a_fs, vetoed = foresight.shield(env, idx_, logits[idx_], args.foresight, args.fs_model, args.fs_pure, args.fs_delay)
                fs_stats[0] += int((a_fs != a[idx_]).sum()); fs_stats[1] += len(idx_)
                a = a.clone(); a[idx_] = a_fs
    if args.focus_mode == "never":
        a = torch.where(a < 18, a % 9, a)
    elif args.focus_mode == "always":
        a = torch.where(a < 18, a % 9 + 9, a)
    if args.act_repeat > 1 and hold["a"] is not None and hold["n"] % args.act_repeat != 0:
        a = hold["a"]
    if hold["a"] is not None:
        hold["changes"] += float(((a % 9) != (hold["a"] % 9))[~first_done].sum()); hold["seen"] += float((~first_done).sum())
    hold["a"], hold["n"] = a, hold["n"] + 1
    return a


first_done = torch.zeros(N, dtype=torch.bool, device=dev)
first_len = torch.full((N,), float(args.cap_frames), device=dev)
first_hit = torch.zeros(N, dtype=torch.bool, device=dev)
ysum = torch.zeros(N, device=dev); ycnt = torch.zeros(N, device=dev)      # where the player stands (first episode)
min_bombs = torch.full((N,), env.n_bombs, device=dev)
prof = []                       # --lag_profile: (action, x position at the observation, still in the first episode) per step
t0 = time.time()
for step in range(args.cap_frames*4+2 if env.timing_clock is not None else args.cap_frames // env.fs + 2):
    live = (~first_done).float()
    ysum += env.py * live; ycnt += live
    min_bombs = torch.where(first_done, min_bombs, torch.minimum(min_bombs, env.bombs))
    a_ = act(obs)
    if args.lag_profile:
        prof.append((a_.clone(), env.px.clone(), (~first_done).clone()))
    if bomber is not None:
        with torch.no_grad():
            b_ = torch.softmax(bomber(obs)[0], 1)[:, 1] > args.bomb_thr
            if args.bomb_gate > 0:
                dx_, dy_ = env.bx - env.px[:, None], env.by - env.py[:, None]
                edge = torch.where(env.balive, (dx_ * dx_ + dy_ * dy_).sqrt() - env.brad, torch.full_like(dx_, 1e9)).amin(1)
                b_ &= edge < args.bomb_gate
        obs, r, d, info = env.step(a_, bomb_req=b_)
    else:
        obs, r, d, info = env.step(a_)
    recurrent_starts = d
    new = d & ~first_done
    first_len = torch.where(new, info["ep_frames"].float().clamp(max=args.cap_frames), first_len)
    first_hit |= new & info["hit"]
    first_done |= d
    if step % 100 == 99 and bool(first_done.all()):
        break
print(f"eval took {time.time()-t0:.0f}s over {N} envs")
if hold["seen"]:
    print(f"direction changes: {hold['changes'] / hold['seen'] * 60 / env.fs:.1f} per second")
if safety is not None:
    print(f"safety: {fs_stats[0]} of {fs_stats[1]} decisions changed ({100 * fs_stats[0] / max(fs_stats[1], 1):.1f}%)")
if args.foresight > 0:
    print(f"foresight {args.foresight} frames, model {args.fs_model}{', plan alone' if args.fs_pure else ''}: {fs_stats[0]} of {fs_stats[1]} decisions changed ({100 * fs_stats[0] / max(fs_stats[1], 1):.1f}%)")
if args.lag_profile:                # same statistic as on the real game's play logs (realgame log.npy: t, px, py, a, ...)
    A = torch.stack([p[0] for p in prof]); X = torch.stack([p[1] for p in prof]); L_ = torch.stack([p[2] for p in prof])
    dxa = env.DX[A % 9].float() * torch.where(A >= 9, 2.0, 4.0) * torch.where((env.DX[A % 9] != 0) & (env.DY[A % 9] != 0), 0.7071, 1.0)
    vx = (X[1:] - X[:-1]) / env.fs
    ok = L_[1:] & L_[:-1]
    T = vx.shape[0]
    out = []
    for L in range(-1, 5):
        k = torch.arange(1, T - 5, device=dev)
        u, v, m = dxa[k], vx[k + L], ok[k + L] & ok[k]
        u, v = u[m], v[m]
        out.append(f"L={L}: {torch.corrcoef(torch.stack([u, v]))[0, 1].item():.2f}")
    print("lag profile (corr of action at k with movement between obs k+L and k+L+1): " + "  ".join(out))

cid = torch.tensor(cond_id, device=dev)
res = {}
extra, ext_rows = [], []
print(f"\npolicy ={args.policy}   (cap {args.cap_frames/60:.0f}s, {args.per_cond} episodes per cell; entries: mean survival s | % reaching cap)")
print(f"{'':18s}" + "".join(f"   d={d:<4}        " for d in diffs))
for gi, (gname, _) in enumerate(groups):
    row = f"{gname:18s}"
    for d in diffs:
        ci = conds.index((groups[gi], d))
        m = cid == ci
        surv = first_len[m] / 60
        reached = (~first_hit[m]).float().mean().item() * 100
        row += f"   {surv.mean().item():5.1f}s | {reached:3.0f}%  "
        ymean = (ysum[m] / ycnt[m].clamp(min=1)).mean().item()
        bused = (env.n_bombs - min_bombs[m]).mean().item()
        res[f"{gname.strip()}@{d}"] = dict(mean_s=surv.mean().item(), cap_pct=reached, mean_y=ymean, bombs_used=bused)
        extra.append(f"y{ymean:4.0f}/b{bused:.1f}")
    print(row)
    ext_rows.append(f"{gname:18s}" + "".join(f"   {e:<15s}" for e in extra))
    extra = []
print("\nwhere the player stands (mean y, game px; 384 = spawn, 16 = top) / bombs used per episode:")
print("\n".join(ext_rows))
if args.json:
    json.dump(res, open(args.json, "w"), indent=1)
