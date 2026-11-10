"""PPO baseline on the procedural danmaku env (training families only; held-out families are for eval)."""
import argparse, os, time, json
import torch
import torch.nn.functional as F
from danmaku_env import DanmakuEnv, TRAIN_FAMILIES, TRAIN_V2, TRAIN_V5, TRAIN_V5C, TRAIN_V6, N_ACTIONS, TRAIN_V9, TRAIN_V11
from policy import ActorCritic

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="baseline")
ap.add_argument("--train_set", default="v1", choices=["v1", "v2", "v5", "v5c", "v6", "v9", "v11"],
                help="v1: 5 families; v2: 12 broader families; v5: v2 + 3 boss-spell-like families")
ap.add_argument("--art_p", type=float, default=0.0, help="fraction of episodes with harmless self-sprite / item / enemy-blob artifacts")
ap.add_argument("--shot_p", type=float, default=0.0, help="fraction of episodes with the player's own amulet shots drawn into the observation (harmless)")
ap.add_argument("--art_lib", default="", help="optional .pt with key 'lib' [2,M,64,64] uint8: real sprite-residue masks")
ap.add_argument("--lat_p", type=float, default=0.0, help="fraction of episodes where the action acts one decision late")
ap.add_argument("--prev_act", type=int, default=0, help="1: the policy sees its previous action (vector input 3 -> 21)")
ap.add_argument("--lagf", default="", help="probabilities of 0,1,2,.. game frames of lag drawn every step (jittery lag, as measured on the real game)")
ap.add_argument("--lat2_frac", type=float, default=0.0, help="among late episodes, the share that is two decisions late")
ap.add_argument('--timing_profile',default='',help='optional measured joint sensor/action delay profile JSON')
ap.add_argument('--timing_trace',default='',help='empirical contiguous replay of observation intervals and conditional delays')
ap.add_argument("--fs_jitter", type=float, default=0.0, help="probability per step that the decision interval is 3 frames instead of 2")
ap.add_argument("--sw_dir", type=float, default=0.0, help="reward penalty for changing the move direction")
ap.add_argument("--sw_focus", type=float, default=0.0, help="reward penalty for toggling Shift")
ap.add_argument("--d_max", type=float, default=1.0, help="top of the difficulty scale (1.0 = old)")
ap.add_argument("--d_lo", type=float, default=0.0, help="lowest difficulty of new episodes (0 = old: uniform from 0)")
ap.add_argument("--d_start", type=float, default=0.2, help="max difficulty at the start of the curriculum (warm starts can begin higher)")
ap.add_argument("--stack", type=int, default=4, help="frames stacked in the observation (a warm start needs the same value)")
ap.add_argument("--envs", type=int, default=512)
ap.add_argument("--rollout", type=int, default=64)
ap.add_argument("--total", type=float, default=100e6, help="total decision steps (env-steps)")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--gamma", type=float, default=0.99)
ap.add_argument("--lam", type=float, default=0.95)
ap.add_argument("--frame_skip", type=int, default=2, help="game frames per decision (2 = 30 decisions per second, 3 = 20); the alive reward per second is kept, gamma and lam are per decision")
ap.add_argument("--clip", type=float, default=0.2)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--minibatches", type=int, default=4)
ap.add_argument("--ent", type=float, default=0.01)
ap.add_argument("--curriculum", type=float, default=0.4, help="fraction of training over which max difficulty ramps 0.2 -> 1")
ap.add_argument("--save_every", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume", default="")
ap.add_argument("--wall_shaping", type=int, default=0, help="1: add the gap-proximity reward for walls (training only)")
ap.add_argument("--bomb", type=int, default=0, help="1: add the bomb action (19 actions, 5-dim vector input)")
ap.add_argument("--ymin_play", type=float, default=16.0, help="player may not go above this y (game px); 16 = no limit")
ap.add_argument("--high_pen", type=float, default=0.0, help="soft penalty per step at the top edge (0 at y=high_ref)")
ap.add_argument("--high_ref", type=float, default=320.0)
ap.add_argument("--bomb_hold", type=float, default=0.0, help="reward per bomb in stock per step (conserve bombs)")
ap.add_argument("--bomb_cost", type=float, default=0.15, help="reward cost of every bomb")
ap.add_argument("--bomb_waste", type=float, default=0.0, help="extra cost of a bomb when no bullet was heading for the player (could have dodged)")
ap.add_argument("--bomb_keep", type=float, default=0.0, help="extra reward per bomb left when the time limit is reached")
ap.add_argument("--bomb_value", type=float, default=0.0, help="worth of a bomb in stock, settled at once: spending costs it, dying with b bombs costs b times it")
ap.add_argument("--blind", default="", help="x_half,up,down game px: blank bullets around the player like the real screen reader (19.33,10.67,22.67)")
ap.add_argument("--lags", default="", help="spaced frame history, e.g. 30,16,8,4,2,1,0 (decisions back); empty = last --stack frames")
ap.add_argument("--init_from", default="", help="warm start from an older checkpoint (sizes may differ)")
ap.add_argument("--future", type=int, default=0, help="1: four more input channels, where the bullets will go during the next second as the image tracker could estimate it (future_env.py)")
ap.add_argument("--teach", type=int, default=0, help=">0: label this many envs per decision with the actions that the search on the true future calls doomed (foresight.doomed_actions); the policy is pushed to rank them below the others")
ap.add_argument("--teach_w", type=float, default=0.5, help="--teach: weight of that loss")
ap.add_argument("--teach_margin", type=float, default=0.5, help="--teach: how far (logits) a doomed action has to stay below the best other one")
ap.add_argument("--teach_h", type=int, default=30, help="--teach: frames the search looks ahead")
ap.add_argument("--teach_delay", type=int, default=0, help="--teach: frames until an action chosen now acts, for the search")
ap.add_argument("--out_root", default=os.environ.get("TOUHOU_RUNS", r"C:\touhou-runs"), help="kept outside any synced folder: checkpoints are ~40MB")
ap.add_argument("--arch", default="cnn", choices=["cnn", "fly"], help="fly: fruit-fly connectome core (flybrain/flypolicy.py)")
ap.add_argument("--fly_graph", default="real", choices=["real", "rewired", "random"])
ap.add_argument("--fly_th", type=int, default=5, help="keep connections with at least this many synapses")
ap.add_argument("--fly_micro", type=int, default=8, help="recurrent steps of the connectome per decision")
ap.add_argument("--fly_seed", type=int, default=0, help="seed of the rewired / random graph")
ap.add_argument("--chunk", type=int, default=0, help=">0: split every minibatch into chunks of this size (gradient accumulation; big nets)")
ap.add_argument("--big_draw", default="24,64", help="big bullets drawn at full size: nearest N in the local view, biggest M in the global view")
ap.add_argument("--dodger", default="", help="two-model setup: this checkpoint (frozen, argmax) moves; the trained net only decides bomb / no bomb")
ap.add_argument("--hit_scale", type=float, default=0.8, help="bullet hit radius = this x its drawn radius (+ the player's)")
ap.add_argument("--graze_reward", type=float, default=0.0, help="per step, not hit and not invulnerable: this x (closeness of the "
                "nearest bullet)^2 -- surviving a close call without bombing pays, the closer the more")
ap.add_argument("--graze_dist", type=float, default=12.0, help="game px beyond the hit distance where a graze starts to count")
ap.add_argument("--plateau", type=float, default=-1, help="stop (after saving) once the smoothed mean reward per step has not "
                "improved by 2%% for this many M steps; -1 = 3 for bomb models (--dodger), off otherwise; 0 = off")
ap.add_argument("--plateau_min", type=float, default=2.0, help="M steps before --plateau may stop a run")
ap.add_argument('--validate_every',type=int,default=0,help='optional fixed validation interval in env steps; 0 keeps original trainer behaviour')
ap.add_argument('--validation_lagf',default='',help='common deployment-delay distribution for validation, independent of training arm')
ap.add_argument('--validation_seed',type=int,default=7001)
ap.add_argument('--validation_timing_profile',default='',help='common measured joint timing profile for fixed validation')
ap.add_argument('--validation_timing_trace',default='',help='common held-out observation-timing replay for validation')
ap.add_argument('--validation_per_condition',type=int,default=4)
ap.add_argument('--validation_frames',type=int,default=2400)
args = ap.parse_args()
if args.plateau < 0:
    args.plateau = 3.0 if args.dodger else 0.0
if args.dodger:
    args.bomb = 1

torch.manual_seed(args.seed)
dev = torch.device("cuda")
out_dir = os.path.join(args.out_root, args.name)
os.makedirs(out_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=1)

env = DanmakuEnv(n_envs=args.envs, frame_skip=args.frame_skip, alive_reward=0.02 * args.frame_skip / 2, wall_shaping=bool(args.wall_shaping), gamma=args.gamma,
                 bomb_mode=bool(args.bomb), ymin_play=args.ymin_play, high_pen=args.high_pen, high_ref=args.high_ref,
                 bomb_hold=args.bomb_hold, bomb_keep=args.bomb_keep, bomb_cost=args.bomb_cost, bomb_waste=args.bomb_waste, art_p=args.art_p,
                 art_lib=torch.load(args.art_lib)["lib"] if args.art_lib else None, lat_p=args.lat_p,
                 fs_jitter=args.fs_jitter, sw_dir=args.sw_dir, sw_focus=args.sw_focus, d_max=args.d_max, stack=args.stack, lags=[int(x) for x in args.lags.split(',')] if args.lags else None,
                 blind=[float(x) for x in args.blind.split(',')] if args.blind else None,
                 shot_p=args.shot_p, lat2_frac=args.lat2_frac, prev_act_obs=bool(args.prev_act),
                 lagf=args.lagf.split(",") if args.lagf else None, bomb_value=args.bomb_value,
                 big_draw=[int(x) for x in args.big_draw.split(",")], hit_scale=args.hit_scale,
                 graze_reward=args.graze_reward, graze_dist=args.graze_dist,timing_profile=args.timing_profile or None,
                 timing_trace=args.timing_trace or None)
env.set_families({"v1": TRAIN_FAMILIES, "v2": TRAIN_V2, "v5": TRAIN_V5, "v5c": TRAIN_V5C, "v6": TRAIN_V6, "v9": TRAIN_V9,
                  "v11": TRAIN_V11}[args.train_set])
if args.future:
    import future_env
    future_env.attach(env, args.seed)
if args.arch == "fly":
    from flybrain.flypolicy import FlyActorCritic
    net = FlyActorCritic(env.n_actions, env.vec_dim, env.lags, graph=args.fly_graph, th=args.fly_th, micro=args.fly_micro,
                         seed=args.fly_seed).to(dev)
    if not args.resume:
        net.calibrate()
else:
    net = ActorCritic(2 if args.dodger else env.n_actions, env.vec_dim, env.stack, env.lags).to(dev).to(memory_format=torch.channels_last)
print(f"{args.arch}: {sum(p.numel() for p in net.parameters()):,} trainable parameters")
if args.init_from:
    src = torch.load(args.init_from, map_location=dev)
    net.warm_start_from(src["net"], src.get("lags") or list(range(src.get("stack", 4) - 1, -1, -1)))
    print(f"warm start from {args.init_from}  (actions {2 if args.dodger else env.n_actions}, vec {env.vec_dim})")
if args.future:
    future_env.widen(net)
dodger = None
if args.dodger:
    dk = torch.load(args.dodger, map_location=dev)
    dodger = ActorCritic(dk["n_actions"], dk["vec_dim"], dk["stack"], dk["lags"]).to(dev).to(memory_format=torch.channels_last)
    dodger.load_state_dict(dk["net"]); dodger.eval()
    assert dk["n_actions"] == N_ACTIONS and dodger.lags == env.lags, "the dodger must be a move-only policy with the same frame history"
    if not args.resume and not (args.init_from and src.get("n_actions") == 2):   # (from a bomb model: keep its head)
        # fresh bomb head (warm-started rows would be move actions): 0 = no bomb, 1 = bomb, rare at first
        torch.nn.init.orthogonal_(net.pi.weight, 0.01)
        with torch.no_grad():
            net.pi.bias.copy_(torch.tensor([0.0, -6.0]))
    print(f"two-model setup: dodger {args.dodger} (frozen), training the bomb model only")


def dodge(o):
    """The frozen dodger's move (argmax, as on the real game); it sees only the inputs it was trained with."""
    with torch.no_grad():
        return dodger(dict(loc=o["loc"], glob=o["glob"], vec=o["vec"][:, :dodger.vec_dim]))[0].argmax(1)


if dodger is not None and args.init_from and not args.resume and src.get("n_actions") == N_ACTIONS + 1:
    # Start the bomb head from the warm-start policy's own sense of when to bomb (a fresh head learns 'never bomb' long before
    # it stumbles on a bomb just before a hit): fit that policy's log-odds of bombing vs any move as a linear function of the
    # shared hidden layer, on states visited with the dodger moving and that policy bombing.
    src_net = ActorCritic(src["n_actions"], src["vec_dim"], src["stack"], src["lags"]).to(dev).to(memory_format=torch.channels_last)
    src_net.load_state_dict(src["net"]); src_net.eval()
    env.set_difficulty(args.d_lo, args.d_start)
    o, Hs, Zs = env.reset(), [], []
    for _ in range(128):
        with torch.no_grad():
            l, _ = src_net(o)
            ok = (o["vec"][:, 3] > 0) & (o["vec"][:, 4] <= 0)          # bombs left and not invulnerable
            Hs.append(net.hidden(o)[ok]); Zs.append((l[:, N_ACTIONS] - torch.logsumexp(l[:, :N_ACTIONS], 1))[ok].clamp(-12, 4))
            b = torch.distributions.Categorical(logits=l).sample() == N_ACTIONS
        o, _, _, _ = env.step(dodge(o), bomb_req=b)
    X = torch.cat([torch.cat(Hs), torch.ones(sum(len(h) for h in Hs), 1, device=dev)], 1).double()
    Z = torch.cat(Zs).double()
    w = torch.linalg.solve(X.T @ X + 1e-3 * torch.eye(X.shape[1], device=dev, dtype=torch.float64), X.T @ Z)
    with torch.no_grad():
        net.pi.weight[0].zero_(); net.pi.bias[0] = 0
        net.pi.weight[1] = w[:-1].float(); net.pi.bias[1] = w[-1].float()
    zf = X @ w
    print(f"bomb head fitted to {args.init_from} on {len(Z)} states: corr {torch.corrcoef(torch.stack([zf, Z]))[0, 1].item():.3f}, "
          f"mean P(bomb) {torch.sigmoid(Z).mean().item():.4f} -> {torch.sigmoid(zf).mean().item():.4f}")
    env.pop_stats()
opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
update0, steps_done = 0, 0
if args.resume:
    ck = torch.load(args.resume, map_location=dev)
    net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
    update0, steps_done = ck["update"], ck["steps"]
    print(f"resumed from {args.resume} at update {update0}")

T, N = args.rollout, args.envs
batch = T * N
n_updates = int(args.total // batch)
obs = env.reset()
buf = dict(loc=torch.zeros(T, N, obs["loc"].shape[1], 64, 64, dtype=torch.uint8, device=dev),
           glob=torch.zeros(T, N, obs["glob"].shape[1], 56, 48, dtype=torch.uint8, device=dev),
           vec=torch.zeros(T, N, env.vec_dim, device=dev))
act_b = torch.zeros(T, N, dtype=torch.long, device=dev)
logp_b, val_b, rew_b, done_b = (torch.zeros(T, N, device=dev) for _ in range(4))
if args.teach:
    import foresight
    doom_b = torch.zeros(T, N, env.n_actions, dtype=torch.bool, device=dev)
    teach_at = 0
log = open(os.path.join(out_dir, "log.txt"), "a")
t_start = time.time()
r_ema, r_best, best_at = None, None, 0       # --plateau: smoothed mean reward per step, its best, and when it last improved
steps0 = steps_done                          # (counted from this start: every env begins a fresh episode here, and those first
                                             # rollouts have no hits yet -> their reward is too high to serve as a 'best')

training_update=update0
monitor=None
train_seconds=0.


def pack_checkpoint(include_optimizer=True):
    result=dict(net=net.state_dict(),update=training_update,steps=steps_done,args=vars(args),
                n_actions=2 if args.dodger else env.n_actions,vec_dim=env.vec_dim,stack=env.stack,lags=env.lags,
                arch=args.arch,dodger=args.dodger,future=args.future,
                fly=dict(graph=args.fly_graph,th=args.fly_th,micro=args.fly_micro,seed=args.fly_seed))
    if include_optimizer:result['opt']=opt.state_dict()
    return result


if args.validate_every:
    from validation_monitor import ValidationMonitor
    monitor=ValidationMonitor(args,env,out_dir)
    monitor.check(net,steps_done,training_update,pack_checkpoint,force=True)

for update in range(update0, n_updates):
    frac = update / n_updates
    d_hi = min(args.d_max, args.d_start + (args.d_max - args.d_start) * frac / args.curriculum)
    env.set_difficulty(min(args.d_lo, d_hi), d_hi)
    for g in opt.param_groups:
        g["lr"] = args.lr * (1 - 0.9 * frac)
    t0 = time.time()
    # ---------------- rollout
    for t in range(T):
        for k in buf:
            buf[k][t] = obs[k]
        with torch.no_grad():
            logits, v = net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
        act_b[t], logp_b[t], val_b[t] = a, dist.log_prob(a), v
        if args.teach:                                   # (a different part of the envs at every decision)
            who = (teach_at + torch.arange(args.teach, device=dev)) % N
            teach_at = (teach_at + args.teach) % N
            doom_b[t] = False
            doom_b[t, who] = foresight.doomed_actions(env, who, args.teach_h, args.teach_delay)
        obs, r, d, _ = env.step(dodge(obs), bomb_req=a == 1) if dodger is not None else env.step(a)
        rew_b[t], done_b[t] = r, d.float()
    t_roll = time.time() - t0
    # ---------------- GAE
    with torch.no_grad():
        _, last_v = net(obs)
        adv = torch.zeros(T, N, device=dev)
        lastgae = 0
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val_b[t + 1]
            nonterm = 1.0 - done_b[t]
            delta = rew_b[t] + args.gamma * nv * nonterm - val_b[t]
            lastgae = delta + args.gamma * args.lam * nonterm * lastgae
            adv[t] = lastgae
        ret = adv + val_b
    # ---------------- optimise
    flat = lambda x: x.reshape(T * N, *x.shape[2:])
    f_loc, f_glob, f_vec = flat(buf["loc"]), flat(buf["glob"]), flat(buf["vec"])
    f_act, f_logp, f_adv, f_ret, f_val = flat(act_b), flat(logp_b), flat(adv), flat(ret), flat(val_b)
    mb = batch // args.minibatches
    stats = torch.zeros(5, device=dev)
    nmb = 0
    if args.teach:
        f_doom, t_stats = flat(doom_b), torch.zeros(3, device=dev)
    for ep in range(args.epochs):
        perm = torch.randperm(batch, device=dev)
        for i in range(args.minibatches):
            idx = perm[i * mb:(i + 1) * mb]
            adv_mb = f_adv[idx]
            adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
            opt.zero_grad(set_to_none=True)
            ch = args.chunk if args.chunk > 0 else mb
            st_mb = torch.zeros(4, device=dev)
            for c0 in range(0, mb, ch):                  # gradient accumulation over chunks (one chunk = the whole minibatch by default)
                j = idx[c0:c0 + ch]
                logits, v = net(dict(loc=f_loc[j], glob=f_glob[j], vec=f_vec[j]))
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(f_act[j])
                ratio = (lp - f_logp[j]).exp()
                a_ = adv_mb[c0:c0 + ch]
                pg = -torch.min(ratio * a_, ratio.clamp(1 - args.clip, 1 + args.clip) * a_).mean()
                vl = 0.5 * F.mse_loss(v, f_ret[j])
                ent = dist.entropy().mean()
                loss = (pg + 0.5 * vl - args.ent * ent) * (len(j) / mb)
                if args.teach and bool(f_doom[j].any()):
                    dm = f_doom[j]
                    dm = dm[dm.any(1)]
                    lg = logits[f_doom[j].any(1)]
                    bad, good = lg.masked_fill(~dm, -1e6).max(1).values, lg.masked_fill(dm, -1e6).max(1).values
                    tl = F.relu(bad - good + args.teach_margin).mean()
                    loss = loss + args.teach_w * tl * (len(j) / mb)
                    t_stats += torch.stack([tl.detach(), (bad >= good).float().mean(), dm.shape[0] / len(j) * torch.ones_like(tl)])
                loss.backward()
                st_mb += torch.stack([pg.detach(), vl.detach(), ent.detach(), ((ratio - 1).abs() > args.clip).float().mean()]) * (len(j) / mb)
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            stats += torch.cat([st_mb, gn[None]])
            nmb += 1
    steps_done += batch
    training_update=update+1
    if monitor is not None:
        torch.cuda.synchronize()
    update_train_seconds=time.time()-t0
    train_seconds+=update_train_seconds
    r_now = rew_b.mean().item()
    r_ema = r_now if r_ema is None else 0.95 * r_ema + 0.05 * r_now
    if steps_done - steps0 >= args.plateau_min * 1e6 and (r_best is None or r_ema > r_best + 0.02 * abs(r_best)):
        r_best, best_at = r_ema, steps_done
    stop = args.plateau > 0 and r_best is not None and steps_done - best_at >= args.plateau * 1e6
    if monitor is not None:
        monitor.train_seconds=train_seconds
        stop=monitor.check(net,steps_done,training_update,pack_checkpoint,force=update==n_updates-1)
    # ---------------- logging
    if update % 5 == 0 or update == n_updates - 1 or stop:
        st = env.pop_stats()
        s = (stats / nmb).tolist()
        dt = update_train_seconds if monitor is not None else time.time()-t0
        line = (f"upd {update:5d}/{n_updates} steps {steps_done/1e6:7.2f}M | d_hi {d_hi:.2f} | ep_len {st['ep_frames']/60:6.1f}s "
                f"(n={int(st['episodes'])}) | pg {s[0]:+.4f} vl {s[1]:.4f} ent {s[2]:.3f} clipfrac {s[3]:.3f} gn {s[4]:.2f} | "
                f"overwrites {int(st['overwrites'])} | bombs {int(st['bombs'])} wasted {int(st['wasted'])} | {batch/dt:,.0f} sps (rollout {t_roll:.1f}s of {dt:.1f}s) | {(time.time()-t_start)/60:.1f} min"
                + (f" | teach loss {t_stats[0] / nmb:.3f}, top action doomed {100 * t_stats[1] / nmb:.1f}%, labelled {100 * t_stats[2] / nmb:.2f}%" if args.teach else ""))
        print(line, flush=True)
        log.write(line + "\n"); log.flush()
    if (update + 1) % args.save_every == 0 or update == n_updates - 1 or stop:
        torch.save(pack_checkpoint(True),os.path.join(out_dir,"ckpt.pt"))
    if stop:
        line = (f"PLATEAU: fixed validation anchor {monitor.anchor:.3f}s since {monitor.anchor_steps/1e6:.2f}M "
                f"-> stopped at {steps_done/1e6:.2f}M" if monitor is not None else
                f"PLATEAU: smoothed reward per step {r_ema:.5f} has not beaten {r_best:.5f} (+2%) since {best_at/1e6:.2f}M steps -> stopped at {steps_done/1e6:.2f}M")
        print(line, flush=True); log.write(line + "\n"); log.flush()
        break
if monitor is not None:
    json.dump(dict(steps=steps_done,best_steps=monitor.best_steps,best_score=monitor.best,
                   stop_reason='validation_plateau' if stop else 'max_steps',train_seconds=train_seconds,
                   wall_seconds=time.time()-t_start,steps_per_train_second=(steps_done-steps0)/max(train_seconds,1e-9),
                   peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9),open(os.path.join(out_dir,'result.json'),'w'),indent=2)
    if env.timing_clock is not None:
        json.dump(env.timing_clock.summary(),open(os.path.join(out_dir,'clock_summary.json'),'w'),indent=2)
print("done")
