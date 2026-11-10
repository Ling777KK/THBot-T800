"""Train TrajNet (traj_net.py) on the real game's recordings: bench\\*_ai5 and *_rec1, half-size masks of every decision of AI-mode LUNATIC spell
practice. The label is the mask of the future frame, pooled to 4 game px cells, so the bullets fired later are in it too. Cards in --holdout
are never trained on: only these numbers say anything about a card it has not seen. Left-right flips double the data.
Measure as in realgame/predictor_eval.py, on cells: of the bullet cells in the ring of 8..60 game px around where the player really is at that
future moment, the share that was predicted (recall), the share of the predicted cells that were right (precision), the share of the ring
marked (area). Doing nothing ('keep') and the tracker are compared on the same frames by traj_eval.py.
  traj_train.py [--name traj0] [--holdout s1c2,s5c1,s6c3] [--iters 20000]"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from traj_net import TrajNet, TrajRNN, LAGS, T_MAX, H_HALF, W_HALF, H_CELL, W_CELL, CELL, player_marker

RUNS = os.environ.get("TOUHOU_RUNS", r"C:\touhou-runs")
BENCH = os.path.join(RUNS, "bench")
CACHE = os.path.join(RUNS, "traj_cache")


def load_cache(runs=("_ai5", "_rec1"), rate_ms=36.0, name="real"):
    """-> dict of arrays over all usable decisions: half [N, 12096] packed masks, cell [N, 1344] packed 4 px cells, xy [N, 2], start / end [N] the
    first and last index of the attempt, card [N] index into cards. Built once into <name>.npz."""
    f = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    os.makedirs(CACHE, exist_ok=True)
    half, cell, xy, start, end, card, run_of, cards, runs_used = [], [], [], [], [], [], [], [], []
    N = 0
    for m_f in sorted(glob.glob(os.path.join(BENCH, "*", "play", "dump_mask.npy"))):
        d = os.path.dirname(m_f)
        name = os.path.basename(os.path.dirname(d))
        if not any(r in name for r in runs) or not os.path.exists(os.path.join(d, "attempts.npy")) or not os.path.exists(os.path.join(d, "dump_meta.npy")):
            continue
        M = np.load(m_f, mmap_mode="r")
        meta = np.load(os.path.join(d, "dump_meta.npy"))
        A = np.load(os.path.join(d, "attempts.npy"))
        n = min(len(meta), M.shape[0])
        t = meta[:n, 0]
        ms = np.median(np.diff(t)) * 1e3
        if ms > rate_ms:
            print(f"{name}: {ms:.1f} ms per decision, skipped", flush=True)
            continue
        c = name.split("_")[0]
        if c not in cards:
            cards.append(c)
        runs_used.append(name)
        kept = 0
        for e, dur in A[:, :2].astype(float):
            idx = np.where((t > e - dur) & (t < e - 1.5))[0]
            if len(idx) < 60:
                continue
            m = np.array(M[idx])
            if m.shape[1] == 672:
                m = m[:, ::2, ::2]
            assert m.shape[1:] == (H_HALF, W_HALF), m.shape
            cl = torch.from_numpy(m).float()[:, None]
            cl = (F.max_pool2d(cl, 3, 3)[:, 0] > 0).numpy()
            half.append(np.packbits(m.reshape(len(m), -1), axis=1))
            cell.append(np.packbits(cl.reshape(len(m), -1), axis=1))
            xy.append(meta[idx, 1:3].astype(np.float32))
            start.append(np.full(len(m), N, np.int64))
            end.append(np.full(len(m), N + len(m) - 1, np.int64))
            card.append(np.full(len(m), cards.index(c), np.int16))
            run_of.append(np.full(len(m), len(runs_used) - 1, np.int16))
            N += len(m)
            kept += len(m)
        print(f"{name}: {ms:.1f} ms per decision, {kept} decisions used, total {N}", flush=True)
    out = dict(half=np.concatenate(half), cell=np.concatenate(cell), xy=np.concatenate(xy), start=np.concatenate(start), end=np.concatenate(end),
               card=np.concatenate(card), run=np.concatenate(run_of), cards=np.array(cards), runs=np.array(runs_used))
    np.savez(f, **out)
    return out


def merge(a, b):
    """Two caches of load_cache as one: b's card and run indices and attempt bounds moved to follow a's."""
    cards = [str(c) for c in a["cards"]]
    cards += [str(c) for c in b["cards"] if str(c) not in cards]
    remap = np.array([cards.index(str(c)) for c in b["cards"]], np.int16)
    N = len(a["xy"])
    cat = lambda k: np.concatenate([a[k], b[k]])
    return dict(half=cat("half"), cell=cat("cell"), xy=cat("xy"), start=np.concatenate([a["start"], b["start"] + N]), end=np.concatenate([a["end"], b["end"] + N]),
                card=np.concatenate([a["card"], remap[b["card"]]]), run=np.concatenate([a["run"], b["run"] + len(a["runs"])]), cards=np.array(cards), runs=cat("runs"))


def shrunk_cells(half, K, dev, chunk=1024):
    """half: [N, 12096] packed half-size masks -> [N, 1344] packed 4 px cells of the masks shrunk toward the bullets' cores (2026-11-04: the
    masks hold the glow too, so on dense cards every gap looked closed and the search saw 'no way out' far too often). Every blob loses K
    half-size px of its rim; a blob thinner than that keeps its middle (the pixels as deep as the deepest within K px), so no bullet vanishes."""
    out = []
    for b in range(0, len(half), chunk):
        m = torch.from_numpy(np.unpackbits(half[b:b + chunk], axis=1).reshape(-1, 1, H_HALF, W_HALF)).to(dev).float()
        d, e = m.clone(), m
        for _ in range(K):
            e = -F.max_pool2d(-e, 3, 1, 1)
            d = d + e
        top = F.max_pool2d(d, 2 * K + 1, 1, K).clamp(max=K + 1)
        keep = (m > 0) & (d >= top)
        cl = (F.max_pool2d(keep.float(), 3, 3)[:, 0] > 0).reshape(len(m), -1)
        out.append(np.packbits(cl.cpu().numpy(), axis=1))
    return np.concatenate(out)


def shrunk_cache(name, half, K, dev):
    """The cells of shrunk_cells for real.npz / sim.npz, built once into <name>_shrink<K>.npz."""
    f = os.path.join(CACHE, f"{name}_shrink{K}.npz")
    if not os.path.exists(f):
        t0 = time.time()
        np.savez(f, cell=shrunk_cells(half, K, dev))
        print(f"{f}: built in {time.time() - t0:.0f} s", flush=True)
    return np.load(f)["cell"]


class Data:
    def __init__(self, z, dev, lags=LAGS, T=T_MAX):
        self.z, self.dev, self.T = z, dev, T
        self.half, self.cell, self.xy, self.start, self.end = z["half"], z["cell"], z["xy"], z["start"], z["end"]
        self.lags = np.array(lags)
        self.k = np.arange(1, T + 1)

    def inputs(self, i):
        """i: int array [n] -> float [n, len(lags) + 1, 336, 288]"""
        J = np.maximum(i[:, None] - self.lags[None], self.start[i][:, None])    # before the attempt: its first frame, as the game loop does at a reset
        x = np.unpackbits(self.half[J.ravel()], axis=1).reshape(len(i), len(self.lags), H_HALF, W_HALF)
        x = torch.from_numpy(x).to(self.dev, non_blocking=True)
        xy = torch.from_numpy(self.xy[i]).to(self.dev)
        return torch.cat([x, player_marker(xy[:, 0], xy[:, 1])[:, None]], 1).float()

    def targets(self, i):
        """-> [n, T, 112, 96] float, valid [n, T] float (the future frame is still in this attempt)."""
        J = i[:, None] + self.k[None]
        ok = J <= self.end[i][:, None]
        y = np.unpackbits(self.cell[np.minimum(J, self.end[i][:, None]).ravel()], axis=1).reshape(len(i), self.T, H_CELL, W_CELL)
        return torch.from_numpy(y).to(self.dev, non_blocking=True).float(), torch.from_numpy(ok).to(self.dev).float()


def ring_of(xy, dev):
    """xy: [n, 2] game px -> [n, 112, 96] bool, the ring of 8..60 game px around it, by the cells' centres."""
    yy, xx = torch.meshgrid((torch.arange(H_CELL, device=dev).float() + 0.5) * CELL, (torch.arange(W_CELL, device=dev).float() + 0.5) * CELL, indexing="ij")
    r = torch.hypot(xx[None] - xy[:, 0, None, None], yy[None] - xy[:, 1, None, None])
    return (r >= 8) & (r <= 60)


THR = (0.1, 0.2, 0.3, 0.5, 0.7)
HORIZONS = (4, 8, 15, 30)


@torch.no_grad()
def validate(net, data, idx, batch=32):
    """idx: the decisions to test on. -> per card and overall {h: [tp, predicted, truth, ring] for each threshold and 'keep'}, mean bce."""
    dev = data.dev
    res = {}
    bce, nb = 0.0, 0
    net.eval()
    card = data.z["card"]
    for b in range(0, len(idx), batch):
        i = idx[b:b + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(data.inputs(i))
        lg = lg.float()[:, :T_MAX]
        y, ok = data.targets(i)
        bce += float((F.binary_cross_entropy_with_logits(lg, y, reduction="none").mean((2, 3)) * ok).sum() / ok.sum().clamp(min=1))
        nb += 1
        p = torch.sigmoid(lg)
        now = torch.from_numpy(np.unpackbits(data.cell[i], axis=1).reshape(len(i), H_CELL, W_CELL)).to(dev) > 0
        for h in HORIZONS:
            good = ok[:, h - 1] > 0
            if not bool(good.any()):
                continue
            j = np.minimum(i + h, data.end[i])
            ring = ring_of(torch.from_numpy(data.xy[j]).to(dev), dev)
            tru = (y[:, h - 1] > 0) & ring
            counts = {}
            for m, q in (("keep", now), *((th, p[:, h - 1] > th) for th in THR)):
                q = q & ring
                counts[m] = torch.stack([(q & tru).sum((1, 2)), q.sum((1, 2)), tru.sum((1, 2)), ring.sum((1, 2))], 1).float().cpu().numpy()
            g = good.cpu().numpy()
            for key in (*np.unique(card[i]).tolist(), -1):
                s = g & (card[i] == key if key >= 0 else True)
                r = res.setdefault((int(key), h), {m: np.zeros(4) for m in counts})
                for m, v in counts.items():
                    r[m] += v[s].sum(0)
    return res, bce / max(nb, 1)


def table(res, cards, keys=None):
    lines = []
    for (key, h), r in sorted(res.items(), key=lambda kv: (kv[0][0] != -1, kv[0][0], kv[0][1])):
        if keys is not None and key not in keys:
            continue
        nm = "all" if key == -1 else str(cards[key])
        parts = []
        for m in ("keep", *THR):
            tp, pa, tr, rg = r[m]
            parts.append(f"{m}: R{100 * tp / max(tr, 1):3.0f} P{100 * tp / max(pa, 1):3.0f} A{100 * pa / max(rg, 1):3.0f}")
        lines.append(f"  {nm:5s} {h * 2:3d}f | " + " | ".join(parts))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="traj0")
    ap.add_argument("--out_root", default=os.path.join(RUNS, "traj"))
    ap.add_argument("--holdout", default="s1c2,s5c1,s6c3", help="cards never trained on; empty: every card (only after the held-out numbers are known)")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--val_n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="unet", choices=["unet", "rnn"])
    ap.add_argument("--every", type=int, default=2, help="rnn: decisions per recurrent step")
    ap.add_argument("--sim_share", type=float, default=0.0, help="share of each batch from the simulator's recording (traj_sim_record.py)")
    ap.add_argument("--shift", type=int, default=0, help="move the whole picture by up to this many cells, per batch")
    ap.add_argument("--shrink", type=int, default=0, help="labels from masks shrunk by this many half-size px (shrunk_cells); the input stays as seen")
    ap.add_argument("--base_runs", default="real=_ai5,_rec1", help="name=part,part,...: the main recordings (bench dirs whose name contains a "
                    "part), cached as <name>.npz; memory masks (play.py --memperc, 2026-11-05): memrec=_memrec,_mm2N")
    ap.add_argument("--init", default="", help="start from this checkpoint's weights (same arch / base / every) instead of from scratch")
    ap.add_argument("--extra", default="", help="name=part,part,...: more recordings (bench dirs whose name contains a part), cached as <name>.npz, "
                    "e.g. dense1=_traj_dump1,_traj_ab (2026-11-04: the dense stage 4-6 cards with the orbs visible, never in real.npz)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")
    out = os.path.join(args.out_root, args.name)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, "args.json"), "w"), indent=1)
    bn, bparts = args.base_runs.split("=", 1)
    z = load_cache(tuple(bparts.split(",")), name=bn)
    if args.shrink:
        z["cell"] = shrunk_cache(bn, z["half"], args.shrink, dev)
    if args.extra:
        nm, parts = args.extra.split("=", 1)
        e = load_cache(tuple(parts.split(",")), name=nm)
        if args.shrink:
            e["cell"] = shrunk_cache(nm, e["half"], args.shrink, dev)
        z = merge(z, e)
    cards = [str(c) for c in z["cards"]]
    N = len(z["xy"])
    sim_idx = None
    if args.sim_share > 0:
        q = dict(np.load(os.path.join(CACHE, "sim.npz")))
        if args.shrink:
            q["cell"] = shrunk_cache("sim", q["half"], args.shrink, dev)
        sim_idx = np.arange(N, N + len(q["xy"]))
        z = dict(z, half=np.concatenate([z["half"], q["half"]]), cell=np.concatenate([z["cell"], q["cell"]]), xy=np.concatenate([z["xy"], q["xy"]]),
                 start=np.concatenate([z["start"], q["start"] + N]), end=np.concatenate([z["end"], q["end"] + N]),
                 card=np.concatenate([z["card"], np.full(len(q["xy"]), -2, np.int16)]))
        print(f"simulator: {len(sim_idx)} decisions, {args.sim_share:.0%} of each batch", flush=True)
    print(f"{N} decisions ({N / 30 / 60:.0f} min of play), cards {cards}, {len(z['runs'])} recordings", flush=True)
    hold = [cards.index(c) for c in args.holdout.split(",") if c in cards]
    is_hold = np.isin(z["card"], hold)
    tr_idx = np.where(~is_hold & (z["card"] >= 0))[0]
    va_all = np.where(is_hold)[0] if hold else np.where(z["card"] >= 0)[0]
    va_idx = np.sort(np.random.default_rng(12345).choice(va_all, min(args.val_n, len(va_all)), replace=False))
    print(f"train {len(tr_idx)} decisions, test {len(va_idx)} of {len(va_all)} on {[cards[c] for c in hold]}", flush=True)
    data = Data(z, dev)
    net = (TrajRNN(base=args.base, every=args.every) if args.arch == "rnn" else TrajNet(base=args.base)).to(dev).to(memory_format=torch.channels_last)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters", flush=True)
    if args.init:
        ik = torch.load(args.init, map_location=dev)
        assert ik["arch"] == args.arch and ik["base"] == args.base and ik.get("every", args.every) == args.every, "--init: other arch / base / every"
        net.load_state_dict(ik["net"])
        print(f"weights from {args.init} ({ik['iters']} iterations)", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.iters, pct_start=0.05)
    log = open(os.path.join(out, "log.txt"), "a", encoding="utf-8")
    t0 = time.time()
    run_loss, run_n, best = 0.0, 0, 1e9

    def say(s):
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    def save(nm, it):
        torch.save(dict(net=net.state_dict(), arch=args.arch, every=args.every, lags=LAGS, T=T_MAX, base=args.base, iters=it, holdout=args.holdout, args=vars(args)), os.path.join(out, nm))

    for it in range(1, args.iters + 1):
        net.train()
        i = tr_idx[rng.integers(0, len(tr_idx), args.batch)]
        if sim_idx is not None:
            n_s = int(args.batch * args.sim_share)
            i[:n_s] = sim_idx[rng.integers(0, len(sim_idx), n_s)]
        x = data.inputs(i)
        y, ok = data.targets(i)
        f = torch.rand(args.batch, device=dev) < 0.5
        x[f] = x[f].flip(-1)
        y[f] = y[f].flip(-1)
        if args.shift:
            sy, sx = rng.integers(-args.shift, args.shift + 1, 2).tolist()
            x, y = torch.roll(x, (3 * sy, 3 * sx), (2, 3)), torch.roll(y, (sy, sx), (2, 3))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(x)
        loss = (F.binary_cross_entropy_with_logits(lg.float()[:, :T_MAX], y, reduction="none").mean((2, 3)) * ok).sum() / ok.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss += loss.item(); run_n += 1
        if it % 200 == 0:
            say(f"iter {it:6d}/{args.iters} bce {run_loss / run_n:.4f} lr {sched.get_last_lr()[0]:.2e} | {it / (time.time() - t0):.1f} it/s | {(time.time() - t0) / 60:.1f} min")
            run_loss, run_n = 0.0, 0
        if it % args.val_every == 0 or it == args.iters:
            res, vb = validate(net, data, va_idx)
            say(f"== iter {it} test bce {vb:.4f}\n" + table(res, cards))
            if vb < best:
                best = vb
                save("best.pt", it)
            save("last.pt", it)
    say("done")
