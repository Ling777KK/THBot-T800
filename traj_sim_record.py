"""More variety for traj_train.py: the policy playing in the simulator, recorded in the format of the real recordings (traj_sim.raster). The real ones have 10
cards; the network starts to learn those instead of bullets after a few thousand iterations.
  traj_sim_record.py [--envs 128] [--steps 1000]   -> touhou-runs\\traj_cache\\sim.npz"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F

from danmaku_env import DanmakuEnv, TRAIN_V11
from policy import ActorCritic
from traj_sim import raster
from traj_train import CACHE

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "v11a_policy.pt"))
ap.add_argument("--envs", type=int, default=128)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
torch.manual_seed(args.seed)
dev = torch.device("cuda")
ck = torch.load(args.policy, map_location=dev)
pol = ActorCritic(ck["n_actions"], ck["vec_dim"], ck["stack"], ck["lags"]).to(dev).to(memory_format=torch.channels_last)
pol.load_state_dict(ck["net"]); pol.eval()
env = DanmakuEnv(n_envs=args.envs, frame_skip=2, ymin_play=16.0, d_max=1.4, stack=ck["stack"], lags=ck["lags"], blind=[5.3, 5.3, 5.3], big_draw=[48, 160])
env.set_families(TRAIN_V11)
env.set_difficulty(0.4, 1.4)
obs = env.reset()
half = [[[]] for _ in range(args.envs)]                                     # per env, per episode: (half packed, cell packed, x, y)
samp = torch.arange(args.envs, device=dev) % 4 == 0
for t in range(args.steps):
    with torch.no_grad():
        logits = pol(obs)[0]
        a = torch.where(samp, torch.distributions.Categorical(logits=logits).sample(), logits.argmax(1))
        m = raster(env)
        cl = F.max_pool2d(m[:, None].float(), 3, 3)[:, 0] > 0
    hp = np.packbits(m.cpu().numpy().reshape(args.envs, -1) > 0, axis=1)
    cp = np.packbits(cl.cpu().numpy().reshape(args.envs, -1), axis=1)
    xy = torch.stack([env.px, env.py], 1).cpu().numpy().astype(np.float32)
    for n in range(args.envs):
        half[n][-1].append((hp[n], cp[n], xy[n]))
    obs, _, done, _ = env.step(a)
    for n in done.nonzero()[:, 0].tolist():
        half[n].append([])
    if t % 200 == 0:
        print(f"step {t}", flush=True)
H, C, XY, ST, EN = [], [], [], [], []
N = 0
for per in half:
    for ep in per:
        if len(ep) < 60:
            continue
        H.append(np.stack([e[0] for e in ep])); C.append(np.stack([e[1] for e in ep])); XY.append(np.stack([e[2] for e in ep]))
        ST.append(np.full(len(ep), N, np.int64)); EN.append(np.full(len(ep), N + len(ep) - 1, np.int64))
        N += len(ep)
np.savez(os.path.join(CACHE, "sim.npz"), half=np.concatenate(H), cell=np.concatenate(C), xy=np.concatenate(XY), start=np.concatenate(ST), end=np.concatenate(EN))
print(f"{N} decisions in {len(H)} episodes")
