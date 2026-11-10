"""Vision worker for play.py --vision_reflex (2026-11-06). A separate process that grabs the playfield on EVERY game frame it can
(~60 Hz; every 2nd in the densest scenes) and keeps the bullet table from the picture (vision_table.VisionTableCPU) up to date.
Why: from the decision loop's own pictures (2-4 frames apart, the exact gap unknown) the tracker matched each bullet of a regular
stream to the one behind it (std_lunatic_vis_smoke, death at 88 s: Rumia's column measured as crawling UP), and the table cost the
loop ~6-8 ms. At one frame apart a 6.5 px/frame bullet moves far less than a stream's spacing, and the gap is known.
Memory: only the player's position (read right after each grab), as play.py. CPU only (numpy): a second CUDA context would
time-slice the GPU with play.py's networks.
The newest table sits in shared memory (two slots + a sequence number); play.py copies it when the reflex needs it.
   started by play.py (VisionProc); standalone test:  python vision_proc.py --test 10"""
import argparse, os, sys, time
import numpy as np
from multiprocessing import shared_memory

HDR = 64                    # float64 header: [0] seq of the newest complete table slot, [1] stop flag, [2] parent heartbeat (time.time()),
SLOT_H = 16                 #   [3] parent beating yet, [4] seq of the newest picture, [5], [6] t_grab of picture slot 0 / 1, [7] dump requests
                            #   [8 + 16 * s ...] table slot s: n rows, t_grab (perf_counter), x, y, ms, n_moving, n_dots, frames, n_big_px, gap
MAXR = 40000                # rows per table slot: x, y, vx, vy, hb (float32)
WIDE = (340, 24, 920, 678)  # the grab (client x0, y0, w, h): play.py's wide grab = playfield + HUD + left frame (WIDE_X0 ... in play.py)


def _sizes():
    return HDR * 8, MAXR * 5 * 4, WIDE[3] * WIDE[2] * 4


class VisionProc:
    """parent side: starts vision_proc.py, reads the newest table."""

    def __init__(self, cfg_args, log_path=None):
        import subprocess
        h, d, im = _sizes()
        self.shm = shared_memory.SharedMemory(create=True, size=h + 2 * d + 2 * im)
        self.h = np.ndarray((HDR,), np.float64, buffer=self.shm.buf[:h])
        self.h[:] = 0
        self.h[2] = time.time()
        self.d = [np.ndarray((MAXR, 5), np.float32, buffer=self.shm.buf[h + s * d:h + (s + 1) * d]) for s in (0, 1)]
        o = h + 2 * d
        self.im = [np.ndarray((WIDE[3], WIDE[2], 4), np.uint8, buffer=self.shm.buf[o + s * im:o + (s + 1) * im]) for s in (0, 1)]
        here = os.path.dirname(os.path.abspath(__file__))
        self.log = open(log_path, "w") if log_path else subprocess.DEVNULL
        self.p = subprocess.Popen([sys.executable, os.path.join(here, "vision_proc.py"), "--shm", self.shm.name] + list(cfg_args),
                                  cwd=here, stdout=self.log, stderr=subprocess.STDOUT)
        self.last_seq = 0

    def dump_ring(self):
        """ask the worker to save its last ~3 s of pictures (vring_NN.npz in --out): play.py calls it at a death."""
        self.h[7] += 1

    def beat(self):
        self.h[2] = time.time()
        self.h[3] = 1

    def picture(self):
        """-> (t_grab, BGRA [h, w, 4] copy of the newest wide grab), or None. play.py takes every picture from here (only one process
        may grab: a second grab in the same frame waits for the next one, and the two then share the frames)."""
        for _ in range(3):
            seq = int(self.h[4])
            if seq < 1:
                return None
            s = seq % 2
            tg = float(self.h[5 + s])
            img = self.im[s].copy()
            if int(self.h[4]) == seq:
                return tg, img
        return None

    def latest(self):
        """-> (t_grab, (x, y), bullets dict, stats dict) of the newest table, or None."""
        self.h[2] = time.time()
        for _ in range(3):
            seq = int(self.h[0])
            if seq < 1:
                return None
            s = seq % 2
            hd = self.h[8 + SLOT_H * s:8 + SLOT_H * (s + 1)].copy()
            n = int(hd[0])
            rows = self.d[s][:n].copy()
            if int(self.h[0]) == seq:
                b = dict(x=rows[:, 0], y=rows[:, 1], vx=rows[:, 2], vy=rows[:, 3], hb=rows[:, 4])
                st = dict(seq=seq, ms=hd[4], n_moving=int(hd[5]), n_dots=int(hd[6]), frames=hd[7], n_big_px=int(hd[8]), gap=hd[9])
                return float(hd[1]), (float(hd[2]), float(hd[3])), b, st
        return None

    def close(self):
        try:
            self.h[1] = 1
            self.p.wait(timeout=3)
        except Exception:
            self.p.kill()
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:
            pass


def _cpu_boost():
    import ctypes
    from ctypes import wintypes

    class PPTS(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    h = k32.GetCurrentProcess()
    s = PPTS(1, 0x1, 0x0)
    return bool(k32.SetProcessInformation(wintypes.HANDLE(h), 4, ctypes.byref(s), ctypes.sizeof(s))) and \
        bool(k32.SetPriorityClass(wintypes.HANDLE(h), 0x8000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shm", default="")
    ap.add_argument("--core", type=int, default=200)
    ap.add_argument("--lags", default="0,0.6,1.2,1.8")
    ap.add_argument("--reach_pf", type=float, default=6.0)
    ap.add_argument("--use2", type=int, default=1)
    ap.add_argument("--big", type=int, default=1)
    ap.add_argument("--hb_min", type=float, default=5.0)
    ap.add_argument("--out", default="", help="folder for vring_NN.npz (the last ~3 s of the worker's own half-size pictures at a death)")
    ap.add_argument("--test", type=float, default=0.0, help="standalone: run this many s against the open game, print the rate")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print("cpu boost:", _cpu_boost(), flush=True)
    import mss
    import torch
    torch.set_num_threads(1)
    import winutil as W
    import gamemem as G
    import vision_table as VT
    from perception import PF_X0, PF_Y0, PF_W, PF_H
    assert WIDE[1] == PF_Y0 and WIDE[3] == PF_H + 6, "WIDE must match play.py's wide grab"
    px0 = PF_X0 - WIDE[0]
    G.PLAYER_RVAS[:] = [0x551720, 0x551980]
    vt = VT.VisionTableCPU(core=a.core, lags=tuple(float(v) for v in a.lags.split(",")), reach_pf=a.reach_pf,
                           use2=bool(a.use2), big=bool(a.big), hb_min=a.hb_min)
    if a.shm:
        shm = shared_memory.SharedMemory(name=a.shm)
        hsz, dsz, isz = _sizes()
        H = np.ndarray((HDR,), np.float64, buffer=shm.buf[:hsz])
        D = [np.ndarray((MAXR, 5), np.float32, buffer=shm.buf[hsz + s * dsz:hsz + (s + 1) * dsz]) for s in (0, 1)]
        o = hsz + 2 * dsz
        IM = [np.ndarray((WIDE[3], WIDE[2], 4), np.uint8, buffer=shm.buf[o + s * isz:o + (s + 1) * isz]) for s in (0, 1)]
    else:
        H = np.zeros(HDR); D = [np.zeros((MAXR, 5), np.float32) for _ in (0, 1)]
        IM = [np.zeros((WIDE[3], WIDE[2], 4), np.uint8) for _ in (0, 1)]
        H[2] = time.time() + 1e9
    hwnd = mem = None
    from collections import deque
    ring = deque(maxlen=200)                                    # (t_grab, half-size RGB, player xy): ~3 s at 60 Hz, ~58 MB
    dumped = 0
    last_t = None
    seq = 0
    t_start = time.perf_counter()
    gaps, mss_ = [], []
    print("vision worker ready", flush=True)
    with mss.MSS() as sct:
        while True:
            if H[1] or time.time() - H[2] > (20 if H[3] else 300) or (a.test and time.perf_counter() - t_start > a.test):
                break                                           # (stop flag / parent gone; before the parent's loop runs: game start-up)
            if hwnd is None:
                hwnd, _ = W.find_window()
                if not hwnd:
                    time.sleep(0.2); continue
            if mem is None:
                m_ = G.GameMem()
                if m_.ok and m_.player_xy():
                    mem = m_
                    print("vision worker: game found", flush=True)
                else:
                    time.sleep(0.2); continue
            try:
                l, t, r, b = W.client_rect(hwnd)
            except Exception:
                hwnd = mem = None; continue
            shot = np.asarray(sct.grab(dict(left=l + WIDE[0], top=t + WIDE[1], width=WIDE[2], height=WIDE[3])))
            tg = time.perf_counter()
            si = (int(H[4]) + 1) % 2                            # publish the picture first (play.py's policy reads it)
            IM[si][:] = shot
            H[5 + si] = tg
            H[4] = H[4] + 1
            xy = mem.player_xy()
            if not xy:
                continue
            half = np.ascontiguousarray(shot[:PF_H:2, px0:px0 + PF_W:2, 2::-1])
            ring.append((tg, half, xy))
            if a.out and int(H[7]) > dumped:                    # a death in play.py: keep what the worker saw (offline replays of the table)
                dumped = int(H[7])
                try:
                    np.savez(os.path.join(a.out, f"vring_{dumped:02d}.npz"), t=np.array([r_[0] for r_ in ring]),
                             img=np.stack([r_[1] for r_ in ring]), xy=np.array([r_[2] for r_ in ring], np.float32))
                    print(f"vision worker: saved vring_{dumped:02d}.npz ({len(ring)} pictures)", flush=True)
                except Exception as ex:
                    print("vision worker: ring not saved:", repr(ex)[:120], flush=True)
            if last_t is None or tg - last_t > 0.25:
                vt.reset()
            gap = (tg - last_t) * 60 if last_t is not None else 0.0
            last_t = tg
            t0 = time.perf_counter()
            try:
                tab = vt.update(half, tg, xy)
            except Exception as ex:
                print("vision worker: update failed:", repr(ex)[:200], flush=True)
                vt.reset(); continue
            ms = (time.perf_counter() - t0) * 1000
            bb = tab["bullets"]
            n = min(len(bb["x"]), MAXR)
            s = (seq + 1) % 2
            D[s][:n, 0], D[s][:n, 1], D[s][:n, 2], D[s][:n, 3], D[s][:n, 4] = bb["x"][:n], bb["y"][:n], bb["vx"][:n], bb["vy"][:n], bb["hb"][:n]
            L = vt.last
            H[8 + SLOT_H * s:8 + SLOT_H * s + 10] = (n, tg, xy[0], xy[1], ms, L.get("n_moving", 0), L.get("n_dots", 0),
                                                      L.get("frames", 0), L.get("n_big_px", 0), gap)
            seq += 1
            H[0] = seq
            gaps.append(gap); mss_.append(ms)
            if a.test and seq % 120 == 0:
                print(f"{seq} tables, gap p50 {np.median(gaps[-120:]):.2f} frames, update {np.median(mss_[-120:]):.1f} ms, {L}", flush=True)
    if gaps:
        g = np.round(np.array(gaps[1:]))
        print(f"vision worker: {seq} tables in {time.perf_counter() - t_start:.0f} s; frames between pictures 1: {np.mean(g == 1):.0%}, "
              f"2: {np.mean(g == 2):.0%}, 3+: {np.mean(g >= 3):.0%}; update ms p50 {np.median(mss_):.1f} p90 {np.percentile(mss_, 90):.1f}", flush=True)


if __name__ == "__main__":
    main()
