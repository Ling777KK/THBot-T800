"""Closed loop on the real game: screen -> bullet maps -> policy -> keys, at 30 decisions per second.

Safety: keys are injected ONLY while the game window is the foreground window; every key is released when the
loop stops (including on errors / Ctrl-C). Policies: path to a ckpt, or  stay | random.
"""
import argparse, os, sys, time
import numpy as np
import mss
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # touhou-agent/ for policy.py
import winutil as W
import keys as K
import difflabel
from perception import Perceiver, overlay, PF_X0, PF_Y0, PF_W, PF_H, S
from policy import ActorCritic

ap = argparse.ArgumentParser()
ap.add_argument("policy", help="ckpt path | stay | random")
ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--dry", action="store_true", help="do not press any keys, only think")
ap.add_argument("--out", default="", help="folder for overlay snapshots and the log")
ap.add_argument("--snap_every", type=float, default=4.0)
ap.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
ap.add_argument("--hz", type=float, default=30.0)
ap.add_argument("--predict", type=float, default=0.0, help=">0: show the policy the bullets this many game frames ahead, estimated from "
                "the picture alone (tracker.py), so that it acts on what will be there when its keys take effect")
ap.add_argument("--predict_model", default="turn", help="tracker.py path model: line / quad / turn")
ap.add_argument("--predict_self", type=float, default=-1.0, help="frames to roll the player forward by its last action (default: --predict - 1, "
                "the memory value is about a frame newer than the picture)")
ap.add_argument("--shield", type=int, default=0, help=">0: the policy keeps its choice unless that shortens how long the player can foreseeably stay alive, "
                "judged from the tracked bullets' paths over this many game frames (shield.py; the policy still sees the picture as it is)")
ap.add_argument("--shield_delay", type=int, default=3, help="--shield: game frames from the picture until an action chosen now acts")
ap.add_argument("--safety", default="", help="safety network (safety_train.py): overrules the policy's choice when it calls that action doomed; no tracker")
ap.add_argument("--safety_thr", type=float, default=0.5, help="--safety: the probability of being doomed from which an action is out")
ap.add_argument("--traj", default="", help="prediction network (traj_train.py): the policy keeps its choice unless the bullets predicted from the past frames make it clearly "
                "less survivable than the best action (traj_net.SoftSearch); no tracker")
ap.add_argument("--traj_h", type=int, default=30, help="--traj: game frames to look ahead")
ap.add_argument("--traj_margin", type=float, default=0.2, help="--traj: an action is out when its chance to survive is this share below the best action's")
ap.add_argument("--traj_delay", type=int, default=3, help="--traj: game frames from the picture until an action chosen now acts")
ap.add_argument("--traj_dilate", type=int, default=0, help="--traj: grow the predicted maps by this many cells before the search")
ap.add_argument("--traj_yield", type=float, default=-1.0, help="--traj: >= 0: wait for the GPU between the network's parts and then sleep this many ms, so that the "
                "game's own frames are not queued behind one long burst of work (2026-11-03: without it the response went from 28 to 40 ms)")
ap.add_argument("--pos_comp", type=float, default=0.0, help=">0: read the player position right after the screen grab and move it "
                "back by this many game frames of its own motion (the captured picture is ~1 frame older than the memory value: "
                "64%% of 243 measured pairs 1 frame, 15%% 2, 20%% 0; timing_fit_20261028 joint_profile.json)")
ap.add_argument("--no_mem", action="store_true", help="vision-only player tracking")
ap.add_argument("--memperc", default="", help="bullets / lasers / enemies from the game's memory (memperception.py, read-only) instead of the "
                "picture: 'default' or a memcalib_fit.py json of drawing sizes. The screen is then grabbed only every 3rd step (HUD, menus)")
ap.add_argument("--lang", default="en", help="input language of the game window at the start (setlang.py: en | zh | ja; none = leave it). With the Japanese input method "
                "the agent answered ~10 ms later (38-39 ms vs 27-29 ms, s1c1 1/10 vs 8/10, 2026-11-03); English (US, no input method) = Chinese")
ap.add_argument("--percep", type=int, default=3, choices=[1, 2, 3, 4, 5], help="perception version (1 = original, 3 = component clean-up, 4 = + colour bullets without a white core + lasers, 5 = game in AI mode: black background, no blind zone)")
ap.add_argument("--shoot", action="store_true", help="hold Z (fire) the whole time")
ap.add_argument("--ymin_play", type=float, default=-1.0, help="override the play-zone limit (game px); -1 = from checkpoint")
ap.add_argument("--standard", action="store_true", help="standard mode (3 lives): watch the life hearts instead of the MISS counter and stop at the continue prompt (game over)")
ap.add_argument("--continues", type=int, default=0, help="with --standard: accept the continue prompt this many times (Z on the default 'yes') instead of stopping")
ap.add_argument("--items", action="store_true", help="steer toward power (P) items when no bullet is near: adds a bonus to the policy logits")
ap.add_argument("--item_beta", type=float, default=2.5)
ap.add_argument("--shot_filter", action="store_true", help="drop the player's own shots from the bullet mask by frame-to-frame motion")
ap.add_argument("--reflex", type=int, default=0, help="--memperc: reflex layer (reflex.py), frames ahead: from the bullets in memory moved in straight lines and "
                "the TRUE hitboxes, keep the policy's action unless it is not safe in the next 2 frames + an escape after; then a safe one (0 = off)")
ap.add_argument("--dialogue_max_bullets", type=int, default=8, help="--memperc: the portrait rule's 'dialogue' (keys released, standing "
                "still) only counts with at most this many bullets in memory (2026-11-05: Patchouli's red fireballs filled the portrait "
                "corner and the agent stood still for 1-1.7 s, 3 deaths; real dialogues had 0 bullets)")
ap.add_argument("--laser_extend", type=float, default=0.0, help="--memperc: draw / check every laser from its origin at least this long "
                "(game px; 800 = past any screen edge; std_lunatic_laser1: 0 laser deaths but 3 bombs in 11 s on Cirno). 0 (default, "
                "2026-11-06) = the real beam: origin + [start, end] fields, moving on at its speed (memperception.py, reflex.laser_span)")
ap.add_argument("--reflex_hold", type=int, default=0, help=">0: timing-robust reflex (reflex.choose_robust): the key may act 1..3 frames "
                "after the read and the action stay on up to this many frames (2026-11-05: 33 ms between reads = 1-4 game frames). "
                "Offline (reflex_timing_check.py) hold 4: warned in time before 60%% of the standard-mode hits (old 0%%), 34%% of the "
                "Spell Practice hits (old 18%%), objected on 0.4%% of survived decisions. Use with --reflex 10")
ap.add_argument("--reflex_enemy_static", action="store_true", help="--reflex: treat enemies as standing still (before 2026-11-08); by default "
                "they move on at the velocity measured per slot between two memory reads (Remilia's bats fly ~4.5 px / frame)")
ap.add_argument("--reflex_delays", default="0,1,2", help="--reflex_hold: the key timings checked (game frames after the read during which the "
                "old action is still on). 2026-11-06 (delay_measure.py, exact frames from the bullets' own frame counters): the new key acts "
                "0 frames after the read ~30%%, 1 frame ~50%%, 2+ ~1%%; the old '1,2,3' never checked 0 -> 15 of 17 hits had been called "
                "safe on the decision before. '0,1,2': objects on 0.2-0.3%% of survived decisions, 'no way out' 0.1%% -> 0")
ap.add_argument("--reflex_delay", type=int, default=2, help="--reflex: game frames from the memory read until the keys act")
ap.add_argument("--dump_from", type=float, default=-1, help="save every decision's half-size frame + mask from this time (s) on (offline flicker analysis)")
ap.add_argument("--dump_secs", type=float, default=60)
ap.add_argument("--dump_no_img", action="store_true", help="--dump_from: masks + positions only, no raw frames (200 s: 0.6 instead of 2.3 GB)")
ap.add_argument("--grab_all", action="store_true", help="--memperc: grab the picture on every step (normally every 3rd), so a --dump_from "
                "recording has consecutive pictures next to the memory tables (2026-11-06: data for a vision-only reflex)")
ap.add_argument("--dump_tables", action="store_true", help="--dump_from + --memperc: also every dumped step's memory tables (bullets x y vx vy + sprite id, "
                "lasers x y angle length width state, enemies x y w h) -> dump_tables.npz, to redraw masks or make hitbox labels later")
ap.add_argument("--miss_back", type=float, default=0, help="death post-mortems reach this many s back (default: the last 1.3 s). In Spell Practice the "
                "'Failed' screen = the signal comes > 1 s after the hit, so the old montage showed only the time after it. Also saves "
                "miss_NN.npz (half-size raw frames + masks + player position of that whole window) for offline analysis")
ap.add_argument("--dump_full", action="store_true", help="dump full-resolution frames + masks (for perception work) instead of half size")
ap.add_argument("--start_tap", default="", help="press this key once after loading (e.g. z to start the selected spell card)")
ap.add_argument("--expect_label", default="", help="with --start_tap: the HUD difficulty label must read this (EASY / NORMAL / LUNATIC), else exit 3 without playing")
ap.add_argument("--spell_attempts", type=int, default=0, help="Spell Practice bench: after each attempt (result menu) log it, press Retry and go on, N attempts")
ap.add_argument("--stop_on_success", action="store_true", help="with --spell_attempts: stop at the first captured card")
ap.add_argument("--end_after_no_field", type=float, default=0, help=">0: stop when the golden playfield frame has been gone this many s (game over / ending / menus); keys are released as soon as it is gone for 2 s")
ap.add_argument("--focus_bias", type=float, default=0.0, help="add this to the logits of the Shift (slow) actions while a bullet is within --focus_radius")
ap.add_argument("--focus_radius", type=float, default=40.0, help="game px")
ap.add_argument("--sticky", type=float, default=0.0, help="add this to the logit of the previous action (less flicker)")
ap.add_argument("--align", action="store_true", help="when a boss is on screen (red glow on the bottom frame) and no bullet is near, favour moving under it")
ap.add_argument("--align_beta", type=float, default=2.0)
ap.add_argument("--enemy_align", action="store_true", help="--memperc, no boss on screen (road sections): when no bullet is within "
                "--enemy_gate, favour moving under an enemy above the player (memory positions; nearest by |dx| + 0.3 x height, kept "
                "until another is clearly better); no enemy above: pull back toward the middle when more than 96 px off it. "
                "2026-11-06, std_lunatic_laser1: in the road sections the player sat in a 64 px side strip 59%% of the time (boss "
                "fights 17%%) and right under an enemy only 24%%, while one was above it 74%%. Soft logit bonus only, no area limit")
ap.add_argument("--enemy_beta", type=float, default=2.0)
ap.add_argument("--enemy_gate", type=float, default=36.0, help="--enemy_align: nearest bullet at least this far (game px)")
ap.add_argument("--seek_bombs", action="store_true", help="--memperc: when a bomb (B) or 1UP item is on screen, not below the player, "
                "and no bullet within --seek_gate game px, favour moving under it (or down to it) so it falls onto the player; "
                "the traj veto and the reflex still decide (a bomb is worth a life). Items read from memory (read-only)")
ap.add_argument("--seek_beta", type=float, default=3.0)
ap.add_argument("--seek_gate", type=float, default=30.0)
ap.add_argument("--seek_types", default="3,5", help="--seek_bombs: item types (3 bomb, 5 1UP; 0 small P, 1 point, 2 big P, 4 full power)")
ap.add_argument("--center_beta", type=float, default=1.0, help="--enemy_align: the pull toward the middle (no enemy above)")
ap.add_argument("--record_every", type=int, default=0, help="save every Nth playfield frame as PNG (+ player pos) for offline perception work")
ap.add_argument("--bomber", default="", help="two-model setup: this bomb model (2 actions) decides when to press X, the policy (dodger) moves. "
                "Bombs left are read from the HUD stars. Only with the user's OK to press X.")
ap.add_argument("--doom_bomb", type=float, default=0.0, help=">0 (needs --traj): press X when the prediction network's doom (1 - the best "
                "action's chance to survive the next --traj_h frames) exceeds this; same locks / HUD star check as --bomber. Only with the user's OK per run")
ap.add_argument("--reflex_bomb", action="store_true", help="(needs --reflex) press X when the reflex layer finds NO safe action this decision "
                "(every move hits within --reflex frames, bullets in memory in straight lines); same locks / HUD star check as --bomber. "
                "Offline on the reflex A/B (reflex_bomb_check.py): 1.2 bombs / min, saves ~half the hits. Only with the user's OK per run")
ap.add_argument("--reflex_bomb_best", type=int, default=1, help="--reflex_bomb with --reflex_hold: X when the best action survives at most this "
                "many of the --reflex_delays timings (2026-11-06: with the old '1,2,3' 'none' (0) fired 10-19 times a game; with '0,1,2' "
                "it almost never fires -- std_lunatic_dfix1 died 3 times holding 3 bombs. '<= 1' fires about as often as the old rule "
                "(scratchpad bomb_trigger.py: 10-23 a game)")
ap.add_argument("--vision_reflex", action="store_true", help="the reflex layer (--reflex) from the PICTURE alone (vision_table.VisionTable2: "
                "white bullet cores tracked, the lag of the picture covered, the rest of the lit pixels still obstacles) instead of the bullets in "
                "memory; memory only for the player's position (2026-11-06, user: pure-vision scheme + reflex). Not with --memperc")
ap.add_argument("--vr_core", type=int, default=200, help="--vision_reflex: a bullet's core = min(R,G,B) at least this")
ap.add_argument("--vr_hb_min", type=float, default=5.0, help="--vision_reflex: smallest hitbox of a tracked core (game px)")
ap.add_argument("--vr_lags", default="0,0.6,1.2,1.8", help="--vision_reflex: frames the picture may be behind the position read (each moving bullet put in at all)")
ap.add_argument("--vr_reach_pf", type=float, default=6.0, help="--vision_reflex: a new track may jump this many half-size px per frame that passed")
ap.add_argument("--vr_use2", type=int, default=1, help="--vision_reflex: 1 = motion from 2 sightings already (the core also stays a still obstacle)")
ap.add_argument("--vr_big", type=int, default=1, help="--vision_reflex: 1 = big core blobs (merged cores) tracked as rigid groups")
ap.add_argument("--watch_mem", action="store_true", help="with --vision_reflex: ALSO read the bullet tables from memory after each decision, ONLY "
                "to dump them (--dump_tables) for the offline vision-vs-memory check; never used for any decision")
ap.add_argument("--reflex_bomb_consec", type=int, default=1, help="--reflex_bomb: X only when the best action survives <= --reflex_bomb_best "
                "timings on this many decisions in a row -- or none at all now (2026-11-06, --vision_reflex: the picture table is more "
                "cautious than memory; '<= 1' alone fired at 52 and 81 s of stage 1 where memory had 5-18 fully safe actions)")
ap.add_argument("--reflex_bomb_sum", type=int, default=-1, help=">= 0 (--reflex_bomb, --reflex_hold): X when the best action survives at most "
                "2 timings now and (now + the previous decision's best) <= this -- with 3: two shaky decisions in a row, one of them <= 1, "
                "or none at all now. 2026-11-06, vision games 1-2 (vision_game_check / dump_nok): '<= 1' once fired 5-7 times a minute, "
                "'<= 1' twice in a row caught 4 of 10 deaths, this one 6 of 10 at 1.2-2.1 a minute (memory reflex: ~0.2)")
ap.add_argument("--bomb_thr", type=float, default=0.5, help="the bomb model bombs when P(bomb) exceeds this")
ap.add_argument("--old_grabs", action="store_true", help="the grabs as before 2026-11-03 evening (separate HUD / prompt / field-frame grabs, "
                "each waiting ~14 ms for the next frame: 22 decisions / s in standard mode), for A/B runs")
ap.add_argument("--tint_q", type=float, default=0.0, help="perception v5: measure the 'whole-screen tint' at this quantile of the frame (0.1) "
                "instead of the median: under Remilia's pale bullets the median rose to 80-137 and the threshold with it (2026-11-03)")
ap.add_argument("--bomb_watch", default="", help="WATCH ONLY: this bomb model's P(bomb) ('3 bombs in hand, not invulnerable') is computed every "
                "decision after the keys are sent (no response time) and saved to bomb_watch.npy; X is never pressed. For Spell Practice, "
                "where there are no bombs: were its 'now' moments right before the hits?")
ap.add_argument("--bomb_head", default="", help="real-game calibrated decision (bomb_calib_fit.py .npz: weights on the bomb "
                "model's hidden layer + threshold) instead of the simulator head and --bomb_thr")
ap.add_argument("--bomb_lock", type=float, default=5.0, help="minimum s between two bombs (the bomb lasts ~5 s)")
ap.add_argument("--respawn_lock", type=float, default=4.0, help="s after a respawn (or the start) with no bomb (invulnerable anyway)")
ap.add_argument("--bomb_gate", type=float, default=0.0, help=">0: bomb only if a bullet pixel is within this many game px")
ap.add_argument("--newgame", default="", help="'DIFF MODE OUTDIR': after loading and warming up the models, start a fresh run with "
                "newgame.py (closes / relaunches the game) and take over the moment it is running")
ap.add_argument("--cpu_boost", type=int, default=1, help="1: this process only -- opt out of Windows' power throttling (EcoQoS) and run at "
                "ABOVE_NORMAL priority. 2026-11-06: with the game in the foreground the agent ran at 20-23 decisions/s (CPU-only reflex "
                "2.9 ms offline, 7.9 ms in the game) on the i5-13600K's hybrid cores; no system setting is touched")
args = ap.parse_args()


def cpu_boost():
    """-> text: our own process out of power throttling (EcoQoS -> efficiency cores / low clocks) and at above-normal priority."""
    import ctypes
    from ctypes import wintypes

    class PPTS(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    h = k32.GetCurrentProcess()
    s = PPTS(1, 0x1, 0x0)                                  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED controlled, state off
    ok1 = k32.SetProcessInformation(wintypes.HANDLE(h), 4, ctypes.byref(s), ctypes.sizeof(s))      # 4 = ProcessPowerThrottling
    ok2 = k32.SetPriorityClass(wintypes.HANDLE(h), 0x8000)                                          # ABOVE_NORMAL_PRIORITY_CLASS
    return f"power throttling off: {bool(ok1)}, above-normal priority: {bool(ok2)}"


if args.cpu_boost:
    print("cpu boost (this process): " + cpu_boost(), flush=True)

DX = [0, 0, 0, -1, 1, -1, 1, -1, 1]
DY = [0, -1, 1, 0, 0, -1, -1, 1, 1]


def keys_for(action):
    if action >= 18:                      # bomb: no movement keys (the X key is added by the caller)
        return set()
    m, focus = action % 9, action // 9
    ks = set()
    if DX[m] < 0: ks.add("left")
    if DX[m] > 0: ks.add("right")
    if DY[m] < 0: ks.add("up")
    if DY[m] > 0: ks.add("down")
    if focus: ks.add("shift")
    return ks


dev = torch.device("cuda")
net = None
n_act, vdim, ymin_play, stack_n, lags_n = 18, 3, 16.0, 4, None
if args.policy not in ("stay", "random"):
    ck = torch.load(args.policy, map_location=dev)
    n_act, vdim = ck.get("n_actions", 18), ck.get("vec_dim", 3)
    stack_n = ck.get("stack", 4)
    lags_n = ck.get("lags")
    ymin_play = float(ck.get("args", {}).get("ymin_play", 16.0))
    if ck.get("arch") == "cnn_gru":
        from memory_policy import build_policy
        net = build_policy(ck, dev)
    else:
        net = ActorCritic(n_act, vdim, stack_n, lags_n).to(dev).to(memory_format=torch.channels_last)
        if ck.get("future"):
            import future_env
            future_env.widen(net)
            assert args.predict > 0, "a model with future channels needs the tracker: --predict"
    net.load_state_dict(ck["net"]); net.eval()
    print(f"policy: {args.policy} (update {ck['update']}, {ck['steps']/1e6:.1f}M steps, {n_act} actions, vec {vdim}, "
          f"play zone y>={ymin_play:g})")
recurrent_policy = net is not None and getattr(net, "is_recurrent", False)
recurrent_state = None
if args.ymin_play >= 0:
    ymin_play = args.ymin_play
BOMB_INVULN_S = 3.0                         # same as the simulator (180 frames)
bombs_left, invuln_until, bombs_used = 0, 0.0, 0     # without --bomber: a bomb-aware policy sees 'no bombs, ever'
bomber = None
if args.bomber:
    K.VK["x"] = 0x58                        # X (bomb) exists only in this mode (keys.py leaves it out on purpose)
    bk = torch.load(args.bomber, map_location=dev)
    assert bk.get("n_actions") == 2 and bk.get("vec_dim") == 5, "not a bomb model"
    assert net is not None and list(bk.get("lags") or []) == list(lags_n or []) and bk.get("stack", 4) == stack_n, \
        "bomb model and dodger see different inputs"
    bomber = ActorCritic(2, 5, bk.get("stack", 4), bk.get("lags")).to(dev).to(memory_format=torch.channels_last)
    bomber.load_state_dict(bk["net"]); bomber.eval()
    bombs_left = 3                          # every life starts with 3; corrected from the HUD stars
    head_w = None
    if args.bomb_head:                      # P = sigmoid(h . w + b) on the bomb model's 256-d hidden layer, fitted on real footage
        hz = np.load(args.bomb_head)
        head_w, head_b = torch.tensor(hz["w"], device=dev, dtype=torch.float32), float(hz["b"])
        args.bomb_thr = float(hz["thr"])
    print(f"bomb model: {args.bomber} (update {bk['update']}, {bk['steps']/1e6:.1f}M steps), presses X when "
          f"{'calibrated ' if head_w is not None else ''}P(bomb) > {args.bomb_thr:.3g}; at least {args.bomb_lock:g} s between bombs, "
          f"none for {args.respawn_lock:g} s after a respawn" + (f", gate {args.bomb_gate:g} px" if args.bomb_gate else ""))
bombing = bomber is not None or args.doom_bomb > 0 or args.reflex_bomb     # who may press X: bomb model / traj doom / reflex 'no way out'
if args.reflex_bomb:
    assert args.reflex > 0 and bomber is None and args.doom_bomb <= 0, "--reflex_bomb needs --reflex, and no --bomber / --doom_bomb"
    K.VK["x"] = 0x58                        # X (bomb) exists only in this mode (keys.py leaves it out on purpose)
    bombs_left = 3
    print(f"reflex bombing: X when the best action survives <= {args.reflex_bomb_best} of the key timings"
          f" ({args.reflex} frames ahead); at least {args.bomb_lock:g} s between bombs, "
          f"none for {args.respawn_lock:g} s after a respawn")
if args.doom_bomb > 0:
    assert args.traj and bomber is None, "--doom_bomb needs --traj and no --bomber"
    K.VK["x"] = 0x58                        # X (bomb) exists only in this mode (keys.py leaves it out on purpose)
    bombs_left = 3
    print(f"doom bombing: X when the traj network's doom > {args.doom_bomb:g}; at least {args.bomb_lock:g} s between bombs, "
          f"none for {args.respawn_lock:g} s after a respawn")
STAR_X0, STAR_DX, STAR_Y = 1088, 20, 224    # client px: centre of the first of the 8 bomb stars, spacing, row (filled = white-green)
star_ok = None                              # does the star reader agree with our own count at the start? (else: count bombs ourselves)
pending_bomb = None                         # (time, stars before) of the last X press, until the HUD confirms it
bomb_events = []                            # (time s, P(bomb), bombs before)
watcher, watch_log = None, []               # --bomb_watch: the bomb model only watches (X is not even in K.VK), (time s, P) per decision
if args.bomb_watch:
    wk = torch.load(args.bomb_watch, map_location=dev)
    assert wk.get("n_actions") == 2 and wk.get("vec_dim") == 5, "not a bomb model"
    assert net is not None and list(wk.get("lags") or []) == list(lags_n or []) and wk.get("stack", 4) == stack_n, \
        "bomb model and dodger see different inputs"
    watcher = ActorCritic(2, 5, wk.get("stack", 4), wk.get("lags")).to(dev).to(memory_format=torch.channels_last)
    watcher.load_state_dict(wk["net"]); watcher.eval()
    print(f"bomb model WATCH ONLY: {args.bomb_watch} (update {wk['update']}), P(bomb) logged every decision, X never pressed")
lock_until = 0.0                            # no bomb decision before this (our bomb still running / just respawned)
cutin_t = -1e9                              # time of our last X press not known to have been ignored (bomb cut-in on screen)


def bomb_stars(sct_, l_, t_):
    """Number of filled bomb stars on the HUD (filled stars are near-white green, empty ones dark green)."""
    g_ = np.asarray(sct_.grab(dict(left=l_ + STAR_X0 - 8, top=t_ + STAR_Y - 8, width=8 * STAR_DX, height=16)))[..., :3].min(axis=2)
    return sum(int((g_[:, k * STAR_DX:k * STAR_DX + 16] > 150).sum() >= 10) for k in range(8))


last_xy = None
P = Perceiver(version=args.percep, stack=stack_n, shot_filter=args.shot_filter, lags=lags_n)
P.tint_q = args.tint_q
if args.predict > 0 or args.shield > 0:
    from tracker import BulletTracker
    P.tracker, P.lead, P.track_model = BulletTracker(), args.predict, args.predict_model
    P.future = bool(net is not None and ck.get("future"))
guard = None
if args.shield > 0:
    from shield import Shield
    guard = Shield(args.shield, args.shield_delay, dev)
safety_net, n_safety, n_safety_all = None, 0, 0
if args.safety:
    sk = torch.load(args.safety, map_location=dev)
    assert list(sk.get("lags") or []) == list(lags_n or []) and sk.get("stack", 4) == stack_n, "safety network and policy see different inputs"
    safety_net = ActorCritic(sk["n_actions"], sk["vec_dim"], sk["stack"], sk["lags"]).to(dev).to(memory_format=torch.channels_last)
    safety_net.load_state_dict(sk["net"]); safety_net.eval()
    print(f"safety network: {args.safety} ({sk['update']} cycles), action out when P(doomed) > {args.safety_thr:g}")
traj_net, traj_hist, traj_V, n_traj, n_traj_all = None, None, None, 0, 0
doom_log = []                                   # --traj: (time s, 1 - best survival chance over the 18 actions) per decision -> doom.npy
if args.traj:
    from traj_net import build as build_traj, SoftSearch, player_marker
    tk = torch.load(args.traj, map_location=dev)
    traj_net = build_traj(tk)
    traj_search = SoftSearch(args.traj_h // 2 - 1, dev)
    traj_first = (args.traj_delay + 2) // 2                                 # the maps are computed after the keys of the last decision and used at the next one
    traj_steps = (traj_first + traj_search.steps + traj_net.every - 1) // traj_net.every     # only as far as the search looks
    traj_V = None
    traj_pause = (lambda: (torch.cuda.synchronize(), time.sleep(args.traj_yield / 1000.0) if args.traj_yield > 0 else None)) if args.traj_yield >= 0 else None
    print(f"traj network: {args.traj} ({tk['iters']} iterations, held out {tk['holdout'] or 'nothing'}), {args.traj_h} frames ahead, margin {args.traj_margin:g}")
vproc = None                                     # --vision_reflex: the vision worker process (vision_proc.py), started before the game
if args.vision_reflex:
    assert not args.memperc, "--vision_reflex is the vision-only scheme: not with --memperc"
    assert args.reflex > 0 and not args.no_mem, "--vision_reflex needs --reflex and the player's position from memory"
    import atexit
    from vision_proc import VisionProc
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    vproc = VisionProc(["--core", str(args.vr_core), "--lags", args.vr_lags, "--reach_pf", str(args.vr_reach_pf), "--use2", str(args.vr_use2),
                        "--big", str(args.vr_big), "--hb_min", str(args.vr_hb_min)] + (["--out", args.out] if args.out else []),
                       log_path=os.path.join(args.out, "vision_worker.txt") if args.out else None)
    atexit.register(vproc.close)
    print(f"vision reflex: bullet table from the picture, worker process grabbing every frame (cores >= {args.vr_core}, lags {args.vr_lags}, "
          f"reach {args.vr_reach_pf}/frame, 2-sighting motion {bool(args.vr_use2)}, big groups {bool(args.vr_big)}, hitbox >= {args.vr_hb_min}); "
          f"memory only for the player's position" + ("; memory bullet tables read at the reflex for the dump only (--watch_mem)" if args.watch_mem else ""), flush=True)
if args.newgame:
    # Start the game only now that everything is loaded and warm: when newgame.py ran first, the game was already running for
    # ~10 s (newgame's own wait + loading torch and the models here) with nobody at the keys -- LUNATIC's first fairies killed
    # the idle player at the spawn point (2026-10-27).
    import subprocess
    with torch.no_grad():
        wx_ = (1.0, 0.0) if (vdim == 5 or bomber is not None) else tuple(np.eye(18)[0].tolist()) if vdim == 21 else ()
        o_, _ = P.observe(np.zeros((PF_H, PF_W, 3), np.uint8), focus=0, reset=True, mem_xy=(192.0, 384.0), extra_vec=wx_)
        if net is not None:
            warm_obs_ = dict(o_, vec=o_["vec"][:, :vdim]) if bomber is not None else o_
            if recurrent_policy:
                net.forward_step(warm_obs_, None)
            else:
                net(warm_obs_)
        if bomber is not None:
            bomber.pi(bomber.hidden(o_))
        if traj_net is not None:                     # (its first call builds the kernels: ~1 s that LUNATIC's first wave would get)
            h0_ = torch.zeros((PF_H // 2, PF_W // 2), dtype=torch.uint8, device=dev)
            x0_ = torch.stack([h0_] * len(traj_net.lags) + [player_marker(torch.tensor([192.0], device=dev), torch.tensor([384.0], device=dev))[0]], 0)[None].float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p0_ = torch.sigmoid(traj_net(x0_, traj_steps, None).float())
            traj_search.values(p0_, traj_first, None)
        torch.cuda.synchronize()
    ng = args.newgame.split()
    for try_ in range(2):
        r_ = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "newgame.py"), ng[0],
                             os.path.join(ng[2], f"newgame{try_ + 1}") if len(ng) > 2 else "", ng[1] if len(ng) > 1 else "standard"],
                            cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
        for line_ in (r_.stdout + r_.stderr).splitlines():
            print("newgame: " + line_, flush=True)
        if r_.returncode == 0:
            break
    else:
        K.release_all()
        sys.exit(4)

hwnd, _ = W.find_window()
if not hwnd:
    sys.exit("game window not found")
if args.lang and args.lang != "none":           # input language of the game window only: with Japanese the agent answered ~10 ms later (2026-11-03)
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "setlang.py"), args.lang])
l, t, r, b = W.client_rect(hwnd)
box = dict(left=l + PF_X0, top=t + PF_Y0, width=PF_W, height=PF_H)
removed_px = 0
dump = None
mem = None
if not args.no_mem:
    import gamemem as G
    G.PLAYER_RVAS[:] = [0x551720, 0x551980]          # player (x, y) globals in th06nc.exe v1.03 (read-only)
    mem = G.GameMem()
    if not (mem.ok and mem.player_xy()):
        print("memory read unavailable -> falling back to vision for the player position"); mem = None
    else:
        print("player position from memory: ", mem.player_xy())
mreader = mmask = None
if args.memperc:                                  # bullets / lasers / enemies from memory (read-only); the screen only for HUD / menus
    import json
    import memperception as MP
    if mem is None:
        print("--memperc needs the memory reader"); sys.exit(2)
    mreader = MP.MemReader(mem)
    draw_ = dict(MP.DRAW)
    if args.memperc != "default":                 # memcalib_fit2.py json: 'keys' ("sprite id,state" -> half sizes) stay strings
        for k, v in json.load(open(args.memperc)).items():
            draw_[k] = {int(a): b for a, b in v.items()} if k in ("along", "across", "laser_half", "laser_rel") else v
    mmask = MP.MemMask(draw=draw_)
    print(f"perception from memory ({args.memperc}): templates sprite {mreader.sprite.tolist()}")
    if args.items:
        print("WARNING: --items has no item list with --memperc (items come from the v5 picture)")
vtable = vproc                                    # (--vision_reflex: the worker)
if vproc is not None and args.watch_mem:
    assert mem is not None, "--watch_mem needs the memory reader"
    import memperception as MP
    mreader = MP.MemReader(mem)
vis_tab = None
watch_tab = None                                  # --watch_mem: the memory tables read at the reflex (dump only)
E0_ = dict(x=np.zeros(0, np.float32), y=np.zeros(0, np.float32), w=np.zeros(0, np.float32), h=np.zeros(0, np.float32))
vis_age = []                                      # --vision_reflex: frames from the worker's grab to the reflex, per decision
shaky_n = 0                                       # --reflex_bomb_consec: decisions in a row with a shaky best escape
prev_best = 3                                     # --reflex_bomb_sum: the previous decision's best escape
vis_nok = None                                    # this decision's reflex verdict (n_ok per action) for the dump (dump_nok.npy)
vis_meta = None                                   # (worker grab time of the table used, time of the shift, reflex x, y) -> dump_vismeta.npy
pic_age = []                                      # --vision_reflex: frames from the worker's grab to the start of the step (policy's picture)
reflex, reflex_log = None, []                     # (time s, policy's action, final action, safe actions) per decision the reflex looked at
reflex_none = False                               # --reflex_bomb: set by this step's reflex check, cleared after the bomb decision
if args.reflex > 0:
    if mreader is None and vtable is None:
        print("--reflex needs --memperc or --vision_reflex"); sys.exit(2)
    from reflex import Reflex, VX as RVX, VY as RVY
    reflex_delays = tuple(int(v) for v in args.reflex_delays.split(","))
    if args.reflex_hold > 0:
        reflex = Reflex(args.reflex, 3)             # horizon as validated offline (delay 3 + reflex frames)
        print(f"reflex layer (timing-robust): keys act {reflex_delays} frames after the read, action held up to {args.reflex_hold} frames, "
              f"escape searched {args.reflex} frames after a 3-frame delay, true hitboxes")
    else:
        reflex = Reflex(args.reflex, args.reflex_delay)
        print(f"reflex layer: {args.reflex} frames ahead, keys act {args.reflex_delay} frames after the read, true hitboxes")
    reflex.move_enemies = not args.reflex_enemy_static
    print("reflex: enemies " + ("stand still (--reflex_enemy_static)" if args.reflex_enemy_static else
                                "move on at their velocity (per slot, from two memory reads)"))
held = set()
cur_focus = 0
last_a = 0
if args.out:
    os.makedirs(args.out, exist_ok=True)
def save_hud(name):
    """HUD crop (MISS counter) from the live screen with mss; PrintWindow can return a stale cached frame."""
    l_, t_, r_, b_ = W.client_rect(hwnd)
    with mss.MSS() as s_:
        img = np.asarray(s_.grab(dict(left=l_ + 1100, top=t_ + 150, width=180, height=80)))[..., :3][..., ::-1]
    Image.fromarray(np.ascontiguousarray(img)).resize((540, 240)).save(os.path.join(args.out, name))


if args.out:
    save_hud("hud_start.png")
from collections import deque
from PIL import ImageDraw
ring = deque(maxlen=max(60, int(30 * (args.miss_back + 0.5))))   # last 2 s (or --miss_back) of (raw, overlay, nearest-bullet distance, action) for death post-mortems


def save_window(tag):
    """--miss_back: the last miss_back s of the ring -> <tag>.npz (half-size raw + mask + player position, for miss_zoom.py) and <tag>.jpg
    (8 tiles: raw above, mask in green + player below). Called when a Spell Practice attempt ends 'Failed' (~1 s after the hit; the HUD
    MISS signal comes ~3 s later still)."""
    n_ = min(len(ring), int(30 * args.miss_back) + 1)
    if n_ < 12:
        return
    seq = list(ring)[-n_:]
    np.savez(os.path.join(args.out, tag + ".npz"), raw=np.stack([e[0] for e in seq]), mask=np.packbits(np.stack([e[1] for e in seq]), axis=-1),
             nearest=np.array([e[2] for e in seq]), action=np.array([e[3] for e in seq]), pos=np.array([e[4] for e in seq]),
             age=np.array([e[6] for e in seq]))
    # --memperc grabs the screen only every 3rd step: a tile's picture would be up to 2 steps (~4 game frames) older than its mask, and
    # the green would sit ahead of the bullets. Each tile is moved back to the step whose picture was grabbed in that step.
    idx = [max(int(i) - seq[int(i)][6], 0) for i in np.linspace(0, n_ - 1, 8).round().astype(int)]
    tiles_raw, tiles_ov = [], []
    for i in idx:
        im_ = seq[i][0].copy(); mk_ = seq[i][1]
        tiles_raw.append(seq[i][0])
        im_[mk_] = (0.4 * im_[mk_] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        cxp, cyp = int(seq[i][4][0]), int(seq[i][4][1])
        im_[max(cyp - 1, 0):cyp + 2, max(cxp - 5, 0):cxp + 6] = (255, 255, 0)
        im_[max(cyp - 5, 0):cyp + 6, max(cxp - 1, 0):cxp + 2] = (255, 255, 0)
        tiles_ov.append(im_)
    sheet = Image.fromarray(np.concatenate([np.concatenate(tiles_raw, axis=1), np.concatenate(tiles_ov, axis=1)], axis=0))
    dr = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        dr.text((k * tiles_raw[0].shape[1] + 4, 4), f"{(i - n_ + 1) / 27:+.1f}s  nearest {seq[i][2]:.0f}px", fill=(255, 255, 0))
    sheet.save(os.path.join(args.out, tag + ".jpg"), quality=80)


n_deaths = 0                         # respawn jumps seen in memory (secondary signal)
rec_meta = []
miss_events = []                     # (time, min nearest-bullet distance 0.7..0.2 s before) from the HUD MISS counter
hud_prev, last_miss_t = None, -1e9
HUD_BOX = (1078, 176, 162, 26) if args.standard else (1200, 174, 60, 32)   # standard: the life hearts; challenge: the MISS number (client px x, y, w, h); a change = one more death
HUD_THR = 2.5 if args.standard else 6.0
game_over_t, go_hits = None, 0
n_continues = 0
_PROMPT = np.asarray(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "continue_prompt.png")).convert("L")).astype(np.float32)


# A second screen grab in the same decision waits for the next composed frame: the HUD grab cost 13.9 ms every 3rd step, the continue-prompt
# grab another 13.9 (standard mode: 22 decisions / s; step_bench.py, 2026-11-03). On those steps one WIDE grab (playfield + HUD + the
# frame left of it) replaces them all; the other reads are cut out of it. Client px:
WIDE_X0, WIDE_Y0, WIDE_W, WIDE_H = 340, PF_Y0, 920, PF_H + 6


def wide_cut(wide, x, y, w, h):
    """BGRA cut of the wide grab at client (x, y), w x h."""
    return wide[y - WIDE_Y0:y - WIDE_Y0 + h, x - WIDE_X0:x - WIDE_X0 + w]


def prompt_score(sct_, l_, t_, wide=None):
    """Normalised correlation of the screen with the continue prompt's text ("Continue?", client (555, 262), 157 x 42)."""
    g_ = (wide_cut(wide, 555, 262, 157, 42) if wide is not None else
          np.asarray(sct_.grab(dict(left=l_ + 555, top=t_ + 262, width=157, height=42))))[..., :3].astype(np.float32)
    g_ = g_[..., 2] * 0.299 + g_[..., 1] * 0.587 + g_[..., 0] * 0.114            # BGRA -> L
    a_, b_ = g_ - g_.mean(), _PROMPT - _PROMPT.mean()
    return float((a_ * b_).sum() / (np.sqrt((a_ * a_).sum() * (b_ * b_).sum()) + 1e-6))


def continue_rows(sct_, l_, t_):
    """Brightness of the 'yes' / 'no' rows of the continue prompt (the highlighted one pulses 170-240, the other ~96)."""
    f_ = np.asarray(sct_.grab(dict(left=l_ + 620, top=t_ + 368, width=48, height=93)))[..., :3].max(axis=2)
    return float(np.percentile(f_[:30], 95)), float(np.percentile(f_[63:], 95))
log = []
period = 1.0 / args.hz
t_start = time.perf_counter()
next_t = t_start
last_snap = -1e9
attempts, att_start, static_n, prev_small, last_static_check = [], time.perf_counter(), 0, None, -1e9
prev_pos_s = None
mxy_grab, t_mem, comp_prev = None, 0.0, None       # --pos_comp: position read at the grab, its time, the previous one
last_field_t = time.perf_counter()
last_field_check = -1e9
field_ok = True
ended_at = None
last_hud_snap = -1e9
n_item_steps = 0
n_seek_steps, seek_seen, seek_log, seek_err = 0, set(), [], False      # --seek_bombs
seek_types = {int(v) for v in args.seek_types.split(",") if v.strip()}
n_align_steps = 0
n_enemy_steps = n_center_steps = 0
enemy_tgt = None                                     # --enemy_align: slot of the enemy being lined up
boss_x = None
last_z = -1e9
last_front = -1e9
if not args.dry:
    W.bring_to_front(hwnd); time.sleep(0.3)
    if args.start_tap and W.is_foreground(hwnd):
        K.tap(args.start_tap, 0.15); time.sleep(1.0)       # (0.08 s was not taken on the practice stage menu, 2026-10-28)
        for try_ in range(3):                          # the HUD difficulty label (EASY / NORMAL / HARD / LUNATIC): record and check it
            with mss.MSS() as s_:
                lab_ = np.asarray(s_.grab(dict(left=l + 990, top=t + 10, width=280, height=80)))[..., :3][..., ::-1]
            name_, sc_, _ = difflabel.classify(lab_)
            if name_ or not args.expect_label:
                break
            time.sleep(0.5)                            # not shown yet (still fading in)
        if args.out:
            Image.fromarray(np.ascontiguousarray(lab_)).save(os.path.join(args.out, "difficulty_label.png"))
        print(f"difficulty label: {name_ or 'UNKNOWN'} (score {sc_:.2f})")
        if args.expect_label and name_ != args.expect_label.upper():
            K.release_all()
            print(f"could not verify difficulty: expected {args.expect_label.upper()}, read {name_ or 'UNKNOWN'} -> not playing")
            sys.exit(3)
        t_start = att_start = next_t = time.perf_counter()
n_dialogue = n_nofg = n_dlg_veto = 0
dlg_active, dlg_in, dlg_out, dlg_t0 = False, 0, 0, 0.0      # dialogue state with hysteresis (3 frames in, 5 frames out)
dlg_segs = []                                               # duration (s) of every dialogue segment
first = True
tm = dict(grab=0.0, hud=0.0, observe=0.0, policy=0.0, post=0.0)      # where does each 33 ms step go?
full_ = pf = strip = None
pf_age = 0                           # steps since pf was grabbed (--memperc reuses it on 2 of 3 steps; only the post-mortem pictures care)
try:
    with mss.MSS() as sct:
        while time.perf_counter() - t_start < args.seconds:
            now = time.perf_counter()
            l, t, r, b = W.client_rect(hwnd)               # the user may move the window: re-read every frame
            field_due = args.end_after_no_field > 0 and now - last_field_check > 0.5
            grabbed = True
            vpic_ = vproc.picture() if vproc is not None and not args.old_grabs else None
            if vpic_ is not None and now - vpic_[0] < 0.1:     # --vision_reflex: the worker grabs every frame; the newest picture
                wide = vpic_[1]                                # (a grab here would wait for the next frame and take every 2nd
                full_ = wide[:, PF_X0 - WIDE_X0:PF_X0 - WIDE_X0 + PF_W, :3][..., ::-1]     # frame away from the worker)
                pic_age.append((now - vpic_[0]) * 60.0)
            elif mmask is not None and not args.grab_all and not (len(log) % 3 == 0 or field_due or full_ is None):
                grabbed, wide = False, None                    # --memperc: no picture needed this step (the last one is kept)
            elif (len(log) % 3 == 0 or field_due) and not args.old_grabs:    # HUD / prompt / field-frame step: one wide grab instead of several (see WIDE_X0)
                wide = np.asarray(sct.grab(dict(left=l + WIDE_X0, top=t + WIDE_Y0, width=WIDE_W, height=WIDE_H)))
                full_ = wide[:, PF_X0 - WIDE_X0:PF_X0 - WIDE_X0 + PF_W, :3][..., ::-1]
            else:
                wide = None
                box = dict(left=l + PF_X0, top=t + PF_Y0, width=PF_W, height=PF_H + 6)     # +6 rows: the frame below the field (boss marker)
                full_ = np.asarray(sct.grab(box))[..., :3][..., ::-1]
            if grabbed:
                pf = np.ascontiguousarray(full_[:PF_H])
                strip = full_[PF_H + 1:PF_H + 5].astype(np.float32)
            pf_age = 0 if grabbed else pf_age + 1
            if vproc is not None:
                vproc.beat()                                   # (the worker stops when play.py has been silent for 20 s)
            if args.pos_comp > 0 and mem:                  # the position as close to the picture as possible (not after the HUD)
                mxy_grab, t_mem = mem.player_xy(), time.perf_counter()
            tm["grab"] += time.perf_counter() - now
            if field_due:                                   # golden playfield frame = we are in a stage
                last_field_check = now
                g_ = (wide_cut(wide, 340, 100, 20, 500) if wide is not None else
                      np.asarray(sct.grab(dict(left=l + 340, top=t + 100, width=20, height=500))))[..., :3][..., ::-1].astype(int)
                field_ok = ((g_[..., 0] > 150) & (g_[..., 1] > 100) & (g_[..., 2] < 110) & (g_[..., 0] - g_[..., 2] > 80)).mean() > 0.05
                if field_ok or not W.is_foreground(hwnd):     # another window in front covers the game: that is not 'gameplay
                    last_field_t = now                         # ended' (2026-10-27: the Claude window ended a run at 405 s)
            if args.end_after_no_field > 0 and now - last_field_t > 2.0:      # menu / ending: hands off the keyboard
                for k in list(held) + ["z", "ctrl"]:
                    K.up(k)
                held.clear(); first = True; last_a = 0
                if now - last_field_t > args.end_after_no_field:
                    ended_at = now - t_start
                    print(f"GAMEPLAY ENDED (no playfield for {args.end_after_no_field:.0f} s) at {ended_at:.1f}s")
                    break
                next_t += period; time.sleep(max(0, next_t - time.perf_counter())); continue
            if args.spell_attempts and now - last_static_check > 0.12 and grabbed and W.is_foreground(hwnd):     # (a backgrounded game also stops moving)
                last_static_check = now
                small = pf[::8, ::8].mean(axis=2)
                # the game is paused on the result menu: the picture AND the player's position stop. A still picture alone also
                # happens in Perfect Freeze's frozen phase and on the 'Spell Card Bonus!' screen (2026-10-28), while the game runs
                pos_s = mem.player_xy() if mem else None
                pos_still = pos_s is None or (prev_pos_s is not None and abs(pos_s[0] - prev_pos_s[0]) + abs(pos_s[1] - prev_pos_s[1]) < 0.01)
                prev_pos_s = pos_s
                static_n = static_n + 1 if prev_small is not None and np.abs(small - prev_small).mean() < 0.8 and pos_still else 0
                prev_small = small
                if static_n >= 5 and now - att_start > 1.5:                  # ~0.6 s without any change = paused on the result menu
                    for k in list(held) + ["z", "shift", "ctrl", "left", "right", "up", "down"]:
                        K.up(k)
                    held.clear()
                    dur = now - att_start - 0.6
                    time.sleep(0.3)                                           # let the menu finish fading in
                    def rows_():
                        c_ = np.asarray(sct.grab(dict(left=l, top=t, width=1280, height=720)))[..., :3].astype(int)
                        title = int((c_[268:312, 560:740].min(axis=2) > 170).sum())
                        return title, [int(c_[y0:y0 + 28, 540:740].max()) for y0 in (348, 396, 442)]   # the three option rows (Failed and Success menus)
                    title, rows = rows_()
                    for _w in range(12):        # a captured card first shows a still 'Spell Card Bonus!' screen: wait for the menu
                        if max(rows) > 60:
                            break
                        time.sleep(0.3); title, rows = rows_()
                    if max(rows) <= 60:         # no menu after all: not the end of an attempt
                        static_n, prev_small = 0, None
                        continue
                    for _w in range(6):         # the menu is still fading in: read the 'Failed...' title at its brightest
                        time.sleep(0.25)        # (read too early, a failure was logged as a capture -- 2026-10-28)
                        t2_, rows = rows_(); title = max(title, t2_)
                    outcome = "failed" if title > 150 else "success"
                    if args.out:                # the whole menu screen, to check the outcome by eye
                        Image.fromarray(np.ascontiguousarray(np.asarray(sct.grab(dict(left=l, top=t, width=1280, height=720)))[..., :3][..., ::-1])).resize(
                            (640, 360)).save(os.path.join(args.out, f"attempt_{len(attempts) + 1:02d}_menu.jpg"), quality=80)
                    attempts.append((now - t_start, dur, outcome))
                    print(f"attempt {len(attempts)}: {dur:5.1f}s  {outcome}  (menu rows {rows})", flush=True)
                    if args.out and args.miss_back > 0 and outcome == "failed" and len(attempts) <= 30:
                        save_window(f"fail_{len(attempts):02d}_t{now - t_start:05.1f}")
                    if args.out and len(attempts) <= 30:
                        Image.fromarray(np.ascontiguousarray(pf)).save(os.path.join(args.out, f"attempt_{len(attempts):02d}_end.jpg"), quality=80)
                    if len(attempts) >= args.spell_attempts or (args.stop_on_success and outcome == "success"):
                        break                                         # (the result menu stays up: its Save Replay is used by hand)
                    ok_ = False
                    for _ in range(4):                                         # make sure the cursor is on Retry (never on Save Replay)
                        if not W.is_foreground(hwnd):                          # never type into another window
                            break
                        title, rows = rows_()
                        if int(np.argmax(rows)) == 2:
                            ok_ = True; break
                        K.tap("down", 0.08); time.sleep(0.35)
                    if not ok_ or not W.is_foreground(hwnd):
                        print("could not put the cursor on Retry (or the game lost focus) -> stop"); break
                    K.tap("z", 0.08); time.sleep(1.0)
                    first, static_n, prev_small, last_a = True, 0, None, 0
                    P.prev_raw = None
                    att_start = time.perf_counter(); next_t = att_start
                    continue
            fg = W.is_foreground(hwnd)
            if not fg:                                   # never type into another window
                for k in list(held):
                    K.up(k)
                held.clear(); n_nofg += 1; first = True; last_a = 0
                # take the foreground back only if the user is not typing/clicking somewhere else
                if not args.dry and now - last_front > 1.0 and W.user_idle_seconds() > 2.0:
                    W.bring_to_front(hwnd); last_front = now
                next_t += period; time.sleep(max(0, next_t - time.perf_counter())); continue
            # --- death detector: the HUD MISS number changes the moment the player is hit
            t_h = time.perf_counter()
            hud_diff = 0.0
            redness = (strip[..., 0] - 0.5 * (strip[..., 1] + strip[..., 2])).mean(axis=0)
            redness = np.convolve(redness, np.ones(9) / 9, mode='same')
            boss_x = float(redness.argmax()) / S if redness.max() - np.median(redness) > 25 else None       # game px, None = no boss
            if len(log) % 3 == 0:                                    # every 3rd step is plenty (100 ms) and saves ~4 ms
                hud = (wide_cut(wide, *HUD_BOX) if wide is not None else
                       np.asarray(sct.grab(dict(left=l + HUD_BOX[0], top=t + HUD_BOX[1], width=HUD_BOX[2], height=HUD_BOX[3]))))[..., :3].astype(np.int16)
                hud_diff = float(np.abs(hud - hud_prev).mean()) if hud_prev is not None else 0.0
                hud_prev = hud
                if bombing and now - t_start > 1.0 and (len(log) % 9 == 0 or pending_bomb is not None):
                    st_ = bomb_stars(sct, l, t)                  # every 9th step (~0.3 s), or every 3rd while an X press is unconfirmed
                    if star_ok is None:
                        star_ok = st_ == 3 - bombs_used
                        print(f"bomb stars at the start: {st_} (pressed {bombs_used}) -> " + ("bombs read from the HUD" if star_ok else
                              "WARNING: star reader off, counting bombs ourselves"), flush=True)
                    if star_ok:
                        if pending_bomb is not None and now - pending_bomb[0] > 0.5:
                            if st_ >= pending_bomb[1]:               # no star went away: the press did not register
                                invuln_until = lock_until = 0.0; bombs_used -= 1; cutin_t = -1e9
                                print(f"X at {pending_bomb[0] - t_start:.1f}s did not register (stars {pending_bomb[1]} -> {st_})", flush=True)
                            pending_bomb = None
                        if pending_bomb is None:
                            bombs_left = st_
                if args.standard and now - t_start > 5.0:                # game over = the continue prompt ("Continue?") is on screen
                    go_hits = go_hits + 1 if prompt_score(sct, l, t, wide) > 0.8 else 0     # (template match: white bullets there fooled the old pixel count)
                    if go_hits >= 2 and n_continues < args.continues and W.is_foreground(hwnd):
                        for k in list(held) + ["z", "shift", "ctrl", "left", "right", "up", "down"]:
                            K.up(k)
                        held.clear()
                        if args.out:                                 # what the prompt looked like (for the record)
                            c_ = np.asarray(sct.grab(dict(left=l, top=t, width=1280, height=720)))[..., :3][..., ::-1]
                            Image.fromarray(np.ascontiguousarray(c_)).save(os.path.join(args.out, f"continue_{n_continues + 1:02d}.png"))
                        time.sleep(0.8)
                        on_yes = False                               # the cursor starts on 'no': move it to 'yes' and VERIFY before Z
                        for _ in range(4):
                            yes_, no_ = continue_rows(sct, l, t)
                            if yes_ > no_ + 30:
                                on_yes = True; break
                            K.tap("up", 0.08); time.sleep(0.5)
                        if not on_yes or not W.is_foreground(hwnd):
                            print(f"could not put the cursor on 'yes' at {now - t_start:.1f}s -> stop")
                            game_over_t = now - t_start
                            break
                        K.tap("z", 0.1); time.sleep(2.5)
                        if prompt_score(sct, l, t) > 0.5:
                            print(f"continue prompt still there after Z at {now - t_start:.1f}s -> stop")
                            game_over_t = now - t_start
                            break
                        n_continues += 1
                        bombs_left, pending_bomb = (3 if bombing else 0), None
                        lock_until = time.perf_counter() + args.respawn_lock
                        print(f"CONTINUE {n_continues} at {now - t_start:.1f}s (deaths so far {n_deaths})", flush=True)
                        go_hits, hud_prev, first, last_a = 0, None, True, 0
                        last_miss_t = time.perf_counter() + 3.0
                        next_t = time.perf_counter()
                        continue
                    if go_hits >= 2:
                        game_over_t = now - t_start
                        print(f"GAME OVER (continue prompt) at {game_over_t:.1f}s")
                        break
            if hud_diff > HUD_THR and now - last_miss_t > 1.5 and now - t_start > 4.0:      # first seconds: restart refreshes the HUD
                if recurrent_policy:
                    recurrent_state = None
                last_miss_t = now
                tt = now - t_start
                recent = [e for e in ring if True][-22:-6]                    # roughly 0.7 s .. 0.2 s before the signal
                md = min([e[2] for e in recent], default=99.0)
                miss_events.append((tt, md))
                if vproc is not None and args.out and len(miss_events) <= 20:
                    vproc.dump_ring()                                         # (the worker's own last ~3 s of pictures -> vring_NN.npz)
                if args.out and len(ring) >= 12 and len(miss_events) <= 20:
                    back = (-40, -34, -28, -22, -16, -10, -4)
                    if args.miss_back > 0:                                           # 7 tiles evenly from miss_back s before to the signal
                        back = tuple(int(round(-30 * args.miss_back + k * (30 * args.miss_back - 4) / 6)) for k in range(7))
                        n_ = min(len(ring), int(30 * args.miss_back) + 1)
                        np.savez(os.path.join(args.out, f"miss_{len(miss_events):02d}.npz"),
                                 raw=np.stack([ring[i][0] for i in range(-n_, 0)]), mask=np.packbits(np.stack([ring[i][1] for i in range(-n_, 0)]), axis=-1),
                                 nearest=np.array([ring[i][2] for i in range(-n_, 0)]), action=np.array([ring[i][3] for i in range(-n_, 0)]),
                                 pos=np.array([ring[i][4] for i in range(-n_, 0)]), age=np.array([ring[i][6] for i in range(-n_, 0)]),
                                 t_signal=tt)
                    idx = [i - ring[i][6] for i in back if -i <= len(ring)]       # (--memperc: the step whose picture is its own, as above)
                    idx = [i for i in idx if -i <= len(ring)]
                    tiles_raw = [ring[i][0] for i in idx]
                    tiles_ov = []
                    for i in idx:                                                    # build the overlay only now (not every step)
                        im_ = ring[i][0].copy(); mk_ = ring[i][1]
                        im_[mk_] = (0.4 * im_[mk_] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
                        if ring[i][5] is not None:
                            rm_ = ring[i][5]; im_[rm_] = (0.4 * im_[rm_] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
                        cxp, cyp = int(ring[i][4][0]), int(ring[i][4][1])
                        im_[max(cyp - 1, 0):cyp + 2, max(cxp - 5, 0):cxp + 6] = (255, 255, 0)
                        im_[max(cyp - 5, 0):cyp + 6, max(cxp - 1, 0):cxp + 2] = (255, 255, 0)
                        tiles_ov.append(im_)
                    sheet = Image.fromarray(np.concatenate([np.concatenate(tiles_raw, axis=1), np.concatenate(tiles_ov, axis=1)], axis=0))
                    dr = ImageDraw.Draw(sheet)
                    for k, i in enumerate(idx):
                        dr.text((k * tiles_raw[0].shape[1] + 4, 4), f"{i / 30:+.1f}s  nearest {ring[i][2]:.0f}px", fill=(255, 255, 0))
                    sheet.save(os.path.join(args.out, f"miss_{len(miss_events):02d}_t{tt:05.1f}.jpg"), quality=80)
            tm["hud"] += time.perf_counter() - t_h
            mxy = (mxy_grab if args.pos_comp > 0 else mem.player_xy()) if mem else None
            t_mxy = time.perf_counter()
            obs_xy = mxy
            if args.pos_comp > 0 and mxy and comp_prev:    # back by pos_comp frames of the player's own velocity (per 60 Hz frame)
                dt_f = (t_mem - comp_prev[0]) * 60.0
                dx_, dy_ = mxy[0] - comp_prev[1], mxy[1] - comp_prev[2]
                if 0.5 < dt_f < 12 and abs(dx_) + abs(dy_) < 30:           # (not across a respawn / stage-start jump)
                    obs_xy = (mxy[0] - float(np.clip(dx_ / dt_f * args.pos_comp, -8, 8)),
                              mxy[1] - float(np.clip(dy_ / dt_f * args.pos_comp, -8, 8)))
            if args.pos_comp > 0 and mxy:
                comp_prev = (t_mem, mxy[0], mxy[1])
            if mxy and last_xy and abs(mxy[0] - last_xy[0]) + abs(mxy[1] - last_xy[1]) > 40 \
                    and abs(mxy[0] - 192) < 6 and abs(mxy[1] - 384) < 6:      # respawn jump (secondary death signal)
                n_deaths += 1
                if recurrent_policy:
                    recurrent_state = None
                if bombing:                                 # new life: 3 bombs, and invulnerable for a while (no bomb needed)
                    pending_bomb, lock_until = None, max(lock_until, now + args.respawn_lock)
                    if not star_ok:
                        bombs_left = 3
            last_xy = mxy
            extra = ((min(bombs_left, 3) / 3.0, max(0.0, invuln_until - now) / BOMB_INVULN_S) if (vdim == 5 or bomber is not None) else
                     tuple(float(v) for v in np.eye(18)[last_a]) if vdim == 21 else ())     # 21: previous action one-hot
            t_o = time.perf_counter()
            if first and recurrent_policy:
                recurrent_state = None
            view_xy = None
            if args.predict > 0 and obs_xy:
                m_, f_ = last_a % 9, args.predict_self if args.predict_self >= 0 else args.predict - 1
                sp_ = (2.0 if last_a >= 9 else 4.0) * (0.7071 if DX[m_] and DY[m_] else 1.0) * f_
                view_xy = (float(np.clip(obs_xy[0] + DX[m_] * sp_, 8, 376)), float(np.clip(obs_xy[1] + DY[m_] * sp_, 16, 432)))
            mo_ = mem_tab = None
            if mmask is not None and obs_xy:              # --memperc: bullets / lasers / enemies read right now, drawn like v5
                mem_tab = mreader.read()
                if args.laser_extend > 0 and len(mem_tab["lasers"]["len"]):          # every laser as long as the screen (mask + reflex)
                    la_ = mem_tab["lasers"]
                    la_["len"] = np.maximum(la_["len"], args.laser_extend).astype(np.float32)
                    la_["start"], la_["end"], la_["speed"] = np.zeros_like(la_["len"]), la_["len"].copy(), np.zeros_like(la_["len"])
                mo_ = mmask.mask(mem_tab, obs_xy)
            obs, (px, py, mask) = P.observe(pf, focus=cur_focus, reset=first, mem_xy=obs_xy, extra_vec=extra, grab_t=now, view_xy=view_xy,
                                            mask_override=mo_)
            tm["observe"] += time.perf_counter() - t_o
            if traj_net is not None:                              # (the half-size masks it was trained on)
                h_ = mask[::2, ::2].to(torch.uint8)
                traj_hist = [h_] * (traj_net.lags[0] + 1) if first or traj_hist is None else traj_hist[1:] + [h_]
                if first:
                    traj_V = None
            if P.tracker is not None:                             # (parts of observe)
                tm["track_update"] = tm.get("track_update", 0.0) + P.t_update
                tm["track_shift"] = tm.get("track_shift", 0.0) + P.t_shift
                tm["track_future"] = tm.get("track_future", 0.0) + P.t_future
            dlg_seen = not args.spell_attempts and P.in_dialogue() and not now - cutin_t < 3.0
            # (our own bomb shows a big Reimu cut-in for ~2 s, which the portrait rule took for a dialogue; only a press the
            # game accepted counts: in a real dialogue X does nothing, and pressing it every 2 s there kept the agent moving)
            if dlg_seen and mem_tab is not None and int(np.sum(np.asarray(mem_tab["bullets"]["key"]) != 0)) > args.dialogue_max_bullets:
                dlg_seen = False; n_dlg_veto += 1        # --memperc: red bullets in the portrait corner (Patchouli's fireballs), not a story scene
            if dlg_seen:
                dlg_in, dlg_out = dlg_in + 1, 0
            else:
                dlg_in, dlg_out = 0, dlg_out + 1
            if not dlg_active and dlg_in >= 3:
                dlg_active, dlg_t0 = True, now
            elif dlg_active and dlg_out >= 5:
                dlg_active = False; dlg_segs.append(now - dlg_t0)
                if not args.dry:
                    K.up("ctrl")
            if dlg_active:                               # story text: no steering; hold Ctrl (the game's skip) and tap Z
                n_dialogue += 1; first = True; last_a = 0
                if not args.dry:
                    for k in list(held):
                        K.up(k)
                    held.clear()
                    K.down("ctrl")
                    if now - last_z > 0.1:
                        K.tap("z", 0.04); last_z = now
                next_t += period; time.sleep(max(0, next_t - time.perf_counter())); continue
            first = False
            t_p = time.perf_counter()
            if args.policy == "stay":
                a = 0
            elif args.policy == "random":
                a = int(np.random.randint(18))
            else:
                with torch.no_grad():
                    policy_obs_ = dict(obs, vec=obs["vec"][:, :vdim]) if bomber is not None else obs
                    if recurrent_policy:
                        logits, _, recurrent_state = net.forward_step(policy_obs_, recurrent_state)
                    else:
                        logits, _ = net(policy_obs_)
                    if args.focus_bias:
                        cx_, cy_ = int(P.pos[0]), int(P.pos[1])
                        x0_, y0_ = max(cx_ - 90, 0), max(cy_ - 90, 0)
                        nz = mask[y0_:cy_ + 90, x0_:cx_ + 90].nonzero()
                        nd2 = float(((nz[:, 1] + x0_ - cx_).float() ** 2 + (nz[:, 0] + y0_ - cy_).float() ** 2).sqrt().min().item() / 1.5) if len(nz) else 99.0
                        if nd2 < args.focus_radius:
                            logits = logits + torch.cat([torch.zeros(9, device=dev), torch.full((9,), args.focus_bias, device=dev)])[None]
                    if args.sticky and last_a < 18:
                        logits = logits.clone(); logits[0, last_a] += args.sticky
                    if args.items and P.items:
                        pw = [it for it in P.items if it[2] and 100 < it[1] < 425 and abs(it[0] - px) < 170 and -230 < it[1] - py < 130]
                        if pw:
                            cx_, cy_ = int(P.pos[0]), int(P.pos[1])
                            x0_, y0_ = max(cx_ - 150, 0), max(cy_ - 150, 0)
                            nz = mask[y0_:cy_ + 150, x0_:cx_ + 150].nonzero()
                            nd0 = float(((nz[:, 1] + x0_ - cx_).float() ** 2 + (nz[:, 0] + y0_ - cy_).float() ** 2).sqrt().min().item() / 1.5) if len(nz) else 99.0
                            if nd0 >= 45.0:                              # quiet: nothing within 45 game px
                                tgt = min(pw, key=lambda it: (it[0] - px) ** 2 + (it[1] - py) ** 2)
                                vx_, vy_ = tgt[0] - px, tgt[1] - py
                                dist_ = (vx_ ** 2 + vy_ ** 2) ** 0.5
                                if dist_ > 6.0:
                                    bonus = np.zeros(18, dtype=np.float32)
                                    for a_ in range(18):
                                        m_ = a_ % 9
                                        if m_ == 0:
                                            continue
                                        nv_ = (DX[m_] ** 2 + DY[m_] ** 2) ** 0.5
                                        bonus[a_] = args.item_beta * (DX[m_] * vx_ + DY[m_] * vy_) / (nv_ * dist_) * (0.7 if a_ >= 9 else 1.0)
                                    logits = logits + torch.from_numpy(bonus).to(dev)[None]
                                    n_item_steps += 1
                    seek_on = False
                    if args.seek_bombs and mreader is not None and obs_xy:      # bomb / 1UP items: get under them (2026-11-09)
                        try:
                            it_ = mreader.read_items()
                        except Exception as ex_:
                            it_ = None
                            if not seek_err:
                                print("read_items failed:", repr(ex_)[:80], flush=True); seek_err = True
                        if it_ is not None and len(it_["x"]):
                            for k_ in range(len(it_["x"])):          # each wanted item once: log it and keep a picture (to check the type)
                                key_ = (int(it_["slot"][k_]), int(it_["type"][k_]))
                                if int(it_["type"][k_]) in seek_types and key_ not in seek_seen:
                                    seek_seen.add(key_)
                                    seek_log.append((now - t_start, int(it_["type"][k_]), float(it_["x"][k_]), float(it_["y"][k_])))
                                    print(f"item type {key_[1]} seen at {now - t_start:.1f}s ({it_['x'][k_]:.0f}, {it_['y'][k_]:.0f}) slot {key_[0]}", flush=True)
                                    if args.out:
                                        try:
                                            qx_, qy_ = int(1.5 * it_["x"][k_]), int(1.5 * it_["y"][k_])
                                            Image.fromarray(np.ascontiguousarray(pf[max(qy_ - 60, 0):qy_ + 60, max(qx_ - 60, 0):qx_ + 60])).save(
                                                os.path.join(args.out, f"item{key_[1]}_{now - t_start:06.1f}.jpg"), quality=90)
                                        except Exception:
                                            pass
                            want_ = np.isin(it_["type"], list(seek_types)) & (it_["y"] > -8) & (it_["y"] < py + 12) & \
                                (it_["x"] > 0) & (it_["x"] < 384)
                            if want_.any():
                                j_ = int(np.argmin(np.where(want_, np.abs(it_["x"] - px), np.inf)))
                                tx_, ty_ = float(it_["x"][j_]), float(it_["y"][j_])
                                cx_, cy_ = int(P.pos[0]), int(P.pos[1])
                                x0_, y0_ = max(cx_ - 150, 0), max(cy_ - 150, 0)
                                nz = mask[y0_:cy_ + 150, x0_:cx_ + 150].nonzero()
                                nd4 = float(((nz[:, 1] + x0_ - cx_).float() ** 2 + (nz[:, 0] + y0_ - cy_).float() ** 2).sqrt().min().item() / 1.5) if len(nz) else 99.0
                                dxs_ = tx_ - px
                                dys_ = (ty_ - py) if ty_ > py else 0.0        # never up toward it: it falls onto the player
                                if nd4 >= args.seek_gate and (abs(dxs_) > 3.0 or dys_ > 3.0):
                                    n_ = (dxs_ ** 2 + dys_ ** 2) ** 0.5
                                    bonus = np.zeros(18, dtype=np.float32)
                                    for a_ in range(18):
                                        m_ = a_ % 9
                                        if m_:
                                            bonus[a_] = args.seek_beta * (DX[m_] * dxs_ + DY[m_] * dys_) / (((DX[m_] ** 2 + DY[m_] ** 2) ** 0.5) * n_) \
                                                * (0.7 if a_ >= 9 else 1.0)
                                    logits = logits + torch.from_numpy(bonus).to(dev)[None]
                                    seek_on = True
                                    n_seek_steps += 1
                    if args.align and boss_x is not None and abs(boss_x - px) > 14.0 and not seek_on:
                        cx_, cy_ = int(P.pos[0]), int(P.pos[1])
                        x0_, y0_ = max(cx_ - 150, 0), max(cy_ - 150, 0)
                        nz = mask[y0_:cy_ + 150, x0_:cx_ + 150].nonzero()
                        nd1 = float(((nz[:, 1] + x0_ - cx_).float() ** 2 + (nz[:, 0] + y0_ - cy_).float() ** 2).sqrt().min().item() / 1.5) if len(nz) else 99.0
                        if nd1 >= 40.0:
                            sgn_ = 1.0 if boss_x > px else -1.0
                            bonus = np.array([args.align_beta * sgn_ * DX[a_ % 9] * (0.7 if a_ >= 9 else 1.0) for a_ in range(18)], dtype=np.float32)
                            logits = logits + torch.from_numpy(bonus).to(dev)[None]
                            n_align_steps += 1
                    if args.enemy_align and boss_x is None and mem_tab is not None and not seek_on:
                        en_ = mem_tab["enemies"]
                        ex_, ey_ = np.asarray(en_["x"], np.float32), np.asarray(en_["y"], np.float32)
                        ok_ = (ey_ > 8) & (ey_ < py - 24) & (ex_ > 4) & (ex_ < 380) & (np.asarray(en_["w"], np.float32) > 0)
                        tgt_x = None
                        if ok_.any():
                            cost_ = np.where(ok_, np.abs(ex_ - px) + 0.3 * (py - ey_), np.inf)
                            j_ = int(np.argmin(cost_))
                            if enemy_tgt is not None:                    # keep the one being lined up unless another is clearly better
                                k_ = np.nonzero(np.asarray(en_["slot"]) == enemy_tgt)[0]
                                if len(k_) and ok_[k_[0]] and cost_[k_[0]] < cost_[j_] + 40.0:
                                    j_ = int(k_[0])
                            enemy_tgt, tgt_x, beta_ = int(en_["slot"][j_]), float(ex_[j_]), args.enemy_beta
                        else:
                            enemy_tgt = None
                            if abs(px - 192.0) > 96.0:
                                tgt_x, beta_ = 192.0, args.center_beta
                        if tgt_x is not None and abs(tgt_x - px) > 6.0:
                            cx_, cy_ = int(P.pos[0]), int(P.pos[1])
                            x0_, y0_ = max(cx_ - 150, 0), max(cy_ - 150, 0)
                            nz = mask[y0_:cy_ + 150, x0_:cx_ + 150].nonzero()
                            nd3 = float(((nz[:, 1] + x0_ - cx_).float() ** 2 + (nz[:, 0] + y0_ - cy_).float() ** 2).sqrt().min().item() / 1.5) if len(nz) else 99.0
                            if nd3 >= args.enemy_gate:
                                sgn_ = 1.0 if tgt_x > px else -1.0
                                bonus = np.array([beta_ * sgn_ * DX[a_ % 9] * (0.7 if a_ >= 9 else 1.0) for a_ in range(18)], dtype=np.float32)
                                logits = logits + torch.from_numpy(bonus).to(dev)[None]
                                if ok_.any():
                                    n_enemy_steps += 1
                                else:
                                    n_center_steps += 1
                    if guard is not None and obs_xy:              # (where it is when this action acts: the memory value is a frame newer than the picture)
                        t_s = time.perf_counter()
                        m_, f_ = last_a % 9, max(args.shield_delay - 1, 0)
                        sp_ = (2.0 if last_a >= 9 else 4.0) * (0.7071 if DX[m_] and DY[m_] else 1.0) * f_
                        xy_, rad_ = P.tracked_paths(args.shield_delay + guard.steps * 2)
                        veto_ = guard.veto(xy_, rad_, torch.tensor(float(np.clip(obs_xy[0] + DX[m_] * sp_, 8, 376)), device=dev),
                                           torch.tensor(float(np.clip(obs_xy[1] + DY[m_] * sp_, 16, 432)), device=dev))
                        if veto_ is not None:
                            guard.seen += 1
                            a0_ = int(logits.argmax(1))
                            logits = logits.masked_fill(veto_[None], float("-inf"))
                            guard.vetoes += int(int(logits.argmax(1)) != a0_)
                        tm["shield"] = tm.get("shield", 0.0) + time.perf_counter() - t_s
                    if safety_net is not None:
                        t_s = time.perf_counter()
                        veto_ = torch.sigmoid(safety_net(obs)[0].float()) > args.safety_thr
                        n_safety_all += 1
                        if bool(veto_.any()) and not bool(veto_.all()):
                            a0_ = int(logits.argmax(1))
                            logits = logits.masked_fill(veto_, float("-inf"))
                            n_safety += int(int(logits.argmax(1)) != a0_)
                        tm["safety"] = tm.get("safety", 0.0) + time.perf_counter() - t_s
                    if traj_V is not None and obs_xy:             # only a look-up: the maps were made after the last decision's keys (below)
                        t_s = time.perf_counter()
                        m_, f_ = last_a % 9, max(args.traj_delay - 1, 0)
                        sp_ = (2.0 if last_a >= 9 else 4.0) * (0.7071 if DX[m_] and DY[m_] else 1.0) * f_
                        xy_ = torch.tensor([[np.clip(obs_xy[0] + DX[m_] * sp_, 8, 376), np.clip(obs_xy[1] + DY[m_] * sp_, 16, 432)]], device=dev, dtype=torch.float32)
                        s_ = traj_search.lookup(traj_V, xy_[:, 0], xy_[:, 1])
                        doom_log.append((now - t_start, 1.0 - float(s_.max())))     # no way out = a bomb moment? (traj_doom.py; watch only)
                        veto_ = s_ < s_.max() * (1 - args.traj_margin)
                        n_traj_all += 1
                        if bool(veto_.any()):
                            a0_ = int(logits.argmax(1))
                            logits = logits.masked_fill(veto_, float("-inf"))
                            n_traj += int(int(logits.argmax(1)) != a0_)
                        tm["traj_lookup"] = tm.get("traj_lookup", 0.0) + time.perf_counter() - t_s
                    a = int(torch.distributions.Categorical(logits=logits).sample() if args.sample else logits.argmax(1))
                    reflex_none = False                                                   # (--reflex_bomb: no safe action this decision)
                    vis_nok = None
                    vis_meta = None
                    rtab_, rxy_ = mem_tab, obs_xy
                    if vproc is not None and obs_xy:                   # --vision_reflex: the worker's newest table; a fresh position read,
                        t_s = time.perf_counter()                      # the moving bullets go on for the time since the worker's grab
                        got_ = vproc.latest()
                        rtab_ = None
                        if got_ is not None and time.perf_counter() - got_[0] < 0.2:
                            rxy_ = mem.player_xy() or obs_xy
                            t_e_ = time.perf_counter()
                            e_ = (t_e_ - got_[0]) * 60.0
                            vis_meta = (got_[0], t_e_, rxy_[0], rxy_[1])
                            b_ = got_[2]
                            rtab_ = dict(bullets=dict(b_, x=b_["x"] + b_["vx"] * e_, y=b_["y"] + b_["vy"] * e_), enemies=E0_, lasers=dict(x=[]))
                            vis_age.append(e_)
                            if args.watch_mem and mreader is not None:
                                watch_tab = mreader.read()             # (dump only: what memory says at the same moment)
                        tm["vtable_read"] = tm.get("vtable_read", 0.0) + time.perf_counter() - t_s
                    if reflex is not None and rtab_ is not None and rxy_ and a < 18:     # last word: the exact short-range check
                        t_s = time.perf_counter()
                        m_ = last_a % 9
                        sp_ = (2.0 if last_a >= 9 else 4.0) * (0.70710678 if DX[m_] and DY[m_] else 1.0) * args.reflex_delay
                        a0_ = a
                        if args.reflex_hold > 0:                                          # timing-robust: key acts 0..2 frames after the read
                            a, n_ok_ = reflex.choose_robust(rtab_, float(rxy_[0]), float(rxy_[1]),
                                                            (RVX[last_a % 18], RVY[last_a % 18]), a, reflex_delays, args.reflex_hold)
                            S_ = n_ok_ == len(reflex_delays)                              # safe whatever the timing
                            if n_ok_.max() == 0:
                                S_ = np.zeros(18, bool)
                            elif not S_.any():                                            # best is partial: log as 'some safe'
                                S_ = n_ok_ == n_ok_.max()
                        else:
                            a, S_ = reflex.choose(rtab_, float(np.clip(rxy_[0] + DX[m_] * sp_, 8, 376)),
                                                  float(np.clip(rxy_[1] + DY[m_] * sp_, 16, 432)), a)
                        reflex_none = not S_.any()
                        if args.reflex_hold > 0:                                          # (bomb trigger: the best escape is shaky)
                            shaky_n = shaky_n + 1 if n_ok_.max() <= args.reflex_bomb_best else 0
                            if shaky_n >= args.reflex_bomb_consec and args.reflex_bomb_sum < 0:
                                reflex_none = True
                            if args.reflex_bomb_sum >= 0 and n_ok_.max() <= 2 and n_ok_.max() + prev_best <= args.reflex_bomb_sum:
                                reflex_none = True
                            prev_best = int(n_ok_.max())
                            vis_nok = n_ok_.astype(np.int8)
                        if not S_.all():
                            reflex_log.append((now - t_start, a0_, a, int(S_.sum())) + ((int(n_ok_.max()),) if args.reflex_hold > 0 else ()))
                        tm["reflex"] = tm.get("reflex", 0.0) + time.perf_counter() - t_s
            bomb_now = False                            # X is pressed only by the bomb model (--bomber) or on doom (--doom_bomb)
            if bombing and bombs_left > 0 and now >= max(invuln_until, lock_until) and pending_bomb is None \
                    and now - t_start > args.respawn_lock:
                if bomber is not None:
                    with torch.no_grad():
                        hb_ = bomber.hidden(obs if head_w is None else      # calibrated head: fitted with 'bombs 3/3, not invulnerable'
                                            dict(obs, vec=torch.cat([obs["vec"][:, :3], obs["vec"].new_tensor([[1.0, 0.0]])], 1)))
                        p_b = float(torch.sigmoid(hb_[0].float() @ head_w + head_b) if head_w is not None else
                                    torch.softmax(bomber.pi(hb_).float(), 1)[0, 1])
                    thr_b = args.bomb_thr
                elif args.reflex_bomb:                   # the reflex found no safe action in this decision
                    p_b, thr_b = (1.0 if reflex_none else 0.0), 0.5
                else:                                    # this decision's doom (from the traj look-up above); none if it was not made
                    p_b = doom_log[-1][1] if doom_log and abs(doom_log[-1][0] - (now - t_start)) < 1e-6 else 0.0
                    thr_b = args.doom_bomb
                if p_b > thr_b and args.bomb_gate > 0:      # nearest bullet pixel (game px), as in the log below
                    cx_, cy_ = int(P.pos[0]), int(P.pos[1])
                    x0_, y0_ = max(cx_ - 150, 0), max(cy_ - 150, 0)
                    nz = mask[y0_:cy_ + 150, x0_:cx_ + 150].nonzero()
                    ndb = float(((nz[:, 1] + x0_ - cx_).float() ** 2 + (nz[:, 0] + y0_ - cy_).float() ** 2).sqrt().min().item() / 1.5) if len(nz) else 99.0
                    if ndb >= args.bomb_gate:
                        p_b = 0.0
                if p_b > thr_b:
                    bomb_now = True
                    bomb_events.append((now - t_start, p_b, bombs_left))
                    pending_bomb = (now, bombs_left) if star_ok else None
                    bombs_left -= 1; bombs_used += 1; invuln_until = now + BOMB_INVULN_S; lock_until = now + args.bomb_lock
                    cutin_t = now
                    print(f"BOMB {bombs_used} at {now - t_start:.1f}s (P {p_b:.2f}, bombs {bombs_left + 1} -> {bombs_left})", flush=True)
                    if args.out and len(bomb_events) <= 40:
                        Image.fromarray(np.concatenate([pf, overlay(pf, mask, P.pos, None)], axis=1)).save(
                            os.path.join(args.out, f"bomb_{len(bomb_events):02d}_t{now - t_start:06.1f}"
                                         + (f"_pic{pf_age}stepsold" if pf_age else "") + ".jpg"), quality=85)   # (--memperc: a reused picture)
            reflex_none = False                          # used up: a later step without a reflex check must not bomb on it
            tm["policy"] += time.perf_counter() - t_p
            if a >= 18:                                  # (a policy trained with a bomb action: treat it as 'do nothing')
                a = 0
            elif ymin_play > 16 and py <= ymin_play + 3:  # optional hard play-zone limit (off by default)
                m_ = a % 9
                a = {1: 0, 5: 3, 6: 4}.get(m_, m_) + 9 * (a // 9) if DY[m_] < 0 else a
            last_a = a if a < 18 else 0
            want = keys_for(a)
            if args.shoot:
                want.add("z")
            if bomb_now:                                 # held for this one decision (~2 game frames), released next step
                want.add("x")
            cur_focus = 1 if (a < 18 and a // 9 == 1) else 0
            if not args.dry:
                for k in held - want: K.up(k)
                for k in want - held: K.down(k)
                held = set(want)
            if watcher is not None and args.policy not in ("stay", "random"):     # after the keys: no response time
                t_s = time.perf_counter()
                with torch.no_grad():
                    wv_ = torch.cat([obs["vec"][:, :3], obs["vec"].new_tensor([[1.0, 0.0]])], 1)      # 3 bombs in hand, not invulnerable
                    p_w = float(torch.softmax(watcher.pi(watcher.hidden(dict(obs, vec=wv_))).float(), 1)[0, 1])
                watch_log.append((now - t_start, p_w))
                tm["bomb_watch"] = tm.get("bomb_watch", 0.0) + time.perf_counter() - t_s
            t_a = time.perf_counter()
            if args.record_every and args.out and len(log) % args.record_every == 0:
                os.makedirs(os.path.join(args.out, "rec"), exist_ok=True)
                Image.fromarray(pf).save(os.path.join(args.out, "rec", f"f{len(log):06d}.png"))
                rec_meta.append((len(log), now - t_start, px, py, cur_focus))
            # distance (game px) from the player to the nearest bullet pixel the policy can see
            cx_, cy_ = int(P.pos[0]), int(P.pos[1])
            x0_, y0_ = max(cx_ - 150, 0), max(cy_ - 150, 0)
            nz = mask[y0_:cy_ + 150, x0_:cx_ + 150].nonzero()
            nd = float(((nz[:, 1] + x0_ - cx_).float() ** 2 + (nz[:, 0] + y0_ - cy_).float() ** 2).sqrt().min().item() / 1.5) if len(nz) else 99.0
            lx_, ly_ = mxy if view_xy is not None else (px, py)   # (--predict: the log keeps where the player really is)
            log.append((now - t_start, lx_, ly_, 18 if bomb_now else a, int(mask.sum().item()), nd, (boss_x - px) if boss_x is not None else float('nan')))
            if args.out:                      # half-size raw + mask for the post-mortem montages (overlay is built lazily)
                ring.append((np.ascontiguousarray(pf[::2, ::2]), mask[::2, ::2].cpu().numpy(), nd, a, P.pos.copy() / 2,
                             P.last_removed[::2, ::2].cpu().numpy() if args.shot_filter and P.last_removed is not None else None, pf_age))
            if args.shot_filter and P.last_removed is not None:
                removed_px += int(P.last_removed.sum().item())
            if args.out and args.dump_from >= 0 and args.dump_from <= now - t_start < args.dump_from + args.dump_secs:
                ds_ = 1 if args.dump_full else 2
                if dump is None:                         # memory-mapped arrays on disk: no per-frame encoding cost
                    nmax = int(args.dump_secs * 32)
                    dump = dict(i=0, n=nmax, img=None if args.dump_no_img else
                                np.lib.format.open_memmap(os.path.join(args.out, 'dump_img.npy'), 'w+', np.uint8, (nmax, PF_H // ds_, PF_W // ds_, 3)),
                                msk=np.lib.format.open_memmap(os.path.join(args.out, 'dump_mask.npy'), 'w+', np.bool_, (nmax, PF_H // ds_, PF_W // ds_)),
                                rem=np.lib.format.open_memmap(os.path.join(args.out, 'dump_removed.npy'), 'w+', np.bool_, (nmax, PF_H // ds_, PF_W // ds_))
                                if args.shot_filter else None,
                                meta=[])
                if dump['i'] < dump['n']:
                    k_ = dump['i']; dump['msk'][k_] = mask[::ds_, ::ds_].cpu().numpy()
                    if dump['img'] is not None:
                        dump['img'][k_] = pf[::ds_, ::ds_]
                    if P.last_removed is not None and dump['rem'] is not None:
                        dump['rem'][k_] = P.last_removed[::ds_, ::ds_].cpu().numpy()
                    tab_d = mem_tab if mmask is not None else (watch_tab if args.watch_mem else None)
                    watch_tab = None
                    if args.dump_tables and tab_d is not None:
                        b_, l_, e_ = tab_d["bullets"], tab_d["lasers"], tab_d["enemies"]
                        dump.setdefault("tab", []).append((
                            np.stack([b_["x"], b_["y"], b_["vx"], b_["vy"]], 1).astype(np.float32), b_["key"].astype(np.int16),
                            np.stack([l_["x"], l_["y"], l_["ang"], l_["len"], l_["width"], l_["state"].astype(np.float32)], 1).astype(np.float32),
                            np.stack([e_["x"], e_["y"], e_["w"], e_["h"]], 1).astype(np.float32),
                            np.stack([b_["slot"], b_["age"], b_["hbf"]], 1).astype(np.float32),     # (2026-11-05: slot = identity, age, hitbox field)
                            l_.get("raw", np.zeros((len(l_["x"]), 166), np.float32)).astype(np.float32),    # whole laser structs (same rows as lasers)
                            np.stack([e_.get("vx", np.zeros(len(e_["x"]))), e_.get("vy", np.zeros(len(e_["x"]))),     # (2026-11-08: enemies'
                                      e_.get("slot", np.full(len(e_["x"]), -1)),                                   # velocity, slot;
                                      e_.get("dt", np.zeros(len(e_["x"])))], 1).astype(np.float32)))              # 11-09: own frames)
                    dump['meta'].append((now - t_start, px, py, a)); dump['i'] += 1
                    dump.setdefault("nok", []).append(vis_nok if vis_nok is not None else np.full(18, -1, np.int8))
                    dump.setdefault("vmeta", []).append(vis_meta if vis_meta is not None else (np.nan,) * 4)
            tm["post"] += time.perf_counter() - t_a
            if args.out and now - last_hud_snap > 30:                 # right-hand HUD panel (Power, score, lives) every 30 s
                try:
                    hud_img = np.asarray(sct.grab(dict(left=l + 1060, top=t + 80, width=200, height=300)))[..., :3][..., ::-1]
                    Image.fromarray(np.ascontiguousarray(hud_img)).save(os.path.join(args.out, f"hudpanel_{now - t_start:05.0f}.jpg"), quality=85)
                except Exception:
                    pass
                last_hud_snap = now
            if args.out and now - last_snap > args.snap_every and pf_age == 0:      # (picture and mask of the same step)
                Image.fromarray(np.concatenate([pf, overlay(pf, mask, P.pos, P.last_removed if args.shot_filter else None)], axis=1)).save(
                    os.path.join(args.out, f"snap_{now - t_start:05.1f}.jpg"), quality=85)
                last_snap = now
            if traj_net is not None and obs_xy:                   # in the time left over: the maps for the next decision (not waited for: the look-up does)
                t_s = time.perf_counter()
                with torch.no_grad():
                    xy_ = torch.tensor([obs_xy], device=dev, dtype=torch.float32)
                    x_ = torch.stack([traj_hist[-1 - L] for L in traj_net.lags] + [player_marker(xy_[:, 0], xy_[:, 1])[0]], 0)[None].float()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p_ = torch.sigmoid(traj_net(x_, traj_steps, traj_pause).float())
                    if args.traj_dilate:
                        p_ = torch.nn.functional.max_pool2d(p_, 2 * args.traj_dilate + 1, 1, args.traj_dilate)
                    traj_V = traj_search.values(p_, traj_first, traj_pause)
                tm["traj_maps"] = tm.get("traj_maps", 0.0) + time.perf_counter() - t_s
            next_t += period
            slack = next_t - time.perf_counter()
            if slack < -0.2:                             # fell behind: do not try to catch up
                next_t = time.perf_counter()
            time.sleep(max(0, slack))
finally:
    if args.out:                                   # HUD at the end (MISS counter), before anything else happens
        try:
            save_hud("hud_end.png")
        except Exception as e:
            print("hud grab failed:", e)
    for k in list(held) + ["left", "right", "up", "down", "shift", "z", "ctrl"] + (["x"] if bombing else []):
        K.up(k)
    if dlg_active:
        dlg_segs.append(time.perf_counter() - dlg_t0)
    el = time.perf_counter() - t_start
    arr = np.array(log) if log else np.zeros((0, 7))
    if game_over_t is not None:
        print(f"STANDARD MODE: game over at {game_over_t:.1f}s after {len(miss_events)} life losses")
    if args.standard:
        print(f"STANDARD MODE: {n_continues} continues used, {n_deaths} deaths by respawn jump, {len(miss_events)} heart changes")
    if args.items:
        print(f"item-seeking bonus applied in {n_item_steps} decisions")
    if args.align:
        print(f"boss-alignment bonus applied in {n_align_steps} decisions")
    if args.enemy_align:
        print(f"enemy-alignment bonus applied in {n_enemy_steps} decisions (toward an enemy), {n_center_steps} (back toward the middle)")
    if args.seek_bombs:
        print(f"bomb-item seeking: bonus in {n_seek_steps} decisions; wanted items seen (time s, type, x, y): "
              f"{[(round(a_, 1), b_, round(c_), round(d_)) for a_, b_, c_, d_ in seek_log]}")
        if args.out:
            np.save(os.path.join(args.out, "seek_items.npy"), np.array(seek_log, np.float64).reshape(-1, 4))
    if safety_net is not None:
        print(f"safety network: the policy's choice changed in {n_safety} of {n_safety_all} decisions")
    if traj_net is not None:
        print(f"traj network: the policy's choice changed in {n_traj} of {n_traj_all} decisions")
    if reflex is not None:
        print(f"reflex: looked at {reflex.n_seen} decisions ({reflex.n_quiet} quiet), changed {reflex.n_changed}, no safe action {reflex.n_none}")
        if pic_age:
            print(f"vision reflex: picture from the worker at the step start, age (frames) p50 {np.median(pic_age):.2f} p90 {np.percentile(pic_age, 90):.2f}")
        if vis_age:
            print(f"vision reflex: worker table age at the reflex (frames since its grab) p10 {np.percentile(vis_age, 10):.2f} "
                  f"p50 {np.median(vis_age):.2f} p90 {np.percentile(vis_age, 90):.2f}, {len(vis_age)} tables used")
    if guard is not None:
        print(f"shield: something at stake in {guard.seen} of {guard.calls} decisions, the policy's choice changed in {guard.vetoes}")
    if bombing:
        print(f"BOMBS: {bombs_used} used, star reader {'ok' if star_ok else 'off'}; (time s, P, bombs before):",
              [(round(a_, 1), round(b_, 2), c_) for a_, b_, c_ in bomb_events])
        if args.out:
            np.save(os.path.join(args.out, "bomb_events.npy"), np.array(bomb_events))
    if attempts:
        d_ = np.array([a_[1] for a_ in attempts])
        print(f"SPELL BENCH: {len(attempts)} attempts, survival mean {d_.mean():.1f}s median {np.median(d_):.1f}s max {d_.max():.1f}s, not failed {sum(a_[2] != 'failed' for a_ in attempts)}")
        if args.out:
            np.save(os.path.join(args.out, "attempts.npy"), np.array([(a_[0], a_[1], a_[2] == "failed") for a_ in attempts]))
    if args.shot_filter:
        print(f"own-shot filter: {removed_px / max(len(log), 1):.0f} mask px removed per decision")
    if dump is not None:
        np.save(os.path.join(args.out, "dump_meta.npy"), np.array(dump["meta"])); dump["msk"].flush(); print(f"dumped {dump['i']} frames")
        if dump.get("nok"):
            np.save(os.path.join(args.out, "dump_nok.npy"), np.stack(dump["nok"]))      # reflex verdict per dumped step (-1: none)
            np.save(os.path.join(args.out, "dump_vismeta.npy"), np.array(dump["vmeta"], np.float64))
        if dump["img"] is None and dump["rem"] is None and dump["i"] < dump["n"]:
            try:                           # the file was made for dump_secs: cut it to the frames used (a dense card's run used 1/5 of it)
                m_ = np.array(dump["msk"][:dump["i"]])
                mm_ = dump.pop("msk"); mm_._mmap.close(); del mm_
                np.save(os.path.join(args.out, "dump_mask.npy"), m_)
                dump["msk"] = m_
            except Exception as ex_:
                print("dump_mask.npy not cut:", repr(ex_)[:80])
        if dump.get("tab"):                # step k's rows: bullets[b_off[k]:b_off[k + 1]] etc. (k = the dump index, as in dump_mask.npy)
            tb_ = dump["tab"]
            off_ = lambda j: np.concatenate([[0], np.cumsum([len(r[j]) for r in tb_])]).astype(np.int64)
            np.savez_compressed(os.path.join(args.out, "dump_tables.npz"), bullets=np.concatenate([r[0] for r in tb_]), key=np.concatenate([r[1] for r in tb_]),
                                lasers=np.concatenate([r[2] for r in tb_]), enemies=np.concatenate([r[3] for r in tb_]),
                                bullet_slot_age_hb=np.concatenate([r[4] for r in tb_]), laser_raw=np.concatenate([r[5] for r in tb_]),
                                enemy_v_slot=np.concatenate([r[6] for r in tb_]),
                                b_off=off_(0), l_off=off_(2), e_off=off_(3), hitbox=mreader.hitbox.copy(), sprite_px=mreader.sprite.copy())
            print(f"dumped memory tables of {len(tb_)} steps")
        if dump["img"] is not None:
            dump["img"].flush()
    print(f"ran {el:.1f}s, {len(log)} decisions ({len(log)/max(el,1e-9):.1f}/s), dialogue frames {n_dialogue}, not-foreground frames {n_nofg}, "
          f"deaths: {len(miss_events)} by HUD MISS counter, {n_deaths} by respawn jump")
    print("dialogue segments (s): " + str([round(s_, 1) for s_ in dlg_segs]) + f"  total {sum(dlg_segs):.1f}s"
          + (f"; portrait-rule frames vetoed by the bullets in memory: {n_dlg_veto}" if mmask is not None else ""))
    if log:
        print("ms per step: " + ", ".join(f"{k} {1000 * v / max(len(log), 1):.1f}" for k, v in tm.items()))
        if args.out:                                 # (also on disk: a runner's log once lost the printed summary)
            t_ = np.array([e[0] for e in log])
            open(os.path.join(args.out, "timing.txt"), "w").write(
                "ms per step: " + ", ".join(f"{k} {1000 * v / max(len(log), 1):.1f}" for k, v in tm.items()) +
                f"\ndecisions {len(log)} in {t_[-1]:.1f} s = {len(log) / max(t_[-1], 1e-9):.1f} / s, step p50 {1000 * np.median(np.diff(t_)):.1f} ms, "
                f"p90 {1000 * np.percentile(np.diff(t_), 90):.1f} ms\n")
    if miss_events:
        print("MISS events (time s, nearest visible bullet 0.7-0.2 s before, game px):",
              [(round(a_, 1), round(b_)) for a_, b_ in miss_events])
        if args.out:
            np.save(os.path.join(args.out, "miss_events.npy"), np.array(miss_events))
    if len(arr):
        acts = np.bincount(arr[:, 3].astype(int), minlength=19)
        print("player x (game px) %.0f..%.0f   y %.0f..%.0f   mean bullet px %.0f   bombs used %d" % (
            arr[:, 1].min(), arr[:, 1].max(), arr[:, 2].min(), arr[:, 2].max(), arr[:, 4].mean(), bombs_used))
        print("action histogram (move 0-8, +9 = focus, 18 = bomb):", acts.tolist())
    if args.out:
        np.save(os.path.join(args.out, "log.npy"), arr)
        if watch_log:
            np.save(os.path.join(args.out, "bomb_watch.npy"), np.array(watch_log))
        if doom_log:
            np.save(os.path.join(args.out, "doom.npy"), np.array(doom_log))
        if reflex_log:
            np.save(os.path.join(args.out, "reflex.npy"), np.array(reflex_log))
        if rec_meta:
            np.save(os.path.join(args.out, "rec_meta.npy"), np.array(rec_meta))
