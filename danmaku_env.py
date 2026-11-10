"""GPU-batched procedural danmaku (bullet-hell) dodging environment.

Design goals (details in NOTES.md):
  * thousands of parallel envs on one GPU; ~60 fps game physics, one decision every `frame_skip` frames
  * bullet patterns are generated procedurally from parametric emitters instead of being scripted per
    stage, so a policy cannot memorise a stage; whole pattern families can be held out to
    measure generalisation
  * the observation is what a screen-reading agent could get from the real game: stacked binary bullet
    maps (no velocities, no emitter state) plus the player's own position

Units follow the Touhou 6 playfield: 384 x 448 px, player 4 px/frame (2 px/frame focused).
Hitbox / bullet sizes are approximations and are randomised per episode / per emitter.
"""
import math
import torch
import torch.nn.functional as F

W, H = 384.0, 448.0
TWO_PI = 2 * math.pi

N_ACTIONS = 18  # 9 move directions x {normal, focus}
_DX = [0, 0, 0, -1, 1, -1, 1, -1, 1]
_DY = [0, -1, 1, 0, 0, -1, -1, 1, 1]

FAMILY_NAMES = ["aimed_fan", "ring", "spiral", "rain", "accel_aim",      # 0-4  v1 training families
                "curve", "retarget", "wall_gap", "mix",                  # 5-8  v1 held-out families
                "g_fan_aim", "g_fan_free", "g_ring", "g_line", "g_scatter", "g_curve", "g_ret",  # 9-15 v2 broad training
                "diag_rain_l", "diag_rain_r", "fast_dense", "fast_ring", "curve_ret",  # 16-20 v2 strict-OOD
                "b_rows", "b_dense", "b_arms",                                        # 21-23 v5 boss-spell-like
                "b_lattice",                                                          # 24    v5c: drifting grid of bullets (Cirno-style)
                "b_freeze", "b_beam", "b_big",                                        # 25-27 v6: stop-then-release, locked fast streams, big bullets
                "b_dash",                                                             # 28    v9: stop, aim at the player, dash (Misdirection knives)
                "b_bigring", "b_bigspiral", "b_lines", "b_sweep", "b_river", "b_stream"]  # 29-34 v11: what v9a dies to in stages 4-6
TRAIN_FAMILIES = [0, 1, 2, 3, 4]
HELDOUT_FAMILIES = [5, 6, 7, 8]
TRAIN_V2 = TRAIN_FAMILIES + [9, 10, 11, 12, 13, 14, 15]
BOSS_V5 = [21, 22, 23]
TRAIN_V5 = TRAIN_V2 + BOSS_V5
LATTICE = [24]
TRAIN_V5C = TRAIN_V5 + LATTICE
BOSS_V6 = [25, 26, 27]
TRAIN_V6 = TRAIN_V5C + BOSS_V6
BOSS_V9 = [28]
TRAIN_V9 = TRAIN_V6 + BOSS_V9
BOSS_V11 = [29, 30, 31, 32, 33, 34]
TRAIN_V11 = TRAIN_V9 + BOSS_V11
# Strict OOD (never in any training set): angle region (diag walls), speed region (fast_dense) and
# *combinations* of dynamics that training only shows separately (curve+wall, curve+retarget).
OOD_V2 = [16, 17, 18, 19, 20]

PARAM_NAMES = ["n", "spread", "aim_p", "ring_p", "base", "rot", "speed", "acc", "smax", "smin", "omega",
               "rad", "interval", "shots", "start", "angjit", "sjit", "posjit", "spacing", "gap_n",
               "gap_shift", "ret_t", "ret_speed", "mvx_p", "ymin", "ymax", "center_x", "ret_rand", "lock"]
SIGNED = {"rot", "omega", "gap_shift"}  # sampled as magnitude, sign is random
_DEFAULT = dict(n=(1, 1), spread=(0, 0), aim_p=(0, 0), ring_p=(0, 0), base=(0, 0), rot=(0, 0), speed=(2, 2),
                acc=(0, 0), smax=(9, 9), smin=(0.4, 0.4), omega=(0, 0), rad=(3, 3), interval=(8, 8),
                shots=(3, 3), start=(0, 0), angjit=(0, 0), sjit=(0, 0), posjit=(0, 0), spacing=(0, 0),
                gap_n=(0, 0), gap_shift=(0, 0), ret_t=(0, 0), ret_speed=(0, 0), mvx_p=(0.3, 0.3),
                ymin=(40, 40), ymax=(160, 160), center_x=(0, 0), ret_rand=(0, 0), lock=(0, 0))
_FAMILIES = [
    dict(n=(1, 7), spread=(0.10, 0.35), aim_p=(1, 1), speed=(2.0, 4.0), rad=(2.5, 5.0), interval=(5, 16),
         shots=(3, 9), start=(0, 20), angjit=(0, 0.03), sjit=(0, 0.2), ymin=(30, 30), ymax=(160, 160)),
    dict(n=(8, 24), ring_p=(1, 1), base=(0, TWO_PI), rot=(0, 0.25), speed=(1.3, 2.8), rad=(2.5, 6.0),
         interval=(10, 28), shots=(3, 9), ymin=(40, 40), ymax=(170, 170)),
    dict(n=(1, 5), ring_p=(1, 1), base=(0, TWO_PI), rot=(0.12, 0.45), speed=(1.6, 3.2), rad=(2.5, 4.0),
         interval=(2, 4), shots=(25, 60)),
    dict(n=(1, 3), spread=(0.0, 0.0), base=(math.pi / 2, math.pi / 2), speed=(1.5, 3.5), rad=(2.5, 4.5),
         angjit=(0.0, 0.12), posjit=(80, 190), interval=(2, 6), shots=(30, 90), ymin=(-8, -8), ymax=(-8, -8)),
    dict(n=(1, 5), spread=(0.12, 0.40), aim_p=(1, 1), speed=(0.6, 1.0), acc=(0.03, 0.07), smax=(3.5, 5.5),
         rad=(3.0, 5.0), interval=(8, 20), shots=(3, 8)),
    # ---- held-out ----
    dict(n=(6, 16), ring_p=(1, 1), base=(0, TWO_PI), omega=(0.008, 0.03), speed=(1.5, 2.8), rad=(3.0, 5.0),
         interval=(10, 25), shots=(3, 8)),
    dict(n=(8, 18), ring_p=(1, 1), base=(0, TWO_PI), speed=(2.2, 3.2), acc=(-0.09, -0.05), smin=(0, 0),
         ret_t=(35, 55), ret_speed=(2.2, 3.6), rad=(3.0, 5.0), interval=(14, 30), shots=(2, 5)),
    # Walls must be impassable except for the gap: bullets are big enough (rad>=5.5 -> 2*(rp+0.8*rad) > spacing)
    # and the row is wider than the playfield. (An earlier version had slits between bullets, and the
    # spawn point x=192 sat exactly on one, so standing still "beat" the wall.)
    dict(n=(36, 36), spacing=(11.0, 11.0), base=(math.pi / 2, math.pi / 2), speed=(1.2, 2.2), rad=(5.5, 6.0),
         gap_n=(4, 7), gap_shift=(0, 5), interval=(45, 80), shots=(2, 5), ymin=(-10, -10), ymax=(-10, -10),
         center_x=(1, 1), mvx_p=(0, 0)),
    dict(n=(1, 20), spread=(0.05, 0.5), aim_p=(0, 1), ring_p=(0.5, 0.5), base=(0, TWO_PI), rot=(0, 0.4),
         speed=(1.2, 4.0), acc=(-0.02, 0.05), omega=(0, 0.02), rad=(2.5, 6.0), interval=(4, 24),
         shots=(3, 20), angjit=(0, 0.2), sjit=(0, 0.3)),
    # ---- v2 broad training families (wide parameter ranges; raw bullet speed <= 4.0) ----
    dict(n=(1, 12), spread=(0.03, 0.6), aim_p=(1, 1), base=(-0.3, 0.3), rot=(0, 0.1), speed=(0.8, 4.0),
         acc=(-0.04, 0.05), rad=(2.5, 6.0), interval=(3, 30), shots=(2, 14), start=(0, 30), angjit=(0, 0.1),
         sjit=(0, 0.3), ymin=(20, 20), ymax=(200, 200)),
    dict(n=(1, 12), spread=(0.03, 0.6), aim_p=(0, 0), base=(0.2, 2.94), rot=(0, 0.15), speed=(0.8, 4.0),
         acc=(-0.04, 0.05), rad=(2.5, 6.0), interval=(3, 30), shots=(2, 14), start=(0, 30), angjit=(0, 0.1),
         sjit=(0, 0.3), ymin=(20, 20), ymax=(200, 200)),
    dict(n=(4, 32), ring_p=(1, 1), aim_p=(0, 0.3), base=(0, TWO_PI), rot=(0, 0.5), speed=(0.8, 3.5),
         acc=(-0.03, 0.05), rad=(2.5, 6.0), interval=(4, 40), shots=(2, 20), sjit=(0, 0.2), ymin=(20, 20),
         ymax=(220, 220)),
    dict(n=(20, 36), spacing=(10.0, 11.3), base=(math.pi / 2 - 0.3, math.pi / 2 + 0.3), speed=(0.8, 3.0),
         rad=(5.5, 6.0), gap_n=(3, 8), gap_shift=(0, 6), interval=(25, 100), shots=(2, 8), ymin=(-10, -10),
         ymax=(-10, -10), center_x=(0.2, 1), mvx_p=(0, 0)),
    dict(n=(1, 6), spread=(0.0, 1.0), aim_p=(0, 0.3), base=(math.pi / 2 - 0.35, math.pi / 2 + 0.35), posjit=(0, 190), angjit=(0.1, 0.6),
         speed=(0.8, 4.0), rad=(2.5, 5.0), interval=(1, 10), shots=(10, 80), ymin=(-8, 120), ymax=(-8, 120)),
    dict(n=(3, 20), spread=(0.1, 0.5), ring_p=(0.5, 0.5), base=(0.2, 2.94), rot=(0, 0.3),
         omega=(0.004, 0.03), speed=(0.8, 3.5), rad=(2.5, 6.0), interval=(4, 30), shots=(2, 14)),
    dict(n=(4, 20), spread=(0.1, 0.5), ring_p=(0.5, 0.5), base=(0.2, 2.94), speed=(1.5, 3.5),
         acc=(-0.1, 0.0), smin=(0, 0.3), ret_t=(20, 70), ret_speed=(1.5, 4.0), rad=(2.5, 6.0), interval=(8, 35),
         shots=(2, 6)),
    # ---- v2 strict-OOD families ----
    dict(n=(1, 3), base=(math.pi / 2 - 0.8, math.pi / 2 - 0.45), speed=(1.5, 3.5), rad=(2.5, 4.5),
         angjit=(0.0, 0.12), posjit=(80, 190), interval=(2, 6), shots=(30, 90), ymin=(-8, -8), ymax=(-8, -8)),
    dict(n=(1, 3), base=(math.pi / 2 + 0.45, math.pi / 2 + 0.8), speed=(1.5, 3.5), rad=(2.5, 4.5),
         angjit=(0.0, 0.12), posjit=(80, 190), interval=(2, 6), shots=(30, 90), ymin=(-8, -8), ymax=(-8, -8)),
    dict(n=(6, 16), spread=(0.05, 0.2), aim_p=(1, 1), speed=(4.5, 6.5), rad=(2.5, 4.0), interval=(6, 14),
         shots=(3, 8), angjit=(0, 0.02), ymin=(30, 30), ymax=(160, 160)),
    dict(n=(12, 28), ring_p=(1, 1), base=(0, TWO_PI), rot=(0, 0.3), speed=(4.0, 5.5), rad=(2.5, 5.0),
         interval=(10, 28), shots=(3, 8), ymin=(40, 40), ymax=(200, 200)),
    dict(n=(4, 20), spread=(0.1, 0.5), ring_p=(0.5, 0.5), base=(0.2, 2.94), omega=(0.004, 0.02),
         speed=(1.5, 3.5), acc=(-0.08, -0.03), smin=(0, 0.3), ret_t=(30, 60), ret_speed=(1.5, 4.0), rad=(2.5, 6.0),
         interval=(8, 35), shots=(2, 6)),
    # ---- v5 boss-spell-like families: what the diagnostics showed the agent dies to (walls with moving gaps in dense
    # succession, thick scatter). Same emitter machinery, just denser / longer than the older families. ----
    dict(n=(30, 36), spacing=(10.5, 11.3), base=(math.pi / 2 - 0.08, math.pi / 2 + 0.08), speed=(1.0, 1.8),
         rad=(5.5, 6.0), gap_n=(7, 10), gap_shift=(0, 2), interval=(24, 40), shots=(5, 12), ymin=(-10, -10),
         ymax=(-10, -10), center_x=(1, 1), mvx_p=(0, 0)),
    dict(n=(6, 10), ring_p=(0.6, 1), base=(0, TWO_PI), rot=(0.05, 0.5), speed=(1.2, 2.4), rad=(2.5, 4.0),
         interval=(5, 9), shots=(12, 28), angjit=(0, 0.15), sjit=(0, 0.25), ymin=(40, 120), ymax=(40, 120)),
    dict(n=(3, 5), ring_p=(1, 1), base=(0, TWO_PI), rot=(0.25, 0.7), speed=(1.4, 2.6), rad=(2.5, 3.5),
         interval=(3, 5), shots=(30, 60), ymin=(50, 130), ymax=(50, 130)),
    # v5c: a slanted lattice (rows of evenly spaced bullets, each row a little later): seen in the stage-2 boss fight
    dict(n=(8, 12), spacing=(24, 34), base=(math.pi / 2 - 0.5, math.pi / 2 + 0.5), speed=(0.9, 1.6), rad=(2.5, 3.5),
         interval=(20, 34), shots=(4, 9), ymin=(-10, -10), ymax=(-10, -10), center_x=(0.4, 1), mvx_p=(0, 0)),
    # ---- v6 (what LUNATIC boss spells have and the simulator lacked) ----
    # freeze-then-release (Perfect Freeze style): bullets brake to a stop, sit still 1.5-3 s, then fly off in random directions or at the player
    dict(n=(8, 24), ring_p=(0.7, 1), base=(0, TWO_PI), rot=(0, 0.3), speed=(1.5, 3.0), acc=(-0.08, -0.04), smin=(0, 0),
         ret_t=(90, 180), ret_speed=(1.0, 2.5), ret_rand=(0.3, 1.0), rad=(2.5, 4.5), interval=(8, 24), shots=(3, 6),
         ymin=(40, 140), ymax=(40, 140)),
    # beam: a dense fast stream aimed once at the player and then locked (laser-like), after a short delay
    # (each bullet starts slow and accelerates, so the stream grows out of the emitter like a laser's warning line)
    dict(n=(1, 1), aim_p=(1, 1), lock=(1, 1), speed=(0.8, 1.5), acc=(0.1, 0.2), smax=(4.0, 6.0),
         rad=(3.0, 5.0), interval=(1, 1), shots=(10, 20), ymin=(30, 140), ymax=(30, 140)),
    # big bullets (radius 8-14 px)
    dict(n=(3, 10), spread=(0.2, 0.6), ring_p=(0.5, 0.5), aim_p=(0.5, 0.5), base=(0, TWO_PI), speed=(0.8, 2.2), rad=(8.0, 14.0),
         interval=(12, 30), shots=(3, 8), ymin=(40, 140), ymax=(40, 140)),
    # ---- v9 ----
    # dash (Sakuya's Misdirection, measured on the real game): knives fan out, brake to a stop (~30 frames), wait, then turn to
    # where the player IS and dash at 4.5-6.5 px/frame raw (x1.2-1.4 at high difficulty = 6-9 px/frame: real ~8-9);
    # small hitbox, drawn as a streak while fast (see _render_local)
    dict(n=(8, 20), spread=(0.15, 0.5), ring_p=(0.5, 1), base=(0, TWO_PI), rot=(0, 0.3), speed=(2.0, 3.5), acc=(-0.12, -0.07),
         smin=(0, 0), smax=(9, 9), ret_t=(40, 90), ret_speed=(4.5, 6.5), ret_rand=(0, 0.15), rad=(2.0, 3.0), interval=(10, 30),
         shots=(2, 5), ymin=(40, 160), ymax=(40, 160)),
    # ---- v11: the 9 late misses of the NORMAL run (stages 4-6) as general pattern types, not the cards themselves ----
    # dense rings of big bullets, layered by speed jitter (Lava Cromlech fireballs, Remilia's big red balls)
    dict(n=(10, 20), ring_p=(1, 1), base=(0, TWO_PI), rot=(0, 0.3), speed=(1.0, 2.2), sjit=(0, 0.25), rad=(5.0, 9.0),
         interval=(14, 28), shots=(3, 7), ymin=(40, 140), ymax=(40, 140)),
    # many-armed spiral of big bullets that fills the screen (Agni Shine)
    dict(n=(3, 5), ring_p=(1, 1), base=(0, TWO_PI), rot=(0.08, 0.3), speed=(1.0, 2.0), sjit=(0, 0.15), rad=(5.0, 8.0),
         interval=(4, 8), shots=(30, 70), ymin=(50, 130), ymax=(50, 130)),
    # long knives: every direction gets a train of bullets fired on consecutive frames = one long object (Sakuya's knife fans)
    dict(n=(5, 12), spread=(0.15, 0.4), ring_p=(0.5, 1), base=(0.3, 2.84), rot=(0, 0.01), speed=(2.5, 4.0),
         rad=(2.0, 3.0), interval=(1, 1), shots=(8, 18), ymin=(30, 140), ymax=(30, 140)),
    # slowly rotating laser-like lines: fast bullets every frame from a slowly turning ring (Patchouli's lasers)
    dict(n=(1, 3), ring_p=(1, 1), base=(0, TWO_PI), rot=(0.002, 0.008), speed=(4.0, 5.5), rad=(2.5, 3.5), interval=(1, 1),
         shots=(40, 100), ymin=(30, 120), ymax=(30, 120)),
    # river: a dense band of small bullets poured in one direction with scatter (the stage-4 rice stream)
    dict(n=(3, 7), spread=(0.03, 0.1), base=(0.5, 2.64), rot=(0, 0.02), speed=(1.4, 2.8), sjit=(0.15, 0.4),
         angjit=(0.03, 0.15), rad=(2.0, 3.0), interval=(3, 5), shots=(30, 70), ymin=(20, 120), ymax=(20, 120)),
    # stream re-aimed at the player every shot, mixed sizes, that follows the player around (Remilia's falling column)
    dict(n=(1, 3), spread=(0.0, 0.25), aim_p=(1, 1), speed=(1.8, 3.2), sjit=(0.1, 0.35), angjit=(0.02, 0.08), rad=(3.0, 8.0),
         interval=(2, 4), shots=(20, 50), ymin=(30, 120), ymax=(30, 120)),
]
assert len(_FAMILIES) == len(FAMILY_NAMES)

E_FLOAT = ["x", "y", "vx", "spread", "base", "rot", "speed", "acc", "smax", "smin", "omega", "rad", "angjit",
           "sjit", "posjit", "spacing", "ret_t", "ret_speed", "ret_rand", "lock"]
E_LONG = ["t", "start", "interval", "shots", "n", "gap_n", "gap_pos0", "gap_shift"]
E_BOOL = ["alive", "ring", "aim"]

LOC, LOC_CELL = 64, 2.0          # local egocentric view: 64x64 cells of 2 px  (128 px window)
GLB_H, GLB_W, GLB_CELL = 56, 48, 8.0   # global view: whole playfield at 8 px / cell
STACK = 4
PATCH = 3                         # local disc patch half-size (cells)
FAST = 4.5                        # px / frame: faster bullets are drawn as streaks (knife-like sprites)


def _disc_grid(fi, fj, rad, alive, Hc, Wc, ph):
    """Binary [R, Hc, Wc] picture of discs. fi / fj / rad: [R, S] centre row, centre column (cells, continuous) and
    radius (cells); alive: [R, S] bool; ph: patch half-size in cells (discs must fit in 2*ph+1 cells)."""
    R = fi.shape[0]
    off = torch.arange(-ph, ph + 1, device=fi.device)
    jj = torch.floor(fj)[..., None, None] + off.view(1, 1, 1, -1)
    ii = torch.floor(fi)[..., None, None] + off.view(1, 1, -1, 1)
    d2 = (jj + 0.5 - fj[..., None, None]) ** 2 + (ii + 0.5 - fi[..., None, None]) ** 2
    m = (d2 <= (rad[..., None, None] + 0.5) ** 2) & alive[..., None, None]
    m &= (ii >= 0) & (ii < Hc) & (jj >= 0) & (jj < Wc)
    idx = torch.where(m, (ii * Wc + jj).long(), torch.full_like(ii, Hc * Wc, dtype=torch.long))
    grid = torch.zeros(R, Hc * Wc + 1, dtype=torch.uint8, device=fi.device)
    grid.scatter_(1, idx.reshape(R, -1), 1)
    return grid[:, :-1].reshape(R, Hc, Wc)


def make_art_lib(real, dev, M=192):
    """Library of harmless 'self-sprite residue' masks in the local view (index 0: normal, 1: focus), player at the
    window centre. Synthetic pieces follow what the screen reader mistook for bullets in the real game (measured
    positions, game px relative to the hitbox centre): crescents of the two yin-yang options (at +-24 px beside the
    player, or (+-9, -33) when focused, or anywhere in between during the 10-frame slide), white shoes below, the
    head highlight above, sparkles of the shot stream above, a few stray specks. Nothing is ever closer than ~10 px
    to the hitbox (the real perception always blanks the body box). `real`: optional [2, Mr, 64, 64] uint8 masks cut
    from recorded frames, mixed in 50:50."""
    g = torch.Generator(device=dev)
    g.manual_seed(4321)
    R = lambda *s: torch.rand(*s, generator=g, device=dev)
    U = lambda lo, hi, *s: lo + (hi - lo) * R(*s)
    out = []
    for foc in (0, 1):
        xs, ys, rs, al = [], [], [], []
        s = torch.where(R(M) < 0.6, torch.full((M,), float(foc), device=dev), R(M))          # slide state of the options
        for sg in (-1.0, 1.0):
            cx, cy = sg * (24.3 + (8.7 - 24.3) * s), -32.7 * s
            a0, span, ro = U(0, TWO_PI, M), U(1.0, 4.0, M), U(7.0, 10.5, M)
            present = R(M) < 0.6
            for k in range(5):
                a = a0 + span * k / 4
                xs.append(cx + ro * torch.cos(a)); ys.append(cy + ro * torch.sin(a)); rs.append(U(2.5, 3.5, M))
                al.append(present & (R(M) < 0.85))
        for sg in (-1.0, 1.0):                                                                # shoes
            xs.append(sg * U(4, 8, M)); ys.append(U(12, 19, M)); rs.append(U(2.5, 4.0, M)); al.append(R(M) < 0.55)
        xs.append(U(-3, 3, M)); ys.append(-U(14, 20, M)); rs.append(U(2.0, 3.5, M)); al.append(R(M) < 0.4)   # head highlight
        for _ in range(4):                                                                    # shot-stream sparkles
            xs.append(U(-12, 12, M)); ys.append(-U(18, 60, M)); rs.append(U(1.5, 2.5, M)); al.append(R(M) < 0.35)
        for _ in range(3):                                                                    # stray specks
            a, rr = U(0, TWO_PI, M), U(12, 36, M)
            xs.append(rr * torch.cos(a)); ys.append(rr * torch.sin(a)); rs.append(U(1.5, 3.0, M)); al.append(R(M) < 0.25)
        x, y, r_, a_ = (torch.stack(v, 1) for v in (xs, ys, rs, al))
        out.append(_disc_grid(LOC / 2 + y / LOC_CELL, LOC / 2 + x / LOC_CELL, r_ / LOC_CELL, a_, LOC, LOC, 3))
    lib = torch.stack(out, 0)                                                                 # [2, M, 64, 64]
    if real is not None:
        real = real.to(dev)
        pick = torch.randint(0, real.shape[1], (M,), generator=g, device=dev)
        lib = torch.cat([lib, real[:, pick]], 1)
    return lib


class DanmakuEnv:
    def __init__(self, n_envs=512, frame_skip=2, max_frames=3600, device="cuda", bullets=1536, emitters=8,
                 k_per_shot=36, k_near=192, alive_reward=0.02, wall_shaping=False, wall_beta=0.6,
                 cross_bonus=0.5, gamma=0.99, wall_slots=8, bomb_mode=False, ymin_play=16.0, n_bombs=3,
                 bomb_cost=0.15, bomb_invuln=180, high_pen=0.0, high_ref=320.0, bomb_hold=0.0, bomb_keep=0.0,
                 art_p=0.0, art_lib=None, lat_p=0.0, fs_jitter=0.0, sw_dir=0.0, sw_focus=0.0, d_max=1.0, stack=STACK,
                 shot_p=0.0, lat2_frac=0.0, prev_act_obs=False, lags=None, blind=None, lagf=None, bomb_waste=0.0,
                 bomb_horizon=30, bomb_value=0.0, big_draw=(24, 64), hit_scale=0.8, graze_reward=0.0, graze_dist=12.0,
                 timing_profile=None, timing_trace=None):
        self.stack = stack                                 # frames stacked in the observation (4 = old; v5b uses 8)
        # lags: spaced history instead of the last `stack` frames, e.g. (30, 16, 8, 4, 2, 1, 0) decisions back = 1 s of memory with
        # 7 channels (oldest first, newest last, like the old stack)
        self.lags = sorted({int(x) for x in lags}, reverse=True) if lags else None
        # blind: (x_half, up, down) game px around the hitbox that the real screen reader blanks (it cannot tell bullets from
        # Reimu's own sprite there: |dx| < 29, -16 < dy < 34 screen px = 19.3 / 10.7 / 22.7 game px). Bullets there are not drawn.
        self.blind = tuple(float(v) for v in blind) if blind else None
        if self.lags:
            self.stack = len(self.lags)
        # shot_p: fraction of episodes in which the player's own amulet shots are drawn into the observation. The real
        # agent holds the fire key all the time, and the screen reader marks those translucent cards as bullets: a pair
        # every 4 frames, 18 px wide, flying straight up at ~15 px/frame, slightly drifting sideways (homing). Never collide.
        self.shot_p = float(shot_p)
        self.lat2_frac = float(lat2_frac)                  # among late episodes: share that is two decisions late instead of one
        self.N, self.fs, self.max_frames, self.dev = n_envs, frame_skip, max_frames, torch.device(device)
        self.B, self.E, self.K, self.k_near = bullets, emitters, k_per_shot, k_near
        self.alive_reward = alive_reward
        # v5 realism knobs (all off by default = the old simulator):
        #   art_p     fraction of episodes with 'artifacts': harmless things the screen reader also sees in the real game
        #             (Reimu's white shoes / yin-yang options / head highlight / shot sparkles stuck to the player, falling
        #             items, passive enemy blobs). They are drawn into the observation only - they never collide.
        #   art_lib   optional [2, M, 64, 64] uint8 library of real sprite residue masks (index 0: normal, 1: focus)
        #   lat_p     fraction of episodes where the action takes effect one decision late (capture + inference + key lag)
        #   fs_jitter probability per step() call that the decision interval is frame_skip+1 frames instead (loop jitter)
        #   sw_dir / sw_focus  small reward penalties for changing the move direction / toggling Shift (anti-flicker)
        #   d_max     upper limit of the difficulty scale (1.0 = old)
        self.art_p, self.lat_p, self.fs_jitter = float(art_p), float(lat_p), float(fs_jitter)
        #   lagf      probabilities of a lag of 0, 1, 2, ... game frames, drawn anew every step for every env (the real loop runs
        #             at 30 Hz unsynchronised with the 60 fps game, so the capture -> key -> movement delay jitters)
        self.lagf = (torch.tensor([float(x) for x in lagf], device=device) if lagf else None)
        if self.lagf is not None and (self.lagf.numel()>7 or not bool(torch.isfinite(self.lagf).all())
                                      or bool((self.lagf<0).any()) or self.lagf.sum().item()<=0):
            raise ValueError('lagf must give nonnegative finite probabilities for 0..6 game frames')
        self.timing_rows=None
        self.timing_clock=None
        if timing_trace is not None:
            if timing_profile is None or fs_jitter or lat_p or lagf:
                raise ValueError('Timing replay requires a joint profile and no legacy latency knobs')
            from jitter_clock import ReplayClock
            self.timing_clock=ReplayClock(timing_trace)
        if timing_profile is not None:
            if self.lagf is not None or lat_p>0:
                raise ValueError('Do not combine timing_profile with lagf/lat_p')
            from timing_profile import load_profile
            self.timing_rows=load_profile(timing_profile)
        self.sw_dir, self.sw_focus, self.d_max = float(sw_dir), float(sw_focus), float(d_max)
        # Bomb: action 18. Clears every bullet and makes the player invulnerable for `bomb_invuln` frames; limited to
        # n_bombs per episode and costs `bomb_cost` reward, so it is only worth using when death is imminent.
        # ymin_play: the player may not go above this line (humans play in the lower part of the field).
        self.bomb_mode, self.ymin_play, self.n_bombs = bomb_mode, float(ymin_play), float(n_bombs)
        self.bomb_cost, self.bomb_invuln = bomb_cost, float(bomb_invuln)
        # bomb_waste: extra cost when nothing was coming at the player (see _threat) -- "could have dodged, bombed anyway"
        self.bomb_waste, self.bomb_horizon = float(bomb_waste), int(bomb_horizon)
        # hit_scale: a bullet hits when its centre is closer than rp + hit_scale * its drawn radius. The real game's hitboxes are
        # much smaller than the sprites (the real agent survives with sprites 5-7 px from its centre all the time).
        # graze_reward (user's idea for the bomb model): every step not hit and not invulnerable, + graze_reward * closeness^2 of
        # the nearest bullet (0 at graze_dist px outside its hit distance, 1 touching it): living through close calls instead of
        # bombing pays, the closer the more; a hit still costs -1 (and the bombs in stock with bomb_value).
        self.hit_scale, self.graze_reward, self.graze_dist = float(hit_scale), float(graze_reward), float(graze_dist)
        # Soft preference for the lower part of the field (it is where enemies are easiest to shoot down, which
        # lowers the total bullet count in the real game; the simulator has no enemies to kill, so this is a proxy):
        # a per-step penalty that grows linearly from 0 at y = high_ref up to `high_pen` at the top edge.
        self.high_pen, self.high_ref = float(high_pen), float(high_ref)
        # Keeping bombs pays: `bomb_hold` per bomb per step while alive (a terminal-only bonus would be invisible
        # with gamma = 0.99), plus `bomb_keep` per bomb still in stock when the time limit is reached alive.
        self.bomb_hold, self.bomb_keep = float(bomb_hold), float(bomb_keep)
        # bomb_value k: the user's "bombs still in hand when you survive score points", settled at once: spending a bomb costs k
        # now and dying with b in hand costs k*b. Every episode sums to the same as a k-per-bomb bonus at the time limit (minus a
        # constant 3k), but the policy feels it when it decides, not up to 60 s later where gamma hides it.
        self.bomb_value = float(bomb_value)
        # big_draw: how many big bullets are drawn at full size (nearest ones in the local view, biggest in the global view);
        # the rest only as small dots. The old 24 / 64 is too few for screens full of big bullets (v11 families).
        self.big_draw = tuple(int(x) for x in big_draw)
        self.n_actions = N_ACTIONS + 1 if bomb_mode else N_ACTIONS
        # prev_act_obs: the policy also sees its own previous action (one-hot, 18): with a lag of one decision that action is
        # what the player is doing right now, so the state is complete (a delayed MDP is only Markov with the action buffer)
        self.prev_act_obs = bool(prev_act_obs)
        self.vec_dim = (5 if bomb_mode else 3) + (N_ACTIONS if prev_act_obs else 0)
        # Training-only reward shaping for walls-with-a-gap (uses simulator knowledge of the gap; the
        # policy itself only ever sees the bullet maps):
        #   * potential-based:  F = gamma*Phi(s') - Phi(s),  Phi = -beta * g(dy) * |x - gap_x| / 192,
        #     for the nearest wall still above the player (g ramps 0 -> 1 as the wall comes within 300 px)
        #   * crossing bonus when a wall reaches the player's row: cross_bonus * (1 - |x-gap_x| / (half_gap+30))
        self.wall_shaping, self.wall_beta, self.cross_bonus, self.gamma = wall_shaping, wall_beta, cross_bonus, gamma
        self.Wn = wall_slots
        d, N, B1, E = self.dev, n_envs, bullets + 1, emitters  # B1: one trash slot at index `bullets`
        f = lambda *s: torch.zeros(*s, device=d)
        # player
        self.px, self.py, self.focus = f(N), f(N), f(N)
        self.pspeed, self.rp = f(N) + 1, f(N) + 2.5
        self.bombs, self.invuln = f(N) + self.n_bombs, f(N)
        self.frame = torch.zeros(N, dtype=torch.long, device=d)
        self.diff = f(N)
        self.spawn_timer = f(N)
        self.d_lo, self.d_hi = f(N), f(N) + 1.0      # per-env difficulty range for new episodes
        # bullets
        for k in ["bx", "by", "bth", "bs", "bacc", "bsmax", "bsmin", "bom", "brad", "bret", "bretsp", "bretr"]:
            setattr(self, k, f(N, B1))
        self.balive = torch.zeros(N, B1, dtype=torch.bool, device=d)
        self.ptr = torch.zeros(N, dtype=torch.long, device=d)
        # emitters
        self.e = {k: f(N, E) for k in E_FLOAT}
        self.e.update({k: torch.zeros(N, E, dtype=torch.long, device=d) for k in E_LONG})
        self.e.update({k: torch.zeros(N, E, dtype=torch.bool, device=d) for k in E_BOOL})
        self.e["interval"] += 1
        # wall-gap tracking (slot Wn is a trash slot for masked scatters)
        self.w = {k: f(N, self.Wn + 1) for k in ["x", "y", "vx", "vy", "hw", "dyprev"]}
        self.w_alive = torch.zeros(N, self.Wn + 1, dtype=torch.bool, device=d)
        self.w_ptr = torch.zeros(N, dtype=torch.long, device=d)
        self.phi_prev = f(N)
        # constants
        self.DX = torch.tensor(_DX, dtype=torch.float32, device=d)
        self.DY = torch.tensor(_DY, dtype=torch.float32, device=d)
        self.idxK = torch.arange(k_per_shot, device=d)
        lo = torch.tensor([[_FAMILIES[i].get(p, _DEFAULT[p])[0] for i in range(len(_FAMILIES))]
                           for p in PARAM_NAMES], dtype=torch.float32, device=d)
        hi = torch.tensor([[_FAMILIES[i].get(p, _DEFAULT[p])[1] for i in range(len(_FAMILIES))]
                           for p in PARAM_NAMES], dtype=torch.float32, device=d)
        self.LO, self.HI = lo, hi
        self.P = {p: i for i, p in enumerate(PARAM_NAMES)}
        self.fam_probs = torch.zeros(N, len(_FAMILIES), device=d)
        self.set_families(TRAIN_FAMILIES)
        # observation stacks
        self.loc_stack = torch.zeros(N, self.stack, LOC, LOC, dtype=torch.uint8, device=d)
        self.glb_stack = torch.zeros(N, self.stack, GLB_H, GLB_W, dtype=torch.uint8, device=d)
        if self.blind:
            ax_ = (torch.arange(LOC, device=d).float() + 0.5 - LOC / 2) * LOC_CELL
            self.blind_loc = ((ax_[None, :].abs() < self.blind[0]) & (ax_[:, None] > -self.blind[1])
                              & (ax_[:, None] < self.blind[2])).to(torch.uint8)            # [row (dy), col (dx)]
        if self.lags:
            self.hist = self.lags[0] + 1
            self.loc_hist = torch.zeros(N, self.hist, LOC, LOC, dtype=torch.uint8, device=d)
            self.glb_hist = torch.zeros(N, self.hist, GLB_H, GLB_W, dtype=torch.uint8, device=d)
            self.h_ptr = 0
        # diagnostics (kept on GPU to avoid syncs)
        self.stat_len = torch.zeros((), device=d)
        self.stat_eps = torch.zeros((), device=d)
        self.stat_hits = torch.zeros((), device=d)
        self.stat_overwrite = torch.zeros((), device=d)
        self.stat_bombs = torch.zeros((), device=d)
        self.stat_waste = torch.zeros((), device=d)
        # v5: artifacts / latency / switch state
        self.prev_act = torch.zeros(N, dtype=torch.long, device=d)
        self.lat = torch.zeros(N, dtype=torch.long, device=d)         # 0: on time, 1 / 2: that many decisions late
        self.prev_act2 = torch.zeros(N, dtype=torch.long, device=d)
        # A third previous decision is needed for the measured 5-frame tail.
        # The old 0..4-frame path is left unchanged.
        self.prev_act3 = (torch.zeros(N,dtype=torch.long,device=d)
                          if self.lagf is not None and self.lagf.numel()>5 else None)
        self.art_ep = torch.zeros(N, dtype=torch.bool, device=d)       # this episode has artifacts at all
        self.art_on = torch.zeros(N, dtype=torch.bool, device=d)       # sprite residue visible right now
        self.art_timer = f(N)
        self.art_idx = torch.zeros(N, dtype=torch.long, device=d)
        self.NA = 4
        self.a_x, self.a_y, self.a_vx, self.a_vy, self.a_rad = (f(N, self.NA) for _ in range(5))
        self.a_alive = torch.zeros(N, self.NA, dtype=torch.bool, device=d)
        self.NS = 20
        self.s_x, self.s_y, self.s_vx = f(N, self.NS), f(N, self.NS), f(N, self.NS)
        self.s_alive = torch.zeros(N, self.NS, dtype=torch.bool, device=d)
        self.s_ptr = torch.zeros(N, dtype=torch.long, device=d)
        self.s_phase, self.s_drift = f(N), f(N)
        self.shot_ep = torch.zeros(N, dtype=torch.bool, device=d)
        self.sensor_history=None
        if self.timing_rows is not None:
            self.timing_sensor=torch.tensor([r[0] for r in self.timing_rows],device=d,dtype=torch.long)
            self.timing_action=torch.tensor([r[1] for r in self.timing_rows],device=d,dtype=torch.long)
            self.timing_prob=torch.tensor([r[2] for r in self.timing_rows],device=d)
            self.sensor_depth=max(r[0] for r in self.timing_rows)+1
            self.sensor_pointer=0
            self.sensor_history={k:getattr(self,k)[:,None].repeat(1,self.sensor_depth,1)
                                 for k in ('bx','by','brad','bs','bth','balive')}
            depth=64 if self.timing_clock is not None else max(r[1] for r in self.timing_rows)//max(1,self.fs)+3
            self.timing_queue_action=torch.zeros(N,depth,dtype=torch.long,device=d)
            self.timing_queue_due=torch.full((N,depth),-1,dtype=torch.long,device=d)
            self.timing_applied=torch.zeros(N,dtype=torch.long,device=d)
            self.timing_last_due=torch.full((N,),-1,dtype=torch.long,device=d)
            self.timing_pending_delay=torch.zeros(N,dtype=torch.long,device=d)
            self.timing_last_image_frame=torch.zeros(N,dtype=torch.long,device=d)
            if self.timing_clock is not None:
                weights=[r['delay_weights'] for r in self.timing_clock.rows]
                if any(len(w)!=len(self.timing_rows) for w in weights):
                    raise ValueError('Timing trace and joint profile tables differ')
                self.timing_trace_prob=torch.tensor(weights,device=d,dtype=torch.float32)
        self.art_lib = None
        if self.art_p > 0:
            self.art_lib = make_art_lib(art_lib, d)                     # [2, M, 64, 64] uint8
            self.art_libg = F.max_pool2d(self.art_lib.flatten(0, 1)[:, None].float(), 4)[:, 0].to(torch.uint8) \
                .view(2, -1, LOC // 4, LOC // 4)

    # ------------------------------------------------------------------ configuration
    def set_families(self, fams):
        """All envs sample emitters uniformly from `fams`."""
        p = torch.zeros(len(_FAMILIES), device=self.dev)
        p[list(fams)] = 1.0
        self.fam_probs = (p / p.sum())[None].repeat(self.N, 1)

    def set_difficulty(self, lo, hi):
        """New episodes draw difficulty ~ U(lo, hi); running episodes are unchanged."""
        self.d_lo = torch.full((self.N,), float(lo), device=self.dev)
        self.d_hi = torch.full((self.N,), float(hi), device=self.dev)

    def set_conditions(self, fam_lists, d_values):
        """Per-env fixed conditions (evaluation): fam_lists[i] = families env i draws from, d_values[i] = difficulty."""
        probs = torch.zeros(self.N, len(_FAMILIES))
        for i, fams in enumerate(fam_lists):
            probs[i, list(fams)] = 1.0 / len(fams)
        self.fam_probs = probs.to(self.dev)
        d = torch.tensor(d_values, dtype=torch.float32, device=self.dev)
        self.d_lo, self.d_hi = d.clone(), d.clone()

    def pop_stats(self):
        s = dict(ep_frames=(self.stat_len / self.stat_eps.clamp(min=1)).item(), episodes=self.stat_eps.item(),
                 hits=self.stat_hits.item(), overwrites=self.stat_overwrite.item(), bombs=self.stat_bombs.item(),
                 wasted=self.stat_waste.item())
        for t in (self.stat_len, self.stat_eps, self.stat_hits, self.stat_overwrite, self.stat_bombs, self.stat_waste):
            t.zero_()
        return s

    # ------------------------------------------------------------------ reset
    @torch.no_grad()
    def reset(self):
        if self.timing_clock is not None:
            self.timing_clock.reset(int(torch.randint(0,2**31-1,(1,),device='cpu').item()))
        m = torch.ones(self.N, dtype=torch.bool, device=self.dev)
        self._reset(m)
        return self._observe(m)

    def _reset(self, m):
        N, d = self.N, self.dev
        mf = m.float()
        self.px = torch.where(m, torch.full_like(self.px, W / 2), self.px)
        self.py = torch.where(m, torch.full_like(self.py, H - 64), self.py)      # real spawn point: (192, 384)
        self.focus = self.focus * (1 - mf)
        self.bombs = torch.where(m, torch.full_like(self.bombs, self.n_bombs), self.bombs)
        self.invuln = self.invuln * (1 - mf)
        self.balive &= ~m[:, None]
        self.ptr = torch.where(m, torch.zeros_like(self.ptr), self.ptr)
        self.e["alive"] &= ~m[:, None]
        self.w_alive &= ~m[:, None]
        self.w_ptr = torch.where(m, torch.zeros_like(self.w_ptr), self.w_ptr)
        self.phi_prev = self.phi_prev * (1 - mf)
        self.frame = torch.where(m, torch.zeros_like(self.frame), self.frame)
        self.diff = torch.where(m, self.d_lo + torch.rand(N, device=d) * (self.d_hi - self.d_lo), self.diff)
        self.diff = self.diff.clamp(0.0, self.d_max)
        self.prev_act = torch.where(m, torch.zeros_like(self.prev_act), self.prev_act)
        self.prev_act2 = torch.where(m, torch.zeros_like(self.prev_act2), self.prev_act2)
        if self.prev_act3 is not None:
            self.prev_act3 = torch.where(m,torch.zeros_like(self.prev_act3),self.prev_act3)
        if self.lat_p > 0:
            lvl = (torch.rand(N, device=d) < self.lat_p).long() * (1 + (torch.rand(N, device=d) < self.lat2_frac).long())
            self.lat = torch.where(m, lvl, self.lat)
        if self.art_p > 0:
            self.art_ep = torch.where(m, torch.rand(N, device=d) < self.art_p, self.art_ep)
            self.art_on = self.art_on & ~m
            self.art_timer = torch.where(m, torch.rand(N, device=d) * 40, self.art_timer)
            self.a_alive &= ~m[:, None]
        if self.shot_p > 0:
            self.shot_ep = torch.where(m, torch.rand(N, device=d) < self.shot_p, self.shot_ep)
            self.s_alive &= ~m[:, None]
            self.s_phase = torch.where(m, torch.rand(N, device=d) * 4, self.s_phase)
            self.s_drift = torch.where(m, (torch.rand(N, device=d) - 0.5) * 1.0, self.s_drift)
        self.rp = torch.where(m, 1.8 + torch.rand(N, device=d) * 1.2, self.rp)
        self.pspeed = torch.where(m, 0.9 + torch.rand(N, device=d) * 0.2, self.pspeed)
        self.spawn_timer = torch.where(m, 20 + torch.rand(N, device=d) * 40, self.spawn_timer)
        if self.sensor_history is not None:
            self.timing_queue_action=torch.where(m[:,None],torch.zeros_like(self.timing_queue_action),self.timing_queue_action)
            self.timing_queue_due=torch.where(m[:,None],torch.full_like(self.timing_queue_due,-1),self.timing_queue_due)
            self.timing_applied=torch.where(m,torch.zeros_like(self.timing_applied),self.timing_applied)
            self.timing_last_due=torch.where(m,torch.full_like(self.timing_last_due,-1),self.timing_last_due)
            self.timing_pending_delay=torch.where(m,torch.zeros_like(self.timing_pending_delay),self.timing_pending_delay)
            self.timing_last_image_frame=torch.where(m,torch.zeros_like(self.timing_last_image_frame),self.timing_last_image_frame)
            for name,history in self.sensor_history.items():
                self.sensor_history[name]=torch.where(m[:,None,None],getattr(self,name)[:,None],history)

    # ------------------------------------------------------------------ pattern generation
    def _sample_params(self, fam):
        N, d = self.N, self.dev
        lo, hi = self.LO[:, fam].T, self.HI[:, fam].T           # [N, P]
        v = lo + torch.rand_like(lo) * (hi - lo)
        sgn = torch.where(torch.rand(N, len(PARAM_NAMES), device=d) < 0.5, -1.0, 1.0)
        out = {}
        for p, i in self.P.items():
            out[p] = v[:, i] * sgn[:, i] if p in SIGNED else v[:, i]
        return out

    def _put(self, name, val, slot, can):
        fld = self.e[name]
        cur = fld.gather(1, slot)
        fld.scatter_(1, slot, torch.where(can[:, None], val[:, None].to(fld.dtype), cur))

    def _schedule(self, fs):
        N, d, e = self.N, self.dev, self.e
        self.spawn_timer -= fs
        cap = 2 + torch.round(4 * self.diff)
        free = ~e["alive"]
        can = (self.spawn_timer <= 0) & free.any(1) & (e["alive"].sum(1) < cap)
        slot = free.float().argmax(1, keepdim=True)
        fam = torch.multinomial(self.fam_probs, 1).squeeze(1)
        p = self._sample_params(fam)
        # Walls need room: one wall emitter at a time, and not while the previous wall is still in the upper
        # part of the field (two walls with unrelated gaps a few pixels apart cannot be crossed by anyone).
        line_busy = (e["alive"] & (e["spacing"] > 0)).any(1) | (self.w_alive & (self.w["y"] < 200)).any(1)
        can = can & ~((p["spacing"] > 0) & line_busy)
        dd = self.diff
        sm, cm, im = 0.8 + 0.45 * dd, 0.7 + 0.6 * dd, 1.25 - 0.5 * dd
        is_line = p["spacing"] > 0
        n = torch.where(is_line, torch.round(p["n"]), torch.round(p["n"] * cm)).clamp(1, self.K)
        ring = torch.rand(N, device=d) < p["ring_p"]
        spread = torch.where(ring, TWO_PI / n, p["spread"])
        gap_n = torch.round(p["gap_n"])
        gap_pos0 = torch.floor(torch.rand(N, device=d) * (n - gap_n + 1))
        ex = 30 + torch.rand(N, device=d) * (W - 60)
        ey = p["ymin"] + torch.rand(N, device=d) * (p["ymax"] - p["ymin"])
        centered = p["center_x"] > 0.5
        ex = torch.where(centered, torch.full_like(ex, W / 2), ex)
        near = ((ex - self.px).abs() < 80) & ((ey - self.py).abs() < 100) & ~centered
        ex = torch.where(near, W - ex, ex)
        mvx = torch.where(torch.rand(N, device=d) < p["mvx_p"],
                          (0.3 + torch.rand(N, device=d) * 0.9) * torch.where(torch.rand(N, device=d) < 0.5, -1.0, 1.0),
                          torch.zeros(N, device=d))
        ret_t = torch.round(p["ret_t"])
        vals = dict(
            x=ex, y=ey, vx=mvx, spread=spread, base=p["base"], rot=p["rot"], speed=p["speed"] * sm,
            acc=p["acc"] * sm, smax=p["smax"] * sm, smin=p["smin"], omega=p["omega"], rad=p["rad"],
            angjit=p["angjit"], sjit=p["sjit"], posjit=p["posjit"], spacing=p["spacing"],
            ret_t=torch.where(ret_t > 0, ret_t, torch.full_like(ret_t, -1.0)), ret_speed=p["ret_speed"] * sm,
            ret_rand=p["ret_rand"], lock=(p["lock"] > 0.5).float(),
            t=torch.zeros(N, device=d), start=torch.round(p["start"]),
            interval=torch.round(p["interval"] * im).clamp(min=1), shots=torch.round(p["shots"]).clamp(min=1),
            n=n, gap_n=gap_n, gap_pos0=gap_pos0, gap_shift=torch.round(p["gap_shift"]),
            alive=torch.ones(N, device=d), ring=ring.float(), aim=(torch.rand(N, device=d) < p["aim_p"]).float(),
        )
        for k, val in vals.items():
            self._put(k, val, slot, can)
        gap = (160 - 120 * dd).clamp(min=25.0) * (0.6 + 0.8 * torch.rand(N, device=d))
        self.spawn_timer = torch.where(can, gap, self.spawn_timer)

    # ------------------------------------------------------------------ dynamics
    def _emit_step(self):
        e, N, E, K, B = self.e, self.N, self.E, self.K, self.B
        rel = e["t"] - e["start"]
        shot_i = torch.div(rel, e["interval"], rounding_mode="floor")
        fire = e["alive"] & (rel >= 0) & (rel % e["interval"] == 0) & (shot_i < e["shots"])
        shot = shot_i.clamp(min=0)
        idx, n = self.idxK, e["n"][..., None]
        idxf = idx.float()
        ring = e["ring"][..., None]
        ctr = torch.where(ring, idxf, idxf - (n.float() - 1) / 2)
        valid = fire[..., None] & (idx < n)
        span = (e["n"] - e["gap_n"] + 1).clamp(min=1)
        gp = torch.remainder(e["gap_pos0"] + shot * e["gap_shift"], span)
        in_gap = (idx >= gp[..., None]) & (idx < (gp + e["gap_n"])[..., None]) & (e["gap_n"][..., None] > 0)
        valid = valid & ~in_gap
        # track where each wall's gap is (needed for wall spacing at spawn time; the shaping reward reads it too)
        Wn = self.Wn
        wm = fire & (e["spacing"] > 0) & (e["gap_n"] > 0)
        cs_w = wm.long().cumsum(1) - 1
        wdst = torch.where(wm, (self.w_ptr[:, None] + cs_w) % Wn, torch.full_like(cs_w, Wn))
        th_w = e["base"] + shot.float() * e["rot"]
        gcx = e["x"] + (gp.float() + (e["gap_n"].float() - 1) / 2 - (e["n"].float() - 1) / 2) * e["spacing"]
        for k, v in dict(x=gcx, y=e["y"], vx=e["speed"] * torch.cos(th_w), vy=e["speed"] * torch.sin(th_w),
                         hw=e["gap_n"].float() * e["spacing"] / 2, dyprev=torch.full_like(gcx, 1e4)).items():
            self.w[k].scatter_(1, wdst, v)
        self.w_alive.scatter_(1, wdst, wm)
        self.w_alive[:, Wn] = False
        self.w_ptr = (self.w_ptr + wm.sum(1)) % Wn
        aim_ang = torch.atan2(self.py[:, None, None] - e["y"][..., None], self.px[:, None, None] - e["x"][..., None])
        base = e["base"][..., None] + torch.where(e["aim"][..., None], aim_ang, torch.zeros_like(aim_ang))
        r = torch.rand(3, N, E, K, device=self.dev) * 2 - 1
        th = base + ctr * e["spread"][..., None] + (shot.float() * e["rot"])[..., None] + e["angjit"][..., None] * r[0]
        sp = e["speed"][..., None] * (1 + e["sjit"][..., None] * r[1])
        x = e["x"][..., None] + (idxf - (n.float() - 1) / 2) * e["spacing"][..., None] + e["posjit"][..., None] * r[2]
        y = e["y"][..., None].expand(N, E, K)

        def flat(t):
            return t.expand(N, E, K).reshape(N, E * K)

        valid = valid.reshape(N, E * K)
        cs = valid.long().cumsum(1) - 1
        dst = torch.where(valid, (self.ptr[:, None] + cs) % B, torch.full_like(cs, B))
        self.stat_overwrite += (self.balive.gather(1, dst) & valid).sum()
        self.ptr = (self.ptr + valid.sum(1)) % B
        src = dict(bx=x, by=y, bth=th, bs=sp, bacc=e["acc"][..., None], bsmax=e["smax"][..., None],
                   bsmin=e["smin"][..., None], bom=e["omega"][..., None], brad=e["rad"][..., None],
                   bret=e["ret_t"][..., None], bretsp=e["ret_speed"][..., None],
                   bretr=(torch.rand(N, E, K, device=self.dev) < e["ret_rand"][..., None]).float())
        for k, v in src.items():
            getattr(self, k).scatter_(1, dst, flat(v))
        self.balive.scatter_(1, dst, valid)
        self.balive[:, B] = False
        # aim-once emitters (beams): after the first shot the direction stays where the player was
        lock_now = fire & (e["lock"] > 0.5) & e["aim"]
        if bool(lock_now.any()):
            aim_c = torch.atan2(self.py[:, None] - e["y"], self.px[:, None] - e["x"])
            e["base"] = torch.where(lock_now, e["base"] + aim_c, e["base"])
            e["aim"] = e["aim"] & ~lock_now
        # emitter bookkeeping
        e["alive"] &= ~(fire & (shot_i + 1 >= e["shots"]))
        e["t"] += 1
        e["x"] = e["x"] + e["vx"]
        bounce = (e["x"] < 10) | (e["x"] > W - 10)
        e["vx"] = torch.where(bounce, -e["vx"], e["vx"])
        e["x"] = e["x"].clamp(10, W - 10)

    def _bullet_step(self):
        act = self.bret > 0
        self.bret = torch.where(act, self.bret - 1, self.bret)
        trig = act & (self.bret == 0)
        ang = torch.atan2(self.py[:, None] - self.by, self.px[:, None] - self.bx)
        ang = torch.where(self.bretr > 0.5, torch.rand_like(ang) * TWO_PI, ang)        # released in a random direction
        self.bth = torch.where(trig, ang, self.bth)
        self.bs = torch.where(trig, self.bretsp, self.bs)
        self.bacc = torch.where(trig, torch.zeros_like(self.bacc), self.bacc)
        self.bom = torch.where(trig, torch.zeros_like(self.bom), self.bom)
        self.bs = torch.minimum(torch.maximum(self.bs + self.bacc, self.bsmin), self.bsmax)
        self.bth = self.bth + self.bom
        self.bx = self.bx + self.bs * torch.cos(self.bth)
        self.by = self.by + self.bs * torch.sin(self.bth)
        out = (self.bx < -40) | (self.bx > W + 40) | (self.by < -40) | (self.by > H + 40)
        self.balive &= ~out
        self.w["x"] = self.w["x"] + self.w["vx"]
        self.w["y"] = self.w["y"] + self.w["vy"]
        self.w_alive &= self.w["y"] < H + 20

    def _threat(self, frames):
        """[N] bool: some live bullet passes within its hit distance (+6 px) of where the player stands now during the next
        `frames` frames, if the player does not move (straight-line extrapolation). Used to call a bomb 'wasted'."""
        rx, ry = self.bx - self.px[:, None], self.by - self.py[:, None]
        vx, vy = self.bs * torch.cos(self.bth), self.bs * torch.sin(self.bth)
        vv = (vx * vx + vy * vy).clamp(min=1e-6)
        ts = (-(rx * vx + ry * vy) / vv).clamp(0, frames)
        dx, dy = rx + vx * ts, ry + vy * ts
        reach = self.rp[:, None] + self.hit_scale * self.brad + 6.0
        return (self.balive & (dx * dx + dy * dy < reach * reach)).any(1)

    def _vel(self, action):
        """-> (vx, vy, focus) per frame for actions [N]."""
        move, focus = action % 9, ((action // 9) == 1).float()
        dxm, dym = self.DX[move], self.DY[move]
        norm = torch.where((dxm != 0) & (dym != 0), 0.70710678, 1.0)
        spd = torch.where(focus > 0, 2.0, 4.0) * self.pspeed * norm
        return dxm * spd, dym * spd, focus

    def _collide(self):
        dx, dy = self.bx - self.px[:, None], self.by - self.py[:, None]
        thr = self.rp[:, None] + self.hit_scale * self.brad
        return (self.balive & (dx * dx + dy * dy < thr * thr)).any(1)

    @torch.no_grad()
    def step(self, action, bomb_req=None):
        """action: long [N] in [0, 18). Returns obs, reward, done, info (auto-resets finished envs).
        bomb_req: optional bool [N] from a separate bomb model (two-model setup: the dodger's move still applies that step)."""
        pol_act = action                                   # what the policy chose (switch penalties look at this)
        if self.lat_p > 0:                                 # late episodes: the previous decision is what actually acts
            action = torch.where(self.lat == 0, action, torch.where(self.lat == 1, self.prev_act, self.prev_act2))
        fs = self.fs + (1 if self.fs_jitter > 0 and torch.rand(()).item() < self.fs_jitter else 0)
        if self.timing_clock is not None:
            fs=self.timing_interval_pending
        vx, vy, focus = self._vel(action)
        if self.lagf is not None:                          # jittery lag: this step's action starts D game frames late
            D = torch.multinomial(self.lagf, self.N, replacement=True)
        if self.sensor_history is not None:
            # Preserve delivery order when random processing times vary.
            due=torch.maximum(self.frame+self.timing_pending_delay,self.timing_last_due)
            changed=(pol_act!=self.prev_act) if self.timing_clock is not None else torch.ones_like(pol_act,dtype=torch.bool)
            self.timing_last_due=torch.where(changed,due,self.timing_last_due)
            qa=torch.cat([self.timing_queue_action[:,1:],pol_act[:,None]],1)
            qd=torch.cat([self.timing_queue_due[:,1:],due[:,None]],1)
            self.timing_queue_action=torch.where(changed[:,None],qa,self.timing_queue_action)
            self.timing_queue_due=torch.where(changed[:,None],qd,self.timing_queue_due)
        bomb = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
        waste = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
        if self.bomb_mode:      # action 18 = bomb (no movement that step), or bomb_req from the bomb model
            bomb = (action == N_ACTIONS if bomb_req is None else bomb_req) & (self.bombs > 0) & (self.invuln <= 0)
            if self.bomb_waste > 0 and bool(bomb.any()):     # a bomb with no bullet heading for the player = wasted (extra cost)
                waste = bomb & ~self._threat(self.bomb_horizon)
            self.bombs = self.bombs - bomb.float()
            self.invuln = torch.where(bomb, torch.full_like(self.invuln, self.bomb_invuln), self.invuln)
            self.balive &= ~bomb[:, None]
        if fs>0 and self.timing_clock is None:self._schedule(fs)
        hit = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
        if fs==0:focus=self.focus
        for f_ in range(fs):
            if self.timing_clock is not None:self._schedule(1)
            if self.sensor_history is not None:
                mature=self.timing_queue_due<=(self.frame+f_)[:,None]
                order=torch.arange(mature.shape[1],device=self.dev)[None]
                latest=torch.where(mature,order,torch.full_like(order,-1)).amax(1)
                chosen=self.timing_queue_action.gather(1,latest.clamp(min=0)[:,None]).squeeze(1)
                self.timing_applied=torch.where(latest>=0,chosen,self.timing_applied)
                vx,vy,focus=self._vel(self.timing_applied)
            if self.lagf is not None:                      # frame f_ of this step is still driven by the decision ceil((D-f_)/2) back
                if self.prev_act3 is None:
                    Lf = torch.div(D - f_ + self.fs - 1, self.fs, rounding_mode="floor").clamp(0, 2)
                    delayed = torch.where(Lf == 0, action, torch.where(Lf == 1, self.prev_act, self.prev_act2))
                else:
                    Lf = torch.div(D-f_+self.fs-1,self.fs,rounding_mode='floor').clamp(0,3)
                    delayed = torch.where(Lf==0,action,torch.where(Lf==1,self.prev_act,
                                                torch.where(Lf==2,self.prev_act2,self.prev_act3)))
                vx, vy, focus = self._vel(delayed)
            self.px = (self.px + vx).clamp(8, W - 8)
            self.py = (self.py + vy).clamp(self.ymin_play, H - 16)
            self._emit_step()
            self._bullet_step()
            if self.sensor_history is not None:
                self.sensor_pointer=(self.sensor_pointer+1)%self.sensor_depth
                for name,history in self.sensor_history.items():
                    history[:,self.sensor_pointer]=getattr(self,name)
            hit |= self._collide() & (self.invuln <= 0)
            self.invuln = (self.invuln - 1).clamp(min=0)
        self.focus = focus
        self.frame += fs
        trunc = self.frame >= self.max_frames
        done = hit | trunc
        reward = self.alive_reward - hit.float() - self.bomb_cost * bomb.float() - self.bomb_waste * waste.float()
        if self.bomb_mode and self.bomb_value > 0:
            reward = reward - self.bomb_value * (bomb.float() + hit.float() * self.bombs)
        if self.graze_reward > 0:
            gx, gy = self.bx - self.px[:, None], self.by - self.py[:, None]
            gap = (gx * gx + gy * gy).sqrt() - (self.rp[:, None] + self.hit_scale * self.brad)
            close = torch.where(self.balive, (1 - gap / self.graze_dist).clamp(0, 1), torch.zeros_like(gap)).amax(1)
            reward = reward + self.graze_reward * close * close * (~hit & (self.invuln <= 0)).float()
        self.stat_bombs += bomb.sum()
        self.stat_waste += waste.sum()
        if self.sw_dir > 0 or self.sw_focus > 0:           # anti-flicker: small cost for changing direction / toggling Shift
            reward = reward - self.sw_dir * ((pol_act % 9) != (self.prev_act % 9)).float() \
                - self.sw_focus * (((pol_act // 9) == 1) != ((self.prev_act // 9) == 1)).float()
        if self.prev_act3 is not None:
            self.prev_act3 = self.prev_act2
        self.prev_act2 = self.prev_act
        self.prev_act = pol_act
        if self.art_p > 0:
            self._art_step(fs)
        if self.shot_p > 0:
            self._shot_step(fs)
        if self.high_pen > 0:
            reward = reward - self.high_pen * ((self.high_ref - self.py) / (self.high_ref - 16.0)).clamp(0, 1)
        if self.bomb_mode and (self.bomb_hold > 0 or self.bomb_keep > 0):
            reward = reward + self.bombs * ((~hit).float() * self.bomb_hold + (trunc & ~hit).float() * self.bomb_keep)
        if self.wall_shaping:
            reward = reward + self._wall_shaping(hit, done)
        ep_frames = self.frame.clone()
        self.stat_len += (ep_frames * done).sum()
        self.stat_eps += done.sum()
        self.stat_hits += hit.sum()
        self._reset(done)
        obs = self._observe(done)
        return obs, reward, done, dict(ep_frames=ep_frames, hit=hit,elapsed_frames=fs)

    def _art_step(self, fs):
        """Advance the harmless distractors: sprite residue blinking on / off, falling items, drifting enemy blobs."""
        N, d = self.N, self.dev
        self.art_timer = self.art_timer - 1
        flip = self.art_ep & (self.art_timer <= 0)
        self.art_on = self.art_on ^ flip
        dur = torch.where(self.art_on, 10 + torch.rand(N, device=d) * 80, 10 + torch.rand(N, device=d) * 110)
        self.art_timer = torch.where(flip, dur, self.art_timer)
        re = self.art_ep & self.art_on & ((torch.rand(N, device=d) < 0.35) | flip)
        self.art_idx = torch.where(re, torch.randint(0, self.art_lib.shape[1], (N,), device=d), self.art_idx)
        self.a_x = self.a_x + self.a_vx * fs
        self.a_y = self.a_y + self.a_vy * fs
        near = (self.a_x - self.px[:, None]) ** 2 + (self.a_y - self.py[:, None]) ** 2 < 14.0 ** 2      # picked up / hidden
        self.a_alive &= (self.a_y < H + 20) & (self.a_y > -30) & (self.a_x > -30) & (self.a_x < W + 30) & ~near
        free = ~self.a_alive
        spawn = self.art_ep & free.any(1) & (torch.rand(N, device=d) < 0.02)
        slot = free.float().argmax(1, keepdim=True)
        enemy = torch.rand(N, device=d) < 0.4                  # enemy blob (slow, big) vs falling item (small, fast)
        new = dict(
            a_x=20 + torch.rand(N, device=d) * (W - 40),
            a_y=torch.where(enemy, 20 + torch.rand(N, device=d) * 140, torch.full((N,), -6.0, device=d)),
            a_vx=torch.where(enemy, (torch.rand(N, device=d) - 0.5) * 1.2, torch.zeros(N, device=d)),
            a_vy=torch.where(enemy, 0.2 + torch.rand(N, device=d) * 0.5, 1.0 + torch.rand(N, device=d) * 1.2),
            a_rad=torch.where(enemy, 8 + torch.rand(N, device=d) * 6, 5 + torch.rand(N, device=d) * 1.5))
        for k, v in new.items():
            fld = getattr(self, k)
            fld.scatter_(1, slot, torch.where(spawn[:, None], v[:, None], fld.gather(1, slot)))
        self.a_alive.scatter_(1, slot, spawn[:, None] | self.a_alive.gather(1, slot))

    def _shot_step(self, fs):
        """Own amulet shots: move up, spawn a pair at the player every 4 frames."""
        N, d = self.N, self.dev
        self.s_y = self.s_y - 15.0 * fs
        self.s_x = self.s_x + self.s_vx * fs
        self.s_alive &= self.s_y > -30
        self.s_phase = self.s_phase + fs
        spawn = self.shot_ep & (self.s_phase >= 4)
        self.s_phase = torch.where(self.s_phase >= 4, self.s_phase - 4, self.s_phase)
        sgn = torch.tensor([-1.0, 1.0], device=d)
        for k in range(2):
            slot = ((self.s_ptr + k) % self.NS)[:, None]
            nx = self.px + sgn[k] * (8.0 + 3.0 * torch.rand(N, device=d))
            ny = self.py - 20.0 - (self.s_phase) * 15.0           # volley born between decisions: already part-way up
            nvx = self.s_drift + 0.2 * (torch.rand(N, device=d) - 0.5)
            for fld, v in ((self.s_x, nx), (self.s_y, ny), (self.s_vx, nvx)):
                fld.scatter_(1, slot, torch.where(spawn[:, None], v[:, None], fld.gather(1, slot)))
            self.s_alive.scatter_(1, slot, spawn[:, None] | self.s_alive.gather(1, slot))
        self.s_ptr = (self.s_ptr + 2 * spawn.long()) % self.NS

    def _shot_layers(self, loc, glb):
        alive = self.s_alive & self.shot_ep[:, None]
        rad = torch.full_like(self.s_x, 9.0)
        fj = (self.s_x - self.px[:, None]) / LOC_CELL + LOC / 2
        fi = (self.s_y - self.py[:, None]) / LOC_CELL + LOC / 2
        loc = torch.maximum(loc, _disc_grid(fi, fj, rad / LOC_CELL, alive, LOC, LOC, 7))
        glb = torch.maximum(glb, _disc_grid(self.s_y / GLB_CELL, self.s_x / GLB_CELL, rad / GLB_CELL, alive, GLB_H, GLB_W, 2))
        return loc, glb

    def _art_layers(self, loc, glb):
        """OR the artifacts into the freshly rendered local / global bullet maps (observation only, no collisions)."""
        N, dev = self.N, self.dev
        show = (self.art_ep & self.art_on)[:, None, None]
        fidx = (self.focus > 0.5).long()
        loc = torch.maximum(loc, self.art_lib[fidx, self.art_idx] * show.to(torch.uint8))
        # global view: the residue patch pooled to 8-px cells (16x16 cells), centred on the player's cell
        spg = self.art_libg[fidx, self.art_idx]
        a = torch.arange(LOC // 4, device=dev)
        ii = torch.floor(self.py / GLB_CELL).long()[:, None, None] - LOC // 8 + a.view(1, -1, 1)
        jj = torch.floor(self.px / GLB_CELL).long()[:, None, None] - LOC // 8 + a.view(1, 1, -1)
        ok = (ii >= 0) & (ii < GLB_H) & (jj >= 0) & (jj < GLB_W) & show & (spg > 0)
        idx = torch.where(ok, ii * GLB_W + jj, torch.full_like(ii, GLB_H * GLB_W))
        g2 = torch.zeros(N, GLB_H * GLB_W + 1, dtype=torch.uint8, device=dev)
        g2.scatter_(1, idx.reshape(N, -1), 1)
        glb = torch.maximum(glb, g2[:, :-1].view(N, GLB_H, GLB_W))
        # items / enemy blobs
        alive = self.a_alive & self.art_ep[:, None]
        fj = (self.a_x - self.px[:, None]) / LOC_CELL + LOC / 2
        fi = (self.a_y - self.py[:, None]) / LOC_CELL + LOC / 2
        loc = torch.maximum(loc, _disc_grid(fi, fj, self.a_rad / LOC_CELL, alive, LOC, LOC, 7))
        glb = torch.maximum(glb, _disc_grid(self.a_y / GLB_CELL, self.a_x / GLB_CELL, self.a_rad / GLB_CELL, alive,
                                            GLB_H, GLB_W, 2))
        return loc, glb

    def _wall_shaping(self, hit, done):
        """Extra reward: crossing bonus + potential-based guidance toward the gap of the nearest incoming wall."""
        w = self.w
        dy = self.py[:, None] - w["y"]                      # > 0: wall is still above the player
        dx = self.px[:, None] - w["x"]                      # signed offset from the gap centre
        crossed = self.w_alive & (w["dyprev"] > 0) & (dy <= 0)
        bonus = (self.cross_bonus * (1 - dx.abs() / (w["hw"] + 30)).clamp(0, 1) * crossed.float()).sum(1)
        bonus = bonus * (~hit).float()
        w["dyprev"] = dy
        dyn, idx = torch.where(self.w_alive & (dy > 0), dy, torch.full_like(dy, float("inf"))).min(1)
        has = torch.isfinite(dyn)
        dxn = dx.gather(1, idx[:, None]).squeeze(1).abs()
        g = ((300.0 - dyn) / 270.0).clamp(0, 1)
        phi = torch.where(has, -self.wall_beta * g * (dxn / 192.0).clamp(max=1.0), torch.zeros_like(dyn))
        phi_next = torch.where(done, torch.zeros_like(phi), phi)   # terminal state has potential 0
        extra = bonus + self.gamma * phi_next - self.phi_prev
        self.phi_prev = phi_next
        return extra

    # ------------------------------------------------------------------ observation
    def _streak(self, ids):
        """(dx, dy) game px from a bullet's centre to its streak ends: fast bullets (> FAST px/frame) are drawn as a line of
        three discs along their motion, half-length = 0.6 * speed (clamped 4..9 px) -- like a knife / kunai sprite, whose
        visible blade is much longer than its hitbox. Visual only (collisions use the centre disc)."""
        s = self.bs.gather(1, ids)
        th = self.bth.gather(1, ids)
        hl = (0.6 * s).clamp(4.0, 9.0) * (s > FAST).float()
        return hl * torch.cos(th), hl * torch.sin(th), s > FAST

    def _splat_local(self, ids, ok, patch, off_xy=None):
        N = self.N
        ox, oy = off_xy if off_xy is not None else (0.0, 0.0)
        fj = (self.bx.gather(1, ids) + ox).sub(self.px[:, None]) / LOC_CELL + LOC / 2
        fi = (self.by.gather(1, ids) + oy).sub(self.py[:, None]) / LOC_CELL + LOC / 2
        rad = self.brad.gather(1, ids)
        off = torch.arange(-patch, patch + 1, device=self.dev)
        jj = torch.floor(fj)[..., None, None] + off.view(1, 1, 1, -1)
        ii = torch.floor(fi)[..., None, None] + off.view(1, 1, -1, 1)
        ddx, ddy = (jj + 0.5 - fj[..., None, None]) * LOC_CELL, (ii + 0.5 - fi[..., None, None]) * LOC_CELL
        m = (ddx * ddx + ddy * ddy <= (rad[..., None, None] + 1.0) ** 2) & ok[..., None, None]
        m &= (ii >= 0) & (ii < LOC) & (jj >= 0) & (jj < LOC)
        idx = torch.where(m, (ii * LOC + jj).long(), torch.full_like(ii, LOC * LOC, dtype=torch.long))
        grid = torch.zeros(N, LOC * LOC + 1, dtype=torch.uint8, device=self.dev)
        grid.scatter_(1, idx.reshape(N, -1), 1)
        return grid[:, :-1].reshape(N, LOC, LOC)

    def _render_local(self):
        dx, dy = self.bx - self.px[:, None], self.by - self.py[:, None]
        d2 = torch.where(self.balive, dx * dx + dy * dy, torch.full_like(dx, float("inf")))
        vals, ids = torch.topk(d2, self.k_near, dim=1, largest=False)
        grid = self._splat_local(ids, torch.isfinite(vals), PATCH)          # patch fits radius <= ~6 px
        sx, sy, fast = self._streak(ids)
        if bool(fast.any()):
            for sg in (-1.0, 1.0):
                grid = torch.maximum(grid, self._splat_local(ids, torch.isfinite(vals) & fast, PATCH, (sg * sx, sg * sy)))
        big = self.brad > 6.0
        if bool(big.any()):                                                  # big bullets need a bigger patch (radius <= 15 px)
            vb, ib = torch.topk(torch.where(big, d2, torch.full_like(d2, float("inf"))), self.big_draw[0], dim=1, largest=False)
            grid = torch.maximum(grid, self._splat_local(ib, torch.isfinite(vb) & (vb < 100.0 ** 2), 8))
        return grid

    def _render_global(self):
        N = self.N
        gi, gj = torch.floor(self.by / GLB_CELL), torch.floor(self.bx / GLB_CELL)
        m = self.balive & (gi >= 0) & (gi < GLB_H) & (gj >= 0) & (gj < GLB_W)
        if self.blind:
            m &= ~self._in_blind()
        idx = torch.where(m, (gi * GLB_W + gj).long(), torch.full_like(gi, GLB_H * GLB_W, dtype=torch.long))
        grid = torch.zeros(N, GLB_H * GLB_W + 1, dtype=torch.uint8, device=self.dev)
        grid.scatter_(1, idx, 1)
        fast = m & (self.bs > FAST)                                          # streak ends of fast bullets (see _streak)
        if bool(fast.any()):
            hl = (0.6 * self.bs).clamp(4.0, 9.0)
            for sg in (-1.0, 1.0):
                fx, fy = self.bx + sg * hl * torch.cos(self.bth), self.by + sg * hl * torch.sin(self.bth)
                fi_, fj_ = torch.floor(fy / GLB_CELL), torch.floor(fx / GLB_CELL)
                mf = fast & (fi_ >= 0) & (fi_ < GLB_H) & (fj_ >= 0) & (fj_ < GLB_W)
                grid.scatter_(1, torch.where(mf, (fi_ * GLB_W + fj_).long(), torch.full_like(fi_, GLB_H * GLB_W, dtype=torch.long)), 1)
        grid = grid[:, :-1].reshape(N, GLB_H, GLB_W)
        big = self.balive & (self.brad > 6.0)
        if self.blind:
            big &= ~self._in_blind()
        if bool(big.any()):                                                  # big bullets cover several 8-px cells
            vb, ib = torch.topk(torch.where(big, self.brad, torch.zeros_like(self.brad)), self.big_draw[1], dim=1)
            grid = torch.maximum(grid, _disc_grid(self.by.gather(1, ib) / GLB_CELL, self.bx.gather(1, ib) / GLB_CELL,
                                                  self.brad.gather(1, ib) / GLB_CELL, vb > 6.0, GLB_H, GLB_W, 2))
        return grid

    def _in_blind(self):
        dx, dy = self.bx - self.px[:, None], self.by - self.py[:, None]
        return (dx.abs() < self.blind[0]) & (dy > -self.blind[1]) & (dy < self.blind[2])

    def _observe(self, reset_mask):
        if self.sensor_history is None:
            loc,glb=self._render_local(),self._render_global()
        else:
            probability=self.timing_prob
            if self.timing_clock is not None:
                row,self.timing_interval_pending=self.timing_clock.next()
                probability=self.timing_trace_prob[row]
            choice=torch.multinomial(probability,self.N,replacement=True)
            requested_age=self.timing_sensor[choice]
            self.timing_pending_delay=self.timing_action[choice]
            image_frame=torch.maximum(self.frame-requested_age,self.timing_last_image_frame).clamp(min=0)
            self.timing_last_image_frame=image_frame
            age=self.frame-image_frame
            index=(self.sensor_pointer-age)%self.sensor_depth
            lanes=torch.arange(self.N,device=self.dev)
            current={name:getattr(self,name) for name in self.sensor_history}
            try:
                for name,history in self.sensor_history.items():setattr(self,name,history[lanes,index])
                # Old bullet image; current player coordinate and crop center.
                loc,glb=self._render_local(),self._render_global()
            finally:
                for name,value in current.items():setattr(self,name,value)
        if self.blind:
            loc = loc * (1 - self.blind_loc)[None]
        if self.art_p > 0:
            loc, glb = self._art_layers(loc, glb)
        if self.shot_p > 0:
            loc, glb = self._shot_layers(loc, glb)
        r = reset_mask[:, None, None, None]
        if self.lags is None:
            self.loc_stack = torch.where(r, loc[:, None].expand_as(self.loc_stack),
                                         torch.cat([self.loc_stack[:, 1:], loc[:, None]], 1))
            self.glb_stack = torch.where(r, glb[:, None].expand_as(self.glb_stack),
                                         torch.cat([self.glb_stack[:, 1:], glb[:, None]], 1))
        else:                                                    # ring buffer of the last lags[0]+1 frames, pick the lags
            self.h_ptr = (self.h_ptr + 1) % self.hist
            self.loc_hist[:, self.h_ptr] = loc
            self.glb_hist[:, self.h_ptr] = glb
            if bool(reset_mask.any()):
                self.loc_hist[reset_mask] = loc[reset_mask][:, None]
                self.glb_hist[reset_mask] = glb[reset_mask][:, None]
            sel = torch.tensor([(self.h_ptr - L) % self.hist for L in self.lags], device=self.dev)
            self.loc_stack = self.loc_hist.index_select(1, sel)
            self.glb_stack = self.glb_hist.index_select(1, sel)
        # walls in the local window (cells whose centre is outside the playfield)
        ax = torch.arange(LOC, device=self.dev).float() + 0.5 - LOC / 2
        cx, cy = self.px[:, None] + ax[None] * LOC_CELL, self.py[:, None] + ax[None] * LOC_CELL
        wall = ((cy < 0) | (cy > H))[:, :, None] | ((cx < 0) | (cx > W))[:, None, :]
        marker = torch.zeros(self.N, GLB_H * GLB_W, dtype=torch.uint8, device=self.dev)
        pidx = (torch.floor(self.py / GLB_CELL).clamp(0, GLB_H - 1) * GLB_W
                + torch.floor(self.px / GLB_CELL).clamp(0, GLB_W - 1)).long()
        marker.scatter_(1, pidx[:, None], 1)
        cols = [self.px / W * 2 - 1, self.py / H * 2 - 1, self.focus]
        if self.sensor_history is not None:
            cols[2]=(self.prev_act//9==1).float()
        if self.bomb_mode:
            cols += [self.bombs / 3.0, self.invuln / self.bomb_invuln]
        vec = torch.stack(cols, 1)
        if self.prev_act_obs:
            vec = torch.cat([vec, F.one_hot(self.prev_act.clamp(max=N_ACTIONS - 1), N_ACTIONS).float()], 1)
        return dict(loc=torch.cat([self.loc_stack, wall[:, None].to(torch.uint8)], 1),
                    glob=torch.cat([self.glb_stack, marker.view(self.N, 1, GLB_H, GLB_W)], 1), vec=vec)

    # ------------------------------------------------------------------ debugging
    def render_rgb(self, i=0, scale=1):
        """Plain RGB picture of env i (for eyeballing the simulator)."""
        import numpy as np
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (int(W) * scale, int(H) * scale), (10, 10, 30))
        dr = ImageDraw.Draw(img)
        alive = self.balive[i].cpu().numpy()
        bx, by, br = self.bx[i].cpu().numpy(), self.by[i].cpu().numpy(), self.brad[i].cpu().numpy()
        for k in np.nonzero(alive)[0]:
            r = br[k] * scale
            dr.ellipse([bx[k] * scale - r, by[k] * scale - r, bx[k] * scale + r, by[k] * scale + r],
                       fill=(255, 120, 160), outline=(255, 255, 255))
        for k in range(self.E):
            if self.e["alive"][i, k]:
                ex, ey = self.e["x"][i, k].item() * scale, self.e["y"][i, k].item() * scale
                dr.rectangle([ex - 5, ey - 5, ex + 5, ey + 5], fill=(80, 200, 255))
        px, py, rp = self.px[i].item() * scale, self.py[i].item() * scale, self.rp[i].item() * scale
        dr.ellipse([px - 8, py - 8, px + 8, py + 8], outline=(120, 255, 120))
        dr.ellipse([px - rp, py - rp, px + rp, py + rp], fill=(255, 255, 255))
        return img


# ---------------------------------------------------------------------- reference policies
def random_action(env):
    return torch.randint(0, getattr(env, "n_actions", N_ACTIONS), (env.N,), device=env.dev)


def stay_action(env):
    return torch.zeros(env.N, dtype=torch.long, device=env.dev)


@torch.no_grad()
def greedy_action(env, horizon=12, k=96):
    """Privileged reference: pick the action whose end position keeps the largest clearance from the
    nearest bullets extrapolated in straight lines for `horizon` frames (ignores curving/acceleration)."""
    N, dev = env.N, env.dev
    dx, dy = env.bx - env.px[:, None], env.by - env.py[:, None]
    d2 = torch.where(env.balive, dx * dx + dy * dy, torch.full_like(dx, float("inf")))
    vals, ids = torch.topk(d2, k, dim=1, largest=False)
    ok = torch.isfinite(vals)
    bx, by, rad = env.bx.gather(1, ids), env.by.gather(1, ids), env.brad.gather(1, ids)
    th, s = env.bth.gather(1, ids), env.bs.gather(1, ids)
    vx, vy = s * torch.cos(th), s * torch.sin(th)
    acts = torch.arange(N_ACTIONS, device=dev)
    move, foc = acts % 9, (acts // 9).float()
    ddx, ddy = env.DX[move], env.DY[move]
    norm = torch.where((ddx != 0) & (ddy != 0), 0.70710678, 1.0)
    spd = torch.where(foc > 0, 2.0, 4.0)[None] * env.pspeed[:, None] * norm[None]   # [N,18]
    nx = (env.px[:, None] + ddx[None] * spd * env.fs).clamp(8, W - 8)               # [N,18]
    ny = (env.py[:, None] + ddy[None] * spd * env.fs).clamp(16, H - 16)
    hs = torch.arange(env.fs, env.fs + horizon + 1, device=dev).float()              # [h]
    fx = bx[:, :, None] + vx[:, :, None] * hs                                         # [N,k,h]
    fy = by[:, :, None] + vy[:, :, None] * hs
    ex = fx[:, None] - nx[:, :, None, None]                                           # [N,18,k,h]
    ey = fy[:, None] - ny[:, :, None, None]
    clear = torch.sqrt(ex * ex + ey * ey) - (env.rp[:, None, None, None] + 0.8 * rad[:, None, :, None])
    clear = torch.where(ok[:, None, :, None], clear, torch.full_like(clear, 60.0))
    score = clear.amin(dim=(2, 3)).clamp(max=40.0)                                    # [N,18]
    score = score - 0.02 * (ny - (H - 70)).abs() * 0.1 - 0.002 * (nx - W / 2).abs()   # mild preference for bottom-centre
    score = score - 0.05 * (move != 0).float()[None]
    return score.argmax(1)
