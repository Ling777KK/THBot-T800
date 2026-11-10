"""Start a fresh CHALLENGE-mode run (Homing Amulet) at a given difficulty, with every critical step verified.
   python newgame.py <difficulty 0-3: EASY NORMAL HARD LUNATIC> [shots_dir]
The game is closed and relaunched through Steam first, so we always begin from the real title screen (blind
key sequences from an unknown menu state do not work). Highlights are checked by brightness before confirming."""
import os, subprocess, sys, time
import numpy as np
import mss
from PIL import Image
import winutil as W
import keys as K

diff = int(sys.argv[1]) if len(sys.argv) > 1 else 3
shots = sys.argv[2] if len(sys.argv) > 2 else ""
mode = sys.argv[3] if len(sys.argv) > 3 else "challenge"         # 'challenge' (infinite lives) | 'standard' (3 lives)
APPID = 4659620
if shots:
    os.makedirs(shots, exist_ok=True)
n = [0]
h = None


def find():
    global h
    h, _ = W.find_window()
    return h


def grab(x0=0, y0=0, x1=None, y1=None):
    l, t, r, b = W.client_rect(h)
    x1 = x1 or (r - l); y1 = y1 or (b - t)
    with mss.MSS() as s:
        return np.asarray(s.grab(dict(left=l + x0, top=t + y0, width=x1 - x0, height=y1 - y0)))[..., :3][..., ::-1]


def shot(tag):
    if shots:
        Image.fromarray(np.ascontiguousarray(grab())).resize((480, 270)).save(os.path.join(shots, f"{n[0]:02d}_{tag}.png"))
        n[0] += 1


def press(k, wait, hold=0.18):
    K.tap(k, hold); time.sleep(wait)


def peak(x0, y0, x1, y1):
    return int(grab(x0, y0, x1, y1).max())


def in_gameplay():
    """The golden playfield frame is only present while playing (or paused)."""
    g = grab(340, 100, 360, 600).astype(int)
    gold = (g[..., 0] > 150) & (g[..., 1] > 100) & (g[..., 2] < 110) & (g[..., 0] - g[..., 2] > 80)
    return gold.mean() > 0.05


def fail(msg):
    K.release_all()
    shot("FAILED")
    sys.exit("newgame: " + msg)


# 1. close the game if it is running, relaunch it through Steam
if find():
    p = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-Process th06nc -ErrorAction SilentlyContinue | ForEach-Object { $null = $_.CloseMainWindow() }; "
                        "Start-Sleep 3; Get-Process th06nc -ErrorAction SilentlyContinue | Stop-Process -Force"], capture_output=True)
    time.sleep(2.0)
subprocess.Popen(["cmd", "/c", "start", "", f"steam://rungameid/{APPID}"])
for _ in range(60):
    time.sleep(1.0)
    if find():
        break
else:
    fail("game did not start")
time.sleep(8.0)
W.bring_to_front(h); time.sleep(0.6)
if not W.is_foreground(h):
    fail("could not bring the game to the front; not sending any key")
shot("title_logo")
try:
    press("enter", 2.2); shot("title_menu")                  # logo -> menu with 'Start' highlighted
    for _try in range(4):                                     # Start -> difficulty list; the title menu sometimes swallows the first Z
        press("z", 2.2)
        if float(grab(0, 0, 1280, 720).mean()) < 120:         # the title menu is bright (pink / white), the difficulty list is dark
            break
        print("still on the title menu, pressing Z again")
    else:
        fail("title menu never accepted Start")
    shot("difficulty_first")                                  # (EASY highlighted)
    # difficulty list: the highlighted entry is a bright white box; verify instead of trusting the key count
    entry_y = [155, 304, 451, 597]                        # EASY NORMAL HARD LUNATIC (client px, box centre)

    def selected():
        m = [float(grab(470, y - 30, 810, y + 30).mean()) for y in entry_y]
        return int(np.argmax(m)), m

    for _ in range(8):
        cur, m = selected()
        print(f"difficulty highlight: {['EASY','NORMAL','HARD','LUNATIC'][cur]}  (brightness {[round(v) for v in m]})")
        if cur == diff:
            break
        press("down" if cur < diff else "up", 1.0)
    else:
        fail("could not select the requested difficulty")
    shot("difficulty_moved")
    press("z", 2.2); shot("character")
    press("z", 2.2); shot("shot_type")
    for _ in range(3):                                        # TOP entry = Homing Amulet must be highlighted
        top, bot = peak(840, 190, 1180, 225), peak(840, 280, 1180, 315)
        print(f"shot type highlight: top {top} bottom {bot}")
        if top >= bot:
            break
        press("up", 1.2)
    shot("shot_type_chosen")
    press("z", 2.5); shot("mode_screen")                      # -> standard / challenge
    for _ in range(3):                                        # FIRST entry = standard, SECOND = challenge: verify the wanted one
        std, chl = peak(840, 190, 1180, 225), peak(840, 270, 1180, 310)
        print(f"mode highlight: standard {std} challenge {chl}  (wanted: {mode})")
        if (chl > std) == (mode == "challenge"):
            break
        press("down" if mode == "challenge" else "up", 1.5)
    shot("mode_chosen")
    press("z", 3.0); shot("ready")                            # 'START GAME?' prompt
    for attempt in range(3):                                  # the game is live from here on: return the moment it is
        press("z", 0.3)
        t_z = time.time()
        while time.time() - t_z < 6.0 and not in_gameplay():
            time.sleep(0.1)
        if in_gameplay():
            break
        print("not in gameplay yet, confirming again")
    else:
        fail("never reached gameplay")
    shot("running")
finally:
    K.release_all()
print("fresh run started: difficulty", ["EASY", "NORMAL", "HARD", "LUNATIC"][diff], mode, "mode")
