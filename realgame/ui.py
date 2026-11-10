"""Manual menu exploration: optionally (re)launch the game, press a sequence of keys, then save a screenshot.
   python ui.py <out.png> [launch] [key[*n] | wait:<s>] ...      e.g.  python ui.py s.png down*3 z wait:2
Keys are the ones in keys.py (no X). Every key is released at the end."""
import os, sys, subprocess, time
import numpy as np
import mss
from PIL import Image
import winutil as W
import keys as K

out = sys.argv[1]
steps = sys.argv[2:]
SAME = 3.0                                           # mean abs difference of two pictures (0..255) below this: the same menu page
try:
    for st in steps:
        if st == "launch":
            if W.find_window()[0]:
                subprocess.run(["powershell", "-NoProfile", "-Command",
                                "Get-Process th06nc -ErrorAction SilentlyContinue | ForEach-Object { $null = $_.CloseMainWindow() }; "
                                "Start-Sleep 3; Get-Process th06nc -ErrorAction SilentlyContinue | Stop-Process -Force"], capture_output=True)
                time.sleep(2)
            subprocess.Popen(["cmd", "/c", "start", "", "steam://rungameid/4659620"])
            for _ in range(60):
                time.sleep(1)
                if W.find_window()[0]:
                    break
            time.sleep(8)
            continue
        if st.startswith("wait:"):
            time.sleep(float(st[5:]))
            continue
        if st.startswith("lang:"):                   # input language of the game window (setlang.py): with Japanese the agent answered ~10 ms later
            subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "setlang.py"), st[5:]])
            continue
        if st == "front":                            # only bring the game to the front (the screenshot must show the game)
            h, _ = W.find_window()
            if h and not W.is_foreground(h):
                W.bring_to_front(h); time.sleep(0.5)
            continue
        h, _ = W.find_window()
        if not h:
            sys.exit("game window not found")
        if not W.is_foreground(h):
            W.bring_to_front(h); time.sleep(0.5)
        if not W.is_foreground(h):
            sys.exit("could not bring the game to the front; not sending keys")
        if st == "z!":                               # Z that must change the picture: on the title the first Z is sometimes lost (see spellbench.ps1)
            shot = lambda: np.asarray(mss.MSS().grab(dict(zip(("left", "top"), W.client_rect(h)[:2]), width=1280, height=720))).astype(int)[..., :3]
            a = shot()
            K.tap("z", 0.15); time.sleep(1.5)
            d = float(np.abs(shot() - a).mean())
            print(f"z!: picture changed by {d:.1f}")
            if d < SAME:
                K.tap("z", 0.15); time.sleep(1.5)
                print("z!: nothing happened, pressed again")
            continue
        key, n = (st.split("*") + ["1"])[:2]
        for _ in range(int(n)):
            K.tap(key, 0.15); time.sleep(0.6)
    time.sleep(0.8)
    h, _ = W.find_window()
    l, t, r, b = W.client_rect(h)
    with mss.MSS() as s:
        img = np.asarray(s.grab(dict(left=l, top=t, width=r - l, height=b - t)))[..., :3][..., ::-1]
    Image.fromarray(np.ascontiguousarray(img)).resize((960, 540)).save(out)
    print("saved", out)
finally:
    K.release_all()
