"""Release every key and close the game (asks the window to close first, forces it only if it is still running after 3 s).
Called at the end of every real-game test script: the user wants the game closed after each test.
   python closegame.py
Does nothing while <TOUHOU_RUNS>\\keep_game_open.flag exists (a run that may end in a clear: the name entry and the
replay save come after the run; delete the flag and close by hand)."""
import os, subprocess, sys, time
import keys as K
import winutil as W

K.release_all()
if os.path.exists(os.path.join(os.environ.get("TOUHOU_RUNS", r"C:\touhou-runs"), "keep_game_open.flag")) and W.find_window()[0]:
    print("game LEFT OPEN (keep_game_open.flag exists)")
    sys.exit(0)
if W.find_window()[0]:
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-Process th06nc -ErrorAction SilentlyContinue | ForEach-Object { $null = $_.CloseMainWindow() }; "
                    "Start-Sleep 3; Get-Process th06nc -ErrorAction SilentlyContinue | Stop-Process -Force"], capture_output=True)
    time.sleep(1.0)
print("game closed" if not W.find_window()[0] else "game is STILL running")
