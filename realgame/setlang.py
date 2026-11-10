"""Set the input language of the game window only (WM_INPUTLANGCHANGEREQUEST, like Win+Space while the game is in front; nothing system-wide).
Windows remembers it for the app, so end a test on the language the user uses (zh on 2026-11-03).
  python setlang.py zh|ja|en"""
import ctypes, sys, time
import ctypes.wintypes as wt
import winutil as W

u32 = ctypes.windll.user32
u32.GetKeyboardLayout.restype = ctypes.c_void_p
u32.LoadKeyboardLayoutW.restype = ctypes.c_void_p
u32.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, ctypes.c_void_p]
KLID = dict(ja="00000411", zh="00000804", en="00000409")             # en: US keyboard, no input method at all
hwnd, _ = W.find_window()
if not hwnd:
    sys.exit("game window not found")
tid = u32.GetWindowThreadProcessId(hwnd, None)
u32.PostMessageW(hwnd, 0x0050, 0, u32.LoadKeyboardLayoutW(KLID[sys.argv[1]], 0))
time.sleep(0.8)
print(f"game window input layout: {hex(u32.GetKeyboardLayout(tid) or 0)}")
