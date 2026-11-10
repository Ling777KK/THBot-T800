"""Keyboard injection via SendInput (virtual-key + scancode). The game polls GetAsyncKeyState, which
reflects injected key states, but it only reacts while its window is in the foreground."""
import ctypes
import ctypes.wintypes as wt
import time

user32 = ctypes.windll.user32
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE = 0x1, 0x2, 0x8

# NOTE: the X key (bomb) is deliberately absent. The user cancelled the bomb mechanic: the agent must never press X,
# and any code that tries raises KeyError instead of silently bombing.
VK = dict(left=0x25, up=0x26, right=0x27, down=0x28, shift=0xA0, z=0x5A, esc=0x1B, enter=0x0D,
          ctrl=0xA2, c=0x43, q=0x51, r=0x52, space=0x20)
_EXT = {0x25, 0x26, 0x27, 0x28}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _U(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _U)]


def _send(vk, up):
    scan = user32.MapVirtualKeyW(vk, 0)
    flags = (KEYEVENTF_KEYUP if up else 0) | (KEYEVENTF_EXTENDEDKEY if vk in _EXT else 0)
    inp = INPUT(INPUT_KEYBOARD, _U(ki=KEYBDINPUT(vk, scan, flags, 0, 0)))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def down(name):
    _send(VK[name], False)


def up(name):
    _send(VK[name], True)


def tap(name, hold=0.08):
    down(name); time.sleep(hold); up(name)


def release_all():
    for n in VK:
        up(n)
