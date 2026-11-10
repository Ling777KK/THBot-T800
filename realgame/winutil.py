"""Windows helpers for the real game: find the window, get its client area on screen, grab a screenshot.

Only the game's own client rectangle is ever captured."""
import ctypes
import ctypes.wintypes as wt

user32 = ctypes.windll.user32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)      # real pixel coordinates on a scaled display
except Exception:
    user32.SetProcessDPIAware()


def find_window(title_part="\u4e1c\u65b9\u7ea2\u9b54\u4e61"):          # (the window title of the game's Chinese edition)
    """Return the hwnd of the first visible top-level window whose title contains title_part."""
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if title_part in buf.value:
                    found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else (None, None)


def client_rect(hwnd):
    """(left, top, right, bottom) of the client area in screen pixels."""
    r = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    pt = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, pt.x + r.right, pt.y + r.bottom


def window_rect(hwnd):
    r = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def is_foreground(hwnd):
    return user32.GetForegroundWindow() == hwnd


def bring_to_front(hwnd):
    """Foreground-lock workaround: a synthetic Alt tap lets this process take the foreground."""
    if is_foreground(hwnd):                 # a lone Alt tap on the active window opens its system menu and eats keys
        return True
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)          # SW_RESTORE
    user32.keybd_event(0x12, 0, 0, 0)       # Alt down
    user32.keybd_event(0x12, 0, 2, 0)       # Alt up
    user32.SetForegroundWindow(hwnd)
    return is_foreground(hwnd)


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wt.UINT), ("dwTime", wt.DWORD)]


def user_idle_seconds():
    """Seconds since the last keyboard/mouse input (injected input counts too)."""
    lii = _LASTINPUTINFO(ctypes.sizeof(_LASTINPUTINFO), 0)
    user32.GetLastInputInfo(ctypes.byref(lii))
    return (ctypes.windll.kernel32.GetTickCount() - lii.dwTime) / 1000.0


def grab(hwnd):
    from PIL import ImageGrab
    return ImageGrab.grab(bbox=client_rect(hwnd), all_screens=True)


def grab_window(hwnd):
    """Occlusion-proof capture of the game's client area via PrintWindow (PW_CLIENTONLY | PW_RENDERFULLCONTENT).
    Returns a PIL RGB image; works even when another window (e.g. the chat app) is on top of the game."""
    import numpy as np
    from PIL import Image
    gdi32 = ctypes.windll.gdi32
    r = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    w, h = r.right, r.bottom
    hdc = user32.GetDC(hwnd)
    mdc = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    old = gdi32.SelectObject(mdc, bmp)
    ok = user32.PrintWindow(hwnd, mdc, 3)

    class BIH(ctypes.Structure):
        _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG), ("biPlanes", wt.WORD),
                    ("biBitCount", wt.WORD), ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                    ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                    ("biClrImportant", wt.DWORD)]

    bi = BIH(ctypes.sizeof(BIH), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)       # top-down BGRA
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bi), 0)
    gdi32.SelectObject(mdc, old); gdi32.DeleteObject(bmp); gdi32.DeleteDC(mdc); user32.ReleaseDC(hwnd, hdc)
    a = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return Image.fromarray(np.ascontiguousarray(a[..., 2::-1])), bool(ok)
