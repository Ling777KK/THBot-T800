"""Read-only access to the game's player position (the ONLY thing we take from memory).

The player's (x, y) hitbox centre lives in a global variable of th06nc.exe, found by the 'scan, move, re-scan'
trick; we store it as an offset from the module base so it survives ASLR. Never writes to the game."""
import ctypes, ctypes.wintypes as wt
import struct
import winutil as W

k32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi
k32.OpenProcess.restype = ctypes.c_void_p
k32.ReadProcessMemory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_size_t)]
psapi.EnumProcessModules.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), wt.DWORD, ctypes.POINTER(wt.DWORD)]
psapi.GetModuleBaseNameW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wt.LPWSTR, wt.DWORD]

# offsets from the module base (filled in by calibrate(); two copies of the player position were found)
PLAYER_RVAS = []


def module_base(h, name="th06nc.exe"):
    arr = (ctypes.c_void_p * 1024)()
    need = wt.DWORD()
    psapi.EnumProcessModules(h, arr, ctypes.sizeof(arr), ctypes.byref(need))
    for i in range(need.value // ctypes.sizeof(ctypes.c_void_p)):
        buf = ctypes.create_unicode_buffer(260)
        psapi.GetModuleBaseNameW(h, arr[i], buf, 260)
        if buf.value.lower() == name:
            return arr[i]
    return None


class GameMem:
    def __init__(self):
        self.hwnd, _ = W.find_window()
        pid = wt.DWORD()
        W.user32.GetWindowThreadProcessId(self.hwnd, ctypes.byref(pid))
        self.h = k32.OpenProcess(0x0410, False, pid.value)         # QUERY_INFORMATION | VM_READ
        self.base = module_base(self.h) if self.h else None
        self.ok = bool(self.h and self.base)

    def read(self, addr, n):
        buf = (ctypes.c_char * n)()
        got = ctypes.c_size_t()
        k32.ReadProcessMemory(self.h, ctypes.c_void_p(addr), buf, n, ctypes.byref(got))
        return bytes(buf[:got.value])

    def player_xy(self, rva=None):
        """(x, y) in game pixels (playfield 384x448) or None."""
        for r in ([rva] if rva is not None else PLAYER_RVAS):
            d = self.read(self.base + r, 8)
            if len(d) == 8:
                x, y = struct.unpack("<2f", d)
                if -20 < x < 404 and -20 < y < 468:
                    return x, y
        return None
