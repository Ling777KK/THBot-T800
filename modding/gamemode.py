"""Switch Touhou 6 New Classic between DISPLAY mode (the original look) and AI mode (what the screen reader should see:
no Reimu / options / own shots, black background, no spell backgrounds, no boss portraits, no effect sprites, no orbs or
name text of Reimu's own bomb).

Both versions of every texture live in the archives at the same time (AI versions = same DDS header, all-zero pixels =
fully transparent, appended once at the end); switching only rewrites the table entries, so it takes a second and is exactly
reversible. Original entries are read from copies of the untouched archives in <TOUHOU_RUNS>\\game_backup, made by this script the
first time it switches (only if it has never switched before).
   python gamemode.py status | ai | display
The game must be closed (it reads the archives at start). Game folder: TH06NC_DIR, else found in the Steam libraries."""
import json, os, re, shutil, sys
from pkgl import Archive

RUNS = os.environ.get("TOUHOU_RUNS", r"C:\touhou-runs")
BACKUP = os.path.join(RUNS, "game_backup")
STATE = os.path.join(RUNS, "mod", "modstate.json")
DATA = None                                                     # <game folder>\data, set by find_game_dir() below


def find_game_dir():
    """TH06NC_DIR, else the Steam library (registry SteamPath + steamapps\\libraryfolders.vdf) that has steamapps\\common\\th06nc."""
    d = os.environ.get("TH06NC_DIR")
    if d:
        if not os.path.exists(os.path.join(d, "th06nc.exe")):
            sys.exit(f"{d} does not contain th06nc.exe (TH06NC_DIR / -GameDir must be the game folder)")
        return d
    steam = r"C:\Program Files (x86)\Steam"
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as k:
            steam = winreg.QueryValueEx(k, "SteamPath")[0]
    except OSError:
        pass
    libs = [steam]
    vdf = os.path.join(steam, "steamapps", "libraryfolders.vdf")
    if os.path.exists(vdf):
        libs += [p.replace("\\\\", "\\") for p in re.findall(r'"path"\s+"([^"]+)"', open(vdf, encoding="utf-8", errors="replace").read())]
    for lib in libs:
        g = os.path.join(lib, "steamapps", "common", "th06nc")
        if os.path.exists(os.path.join(g, "th06nc.exe")):
            return os.path.normpath(g)
    sys.exit("game folder not found: set TH06NC_DIR to the folder that holds th06nc.exe")


def ensure_backup():
    """copy the untouched archives to BACKUP once (never after this script has switched them: they would not be the originals)."""
    os.makedirs(BACKUP, exist_ok=True)
    for an in BLANK:
        b = os.path.join(BACKUP, an + ".dat.orig")
        if os.path.exists(b):
            continue
        if os.path.exists(STATE):
            sys.exit(f"{b} is missing, but the archives were switched before: restore the original {an}.dat "
                     f"(Steam: Properties > Installed Files > Verify integrity) and copy it there")
        shutil.copy2(os.path.join(DATA, an + ".dat"), b)
        print("backup of the original archive:", b)
BLANK = {
    "th06CM": [r"player00(_4k)?\.dds",                 # Reimu, yin-yang options, own amulets, hitbox dot
               r"eff00(_4k)?\.dds",                     # explosions / effect sprites
               # etama4 (big orbs) is NOT touched any more: the 5 black-centred orbs are also BOSS BULLETS (Patchouli's Trilithon Shake
               # yellow, Emerald Megalith green: invisible in AI mode, the player died to nothing on screen; user + capture_card.ps1,
               # 2026-11-03). Without bombs (never X) Reimu's own orbs never appear anyway.
               r"etama3(_4k)?\.dds"],                   # only the star / point score items (see RECTS); every bullet stays
    "th06IN": [r"localization\.msgpack",               # only the zh-CN name of Reimu's bomb (see NOTEXT)
               r"asciis?(_4k)?\.dds"],                  # only the score pop-up numerals and 'PowerUp' (see RECTS)
    "th06ST": [r"stg\dbg(_4k)?\.dds",                   # stage backgrounds
               r"eff0\d(_4k)?\.dds",                    # spell-card backgrounds
               r"face\d\d[a-c](_4k)?\.dds",             # boss portraits (dialogue / spell cut-ins)
               r"stage\d\.std"],                        # stage scripts: fog colour -> black (see nofog)
}
# The game never clears the playfield: the opaque stage / spell background overwrites the last frame. Transparent backgrounds
# therefore left every bullet smeared into long trails (seen 2026-10-26) -> backgrounds become OPAQUE BLACK, the rest transparent.
BLACK = {"th06ST": [r"stg\dbg(_4k)?\.dds", r"eff0\d(_4k)?\.dds"]}     # (th06CM's eff00 = effect sprites: stays transparent)
# Reimu's bomb (Fantasy Seal) orbs: the five big black-centred orbs of etama4 (red, blue, green, yellow, grey; fractions of the
# texture). In the real run of 2026-10-27 they and their huge flashes filled the screen reader's mask for seconds after every
# bomb, and the bomb model bombed again. The small sprites and glows of etama4 are left alone (unknown users).
ORBS = [(0.0, 0.0, 0.85, 0.283), (0.0, 0.283, 0.5625, 0.566),
        # + the right column (x 436-503, y 0-206 of 512): grey orb, swirl and white glow, the burst drawn when an enemy dies --
        # next to the player they made the bomb model bomb (HARD runs 2026-10-28). The small sparkles below (y >= 208) stay.
        (436 / 512, 0.0, 504 / 512, 207 / 512)]
# Score items that fly onto the player by themselves after a boss / stage (user, 2026-10-27: "the grey box with a star inside";
# the bomb model saw a swarm converging on the player and wasted bombs): star item (boxed + small off-screen marker) and
# point item (boxed + marker), in etama3 px / 1024. P / F / B / 1UP items and every bullet are left as they are.
ITEMS = [(947 / 1024, 111 / 1024, 983 / 1024, 145 / 1024), (332 / 1024, 141 / 1024, 366 / 1024, 170 / 1024),
         (835 / 1024, 74 / 1024, 872 / 1024, 108 / 1024), (596 / 1024, 178 / 1024, 628 / 1024, 206 / 1024)]
# Score pop-ups: every item collected shows its value right where it was caught, i.e. on the player, in Chinese numerals
# (zh-CN), plus 'PowerUp'. After a boss whole strings of them sit on the player and the bomb model took them for bullets
# (HARD run 2026-10-27, bombs 1, 2, 9). ascii(s).dds: the numeral column (x 484-507, y 9-223; 'SP' ends at x 482) and
# 'PowerUp' (x 318-406, y 397-418; the row above ends at y 394), in px / 512, on 4x4 block edges. HUD digits are elsewhere.
ASCII = [(484 / 512, 8 / 512, 508 / 512, 224 / 512), (316 / 512, 396 / 512, 408 / 512, 420 / 512)]
RECTS = {"etama3": ITEMS, "ascii": ASCII}           # (ORBS: no longer blanked, see BLANK)
# The bomb's name banner text (zh-CN, a 24-byte msgpack fixstr) -> 24 spaces: same size, nothing drawn.
NOTEXT = [("\u7075\u7b26\u300c\u68a6\u60f3\u5c01\u5370\u300d", "zh-CN")]           # (game data: the zh-CN name of Reimu's bomb "Spirit Sign: Fantasy Seal" -- must match exactly)
# one BC7 block, mode 6: RGB endpoints 0, alpha endpoints 127, p-bits 1, all indices 0 -> every texel (1, 1, 1, 255)
BC7_BLACK = ((1 << 6) | (127 << 49) | (127 << 56) | (1 << 63) | (1 << 64)).to_bytes(16, "little")
# same mode 6 block with every endpoint and p-bit 0 -> every texel (0, 0, 0, 0): explicitly transparent (an all-zero block is a
# reserved mode, which D3D decodes as transparent but other decoders as opaque black)
BC7_CLEAR = (1 << 6).to_bytes(16, "little")


def kind_of(archive, name):
    if name.endswith(".std"):
        return "nofog"
    if name.endswith(".msgpack"):
        return "notext"
    for base in RECTS:
        if name.startswith(base):
            return "rects:" + base
    return "black" if any(re.fullmatch(p, name) for p in BLACK.get(archive, [])) else "clear"


def no_rects(dds, rects):
    """Zero (= fully transparent) the given rectangles (fractions of the texture) of a DDS without mipmaps: uncompressed
    32-bit (texel exact) or BC7 (whole 4x4 blocks, rectangle rounded outwards, filled with BC7_CLEAR)."""
    assert int.from_bytes(dds[28:32], "little") <= 1
    h, w = int.from_bytes(dds[12:16], "little"), int.from_bytes(dds[16:20], "little")
    d = bytearray(dds)
    if dds[84:88] == b"DX10":
        assert int.from_bytes(dds[128:132], "little") in (98, 99) and len(dds) == 148 + w * h
        for x0, y0, x1, y1 in rects:
            bx0, by0 = int(round(x0 * w)) // 4, int(round(y0 * h)) // 4
            bx1, by1 = -(-int(round(x1 * w)) // 4), -(-int(round(y1 * h)) // 4)
            for by in range(by0, by1):
                a = 148 + (by * (w // 4) + bx0) * 16
                d[a:a + (bx1 - bx0) * 16] = BC7_CLEAR * (bx1 - bx0)
        return bytes(d)
    assert dds[84:88] == b"\0\0\0\0" and int.from_bytes(dds[88:92], "little") == 32
    for x0, y0, x1, y1 in rects:
        for y in range(int(y0 * h), int(y1 * h)):
            a = 128 + (y * w + int(x0 * w)) * 4
            d[a:a + (int(x1 * w) - int(x0 * w)) * 4] = bytes((int(x1 * w) - int(x0 * w)) * 4)
    return bytes(d)


def no_text(msg):
    """Replace each NOTEXT string (msgpack fixstr after its language key) by spaces of the same byte length."""
    d = bytes(msg)
    for text, lang in NOTEXT:
        t = text.encode("utf-8")
        assert len(t) < 32
        old = bytes([0xa0 | len(lang)]) + lang.encode() + bytes([0xa0 | len(t)]) + t
        if d.count(old) != 1:
            raise ValueError(f"{text} ({lang}) found {d.count(old)} times")
        d = d.replace(old, old[:-len(t)] + b" " * len(t))
    return d


def nofog(std):
    """Stage script with every fog instruction's colour set to opaque black. With black background textures the distant
    geometry was still tinted by the fog (stage 2 white mist, stage 3 beige / red, stage 5 red ...). EoSD STD layout:
    u16 objects, u16 faces, u32 faces_offset, u32 script_offset, ...; script = [u32 time, u16 type, u16 size, args];
    type 1 = fog: BGRA colour, f32 near, f32 far. Only the 4 colour bytes change (same size)."""
    import struct
    d = bytearray(std)
    off = struct.unpack_from("<I", d, 8)[0]
    n = 0
    while off + 8 <= len(d):
        t, typ, size = struct.unpack_from("<IHH", d, off)
        if t == 0xFFFFFFFF or typ == 0xFFFF:
            break
        if typ == 1:
            d[off + 8:off + 12] = b"\x00\x00\x00\xff"
            n += 1
        off += 8 + size
    if n == 0:
        raise ValueError("no fog instruction found")
    return bytes(d)


def blank(dds, kind="clear"):
    if kind == "nofog":
        return nofog(dds)
    if kind.startswith("rects:"):
        return no_rects(dds, RECTS[kind[6:]])
    if kind == "notext":
        return no_text(dds)
    dx10 = dds[84:88] == b"DX10"
    head = 148 if dx10 else 128
    n = len(dds) - head
    if kind == "black":
        fmt = int.from_bytes(dds[128:132], "little") if dx10 else 0
        if fmt not in (98, 99) or n % 16:                    # only BC7 (UNORM / SRGB) backgrounds are expected
            raise ValueError(f"black fill needs BC7 data (dxgi {fmt}, {n} bytes)")
        return dds[:head] + BC7_BLACK * (n // 16)
    return dds[:head] + bytes(n)                         # all-zero texels: transparent (RGBA) / transparent black (BC7)


def bc7_mode6_texel(block):
    """Decode texel 0 of a BC7 mode-6 block (self-check of BC7_BLACK)."""
    v = int.from_bytes(block, "little")
    assert v & 0x7f == 1 << 6
    f = lambda pos: (v >> pos) & 0x7f
    p0 = (v >> 63) & 1
    return tuple((f(pos) << 1) | p0 for pos in (7, 21, 35, 49))


assert bc7_mode6_texel(BC7_BLACK) == (1, 1, 1, 255)
assert bc7_mode6_texel(BC7_CLEAR) == (0, 0, 0, 0)


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def prepare(state):
    """Make sure every AI texture exists in the archive (append once) and both entry versions are recorded."""
    for an, pats in BLANK.items():
        orig = Archive(os.path.join(BACKUP, an + ".dat.orig"))
        cur = Archive(os.path.join(DATA, an + ".dat"))
        st = state.setdefault(an, {})
        for name in orig.entries:
            if not any(re.fullmatch(p, name) for p in pats):
                continue
            o = {k: orig.entries[name][k] for k in ("flags", "seed", "orig", "stored", "offset")}
            c = {k: cur.entries[name][k] for k in ("flags", "seed", "orig", "stored", "offset")}
            kind = kind_of(an, name)
            if name in st and "ai" in st[name] and st[name].get("kind", "clear") == kind:
                continue
            if name not in st and c != o and kind == "clear":   # patched by hand earlier (player00*.dds): reuse it
                st[name] = dict(display=o, ai=c, kind=kind)
            else:
                st[name] = dict(display=o, ai=cur.append(name, blank(orig.read(name), kind)), kind=kind)
            os.makedirs(os.path.dirname(STATE), exist_ok=True)
            json.dump(state, open(STATE, "w"), indent=1)    # save after every append (never append twice)
    return state


def wanted(an, name):
    return any(re.fullmatch(p, name) for p in BLANK[an])


def switch(mode):
    state = prepare(load_state())
    for an in BLANK:
        ar = Archive(os.path.join(DATA, an + ".dat"))
        # a texture taken out of BLANK (etama4) keeps its recorded AI version but always gets the original
        ar.set_entries({n: v[mode if wanted(an, n) else "display"] for n, v in state[an].items()})
        ar2 = Archive(os.path.join(DATA, an + ".dat"))      # read back: every switched texture must load
        for n, v in state[an].items():
            d = ar2.read(n)
            assert len(d) == v[mode if wanted(an, n) else "display"]["orig"], n
        print(f"{an}: {len(state[an])} textures -> {mode}")


def status():
    state = load_state()
    for an in BLANK:
        ar = Archive(os.path.join(DATA, an + ".dat"))
        modes = []
        for n, v in state.get(an, {}).items():
            c = {k: ar.entries[n][k] for k in ("flags", "seed", "orig", "stored", "offset")}
            modes.append(("ai" if c == v["ai"] else "display" if c == v["display"] else "?") + ("" if wanted(an, n) else " (kept original)"))
        print(an, {m: modes.count(m) for m in set(modes)} or "not prepared")


if __name__ == "__main__":
    import subprocess
    DATA = os.path.join(find_game_dir(), "data")
    if len(sys.argv) > 1 and sys.argv[1] in ("ai", "display"):
        if b"th06nc.exe" in subprocess.run(["tasklist"], capture_output=True).stdout.lower():
            sys.exit("the game is running: close it first")
        ensure_backup()
        switch(sys.argv[1])
    status()
