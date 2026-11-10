"""Read / patch the PKGL archives (data/th06XX.dat) of Touhou 6 New Classic and Classic. Format (reverse engineered from
th06c.exe, see NOTES.md):
  header  "PKGL" + u32 table_bytes;  table (encrypted, seed = crc32 of the archive name without extension, e.g. "th06CM")
  entry   u16 flags (1 = zstd) | u32 seed | u64 orig_size | u64 stored_size | u64 file_offset | u16 name_len | name
  data    at file_offset (16-aligned), encrypted with the entry's seed, then zstd-compressed if flags & 1
  cipher  key = two SplitMix64 outputs from st = (s * 0x9E3779B1 + 1) ^ (s << 32); data[i] ^= key[i % 16]
The game checks nothing but the decompressed size, so a file is replaced by appending it UNcompressed (flags 0) at the end of
the archive and pointing its table entry there (the old bytes stay in place, unused).
   python pkgl.py list <archive>
   python pkgl.py extract <archive> <name> <out_file>
   python pkgl.py replace <archive> <name>=<new_file> [...]      (edits the archive in place: back it up first)"""
import os, struct, sys, zlib

M64 = (1 << 64) - 1


def _mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def key(seed):
    st = (((seed * 0x9E3779B1) + 1) ^ (seed << 32)) & M64
    z1 = (st + 0x9E3779B97F4A7C15) & M64
    z2 = (z1 + 0x9E3779B97F4A7C15) & M64
    return _mix(z1).to_bytes(8, "little") + _mix(z2).to_bytes(8, "little")


def xor(buf, seed):
    import numpy as np
    k = np.frombuffer(key(seed), dtype=np.uint8)
    a = np.frombuffer(bytes(buf), dtype=np.uint8)
    return (a ^ np.resize(k, len(a))).tobytes()


class Archive:
    def __init__(self, path):
        self.path = path
        self.seed = zlib.crc32(os.path.basename(path).split(".")[0].encode())     # "th06CM" (also for th06CM.dat.orig backups)
        with open(path, "rb") as f:
            hdr = f.read(8)
            if hdr[:4] != b"PKGL":
                raise ValueError("not a PKGL archive")
            self.tsize = struct.unpack_from("<I", hdr, 4)[0]
            self.table = bytearray(xor(f.read(self.tsize), self.seed))
        self.entries, off = {}, 0
        while off < self.tsize:
            flags, seed, orig, stored, fofs = struct.unpack_from("<HIQQQ", self.table, off)
            ln = struct.unpack_from("<H", self.table, off + 30)[0]
            name = self.table[off + 32:off + 32 + ln].decode("ascii")
            self.entries[name] = dict(pos=off, flags=flags, seed=seed, orig=orig, stored=stored, offset=fofs)
            off += 32 + ln

    def read(self, name):
        e = self.entries[name]
        with open(self.path, "rb") as f:
            f.seek(e["offset"]); data = xor(f.read(e["stored"]), e["seed"])
        if e["flags"] & 1:
            import zstandard
            data = zstandard.ZstdDecompressor().decompress(data, max_output_size=e["orig"])
        if len(data) != e["orig"]:
            raise ValueError(f"{name}: got {len(data)} bytes, expected {e['orig']}")
        return data

    def append(self, name, data):
        """Write `data` uncompressed (encrypted with the entry's seed) at the end of the archive; the table is NOT changed.
        Returns the entry fields that point at it."""
        e = self.entries[name]
        with open(self.path, "r+b") as f:
            f.seek(0, 2)
            end = f.tell()
            pad = (-end) % 16
            f.write(b"\0" * pad)
            f.write(xor(data, e["seed"]))
        return dict(flags=0, seed=e["seed"], orig=len(data), stored=len(data), offset=end + pad)

    def set_entries(self, metas):
        """metas: {name: dict(flags, seed, orig, stored, offset)} -> rewrite those table entries in place."""
        for name, m in metas.items():
            e = self.entries[name]
            e.update(flags=m["flags"], seed=m["seed"], orig=m["orig"], stored=m["stored"], offset=m["offset"])
            struct.pack_into("<HIQQQ", self.table, e["pos"], m["flags"], m["seed"], m["orig"], m["stored"], m["offset"])
        with open(self.path, "r+b") as f:
            f.seek(8)
            f.write(xor(self.table, self.seed))

    def replace(self, files):
        """files: {name: bytes}. Appends each one uncompressed (same seed) and rewrites the table in place."""
        self.set_entries({name: self.append(name, data) for name, data in files.items()})


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    ar = Archive(path)
    if cmd == "list":
        for n, e in ar.entries.items():
            print(f"{n:32s} flags {e['flags']} orig {e['orig']:>10,} stored {e['stored']:>10,} @ {e['offset']:>11,}")
    elif cmd == "extract":
        open(sys.argv[4], "wb").write(ar.read(sys.argv[3]))
    elif cmd == "replace":
        files = {}
        for a in sys.argv[3:]:
            n, fn = a.split("=", 1)
            files[n] = open(fn, "rb").read()
        ar.replace(files)
        ar2 = Archive(path)                                    # read everything back
        for n, data in files.items():
            assert ar2.read(n) == data, n
        print(f"replaced {', '.join(files)} in {path}; read-back OK")
