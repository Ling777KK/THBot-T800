"""Which difficulty is being played: match the HUD difficulty label (the 280x80 grab at client (990, 10) that play.py saves as
difficulty_label.png) against templates/label_<NAME>.png by normalised cross-correlation (brightness-invariant: the label is
grabbed while the stage still fades in). Known-good captures score >= 0.99 on their own word and <= 0.5 on the others.
   python difflabel.py <difficulty_label.png> [...]          (offline check of saved runs)"""
import glob, os, sys
import numpy as np
from PIL import Image

Y0, Y1, X0, X1 = 8, 48, 20, 230          # the big difficulty word inside the 280x80 grab
MIN_SCORE = 0.8
_T = {os.path.basename(f)[6:-4]: np.asarray(Image.open(f).convert("L")).astype(float)[Y0:Y1, X0:X1]
      for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "label_*.png"))}


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def classify(img):
    """img: 80x280 (x3) uint8 label grab. Returns (name or None, best score, {name: score})."""
    g = np.asarray(img).astype(float)
    if g.ndim == 3:                       # same weights as PIL's convert("L") used for the templates
        g = g[..., :3] @ np.array([0.299, 0.587, 0.114])
    s = {}
    for k, t in _T.items():
        s[k] = max(_ncc(g[Y0 + dy:Y1 + dy, X0 + dx:X1 + dx], t) for dy in range(-4, 5) for dx in range(-6, 7))
    best = max(s, key=s.get)
    return (best if s[best] >= MIN_SCORE else None), s[best], s


if __name__ == "__main__":
    for f in sys.argv[1:]:
        name, sc, s = classify(Image.open(f).convert("L"))
        print(f"{f}: {name or 'UNKNOWN'}  " + "  ".join(f"{k} {v:.2f}" for k, v in sorted(s.items())))
