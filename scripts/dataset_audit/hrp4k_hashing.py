"""Hashing helpers for the HRP4K full audit. dhash64 replicates src/roadsense/data/rdd2022.py
(L, 9x8 bilinear, row-major, msb first). phash64: L, 32x32 bilinear (PIL antialiased), 2-D DCT-II,
top-left 8x8 block, bit = coefficient > median of the 64 (DC included in the median set)."""
import hashlib, sys, csv, os, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

_N = 32
_D = np.array([[np.cos(np.pi * (2 * j + 1) * k / (2 * _N)) for j in range(_N)] for k in range(_N)])

def dhash64(im):
    g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    v = 0
    for b in (g[:, 1:] > g[:, :-1]).flatten():
        v = (v << 1) | int(b)
    return v

def phash64(im):
    g = np.asarray(im.convert("L").resize((_N, _N), Image.BILINEAR), dtype=np.float64)
    dct = _D @ g @ _D.T
    low = dct[:8, :8].flatten()
    med = np.median(low)
    v = 0
    for b in low > med:
        v = (v << 1) | int(b)
    return v

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def hash_one(p):
    r = {"path": p, "size": os.path.getsize(p), "sha256": sha256_file(p)}
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
            r.update(ok=True, w=im.width, h=im.height, mode=im.mode, fmt=im.format,
                     dhash=f"{dhash64(im):016x}", phash=f"{phash64(im):016x}",
                     exif_orient=im.getexif().get(274))
    except Exception as e:  # noqa
        r.update(ok=False, err=repr(e)[:200])
    return r

def hash_dir(paths, workers=6):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(hash_one, paths, chunksize=8))

if __name__ == "__main__":
    # usage: hrp4k_hashing.py OUT.json DIR [DIR ...]   (recursive *.jpg/*.jpeg/*.png)
    out, dirs = sys.argv[1], sys.argv[2:]
    paths = []
    for d in dirs:
        for root, _, fs in os.walk(d):
            paths += [os.path.join(root, f) for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    paths.sort()
    res = hash_dir(paths)
    json.dump(res, open(out, "w"))
    print(len(res), "hashed;", sum(1 for r in res if not r["ok"]), "failed")
