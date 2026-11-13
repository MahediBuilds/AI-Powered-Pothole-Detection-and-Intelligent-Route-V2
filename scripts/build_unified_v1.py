"""Build the versioned datasets ``unified_v1`` (RDD2022 + HRP4K, training/eval) and ``kaggle_probe_v1`` (eval only).

Deterministic and re-runnable into a FRESH directory: it refuses to overwrite an existing output
directory. Output is built in ``<out>.partial`` and renamed only after validation passes.

Nothing is invented: labels are copies of published boxes, images are only ever excluded or
(HRP4K) resized. All sources are read-only.

Usage (from the project root):
    python scripts/build_unified_v1.py --cross-only CANDIDATES.json     # cross-source pairs, no dataset written
    python scripts/build_unified_v1.py                                  # build both datasets
    python scripts/build_unified_v1.py --validate-only data/processed/unified_v1

List-file entry format (see report): ``./../images/<source>/<split>/<file>.jpg``. Ultralytics replaces
the leading ``./`` with the list file's own folder (``lists/``), so the entry resolves to
``<root>/lists/../images/...``; it is identical on Windows and Linux and survives moving the directory.
The yaml files deliberately omit ``path:`` (Ultralytics then uses the yaml's own folder; ``path: .``
would resolve against the current working directory instead).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
HRP_SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}
HRP_SIZE = (1280, 720)
JPEG_QUALITY = 92
PHASH_THR = 8
VERDICTS = ("ok", "reject", "unclear")
EXPECTED_HRP_BOXES = 7217

RDD_DIR = ROOT / "data/processed/rdd2022_pothole"
HRP_DIR = ROOT / "data/external/hrp4k/extracted/HRP4K"
HRP_AUDIT = ROOT / "docs/dataset_audit/hrp4k"
KAG_DIR = ROOT / "data/external/kaggle_pcm_extracted/data"
KAG_REVIEW = ROOT / "docs/dataset_audit/kaggle_pcm/review"
CROSS_EXCLUSIONS = ROOT / "docs/dataset_build/unified_v1_cross_exclusions.json"


# ----------------------------------------------------------------------------- pure logic
_N = 32
_DCT = np.array([[np.cos(np.pi * (2 * j + 1) * k / (2 * _N)) for j in range(_N)] for k in range(_N)])
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash64(im: Image.Image) -> int:
    """Same algorithm as scripts/dataset_audit/hrp4k_hashing.py (L, 32x32 bilinear, DCT, 8x8, > median)."""
    g = np.asarray(im.convert("L").resize((_N, _N), Image.BILINEAR), dtype=np.float64)
    low = (_DCT @ g @ _DCT.T)[:8, :8].flatten()
    v = 0
    for b in low > np.median(low):
        v = (v << 1) | int(b)
    return v


def popcount64(x: np.ndarray) -> np.ndarray:
    """Per-element popcount of a uint64 array (any shape)."""
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return _POP8[x.view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1, dtype=np.int32)


def hex_to_u64(hexes) -> np.ndarray:
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


def pairs_within(a: np.ndarray, b: np.ndarray | None, thr: int, chunk: int = 256):
    """Index pairs (i, j) with Hamming(a[i], b[j]) <= thr. b=None -> pairs i<j inside a."""
    out = []
    same = b is None
    bb = a if same else b
    for s in range(0, len(a), chunk):
        d = popcount64(a[s:s + chunk, None] ^ bb[None, :])
        ii, jj = np.nonzero(d <= thr)
        for i, j in zip(ii + s, jj):
            if not same or i < j:
                out.append((int(i), int(j), int(d[i - s, j])))
    return out


def connected_components(n: int, pairs) -> np.ndarray:
    """Union-find; returns a component label per node, numbered by first appearance (deterministic)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, *_ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(n)]
    ids: dict[int, int] = {}
    return np.array([ids.setdefault(r, len(ids)) for r in roots], dtype=int)


def cluster_group_ids(labels: np.ndarray, prefix: str) -> list[str]:
    return [f"{prefix}{int(c):05d}" for c in labels]


def leakage_guard(splits: list[str], labels: np.ndarray) -> set[int]:
    """Indices of val/test images whose cluster contains at least one train image."""
    train_clusters = {int(c) for s, c in zip(splits, labels) if s == "train"}
    return {i for i, (s, c) in enumerate(zip(splits, labels)) if s != "train" and int(c) in train_clusters}


def valtest_spanning(splits: list[str], labels: np.ndarray, alive: set[int]) -> set[int]:
    """Among surviving images: test images sharing a cluster with a surviving val image (test side dropped)."""
    val_clusters = {int(labels[i]) for i in alive if splits[i] == "val"}
    return {i for i in alive if splits[i] == "test" and int(labels[i]) in val_clusters}


def check_verdicts(verdict_rows: list[dict], id_map_rows: list[dict], expected: int | None = EXPECTED_HRP_BOXES) -> dict:
    """Every box in id_map has exactly one verdict in VERDICTS. Raises ValueError otherwise."""
    tiles = [r["tile_id"] for r in verdict_rows]
    if len(tiles) != len(set(tiles)):
        raise ValueError("duplicate tile_id in verdicts")
    ids = {r["tile_id"] for r in id_map_rows}
    if len(id_map_rows) != len(ids):
        raise ValueError("duplicate tile_id in id_map")
    if expected is not None and len(id_map_rows) != expected:
        raise ValueError(f"id_map has {len(id_map_rows)} boxes, expected {expected}")
    missing = ids - set(tiles)
    extra = set(tiles) - ids
    bad = [r for r in verdict_rows if r["verdict"] not in VERDICTS]
    if missing or extra or bad:
        raise ValueError(f"verdict mismatch: {len(missing)} missing, {len(extra)} unknown, {len(bad)} invalid")
    return dict(Counter(r["verdict"] for r in verdict_rows))


def rejected_images(verdict_rows: list[dict], id_map_rows: list[dict]) -> set[tuple[str, str]]:
    """(split, image_file) of every image with at least one `reject` box (id_map split names: train/valid/test)."""
    v = {r["tile_id"]: r["verdict"] for r in verdict_rows}
    return {(m["split"], m["image_file"]) for m in id_map_rows if v[m["tile_id"]] == "reject"}


def parse_timestamp(name: str):
    """YYYYMMDD_HHMMSS from a file name like 20250216_164325.jpg, else None."""
    stem = Path(name).stem
    try:
        return dt.datetime.strptime(stem[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def time_groups(names: list[str], gap_s: float = 60.0) -> dict[str, tuple[str, bool]]:
    """name -> (group id, no_timestamp flag). Sorted by time; a gap > gap_s starts a new group;
    images without a timestamp each form their own flagged group."""
    stamped = sorted(((parse_timestamp(n), n) for n in names if parse_timestamp(n) is not None))
    out: dict[str, tuple[str, bool]] = {}
    g = -1
    prev = None
    for ts, n in stamped:
        if prev is None or (ts - prev).total_seconds() > gap_s:
            g += 1
        out[n] = (f"t{g:04d}", False)
        prev = ts
    for k, n in enumerate(sorted(n for n in names if parse_timestamp(n) is None)):
        out[n] = (f"u{k:04d}", True)
    return out


def xywh_px_to_yolo(x: float, y: float, w: float, h: float, W: int, H: int):
    """COCO pixel box (top-left x,y,w,h) -> YOLO normalised (cx,cy,w,h), clipped to the image.
    Returns (box, was_clipped)."""
    x1, y1, x2, y2 = x, y, x + w, y + h
    cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
    clipped = (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2)
    bw, bh = cx2 - cx1, cy2 - cy1
    if bw <= 0 or bh <= 0:
        raise ValueError("box has no area inside the image")
    return ((cx1 + bw / 2) / W, (cy1 + bh / 2) / H, bw / W, bh / H), clipped


def yolo_line(box, cls: int = 0) -> str:
    return f"{cls} " + " ".join(f"{v:.6f}" for v in box)


def list_entry(source: str, split: str, fname: str) -> str:
    """Entry resolved by Ultralytics relative to the list file's folder (lists/)."""
    return f"./../images/{source}/{split}/{fname}"


def label_ok(line: str) -> bool:
    t = line.split()
    if len(t) != 5 or t[0] != "0":
        return False
    cx, cy, w, h = (float(v) for v in t[1:])
    eps = 1e-6
    return w > 0 and h > 0 and cx - w / 2 >= -eps and cy - h / 2 >= -eps and cx + w / 2 <= 1 + eps and cy + h / 2 <= 1 + eps


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- io helpers
def read_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows: list[dict], cols: list[str]):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_text(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_json(p: Path, obj):
    write_text(p, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def digest_dir(out: Path, sources: list[str]) -> dict:
    """Same construction as scripts/dataset_digest.py, extended to the images/<source>/<split> layout."""
    res: dict = {"manifest_sha256": sha256_bytes((out / "manifest.csv").read_bytes()),
                 "dataset_yaml_sha256": sha256_bytes((out / "dataset.yaml").read_bytes())}
    for src in sources:
        for split in SPLITS:
            ldir, idir = out / "labels" / src / split, out / "images" / src / split
            labels = hashlib.sha256()
            files = sorted(ldir.glob("*.txt"), key=lambda p: p.name)
            for p in files:
                labels.update(p.name.encode("utf-8") + b"\0" + p.read_bytes() + b"\0")
            names = sorted(p.name for p in idir.iterdir() if p.is_file())
            res[f"{src}/{split}"] = {"n_images": len(names), "n_labels": len(files),
                                     "image_names_sha256": sha256_bytes("\n".join(names).encode("utf-8")),
                                     "labels_sha256": labels.hexdigest()}
    return res


# ----------------------------------------------------------------------------- workers
def _rdd_worker(args):
    src, dst = args
    data = Path(src).read_bytes()
    if dst:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(data)
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return sha256_bytes(data), im.width, im.height, phash64(im)


def _hrp_worker(args):
    src, dst, expect_sha = args
    data = Path(src).read_bytes()
    sha = sha256_bytes(data)
    if sha != expect_sha:
        raise RuntimeError(f"{src}: sha256 differs from the audit hash")
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        ow, oh = im.size
        out = im.convert("RGB").resize(HRP_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=JPEG_QUALITY)
    ob = buf.getvalue()
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_bytes(ob)
    return sha, sha256_bytes(ob), ow, oh


def _verify_worker(path):
    try:
        data = Path(path).read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return path, True, im.width, im.height, sha256_bytes(data)
    except Exception as e:  # noqa: BLE001
        return path, False, 0, 0, str(e)


def pmap(fn, items, workers):
    if workers <= 1:
        return [fn(i) for i in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items, chunksize=16))


# ----------------------------------------------------------------------------- RDD2022 + HRP4K inputs
def load_rdd() -> list[dict]:
    """Included RDD2022 images exactly as in data/processed/rdd2022_pothole (13,788)."""
    rows = [r for r in read_csv(RDD_DIR / "manifest.csv") if r["status"] == "included"]
    items = []
    for r in sorted(rows, key=lambda r: (r["split"], r["source_id"])):
        split, name = r["split"], r["source_id"]
        img = RDD_DIR / "images" / split / f"{name}.jpg"
        lab = RDD_DIR / "labels" / split / f"{name}.txt"
        if not img.is_file() or not lab.is_file():
            raise FileNotFoundError(f"rdd2022 {name}: image or label missing")
        items.append(dict(source="rdd2022", split=split, name=f"{name}.jpg", img=img, lab=lab,
                          group=r["group_id"], manifest_sha=r["image_sha256"]))
    on_disk = sum(1 for _ in (RDD_DIR / "images").rglob("*.jpg"))
    if on_disk != len(items):
        raise RuntimeError(f"rdd2022: {on_disk} images on disk but {len(items)} included in the manifest")
    return items


def load_hrp() -> tuple[list[dict], dict]:
    """All 6,003 HRP4K originals, with phash clusters, box-review and leakage decisions (no files written)."""
    hashes = json.loads((HRP_AUDIT / "full_hashes.json").read_text())
    ent = {}
    for h in hashes:
        split = "val" if h["split"] == "valid" else h["split"]
        ent[(split, int(h["name"]))] = h
    keys = sorted(ent, key=lambda k: (SPLITS.index(k[0]), k[1]))
    splits = [k[0] for k in keys]
    ph = hex_to_u64(ent[k]["phash"] for k in keys)
    pairs = pairs_within(ph, None, PHASH_THR)
    labels = connected_components(len(keys), pairs)
    gids = cluster_group_ids(labels, "hrp4k:c")

    br = HRP_AUDIT / "box_review"
    verdicts = [r for b in "ABCD" for r in read_csv(br / f"verdicts_batch_{b}.csv")]
    idmap = read_csv(br / "id_map.csv")
    totals = check_verdicts(verdicts, idmap)
    rej = rejected_images(verdicts, idmap)
    leak = leakage_guard(splits, labels)

    items = []
    for i, k in enumerate(keys):
        split, num = k
        h = ent[k]
        d = HRP_SPLIT_DIR[split]
        img = HRP_DIR / d / "images" / f"{num}.jpg"
        lab = HRP_DIR / d / "labels" / f"{num}.txt"
        if not img.is_file() or not lab.is_file():
            raise FileNotFoundError(f"hrp4k {split}/{num}: image or label missing")
        reasons = []
        if (d, f"{num}.jpg") in rej:
            reasons.append("box_review_reject")
        if i in leak:
            reasons.append("leakage_guard_phash8_train_cluster")
        items.append(dict(source="hrp4k", split=split, name=f"hrp4k_{split}_{num}.jpg", orig_name=f"{num}.jpg",
                          img=img, lab=lab, group=gids[i], sha=h["sha256"], phash=h["phash"], reasons=reasons,
                          ow=h["w"], oh=h["h"], exif=h.get("exif_orient"), idx=i))
    alive = {it["idx"] for it in items if not it["reasons"]}
    for i in sorted(valtest_spanning(splits, labels, alive)):
        items[i]["reasons"].append("group_spans_val_test_test_side_dropped")
    info = dict(box_verdicts=totals, n_boxes=len(idmap), n_clusters=int(labels.max()) + 1,
                n_pairs_le8=len(pairs), leak_by_split=dict(Counter(splits[i] for i in leak)),
                box_review_rejected_images=len(rej))
    return items, info


def cross_source(rdd: list[dict], hrp: list[dict], rdd_hash: list[tuple]) -> dict:
    """Exact (sha256 of the original files) and near (pHash <= 8) duplicates between rdd2022 and hrp4k."""
    rsha = defaultdict(list)
    for i, h in enumerate(rdd_hash):
        rsha[h[0]].append(i)
    exact = [(rsha[it["sha"]][0], i) for i, it in enumerate(hrp) if it["sha"] in rsha]
    ra = np.array([h[3] for h in rdd_hash], dtype=np.uint64)
    ha = hex_to_u64(it["phash"] for it in hrp)
    near = pairs_within(ha, ra, PHASH_THR)  # (hrp idx, rdd idx, dist)
    pairs = []
    for hi, ri, d in near:
        pairs.append(dict(hrp=hrp[hi]["name"], hrp_split=hrp[hi]["split"], hrp_excluded=bool(hrp[hi]["reasons"]),
                          rdd=rdd[ri]["name"], rdd_split=rdd[ri]["split"], dist=d))
    return dict(exact=[dict(hrp=hrp[i]["name"], rdd=rdd[r]["name"]) for r, i in exact],
                near=sorted(pairs, key=lambda p: (p["dist"], p["hrp"], p["rdd"])))


def load_cross_exclusions() -> list[dict]:
    if CROSS_EXCLUSIONS.is_file():
        return json.loads(CROSS_EXCLUSIONS.read_text())["exclude"]
    return []


# ----------------------------------------------------------------------------- unified_v1
def write_yaml_set(out: Path, eval_pairs: dict[str, str], train_list: str, val_list: str):
    write_text(out / "dataset.yaml",
               "# `path:` is omitted on purpose: Ultralytics then uses this file's folder as the dataset root.\n"
               f"train: {train_list}\nval: {val_list}\nnames:\n  0: pothole\n")
    for name, lst in eval_pairs.items():
        write_text(out / "eval" / f"{name}.yaml",
                   "# `path:` omitted -> root is this file's folder (eval/); list paths are relative to it.\n"
                   f"train: ../lists/{lst}\nval: ../lists/{lst}\ntest: ../lists/{lst}\nnames:\n  0: pothole\n")


def validate_unified(out: Path, workers: int) -> dict:
    """Independent checks on a built dataset directory. Returns a result dict; raises on any failure."""
    res: dict = {"checks": {}}
    man = read_csv(out / "manifest.csv")
    inc = [r for r in man if r["status"] == "included"]
    files = sorted(p for p in (out / "images").rglob("*") if p.is_file())
    dec = pmap(_verify_worker, [str(p) for p in files], workers)
    bad = [d for d in dec if not d[1]]
    res["checks"]["images_decode"] = dict(n=len(files), failed=len(bad))
    info = {Path(p).as_posix(): (w, h, s) for p, ok, w, h, s in dec if ok}
    errors = []
    if bad:
        errors.append(f"{len(bad)} images fail to decode")
    stems = Counter()
    n_boxes = Counter()
    nolabel = badbox = 0
    for p in files:
        r = p.relative_to(out).parts  # images, source, split, file
        lab = out / "labels" / r[1] / r[2] / (p.stem + ".txt")
        stems[(r[1], p.stem)] += 1
        if not lab.is_file():
            nolabel += 1
            continue
        for ln in lab.read_text().splitlines():
            if ln.strip():
                n_boxes[(r[1], r[2])] += 1
                if not label_ok(ln):
                    badbox += 1
    res["checks"]["label_file_per_image"] = dict(missing=nolabel)
    res["checks"]["boxes_valid"] = dict(boxes=sum(n_boxes.values()), invalid=badbox)
    if nolabel or badbox:
        errors.append(f"missing labels {nolabel}, invalid boxes {badbox}")
    # one split per image (per source and across sources)
    split_of = defaultdict(set)
    for p in files:
        r = p.relative_to(out).parts
        split_of[p.stem].add(r[2])
    multi = [s for s, v in split_of.items() if len(v) > 1]
    dup_stem = [k for k, v in stems.items() if v > 1]
    res["checks"]["no_image_in_two_splits"] = dict(violations=len(multi) + len(dup_stem))
    if multi or dup_stem:
        errors.append("image in more than one split / duplicate stems")
    # groups
    spans = {}
    for src in sorted({r["source"] for r in inc}):
        g = defaultdict(set)
        for r in inc:
            if r["source"] == src:
                g[r["group_id"]].add(r["split"])
        spans[src] = sum(1 for v in g.values() if len(v) > 1)
    res["checks"]["groups_spanning_splits"] = spans
    if any(spans.values()):
        errors.append(f"groups span splits: {spans}")
    # manifest vs disk
    mset = {r["output_path"] for r in inc}
    dset = {p.relative_to(out).as_posix() for p in files}
    sha_bad = sum(1 for r in inc if info.get((out / r["output_path"]).as_posix(), (0, 0, ""))[2] != r["sha256_output"])
    res["checks"]["manifest_matches_disk"] = dict(only_in_manifest=len(mset - dset), only_on_disk=len(dset - mset),
                                                 output_sha_mismatch=sha_bad)
    if mset != dset or sha_bad:
        errors.append("manifest and disk differ")
    # lists
    lists = {}
    for lp in sorted((out / "lists").glob("*.txt")):
        entries = [ln for ln in lp.read_text().splitlines() if ln]
        bad_fmt = [e for e in entries if not e.startswith("./../images/") or "\\" in e]
        resolved = {(lp.parent / e[2:]).resolve() for e in entries if e.startswith("./")}
        lists[lp.name] = (entries, resolved, len(bad_fmt))
    expect = {}
    for src in ("rdd2022", "hrp4k"):
        for split in SPLITS:
            expect[(src, split)] = {(out / "images" / src / split / p.name).resolve()
                                    for p in (out / "images" / src / split).iterdir()}
    want = {"train.txt": expect[("rdd2022", "train")] | expect[("hrp4k", "train")],
            "val_pooled.txt": expect[("rdd2022", "val")] | expect[("hrp4k", "val")]}
    for src in ("rdd2022", "hrp4k"):
        for split in ("val", "test"):
            want[f"{src}_{split}.txt"] = expect[(src, split)]
    lres = {}
    for name, exp in want.items():
        entries, resolved, bad_fmt = lists[name]
        ok = resolved == exp and len(entries) == len(exp) == len(set(entries)) and bad_fmt == 0
        lres[name] = dict(entries=len(entries), matches_directory=bool(ok))
        if not ok:
            errors.append(f"list {name} does not match directory contents")
    res["checks"]["lists_match_directories"] = lres
    res["errors"] = errors
    if errors:
        raise RuntimeError("validation failed: " + "; ".join(errors))
    return res


def build_unified(out: Path, workers: int, rdd, hrp, rdd_hash, cross) -> dict:
    manual = {(e["source"], e["image"]): e for e in load_cross_exclusions()}
    rows = []
    # ---- rdd2022
    todo = []
    for it, h in zip(rdd, rdd_hash):
        key = ("rdd2022", it["name"])
        it["reasons"] = ["cross_source_duplicate_confirmed"] if key in manual else []
        it["sha"], it["ow"], it["oh"] = h[0], h[1], h[2]
        if not it["reasons"]:
            todo.append(it)
    mism = sum(1 for it in rdd if it["manifest_sha"] and it["manifest_sha"] != it["sha"])
    for it in todo:
        sp, nm = it["split"], it["name"]
        shutil.copyfile(it["img"], out / "images/rdd2022" / sp / nm)
        (out / "labels/rdd2022" / sp).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(it["lab"], out / "labels/rdd2022" / sp / (Path(nm).stem + ".txt"))
        it["osha"], it["w"], it["h"] = it["sha"], it["ow"], it["oh"]
    # ---- hrp4k
    for it in hrp:
        if ("hrp4k", it["name"]) in manual:
            it["reasons"].append("cross_source_duplicate_confirmed")
    jobs = [it for it in hrp if not it["reasons"]]
    for it in jobs:
        (out / "labels/hrp4k" / it["split"]).mkdir(parents=True, exist_ok=True)
    res = pmap(_hrp_worker, [(str(it["img"]), str(out / "images/hrp4k" / it["split"] / it["name"]), it["sha"])
                             for it in jobs], workers)
    for it, (sha, osha, ow, oh) in zip(jobs, res):
        it["osha"], it["w"], it["h"] = osha, HRP_SIZE[0], HRP_SIZE[1]
        data = Path(it["lab"]).read_bytes()
        (out / "labels/hrp4k" / it["split"] / (Path(it["name"]).stem + ".txt")).write_bytes(data)  # unchanged
    # ---- manifest / lists / groups
    lists: dict[str, list[str]] = defaultdict(list)
    groups: dict[str, list[dict]] = defaultdict(list)
    stats: dict = defaultdict(lambda: dict(images=0, positives=0, boxes=0, negatives=0))
    for it in rdd + hrp:
        inc = not it["reasons"]
        nb = sum(1 for ln in Path(it["lab"]).read_text().splitlines() if ln.strip())
        outp = f"images/{it['source']}/{it['split']}/{it['name']}" if inc else ""
        rows.append(dict(source=it["source"], original_path=rel(it["img"]), output_path=outp, split=it["split"],
                         status="included" if inc else "excluded:" + "+".join(it["reasons"]),
                         sha256_original=it["sha"], sha256_output=it.get("osha", "") if inc else "",
                         width=it["w"] if inc else it["ow"], height=it["h"] if inc else it["oh"],
                         orig_width=it["ow"], orig_height=it["oh"], n_boxes=nb, group_id=it["group"]))
        if inc:
            e = list_entry(it["source"], it["split"], it["name"])
            lists[f"{it['source']}_{it['split']}"].append(e)
            groups[it["source"]].append(dict(source=it["source"], split=it["split"], image=it["name"], group=it["group"]))
            s = stats[f"{it['source']}/{it['split']}"]
            s["images"] += 1
            s["boxes"] += nb
            s["positives" if nb else "negatives"] += 1
    L = {k: sorted(v) for k, v in lists.items()}
    write_text(out / "lists/train.txt", "\n".join(sorted(L["rdd2022_train"] + L["hrp4k_train"])) + "\n")
    write_text(out / "lists/val_pooled.txt", "\n".join(sorted(L["rdd2022_val"] + L["hrp4k_val"])) + "\n")
    for k in ("rdd2022_val", "hrp4k_val", "rdd2022_test", "hrp4k_test"):
        write_text(out / f"lists/{k}.txt", "\n".join(L[k]) + "\n")
    write_yaml_set(out, {k: f"{k}.txt" for k in ("rdd2022_val", "hrp4k_val", "rdd2022_test", "hrp4k_test")},
                   "lists/train.txt", "lists/val_pooled.txt")
    write_yaml_set_extra = {"pooled_val": "val_pooled.txt"}
    for name, lst in write_yaml_set_extra.items():
        write_text(out / "eval" / f"{name}.yaml",
                   f"train: ../lists/{lst}\nval: ../lists/{lst}\nnames:\n  0: pothole\n")
    cols = ["source", "original_path", "output_path", "split", "status", "sha256_original", "sha256_output",
            "width", "height", "orig_width", "orig_height", "n_boxes", "group_id"]
    write_csv(out / "manifest.csv", rows, cols)
    allg = []
    for src in ("rdd2022", "hrp4k"):
        g = sorted(groups[src], key=lambda r: (r["split"], r["image"]))
        write_csv(out / f"groups_{src}.csv", g, ["image", "group", "split"])
        allg += g
    write_csv(out / "groups.csv", allg, ["source", "image", "group", "split"])
    excl = Counter()
    for it in rdd + hrp:
        for r in it["reasons"]:
            excl[(it["source"], it["split"], r)] += 1
    return dict(stats=dict(stats), exclusions={f"{a}/{b}/{c}": n for (a, b, c), n in sorted(excl.items())},
                rdd_manifest_sha_mismatch=mism, rows=len(rows))


# ----------------------------------------------------------------------------- kaggle probe
def build_probe(out: Path) -> dict:
    dec = read_csv(KAG_REVIEW / "image_decisions.csv")
    sel = [r for r in dec if r["decision"] == "POSITIVE" or
           (r["decision"] == "NEGATIVE" and r["negative_has_intact_manhole"] == "1")]
    sel.sort(key=lambda r: r["image"])
    names = [r["image"] for r in sel]
    kept = defaultdict(list)
    for r in read_csv(KAG_REVIEW / "kept_pothole_boxes.csv"):
        kept[r["image"]].append(r)
    pos = {r["image"] for r in sel if r["decision"] == "POSITIVE"}
    stray = set(kept) - pos
    if stray:
        raise RuntimeError(f"kept pothole boxes on non-positive images: {sorted(stray)[:3]}")
    if pos - set(kept):
        raise RuntimeError("POSITIVE images without kept boxes")
    manh = defaultdict(list)
    for r in read_csv(KAG_REVIEW / "annotation_review.csv"):
        if r["class"] == "manhole" and r["verdict"] == "intact":
            manh[r["image"]].append([float(v) for v in r["box_xywh"].split(",")])
    tg = time_groups(names)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()
    (out / "lists").mkdir()
    (out / "eval").mkdir()
    entries, mrows, grows, man = [], [], [], []
    n_pb = n_mb = n_clip = 0
    for r in sel:
        n = r["image"]
        src = KAG_DIR / "images" / n
        shutil.copyfile(src, out / "images" / n)
        with Image.open(src) as im:
            W, H = im.size
        lines = []
        for b in sorted(kept.get(n, []), key=lambda b: int(b["ann_index"])):
            box, c = xywh_px_to_yolo(float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]), W, H)
            n_clip += c
            lines.append(yolo_line(box))
        write_text(out / "labels" / (Path(n).stem + ".txt"), "".join(ln + "\n" for ln in lines))
        n_pb += len(lines)
        for x, y, w, h in manh.get(n, []):
            box, _ = xywh_px_to_yolo(x, y, w, h, W, H)
            mrows.append(dict(image=n, cx=f"{box[0]:.6f}", cy=f"{box[1]:.6f}", w=f"{box[2]:.6f}", h=f"{box[3]:.6f}"))
            n_mb += 1
        entries.append(f"./../images/{n}")
        gid, flag = tg[n]
        grows.append(dict(image=n, group=gid, no_timestamp=int(flag)))
        man.append(dict(image=n, decision=r["decision"], n_pothole_boxes=len(lines), n_intact_manhole_boxes=len(manh.get(n, [])),
                        width=W, height=H, sha256=sha256_file(src), group=gid))
    write_text(out / "lists/probe.txt", "\n".join(sorted(entries)) + "\n")
    write_text(out / "eval/probe.yaml", "# evaluation only; never used in training. `path:` omitted -> root is eval/.\n"
               "train: ../lists/probe.txt\nval: ../lists/probe.txt\ntest: ../lists/probe.txt\nnames:\n  0: pothole\n")
    write_text(out / "dataset.yaml", "# EVALUATION ONLY - never use for training.\n"
               "train: lists/probe.txt\nval: lists/probe.txt\ntest: lists/probe.txt\nnames:\n  0: pothole\n")
    write_csv(out / "manhole_boxes.csv", mrows, ["image", "cx", "cy", "w", "h"])
    write_csv(out / "groups.csv", grows, ["image", "group", "no_timestamp"])
    write_csv(out / "manifest.csv", man, ["image", "decision", "n_pothole_boxes", "n_intact_manhole_boxes", "width", "height", "sha256", "group"])
    rep = dict(images=len(sel), positives=len(pos), negatives=len(sel) - len(pos), pothole_boxes=n_pb,
               pothole_boxes_clipped_to_image=n_clip, intact_manhole_boxes=n_mb,
               distinct_time_groups=len({g["group"] for g in grows}),
               groups_without_timestamp=sum(g["no_timestamp"] for g in grows),
               timestamped_groups=len({g["group"] for g in grows if not g["no_timestamp"]}))
    write_json(out / "build_report.json", rep)
    return rep


def check_probe(out: Path) -> dict:
    files = sorted((out / "images").glob("*.jpg"))
    bad = [p.name for p in files if not _verify_worker(str(p))[1]]
    nolab = [p.name for p in files if not (out / "labels" / (p.stem + ".txt")).is_file()]
    badbox = sum(1 for p in (out / "labels").glob("*.txt") for ln in p.read_text().splitlines() if ln.strip() and not label_ok(ln))
    entries = [ln for ln in (out / "lists/probe.txt").read_text().splitlines() if ln]
    ok = {(out / "lists" / e[2:]).resolve() for e in entries} == {p.resolve() for p in files}
    if bad or nolab or badbox or not ok:
        raise RuntimeError(f"probe validation failed: decode {bad[:3]} nolabel {nolab[:3]} badbox {badbox} list_ok {ok}")
    return dict(images=len(files), decode_failures=0, missing_labels=0, invalid_boxes=0, list_matches_directory=True)


# ----------------------------------------------------------------------------- main
def fresh(p: Path) -> Path:
    part = p.with_name(p.name + ".partial")
    if p.exists() or part.exists():
        sys.exit(f"refusing to overwrite: {p} or {part} already exists (delete it deliberately and re-run)")
    return part


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data/processed/unified_v1"))
    ap.add_argument("--probe-out", default=str(ROOT / "data/processed/kaggle_probe_v1"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cross-only", metavar="JSON", help="write cross-source candidates to JSON and exit")
    ap.add_argument("--validate-only", metavar="DIR")
    ap.add_argument("--skip-probe", action="store_true")
    a = ap.parse_args(argv)

    if a.validate_only:
        print(json.dumps(validate_unified(Path(a.validate_only), a.workers), indent=2))
        return 0

    out, probe = Path(a.out), Path(a.probe_out)
    if not a.cross_only:
        out_part = fresh(out)
        probe_part = fresh(probe) if not a.skip_probe else None
    rdd = load_rdd()
    hrp, hinfo = load_hrp()
    print(f"rdd2022 {len(rdd)} images; hrp4k {len(hrp)}; clusters {hinfo['n_clusters']}; leakage guard {hinfo['leak_by_split']}; "
          f"verdicts {hinfo['box_verdicts']}", flush=True)
    rdd_hash = pmap(_rdd_worker, [(str(it["img"]), None) for it in rdd], a.workers)
    cross = cross_source(rdd, hrp, rdd_hash)
    if a.cross_only:
        write_json(Path(a.cross_only), cross)
        print(f"cross-source: {len(cross['exact'])} exact, {len(cross['near'])} near (pHash<=8) -> {a.cross_only}")
        return 0

    for p in (out_part / "images/rdd2022", out_part / "images/hrp4k"):
        for s in SPLITS:
            (p / s).mkdir(parents=True, exist_ok=True)
    rep = build_unified(out_part, a.workers, rdd, hrp, rdd_hash, cross)
    rep.update(box_review=hinfo["box_verdicts"], hrp4k_box_review_rejected_images=hinfo["box_review_rejected_images"],
               hrp4k_phash_clusters=hinfo["n_clusters"], hrp4k_leakage_guard_candidates=hinfo["leak_by_split"],
               cross_source=dict(exact=cross["exact"], near_total=len(cross["near"]),
                                 near_different_split=[p for p in cross["near"] if p["hrp_split"] != p["rdd_split"]],
                                 manual_exclusions=load_cross_exclusions()),
               built_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               params=dict(hrp_size=HRP_SIZE, jpeg_quality=JPEG_QUALITY, phash_thr=PHASH_THR))
    rep["validation"] = validate_unified(out_part, a.workers)
    write_json(out_part / "build_report.json", rep)
    write_json(out_part / "digest.json", digest_dir(out_part, ["rdd2022", "hrp4k"]))
    out_part.rename(out)
    print("unified_v1 built:", json.dumps(rep["stats"], sort_keys=True))
    if probe_part is not None:
        prep = build_probe(probe_part)
        prep["validation"] = check_probe(probe_part)
        write_json(probe_part / "build_report.json", prep)
        probe_part.rename(probe)
        print("kaggle_probe_v1 built:", json.dumps(prep, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
