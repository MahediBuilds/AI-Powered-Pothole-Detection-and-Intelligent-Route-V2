"""Shared helpers for the validation-only investigations (fp_categories, threshold sensitivity, ensemble).

Everything here is VALIDATION ONLY. ``guard()`` refuses any path or string that contains ``test`` or
``kaggle_probe`` (case-insensitive, after normalising separators). Every function that opens a data file
(image list, image, label, groups CSV) goes through it. Matching, AP and the group bootstrap are imported
from ``scripts/eval_checkpoint.py`` so the numbers follow the project protocol.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_checkpoint as ec  # noqa: E402  (project eval code: match_image, build_preds, ap_all, resample_weights, ...)

UNI = ROOT / "data" / "processed" / "unified_v1"
OUT = ROOT / "runs" / "eval" / "val_investigations_v1"
CACHE = OUT / "cache"
BANNED = ("test", "kaggle_probe")

VAL_SETS = {"rdd2022_val": UNI / "lists" / "rdd2022_val.txt", "hrp4k_val": UNI / "lists" / "hrp4k_val.txt"}
SOURCE_OF_SET = {"rdd2022_val": "rdd2022", "hrp4k_val": "hrp4k"}
GROUP_CSV = {"rdd2022_val": UNI / "groups_rdd2022.csv", "hrp4k_val": UNI / "groups_hrp4k.csv"}
CKPT = {
    "baseline": ROOT / "model_archive" / "baseline_yolov8s_rdd2022" / "weights" / "best.pt",
    "candidate": ROOT / "runs" / "train" / "unified_v1_y8s_1280" / "weights" / "epoch25.pt",
}
CKPT_SHA = {
    "baseline": "2aabdd131d1664089e06e3219fabcaed4cc96610fa4603ff165429f92696d798",
    "candidate": "0a1c19f2586b6b730e9c5bf2bdeff6fcb3ec8f4567d161f1aba0459d09a5adc3",
}
# (model, imgsz) prediction sets that every investigation draws from
PRED_CONFIGS = [("baseline", 640), ("candidate", 640), ("candidate", 960), ("candidate", 1280)]
SET_NAMES = list(VAL_SETS)


# ----------------------------------------------------------------------------- guard
class ForbiddenPath(RuntimeError):
    pass


def _norm(p) -> str:
    s = str(p)
    try:
        q = Path(s)
        if q.is_absolute():
            s = str(q.resolve().relative_to(ROOT.resolve()))
    except (ValueError, OSError):
        pass
    return s.replace("\\", "/").lower()


def guard(p):
    """Raise ForbiddenPath if ``p`` (path or string) names held-out test data or the kaggle probe."""
    s = _norm(p)
    for b in BANNED:
        if b in s:
            raise ForbiddenPath(f"refusing '{p}': contains '{b}' (held-out data is off limits for these investigations)")
    return p


def guard_all(items):
    for i in items:
        guard(i)
    return items


def sha256(p):
    return ec.sha256(guard(p))


def verify_checkpoint(name):
    got = sha256(CKPT[name])
    if got != CKPT_SHA[name]:
        raise RuntimeError(f"checkpoint {name} hash mismatch: {got}")
    return got


# ----------------------------------------------------------------------------- data
def read_list(set_name):
    """Image paths of a validation list (guarded line by line)."""
    lst = guard(VAL_SETS[set_name])
    files = []
    for ln in Path(lst).read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        q = Path(ln.replace("./", str(Path(lst).parent) + os.sep, 1)) if ln.startswith("./") else Path(ln)
        guard(q)
        files.append(q.resolve())
    return sorted(files)


def read_groups(set_name, stems):
    """image stem -> group for the given stems. The CSV also lists other splits; rows are only used for
    requested val stems (rows with split 'test' are skipped by value and never used)."""
    want = set(stems)
    out = {}
    with open(guard(GROUP_CSV[set_name]), newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != "val":
                continue
            st = Path(r["image"]).stem
            if st in want:
                out[st] = r["group"]
    miss = want - set(out)
    if miss:
        raise RuntimeError(f"{len(miss)} images without group in {set_name}, e.g. {sorted(miss)[:3]}")
    return out


class ValSet:
    """One validation list: image stems/paths, GT boxes (pixels, xyxy), image sizes, groups."""

    def __init__(self, name):
        self.name = name
        self.source = SOURCE_OF_SET[name]
        self.files = read_list(name)
        self.stems = [f.stem for f in self.files]
        if len(set(self.stems)) != len(self.stems):
            raise RuntimeError("duplicate stems")
        for f in self.files:
            guard(f)
        from ultralytics.data.utils import img2label_paths
        guard_all(img2label_paths([str(f) for f in self.files]))
        self.gts, self.sizes = ec.load_gt(self.files)
        gm = read_groups(name, self.stems)
        self.groups = [gm[s] for s in self.stems]
        self.n_gt = np.array([len(self.gts[s][0]) for s in self.stems])
        self.path_of = dict(zip(self.stems, self.files))


class Pool:
    """Concatenation of the two validation sets in a fixed order (rdd2022 then hrp4k)."""

    def __init__(self, sets):
        self.sets = sets
        self.stems, self.groups, self.source = [], [], []
        self.gts, self.sizes, self.path_of = {}, {}, {}
        for s in sets:
            self.stems += s.stems
            self.groups += s.groups
            self.source += [s.source] * len(s.stems)
            self.gts.update(s.gts)
            self.sizes.update(s.sizes)
            self.path_of.update(s.path_of)
        self.source = np.array(self.source)
        self.n_gt = np.array([len(self.gts[s][0]) for s in self.stems])
        if len(set(self.stems)) != len(self.stems):
            raise RuntimeError("stem collision across sets")

    def idx(self, set_name):
        return np.where(self.source == SOURCE_OF_SET[set_name])[0]


def load_pool():
    return Pool([ValSet(n) for n in SET_NAMES])


# ----------------------------------------------------------------------------- prediction cache
def pred_path(model, imgsz, set_name):
    return CACHE / "preds" / f"{model}_{imgsz}" / f"{set_name}.npz"


def save_preds(path, preds):
    stems = sorted(preds)
    img, boxes, conf = [], [], []
    for s in stems:
        b, c = preds[s]
        img += [s] * len(c)
        boxes.append(np.asarray(b, float).reshape(-1, 4)); conf.append(np.asarray(c, float))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, img=np.array(img, dtype=str), boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4)),
                        conf=np.concatenate(conf) if conf else np.zeros(0))


def load_preds_npz(path):
    z = np.load(guard(path))
    out = {}
    img, boxes, conf = z["img"], z["boxes"], z["conf"]
    if len(img):
        cuts = np.flatnonzero(img[1:] != img[:-1]) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(img)]):
            out[str(img[a])] = (boxes[a:b].copy(), conf[a:b].copy())
    return out


def load_model_preds(model, imgsz, pool):
    """All images of the pool -> (boxes, conf) with empty arrays where the model predicted nothing."""
    d = {}
    for sn in SET_NAMES:
        d.update(load_preds_npz(pred_path(model, imgsz, sn)))
    unknown = set(d) - set(pool.stems)
    if unknown:
        raise RuntimeError(f"{len(unknown)} predictions for unknown images")
    empty = (np.zeros((0, 4)), np.zeros(0))
    return {s: d.get(s, empty) for s in pool.stems}


# ----------------------------------------------------------------------------- matching / metrics
def build_P(pool, preds, idx=None, iou_thrs=ec.IOU_THRS):
    """Preds object over pool images (or the subset ``idx``, in that order)."""
    stems = pool.stems if idx is None else [pool.stems[i] for i in idx]
    return ec.build_preds(stems, pool.gts, preds, iou_thrs)


def ap50(P, w=None):
    """AP at IoU 0.5 only (weights optional). Requires P built with IoU 0.5 as the first threshold."""
    w = np.ones(P.n_images) if w is None else np.asarray(w, float)
    n_gt = float((w * P.gt_count).sum())
    if n_gt <= 0:
        return float("nan")
    wp = w[P.p_img]
    nz = wp > 0
    if not nz.any():
        return 0.0
    tp0 = P.p_tp[nz, 0]
    t = wp[nz] * tp0
    f = wp[nz] * ~tp0
    tpc, fpc = np.cumsum(t), np.cumsum(f)
    return ec.compute_ap(tpc / n_gt, tpc / np.maximum(tpc + fpc, 1e-12))


class ThrTable:
    """Per-image TP/FP counts at a fixed sorted threshold grid (IoU 0.5 matches from ``P``).

    tp[i, k] = #detections of image i with conf >= T[k] that are matched; fp likewise unmatched.
    Weighted totals for any per-image weight vector are then one matrix product (fast bootstrap)."""

    def __init__(self, P, T):
        self.T = np.asarray(T, float)
        K = len(self.T)
        n = P.n_images
        idx = np.searchsorted(self.T, P.p_conf, side="right")  # number of thresholds <= conf
        tp0 = P.p_tp[:, 0]
        self.tp = self._counts(P.p_img, idx, tp0, n, K)
        self.fp = self._counts(P.p_img, idx, ~tp0, n, K)
        self.gt = P.gt_count.astype(float)
        self.neg = (P.gt_count == 0).astype(float)

    @staticmethod
    def _counts(img, idx, mask, n, K):
        h = np.zeros((n, K + 1))
        np.add.at(h, (img[mask], idx[mask]), 1.0)
        # detections with idx = j count at thresholds k < j
        return np.cumsum(h[:, ::-1], axis=1)[:, ::-1][:, 1:]

    def metrics(self, w=None):
        """dict of arrays over thresholds: tp, fp, precision, recall, f1, fp_per_image, fp_per_empty."""
        w = np.ones(len(self.gt)) if w is None else np.asarray(w, float)
        TP = w @ self.tp
        FP = w @ self.fp
        ngt = float(w @ self.gt)
        nimg = float(w.sum())
        nneg = float(w @ self.neg)
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where(TP + FP > 0, TP / (TP + FP), np.nan)
            rec = TP / ngt if ngt > 0 else np.full_like(TP, np.nan)
            f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), np.where(np.isnan(prec), np.nan, 0.0))
            fpi = FP / nimg if nimg > 0 else np.full_like(TP, np.nan)
            fpn = (w * self.neg) @ self.fp / nneg if nneg > 0 else np.full_like(TP, np.nan)
        return dict(tp=TP, fp=FP, precision=prec, recall=rec, f1=f1, fp_per_image=fpi, fp_per_empty=fpn)


def argmax_first(v):
    """Index of the maximum ignoring NaN (first on ties = lowest threshold)."""
    v = np.asarray(v, float)
    if np.all(np.isnan(v)):
        return 0
    return int(np.nanargmax(v))


# ----------------------------------------------------------------------------- folds
def make_folds(pool, seed=0, n_folds=2):
    """Group-level fold assignment, stratified by source: within each source the sorted unique groups are
    shuffled with ``default_rng(seed)`` and dealt alternately to folds 0/1. No group is split across folds."""
    fold = np.zeros(len(pool.stems), int)
    rng = np.random.default_rng(seed)
    groups = np.array(pool.groups)
    for src in sorted(set(pool.source)):
        m = pool.source == src
        gl = sorted(set(groups[m]))
        perm = rng.permutation(len(gl))
        g2f = {gl[j]: k % n_folds for k, j in enumerate(perm)}
        fold[m] = [g2f[g] for g in groups[m]]
    return fold


def boot_weights(pool, idx=None, n_boot=1000, seed=0):
    """Group-bootstrap per-image weight vectors over the pool (or the subset idx), via the project's
    resample_weights. Identical for anything evaluated on the same images/groups."""
    groups = pool.groups if idx is None else [pool.groups[i] for i in idx]
    return list(ec.resample_weights(groups, n_boot, seed))


def pctl(v, lo=2.5, hi=97.5):
    v = np.asarray(v, float)
    return [float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def progress(msg):
    """Append a line to PROGRESS.md (resumable work log)."""
    import datetime
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "PROGRESS.md", "a", encoding="utf-8") as f:
        f.write(f"- {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} {msg}\n")
