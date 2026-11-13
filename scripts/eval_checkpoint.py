"""Evaluate a YOLO checkpoint on any dataset YAML with a leakage-free operating-point protocol.

Single-class detection only. Everything is computed from the predictions that Ultralytics
``val()`` itself saved (``save_json``), so letterboxing / rect handling is identical to mAP.

Headline numbers
  * mAP50, mAP50-95: ``val()`` (conf 0.001, NMS IoU 0.7); own recomputation (same predictions,
    greedy matcher) is reported next to it with the difference.
  * Operating-point metrics (TP/FP/FN, P, R, F1, FP/image, FP/negative image, share of
    negative images with a detection) at confidence thresholds that are chosen on VALIDATION only.
  * Ultralytics' mp/mr are taken at the max-F1 confidence of the evaluated split itself, so they
    are stored only under ``p_r_at_split_maxF1_DO_NOT_QUOTE_FOR_TEST``.

Modes
  select  (run on a val split)  -> operating_points.json (max-F1 conf, conf for target recall, full sweep)
  apply   (run on a test split) -> --operating-points FILE.json, metrics at those fixed thresholds;
          or --explicit-conf C: metrics at that single confidence (threshold name 'matched_recall')

Pre-declared operating-point rule (identical for every model): select on the POOLED validation list
(data/processed/unified_v1/eval/pooled_val.yaml, same list as dataset.yaml val) at the model's
evaluation imgsz; the max-F1 confidence (``thresholds.max_f1``) is then applied unchanged to every
test set and the manhole probe. Per-source val runs in select mode store the full PR sweep
(operating_points.json ``sweep``) for later matched-recall comparison. Apply mode refuses a different
checkpoint hash or imgsz, and refuses images shared with the selection list (identical list or any
shared image stem) unless --allow-in-sample, which marks the result in_sample=true. Thresholds from
the pooled val list applied to a disjoint per-source test list are allowed and logged under
``threshold_provenance``. ``--manhole-boxes`` adds the manhole-probe metrics (needs --groups).

Group-aware bootstrap (1000 resamples, seed 0, 95% percentile) resamples GROUPS, from --groups CSV
(image,group) or --rdd-manifest (manifest.csv: source_id -> group_id, country). Without either it
falls back to image-level resampling and records that.

    python scripts/eval_checkpoint.py --mode select --checkpoint W.pt --data D.yaml --split val \
        --imgsz 640 --out runs/eval/NAME/val_640 --rdd-manifest data/processed/rdd2022_pothole/manifest.csv
    python scripts/eval_checkpoint.py --mode apply --operating-points runs/eval/NAME/val_640/operating_points.json ...

Refuses to write inside runs/train or model_archive. Offline.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
SIZE_BINS = [(0.0, 0.01, "<1%"), (0.01, 0.02, "1-2%"), (0.02, 0.04, "2-4%"), (0.04, float("inf"), ">4%")]
N_BOOT = 1000
EVAL_BATCH = 8
EXPLICIT_NAME = "matched_recall"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- matching
def _iou_matrix(a, b):
    """a (n,4), b (m,4) xyxy -> (n,m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-12)


def match_image(pred_boxes, pred_conf, gt_boxes, iou_thrs=IOU_THRS):
    """Greedy match for one image: predictions by descending confidence, each takes the unused GT
    with highest IoU if >= threshold (one prediction per GT).

    Returns (order, tp, gt_match_conf): ``order`` sorts the input predictions by descending conf;
    ``tp`` is (n_pred, len(iou_thrs)) bool in that order; ``gt_match_conf`` is (n_gt, len(iou_thrs))
    with the confidence of the prediction matched to each GT (-1 if none).
    """
    pred_boxes = np.asarray(pred_boxes, float).reshape(-1, 4); pred_conf = np.asarray(pred_conf, float)
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    order = np.argsort(-pred_conf, kind="stable")
    pb, pc = pred_boxes[order], pred_conf[order]
    tp = np.zeros((len(pb), len(iou_thrs)), bool)
    gm = -np.ones((len(gt_boxes), len(iou_thrs)))
    if len(pb) and len(gt_boxes):
        ious = _iou_matrix(pb, gt_boxes)
        for k, thr in enumerate(iou_thrs):
            used = np.zeros(len(gt_boxes), bool)
            for i in range(len(pb)):
                row = np.where(used, -1.0, ious[i])
                j = int(row.argmax())
                if row[j] >= thr:
                    used[j] = True; tp[i, k] = True; gm[j, k] = pc[i]
    return order, tp, gm


class Preds:
    """Flat arrays over all images. Predictions sorted by descending confidence globally."""

    def __init__(self, n_images, gt_count, gt_img, gt_size_frac, gt_match_conf, p_img, p_conf, p_tp, iou_thrs=IOU_THRS):
        o = np.argsort(-np.asarray(p_conf), kind="stable")
        self.n_images = n_images
        self.gt_count = np.asarray(gt_count, int)          # (n_images,)
        self.gt_img = np.asarray(gt_img, int)
        self.gt_size_frac = np.asarray(gt_size_frac, float)
        self.gt_match_conf = np.asarray(gt_match_conf, float).reshape(len(self.gt_img), len(iou_thrs))
        self.p_img = np.asarray(p_img, int)[o]
        self.p_conf = np.asarray(p_conf, float)[o]
        self.p_tp = np.asarray(p_tp, bool).reshape(len(o), len(iou_thrs))[o]
        self.iou_thrs = np.asarray(iou_thrs)


def build_preds(images, gts, preds, iou_thrs=IOU_THRS):
    """images: list of ids (len n). gts: {id: (boxes xyxy (m,4), size_frac (m,))}.
    preds: {id: (boxes xyxy (k,4), conf (k,))}. Returns Preds."""
    gt_count, gt_img, gt_sz, gt_mc = [], [], [], []
    p_img, p_conf, p_tp = [], [], []
    for i, im in enumerate(images):
        gb, gs = gts.get(im, (np.zeros((0, 4)), np.zeros(0)))
        pb, pc = preds.get(im, (np.zeros((0, 4)), np.zeros(0)))
        order, tp, gm = match_image(pb, pc, gb, iou_thrs)
        gt_count.append(len(gb))
        for j in range(len(gb)):
            gt_img.append(i); gt_sz.append(gs[j]); gt_mc.append(gm[j])
        pcs = np.asarray(pc, float)[order]
        p_img.extend([i] * len(pcs)); p_conf.extend(pcs.tolist()); p_tp.extend(tp.tolist())
    return Preds(len(images), gt_count, gt_img, gt_sz, np.array(gt_mc).reshape(-1, len(iou_thrs)),
                 p_img, p_conf, np.array(p_tp).reshape(-1, len(iou_thrs)), iou_thrs)


# --------------------------------------------------------------------------- metrics
def _div(a, b):
    return float(a / b) if b > 0 else float("nan")


def op_metrics(P: Preds, thr, w=None):
    """Operating-point metrics at confidence >= thr (IoU 0.5 matches). ``w`` = per-image weights
    (bootstrap counts or 0/1 subset indicator); default all ones."""
    w = np.ones(P.n_images) if w is None else np.asarray(w, float)
    keep = P.p_conf >= thr
    wp = w[P.p_img] * keep
    tp0 = P.p_tp[:, 0]
    TP = float((wp * tp0).sum()); FP = float((wp * ~tp0).sum())
    n_gt = float((w * P.gt_count).sum()); FN = n_gt - TP
    n_img = float(w.sum()); neg = (P.gt_count == 0)
    n_neg = float((w * neg).sum())
    fp_neg = float((wp * neg[P.p_img]).sum())
    det_img = np.bincount(P.p_img[keep], minlength=P.n_images) > 0
    prec, rec = _div(TP, TP + FP), _div(TP, n_gt)
    f1 = _div(2 * prec * rec, prec + rec) if prec == prec and rec == rec else float("nan")
    return dict(conf_threshold=float(thr), TP=TP, FP=FP, FN=FN, precision=prec, recall=rec, f1=f1,
                fp_per_image=_div(FP, n_img), fp_per_negative_image=_div(fp_neg, n_neg),
                negative_images_with_detection=_div(float((w * neg * det_img).sum()), n_neg),
                n_images=n_img, n_negative_images=n_neg, n_gt=n_gt)


def compute_ap(rec, prec):
    """101-point interpolated AP, mirroring ultralytics.utils.metrics.compute_ap (8.4.x): the curve is
    closed with a zero-precision point at the MAXIMUM recall before the (1, 0) sentinel, so no area is
    credited beyond the recall the detector actually reaches."""
    rec = np.asarray(rec, float); prec = np.asarray(prec, float)
    mrec = np.concatenate(([0.0], rec, [rec[-1] if len(rec) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float((getattr(np, "trapezoid", None) or np.trapz)(np.interp(x, mrec, mpre), x))


def ap_all(P: Preds, w=None):
    """AP per IoU threshold (array). Weighted by per-image weights."""
    w = np.ones(P.n_images) if w is None else np.asarray(w, float)
    n_gt = float((w * P.gt_count).sum())
    wp = w[P.p_img]; nz = wp > 0
    out = []
    for k in range(len(P.iou_thrs)):
        if n_gt <= 0 or not nz.any():
            out.append(float("nan") if n_gt <= 0 else 0.0); continue
        t = wp[nz] * P.p_tp[nz, k]; f = wp[nz] * ~P.p_tp[nz, k]
        tpc, fpc = np.cumsum(t), np.cumsum(f)
        out.append(compute_ap(tpc / n_gt, tpc / np.maximum(tpc + fpc, 1e-12)))
    return np.array(out)


def sweep(P: Preds):
    """PR sweep at every distinct confidence (threshold = that confidence, >=)."""
    if len(P.p_conf) == 0:
        return dict(conf=[], precision=[], recall=[], f1=[], tp=[], fp=[])
    tpc = np.cumsum(P.p_tp[:, 0]); fpc = np.cumsum(~P.p_tp[:, 0])
    last = np.r_[P.p_conf[1:] != P.p_conf[:-1], True]  # last index of each tie block
    tp, fp, conf = tpc[last], fpc[last], P.p_conf[last]
    n_gt = max(int(P.gt_count.sum()), 1)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / n_gt
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return dict(conf=conf.tolist(), precision=prec.tolist(), recall=rec.tolist(), f1=f1.tolist(),
                tp=tp.tolist(), fp=fp.tolist())


def select_thresholds(sw, target_recall):
    """Max-F1 confidence and the confidence where recall == target (linear interpolation on the sweep)."""
    conf = np.array(sw["conf"]); rec = np.array(sw["recall"]); f1 = np.array(sw["f1"])
    if len(conf) == 0:
        return dict(max_f1_conf=None, max_f1=None, target_recall=target_recall, target_recall_conf=None,
                    note_target="no predictions")
    i = int(f1.argmax())
    out = dict(max_f1_conf=float(conf[i]), max_f1=float(f1[i]), target_recall=target_recall)
    ok = np.where(rec >= target_recall)[0]  # conf descends, recall ascends
    if len(ok) == 0:
        out.update(target_recall_conf=None, note_target=f"max recall {rec.max():.4f} < target")
    else:
        j = int(ok[0])
        if j == 0:
            c = float(conf[0])
        else:
            r0, r1, c0, c1 = rec[j - 1], rec[j], conf[j - 1], conf[j]
            c = float(c0 + (target_recall - r0) / (r1 - r0) * (c1 - c0)) if r1 > r0 else float(c1)
        out.update(target_recall_conf=c, note_target="linear interpolation between adjacent sweep points")
    return out


def size_breakdown(P: Preds, thr, w=None):
    w = np.ones(P.n_images) if w is None else np.asarray(w, float)
    gw = w[P.gt_img]; matched = P.gt_match_conf[:, 0] >= thr
    res = {}
    for lo, hi, name in SIZE_BINS:
        m = (P.gt_size_frac >= lo) & (P.gt_size_frac < hi)
        n = float((gw * m).sum()); h = float((gw * m * matched).sum())
        res[name] = dict(n_gt=n, n_detected=h, recall=_div(h, n))
    return res


def groups_digest(group_of_image):
    """sha256 of the sorted unique group labels (what determines the resamples together with the seed)."""
    return hashlib.sha256("\n".join(sorted(set(map(str, group_of_image)))).encode()).hexdigest()


def resample_weights(group_of_image, n_boot=N_BOOT, seed=0):
    """Yield per-image weight vectors for group bootstrap resamples. Fully determined by (seed, the
    SORTED unique group labels): group index = position in the sorted list, G draws with replacement per
    resample from default_rng(seed). Two models evaluated on the same image list/groups get identical
    resamples regardless of image order."""
    labels = sorted(set(map(str, group_of_image)))
    pos = {g: i for i, g in enumerate(labels)}
    gid = np.array([pos[str(g)] for g in group_of_image], int)
    G = len(labels)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, G, G), minlength=G)[gid].astype(float)


def group_bootstrap(P: Preds, group_of_image, thresholds, n_boot=N_BOOT, seed=0):
    """Resample GROUPS with replacement. Returns dict of 95% percentile CIs plus the raw ``samples``
    (one entry per resample). thresholds: {name: conf}."""
    G = len(set(map(str, group_of_image)))
    keys = ["precision", "recall", "fp_per_image"]
    ap50, ap5095 = [], []
    opm = {n: {k: [] for k in keys} for n in thresholds}
    for w in resample_weights(group_of_image, n_boot, seed):
        a = ap_all(P, w); ap50.append(a[0]); ap5095.append(float(np.mean(a)))
        for n, t in thresholds.items():
            m = op_metrics(P, t, w)
            for k in keys:
                opm[n][k].append(m[k])

    def ci(v):
        v = np.asarray(v, float)
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return dict(n_boot=n_boot, seed=seed, n_groups=G, ci_level=0.95, method="percentile, groups resampled",
                mAP50=ci(ap50), mAP50_95=ci(ap5095),
                operating_point={n: {k: ci(v) for k, v in d.items()} for n, d in opm.items()},
                samples=dict(mAP50=ap50, mAP50_95=ap5095, operating_point=opm))


def group_bootstrap_op(P, group_of_image, thr, subset_w, n_boot=N_BOOT, seed=0):
    """Group bootstrap CIs for precision/recall at one threshold on a 0/1 image subset."""
    acc = {"precision": [], "recall": []}
    for w0 in resample_weights(group_of_image, n_boot, seed):
        w = w0 * subset_w
        m = op_metrics(P, thr, w)
        for k in acc:
            acc[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in acc.items()}


# --------------------------------------------------------------------------- manhole probe
def manhole_hits(pred_boxes, pred_conf, manhole_boxes, thr, iou_thr=0.3):
    """For each intact-manhole box (xyxy) say whether a pothole prediction with conf >= thr 'hits' it.

    Two separate definitions: ``iou`` = some prediction has IoU >= iou_thr with the manhole box;
    ``centre`` = some prediction's centre lies inside the manhole box (inclusive edges).
    Returns (hit_iou (m,), hit_centre (m,)) bool arrays.
    """
    mb = np.asarray(manhole_boxes, float).reshape(-1, 4)
    pb = np.asarray(pred_boxes, float).reshape(-1, 4)
    pc = np.asarray(pred_conf, float).reshape(-1)
    pb = pb[pc >= thr]
    if len(mb) == 0 or len(pb) == 0:
        return np.zeros(len(mb), bool), np.zeros(len(mb), bool)
    hit_iou = (_iou_matrix(pb, mb) >= iou_thr).any(0)
    cx = (pb[:, 0] + pb[:, 2]) / 2; cy = (pb[:, 1] + pb[:, 3]) / 2
    inside = ((cx[:, None] >= mb[None, :, 0]) & (cx[:, None] <= mb[None, :, 2]) &
              (cy[:, None] >= mb[None, :, 1]) & (cy[:, None] <= mb[None, :, 3]))
    return hit_iou, inside.any(0)


def manhole_probe(stems, sizes, preds, gt_count, mrows, thr, group_of, n_boot=N_BOOT, seed=0):
    """Manhole-probe metrics at a FIXED threshold. mrows: [(stem, cx, cy, w, h) normalised].
    Bootstrap resamples groups (``group_of`` per image in ``stems`` order)."""
    idx = {s: i for i, s in enumerate(stems)}
    n = len(stems)
    m_img, hi, hc = [], [], []
    for s, cx, cy, bw, bh in mrows:
        if s not in idx:
            raise ValueError(f"manhole box for unknown image {s}")
        W, H = sizes[s]
        box = [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]
        pb, pc = preds.get(s, (np.zeros((0, 4)), np.zeros(0)))
        a, b = manhole_hits(pb, pc, [box], thr)
        m_img.append(idx[s]); hi.append(bool(a[0])); hc.append(bool(b[0]))
    m_img = np.array(m_img, int); hi = np.array(hi, bool); hc = np.array(hc, bool)
    npred = np.array([int((preds[s][1] >= thr).sum()) if s in preds else 0 for s in stems], float)
    gt_count = np.asarray(gt_count)
    free = gt_count == 0

    def stats(w):
        mw = w[m_img] if len(m_img) else np.zeros(0)
        nm = mw.sum(); nf = (w * free).sum()
        return dict(
            manhole_hit_share_iou03=_div((mw * hi).sum(), nm),
            manhole_hit_share_centre_inside=_div((mw * hc).sum(), nm),
            preds_per_pothole_free_image=_div((w * free * npred).sum(), nf),
            pothole_free_images_with_pred=_div((w * free * (npred > 0)).sum(), nf))

    pt = stats(np.ones(n))
    G = len(set(map(str, group_of)))
    samples = [stats(w) for w in resample_weights(group_of, n_boot, seed)]
    ci = {k: [float(np.nanpercentile([x[k] for x in samples], 2.5)), float(np.nanpercentile([x[k] for x in samples], 97.5))]
          for k in pt}
    pt.update(conf_threshold=float(thr), n_manhole_boxes=len(m_img),
              n_pothole_free_images=int(free.sum()), n_pothole_free_images_with_pred=int(((npred > 0) & free).sum()),
              total_preds_on_pothole_free_images=int((npred * free).sum()),
              manhole_boxes_on_pothole_free_images=int(free[m_img].sum()) if len(m_img) else 0,
              hits_iou03=int(hi.sum()), hits_centre_inside=int(hc.sum()),
              ci95=ci, _samples=samples, n_groups=G, n_boot=n_boot, seed=seed,
              definition="hit = a pothole prediction (conf >= threshold) with IoU >= 0.3 with the intact manhole box (iou03) "
                         "OR whose centre lies inside it (centre_inside); reported separately")
    return pt


def write_bootstrap_samples(path, ci, manhole_res=None):
    """Per-resample values -> CSV (resample, mAP50, mAP50_95, <thr>__precision/recall/fp_per_image, probe_*)."""
    s = ci["samples"]; n = len(s["mAP50"])
    cols = {"mAP50": s["mAP50"], "mAP50_95": s["mAP50_95"]}
    for name, d in s["operating_point"].items():
        for k, v in d.items():
            cols[f"{name}__{k}"] = v
    if manhole_res is not None and manhole_res.get("_samples"):
        for k in manhole_res["_samples"][0]:
            cols[f"probe_{k}"] = [x[k] for x in manhole_res["_samples"]]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["resample", *cols])
        for i in range(n):
            wr.writerow([i, *(repr(float(cols[c][i])) for c in cols)])


# --------------------------------------------------------------------------- dataset io
def list_images(split_entry):
    from ultralytics.data.utils import IMG_FORMATS
    entries = split_entry if isinstance(split_entry, (list, tuple)) else [split_entry]
    files = []
    for e in entries:
        p = Path(e)
        if p.is_dir():
            files += [q for q in p.rglob("*") if q.suffix[1:].lower() in IMG_FORMATS]
        elif p.is_file():
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                q = Path(ln.replace("./", str(p.parent) + os.sep, 1)) if ln.startswith("./") else Path(ln)
                if q.suffix[1:].lower() in IMG_FORMATS:
                    files.append(q)
        else:
            raise FileNotFoundError(e)
    return sorted(files)


def load_gt(files):
    from PIL import Image
    from ultralytics.data.utils import exif_size, img2label_paths
    gts, sizes = {}, {}
    for f, lf in zip(files, img2label_paths([str(f) for f in files])):
        with Image.open(f) as im:
            w, h = exif_size(im)
        sizes[f.stem] = (w, h)
        boxes, fr = [], []
        if os.path.exists(lf):
            for ln in Path(lf).read_text().splitlines():
                t = ln.split()
                if len(t) >= 5:
                    cx, cy, bw, bh = (float(v) for v in t[1:5])
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                    fr.append(np.sqrt(bw * w * bh * h) / w)
        gts[f.stem] = (np.array(boxes).reshape(-1, 4), np.array(fr))
    return gts, sizes


def load_preds(pred_json, images_set):
    data = json.loads(Path(pred_json).read_text())
    acc = {}
    for d in data:
        iid = str(d["image_id"])
        x, y, w, h = d["bbox"]
        acc.setdefault(iid, ([], []))
        acc[iid][0].append([x, y, x + w, y + h]); acc[iid][1].append(d["score"])
    unknown = set(acc) - images_set
    if unknown:
        raise RuntimeError(f"{len(unknown)} predicted image ids not in image list, e.g. {sorted(unknown)[:3]}")
    return {k: (np.array(b).reshape(-1, 4), np.array(c)) for k, (b, c) in acc.items()}


def read_map_csv(path, keycol, valcol):
    with open(path, newline="") as f:
        return {Path(r[keycol]).stem: r[valcol] for r in csv.DictReader(f)}


def keyed(mapping, stems, default="UNKNOWN"):
    return [mapping.get(s, default) for s in stems]


def breakdown_by_key(P, keys, thr):
    res = {}
    keys = np.asarray(keys)
    for k in sorted(set(keys.tolist())):
        m = op_metrics(P, thr, (keys == k).astype(float))
        res[k] = {x: m[x] for x in ("n_images", "n_gt", "TP", "FP", "FN", "precision", "recall", "f1",
                                    "fp_per_image", "fp_per_negative_image")}
    return res


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=["select", "apply"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=EVAL_BATCH,
                    help="fixed at 8 for all reported runs (rect padding depends on batch); other values are refused")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--target-recall", type=float, default=0.80)
    ap.add_argument("--operating-points", help="operating_points.json from a select run (apply mode)")
    ap.add_argument("--explicit-conf", type=float,
                    help="apply mode, instead of --operating-points: evaluate at this one confidence, stored as threshold "
                         f"'{EXPLICIT_NAME}' (used for matched-recall comparisons; the caller is responsible for having "
                         "chosen it on validation data, which is not checked)")
    ap.add_argument("--groups", help="CSV image,group for the bootstrap")
    ap.add_argument("--breakdown-csv", help="CSV image,key for the per-key breakdown")
    ap.add_argument("--rdd-manifest", help="manifest.csv with source_id, group_id, country (sets groups + breakdown)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--manhole-boxes", help="CSV image,cx,cy,w,h (normalised intact-manhole boxes): adds the manhole-probe "
                                            "metrics at --primary-threshold; needs --groups")
    ap.add_argument("--primary-threshold", default="max_f1", help="name of the threshold used for the manhole probe")
    ap.add_argument("--allow-in-sample", action="store_true",
                    help="apply mode: permit thresholds selected on the same/overlapping images; the result is "
                         "then flagged in_sample=true")
    a = ap.parse_args(argv)
    if a.batch != EVAL_BATCH:
        sys.exit(f"--batch must be {EVAL_BATCH} (fixed for comparability)")
    if a.mode == "apply" and not a.operating_points and a.explicit_conf is None:
        sys.exit("--operating-points (or --explicit-conf) is required in apply mode")
    if a.explicit_conf is not None and (a.mode != "apply" or a.operating_points):
        sys.exit("--explicit-conf is only valid in apply mode and excludes --operating-points")
    if a.explicit_conf is not None and not 0.0 < a.explicit_conf < 1.0:
        sys.exit("--explicit-conf must be in (0, 1)")

    out = Path(a.out).resolve()
    for bad in ("runs/train", "model_archive"):
        if (Path.cwd() / bad).resolve() in [out, *out.parents]:
            sys.exit(f"refusing to write inside {bad}")
    out.parent.mkdir(parents=True, exist_ok=True)

    import torch, ultralytics
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    ckpt = Path(a.checkpoint).resolve(); data = Path(a.data).resolve()
    ck_sha = sha256(ckpt)
    ds = check_det_dataset(str(data), autodownload=False)
    if int(ds["nc"]) != 1:
        sys.exit("single-class datasets only")
    files = list_images(ds[a.split])
    stems = [f.stem for f in files]
    if len(set(stems)) != len(stems):
        sys.exit("duplicate image stems; predictions.json ids would collide")
    images_hash = hashlib.sha256("\n".join(sorted(str(f) for f in files)).encode()).hexdigest()

    model = YOLO(str(ckpt))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    m = model.val(data=str(data), split=a.split, imgsz=a.imgsz, batch=a.batch, conf=a.conf, iou=a.iou,
                  device=a.device, plots=not a.no_plots, save_json=True, project=str(out.parent),
                  name=out.name, exist_ok=True, workers=2, verbose=False, seed=0, deterministic=True)
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    pred_json = Path(m.save_dir) / "predictions.json"

    gts, sizes = load_gt(files)
    preds = load_preds(pred_json, set(stems))
    P = build_preds(stems, gts, preds)

    # grouping / breakdown
    groups_src, key_src = None, None
    gmap, kmap = {}, {}
    if a.rdd_manifest:
        gmap = read_map_csv(a.rdd_manifest, "source_id", "group_id"); kmap = read_map_csv(a.rdd_manifest, "source_id", "country")
        groups_src = key_src = str(Path(a.rdd_manifest).resolve())
    if a.groups:
        gmap = read_map_csv(a.groups, "image", "group"); groups_src = str(Path(a.groups).resolve())
    if a.breakdown_csv:
        kmap = read_map_csv(a.breakdown_csv, "image", "key"); key_src = str(Path(a.breakdown_csv).resolve())
    if gmap:
        miss = [s for s in stems if s not in gmap]
        if miss:
            sys.exit(f"{len(miss)} images without a group, e.g. {miss[:3]}")
        group_of = keyed(gmap, stems); boot_level = "group"
    else:
        group_of = list(range(len(stems))); boot_level = "image (NO group info supplied: CIs are optimistic)"

    # own AP vs Ultralytics
    ap_own = ap_all(P)
    own = dict(mAP50=float(ap_own[0]), mAP50_95=float(ap_own.mean()))
    uly = dict(mAP50=float(m.box.map50), mAP50_95=float(m.box.map))
    agreement = dict(own=own, ultralytics=uly, abs_diff_mAP50=abs(own["mAP50"] - uly["mAP50"]),
                     abs_diff_mAP50_95=abs(own["mAP50_95"] - uly["mAP50_95"]),
                     tolerance_mAP50=0.005, mAP50_within_tolerance=abs(own["mAP50"] - uly["mAP50"]) <= 0.005)

    # thresholds
    sw = sweep(P)
    if a.mode == "select":
        sel = select_thresholds(sw, a.target_recall)
        thresholds = {"max_f1": sel["max_f1_conf"]}
        if sel["target_recall_conf"] is not None:
            thresholds[f"recall_{a.target_recall:g}"] = sel["target_recall_conf"]
        op = dict(selected_on=dict(split=a.split, data=str(data), images_sha256=images_hash, n_images=len(stems),
                                   image_stems=sorted(stems)),
                  rule="conf = max-F1 point of the sweep on the selection image list at this imgsz (IoU 0.5 match)",
                  checkpoint_sha256=ck_sha, imgsz=a.imgsz, nms_iou=a.iou, match_iou=0.5, **sel,
                  thresholds=thresholds, sweep=sw)
        (out / "operating_points.json").write_text(json.dumps(op, indent=2))
        thr_src = f"selected on this split ({a.split}); in-sample"
        in_sample = True; provenance = dict(relation="selected_on_this_list", in_sample=True)
    elif a.explicit_conf is not None:
        thresholds = {EXPLICIT_NAME: float(a.explicit_conf)}
        thr_src = f"explicit confidence {a.explicit_conf:.6f} supplied on the command line (selection not verified here)"
        in_sample = False
        provenance = dict(relation="explicit_conf", in_sample=False, explicit_conf=float(a.explicit_conf),
                          note="threshold chosen by the caller; no overlap check against any selection list")
    else:
        op_in = json.loads(Path(a.operating_points).read_text())
        if op_in["checkpoint_sha256"] != ck_sha:
            sys.exit("operating points were selected for a different checkpoint")
        if op_in["imgsz"] != a.imgsz:
            sys.exit("operating points were selected at a different imgsz")
        thresholds = {k: v for k, v in op_in["thresholds"].items() if v is not None}
        thr_src = f"{Path(a.operating_points).resolve()} (selected on {op_in['selected_on']['split']}, " \
                  f"images_sha256 {op_in['selected_on']['images_sha256'][:12]})"
        sel_on = op_in["selected_on"]
        same = sel_on["images_sha256"] == images_hash
        overlap = sorted(set(sel_on["image_stems"]) & set(stems)) if "image_stems" in sel_on else None
        in_sample = bool(same or overlap)
        if in_sample and not a.allow_in_sample:
            sys.exit("thresholds were selected on this very image set or on overlapping images "
                     f"({'identical list' if same else str(len(overlap)) + ' shared images'}); refusing to apply. "
                     "Pass --allow-in-sample to get a result flagged in_sample=true")
        if same:
            relation = "same_list"
        elif overlap:
            relation = "overlapping_lists"
        elif overlap is None:
            relation = "different_list_overlap_unchecked (operating points predate image_stems)"
        else:
            relation = "disjoint_list (e.g. thresholds selected on pooled val, applied to a per-source test list)"
        provenance = dict(relation=relation, in_sample=in_sample, selected_on_split=sel_on["split"],
                          selected_on_data=sel_on["data"], selected_on_n_images=sel_on["n_images"],
                          n_overlapping_images=None if overlap is None else len(overlap),
                          checkpoint_sha256_matches=True, imgsz_matches=True)
        print(f"[eval] thresholds selected on {sel_on['data']} ({sel_on['split']}, {sel_on['n_images']} images); "
              f"applying to {data} ({a.split}, {len(stems)} images): {relation}; in_sample={in_sample}", file=sys.stderr)

    ci = group_bootstrap(P, group_of, thresholds)
    op_res = {n: dict(op_metrics(P, t), ci95={k: v for k, v in ci["operating_point"][n].items()})
              for n, t in thresholds.items()}
    primary = "max_f1" if "max_f1" in thresholds else next(iter(thresholds))
    by_key = {n: breakdown_by_key(P, keyed(kmap, stems), t) for n, t in thresholds.items()} if kmap else None
    by_size = {n: size_breakdown(P, t) for n, t in thresholds.items()}

    manhole_res, easy, probe_samples = None, None, None
    if a.manhole_boxes:
        if not a.groups:
            sys.exit("--manhole-boxes needs --groups (bootstrap by group)")
        pname = a.primary_threshold if a.primary_threshold in thresholds else (EXPLICIT_NAME if a.explicit_conf is not None else a.primary_threshold)
        if pname not in thresholds:
            sys.exit(f"threshold {a.primary_threshold} not available")
        thr0 = thresholds[pname]
        with open(a.manhole_boxes, newline="") as f:
            mrows = [(Path(r["image"]).stem, float(r["cx"]), float(r["cy"]), float(r["w"]), float(r["h"]))
                     for r in csv.DictReader(f)]
        manhole_res = manhole_probe(stems, sizes, preds, P.gt_count, mrows, thr0, group_of)
        manhole_res["manhole_boxes_source"] = str(Path(a.manhole_boxes).resolve())
        probe_samples = manhole_res.pop("_samples")
        pos = (P.gt_count > 0).astype(float)
        e = op_metrics(P, thr0, pos)
        easy = dict(note="EASY SUBSET (frames with reviewed pothole boxes): NOT a recall estimate for real driving",
                    conf_threshold=thr0, n_images=e["n_images"], n_gt=e["n_gt"], TP=e["TP"], FP=e["FP"], FN=e["FN"],
                    precision=e["precision"], recall=e["recall"],
                    ci95=group_bootstrap_op(P, group_of, thr0, pos))

    write_bootstrap_samples(out / "bootstrap_samples.csv", ci, dict(_samples=probe_samples) if probe_samples else None)
    img_group = sorted(zip(stems, map(str, group_of)))
    with open(out / "bootstrap_groups.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["image", "group"])
        wr.writerows(img_group)
    metrics = dict(
        mode=a.mode, split=a.split, n_images=len(stems), n_instances=int(P.gt_count.sum()),
        n_negative_images=int((P.gt_count == 0).sum()),
        threshold_free=dict(mAP50=uly["mAP50"], mAP50_95=uly["mAP50_95"], source="Ultralytics val(), conf %g NMS IoU %g" % (a.conf, a.iou),
                            mAP50_ci95=ci["mAP50"], mAP50_95_ci95_own=ci["mAP50_95"],
                            note="CIs are around the own recomputation; see ap_agreement"),
        ap_agreement=agreement,
        operating_points=op_res, primary_operating_point=primary, thresholds_source=thr_src,
        threshold_provenance=provenance, in_sample=in_sample, manhole_probe=manhole_res, easy_subset_positive_frames=easy,
        bootstrap=dict(n_boot=ci["n_boot"], seed=ci["seed"], level=boot_level, n_groups=ci["n_groups"], groups_source=groups_src,
                       method=ci["method"]),
        breakdown_by_key=by_key, breakdown_key_source=key_src, breakdown_by_gt_size=dict(
            note="GT recall per bin of sqrt(box area)/image width, at the operating threshold", bins=by_size),
        p_r_at_split_maxF1_DO_NOT_QUOTE_FOR_TEST=dict(precision=float(m.box.mp), recall=float(m.box.mr),
                                                      note="Ultralytics mp/mr at max-F1 conf of THIS split (tuned on it)"),
        speed_ms_per_img=dict(m.speed), peak_gpu_mem_allocated_mib=peak,
        note_peak_mem="torch.cuda.max_memory_allocated in this process; excludes CUDA context")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cfg = dict(checkpoint=str(ckpt), checkpoint_sha256=ck_sha, dataset_yaml=str(data), dataset_yaml_sha256=sha256(data),
               image_list_sha256=images_hash, n_images=len(stems), split=a.split, mode=a.mode, imgsz=a.imgsz,
               batch=a.batch, conf=a.conf, nms_iou=a.iou, iou=a.iou, device=a.device,
               boot_seed=ci["seed"], boot_n=ci["n_boot"], boot_groups_sha256=groups_digest(group_of),
               boot_image_group_sha256=hashlib.sha256("\n".join(f"{i}\t{g}" for i, g in img_group).encode()).hexdigest(),
               boot_samples_file="bootstrap_samples.csv", target_recall=a.target_recall,
               thresholds_source=thr_src, in_sample=in_sample, allow_in_sample=a.allow_in_sample, explicit_conf=a.explicit_conf, thresholds=thresholds, groups_source=groups_src, breakdown_source=key_src,
               predictions_json=str(pred_json), ultralytics=ultralytics.__version__, torch=torch.__version__,
               numpy=np.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               date=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
    (out / "eval_config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({k: metrics[k] for k in ("threshold_free", "ap_agreement", "operating_points", "thresholds_source")}, indent=2))


if __name__ == "__main__":
    main()
