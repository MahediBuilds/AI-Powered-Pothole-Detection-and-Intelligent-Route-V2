"""Mechanical split of false positives (Investigation 1). Pure numpy.

Detections at or above the threshold are matched greedily by descending confidence, one detection per GT
box, IoU >= 0.5 (same matcher as scripts/eval_checkpoint.py). Each unmatched detection (a false positive) is then
assigned by its largest IoU with ANY ground-truth box in that image:
  duplicate  max IoU >= 0.5 (the GT it overlaps was already matched by a more confident detection)
  loc        0.1 <= max IoU < 0.5 (localisation error)
  bg         max IoU < 0.1 (no meaningful overlap; background false positive; also all FPs of empty images)
"""
from __future__ import annotations

import numpy as np

from common import ec


def mechanical_split(pred_boxes, pred_conf, gt_boxes, thr):
    """Returns a list of dicts for detections with conf >= thr, in descending-confidence order:
    {idx (index into the inputs), conf, cls in {tp, duplicate, loc, bg}, max_iou, gt_index (best-overlap GT or -1)}."""
    pb = np.asarray(pred_boxes, float).reshape(-1, 4)
    pc = np.asarray(pred_conf, float).reshape(-1)
    gb = np.asarray(gt_boxes, float).reshape(-1, 4)
    keep = np.where(pc >= thr)[0]
    if len(keep) == 0:
        return []
    order, tp, _ = ec.match_image(pb[keep], pc[keep], gb, np.array([0.5]))
    ious = ec._iou_matrix(pb[keep][order], gb)
    out = []
    for r, o in enumerate(order):
        mi = float(ious[r].max()) if len(gb) else 0.0
        gi = int(ious[r].argmax()) if len(gb) else -1
        if tp[r, 0]:
            cls = "tp"
        else:
            cls = "duplicate" if mi >= 0.5 else ("loc" if mi >= 0.1 else "bg")
        out.append(dict(idx=int(keep[o]), conf=float(pc[keep[o]]), cls=cls, max_iou=mi, gt_index=gi))
    return out
