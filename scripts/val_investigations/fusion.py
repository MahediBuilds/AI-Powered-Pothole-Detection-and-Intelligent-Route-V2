"""Box fusion for the ensemble investigation: Weighted Boxes Fusion (``ensemble-boxes``) and a plain
concatenate-then-NMS comparator. Pure functions on numpy arrays (no data access).

Boxes are pixel xyxy; ``size`` = (width, height) of the image. WBF works on [0, 1] coordinates, so boxes are
normalised, clipped to [0, 1] and mapped back.
"""
from __future__ import annotations

import numpy as np

WBF_CONF_TYPES = ("avg", "max", "box_and_model_avg")


def _norm(boxes, size):
    w, h = size
    b = np.asarray(boxes, float).reshape(-1, 4) / np.array([w, h, w, h], float)
    return np.clip(b, 0.0, 1.0)


def wbf_fuse(boxes_a, conf_a, boxes_b, conf_b, size, iou_thr, skip_box_thr, weights=(1, 1), conf_type="avg"):
    """Fuse two models' detections for one image with Weighted Boxes Fusion.

    Returns (boxes xyxy pixels (n,4), conf (n,)) sorted by descending confidence."""
    from ensemble_boxes import weighted_boxes_fusion
    if conf_type not in WBF_CONF_TYPES:
        raise ValueError(conf_type)
    na, nb = len(conf_a), len(conf_b)
    if na + nb == 0:
        return np.zeros((0, 4)), np.zeros(0)
    bl = [_norm(boxes_a, size).tolist(), _norm(boxes_b, size).tolist()]
    sl = [np.asarray(conf_a, float).tolist(), np.asarray(conf_b, float).tolist()]
    ll = [[0] * na, [0] * nb]
    b, s, _ = weighted_boxes_fusion(bl, sl, ll, weights=list(weights), iou_thr=iou_thr, skip_box_thr=skip_box_thr,
                                    conf_type=conf_type)
    b = np.asarray(b, float).reshape(-1, 4) * np.array([size[0], size[1], size[0], size[1]], float)
    s = np.asarray(s, float)
    o = np.argsort(-s, kind="stable")
    return b[o], s[o]


def nms(boxes, conf, iou_thr):
    """Greedy NMS (by descending confidence; suppress IoU > iou_thr, so boxes at exactly iou_thr survive,
    as in torchvision). Returns kept indices in descending-confidence order."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    conf = np.asarray(conf, float)
    order = np.argsort(-conf, kind="stable")
    keep = []
    suppressed = np.zeros(len(order), bool)
    b = boxes[order]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(order[i])
        x1 = np.maximum(b[i, 0], b[i + 1:, 0]); y1 = np.maximum(b[i, 1], b[i + 1:, 1])
        x2 = np.minimum(b[i, 2], b[i + 1:, 2]); y2 = np.minimum(b[i, 3], b[i + 1:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / np.maximum(area[i] + area[i + 1:] - inter, 1e-12)
        suppressed[i + 1:] |= iou > iou_thr
    return np.array(keep, int)


def concat_nms(boxes_a, conf_a, boxes_b, conf_b, iou_thr):
    """Pool both models' detections and run NMS. Scores unchanged."""
    boxes = np.concatenate([np.asarray(boxes_a, float).reshape(-1, 4), np.asarray(boxes_b, float).reshape(-1, 4)])
    conf = np.concatenate([np.asarray(conf_a, float), np.asarray(conf_b, float)])
    if len(conf) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    k = nms(boxes, conf, iou_thr)
    return boxes[k], conf[k]
