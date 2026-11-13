"""Qualitative per-frame detection on the owner's recordings. NO ground truth exists: nothing here is precision or recall.

    detect:   python scripts/compare_on_recordings.py detect --checkpoint W.pt --imgsz 640 --conf 0.27 --tag baseline_640
    contact:  python scripts/compare_on_recordings.py contact --tags baseline_640 baseline_1280

detect runs plain Ultralytics predict on every decoded frame of every video under --recordings (default
data/recordings; no tracker, pipeline or database) and writes runs/eval/recordings/<tag>/<video>.json (per-frame
boxes xyxy in source pixels + scores) and summary.csv (frames, frames with >=1 detection, total detections,
detections per 100 frames, mean score).

contact renders, per video, a sheet of up to 12 frames where only ONE of the two tags detects anything, ordered
by that detection's top score (descending), model A's boxes in green and B's in orange, with a legend, so a human
can judge which model is right. Output: runs/eval/recordings/contact_<A>__<B>/<video>.jpg
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "runs" / "eval" / "recordings"
COL_A, COL_B = (60, 200, 60), (0, 140, 255)  # BGR: green, orange


def find_videos(rec_dir):
    return sorted(p for p in Path(rec_dir).rglob("*.mp4") if "_contact" not in p.parts)


def disagreement_frames(frames_a, frames_b, k=12):
    """frames_*: list of per-frame score lists (same length). Frames where exactly one model has a detection,
    ordered by that model's top score descending (ties: earlier frame). Returns [(frame_idx, 'A'|'B', top_score)]."""
    out = []
    for i, (a, b) in enumerate(zip(frames_a, frames_b)):
        if bool(a) != bool(b):
            out.append((i, "A" if a else "B", max(a or b)))
    out.sort(key=lambda t: (-t[2], t[0]))
    return out[:k]


def _check_out(p):
    p = p.resolve()
    if OUT_ROOT.resolve() not in [p, *p.parents]:
        sys.exit(f"refusing to write outside {OUT_ROOT}")


def cmd_detect(a):
    import cv2
    import torch
    from ultralytics import YOLO
    out = OUT_ROOT / a.tag; _check_out(out); out.mkdir(parents=True, exist_ok=True)
    ck = Path(a.checkpoint).resolve()
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    model = YOLO(str(ck))
    rows = []
    for v in find_videos(a.recordings):
        cap = cv2.VideoCapture(str(v))
        frames, buf, idx = [], [], 0

        def flush():
            for r in model.predict(buf, imgsz=a.imgsz, conf=a.conf, iou=a.iou, device=a.device, verbose=False, half=torch.cuda.is_available()):
                frames.append(dict(boxes=[[round(float(x), 1) for x in b] for b in r.boxes.xyxy.cpu().tolist()],
                                   scores=[round(float(s), 4) for s in r.boxes.conf.cpu().tolist()]))
            buf.clear()
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            buf.append(fr); idx += 1
            if len(buf) == a.batch:
                flush()
        if buf:
            flush()
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
        for i, f in enumerate(frames):
            f["frame"] = i
        doc = dict(video=v.stem, video_path=str(v), width=w, height=h, n_frames=len(frames), checkpoint=str(ck), checkpoint_sha256=sha,
                   imgsz=a.imgsz, conf=a.conf, nms_iou=a.iou, note="plain per-frame detection; no ground truth; not precision/recall", frames=frames)
        (out / f"{v.stem}.json").write_text(json.dumps(doc))
        sc = [s for f in frames for s in f["scores"]]
        wd = sum(1 for f in frames if f["scores"])
        rows.append(dict(video=v.stem, frames=len(frames), frames_with_detection=wd, total_detections=len(sc),
                         detections_per_100_frames=round(100 * len(sc) / max(len(frames), 1), 2),
                         mean_score=round(sum(sc) / len(sc), 4) if sc else ""))
        print(rows[-1], flush=True)
    with open(out / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    (out / "run_config.json").write_text(json.dumps(dict(tag=a.tag, checkpoint=str(ck), checkpoint_sha256=sha, imgsz=a.imgsz, conf=a.conf,
                                                         nms_iou=a.iou, batch=a.batch, n_videos=len(rows)), indent=2))


def _draw(img, boxes, scores, color, scale):
    import cv2
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(img, p1, p2, color, 2)
        cv2.putText(img, f"{s:.2f}", (p1[0], max(p1[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def cmd_contact(a):
    import cv2
    import numpy as np
    ta, tb = a.tags
    da, db = OUT_ROOT / ta, OUT_ROOT / tb
    out = OUT_ROOT / f"contact_{ta}__{tb}"; _check_out(out); out.mkdir(parents=True, exist_ok=True)
    tw, th, cols = 480, 270, 4
    for fa in sorted(da.glob("*.json")):
        if fa.name == "run_config.json" or not (db / fa.name).exists():
            continue
        A, B = json.loads(fa.read_text()), json.loads((db / fa.name).read_text())
        if A["n_frames"] != B["n_frames"]:
            print(f"skip {fa.stem}: frame counts differ"); continue
        sel = disagreement_frames([f["scores"] for f in A["frames"]], [f["scores"] for f in B["frames"]], a.max_frames)
        if not sel:
            print(f"{fa.stem}: no frames where only one model detects"); continue
        cap = cv2.VideoCapture(A["video_path"]); tiles = []
        for i, who, top in sel:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read()
            if not ok:
                continue
            sc = tw / fr.shape[1]
            img = cv2.resize(fr, (tw, int(fr.shape[0] * sc)))
            _draw(img, A["frames"][i]["boxes"], A["frames"][i]["scores"], COL_A, sc)
            _draw(img, B["frames"][i]["boxes"], B["frames"][i]["scores"], COL_B, sc)
            cv2.rectangle(img, (0, img.shape[0] - 18), (tw, img.shape[0]), (0, 0, 0), -1)
            cv2.putText(img, f"frame {i}  only {ta if who == 'A' else tb}  top {top:.2f}", (4, img.shape[0] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.resize(img, (tw, th)))
        cap.release()
        if not tiles:
            continue
        while len(tiles) % cols:
            tiles.append(np.zeros((th, tw, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
        legend = np.zeros((34, grid.shape[1], 3), np.uint8)
        cv2.putText(legend, f"{fa.stem}   GREEN = A: {ta}   ORANGE = B: {tb}   (frames where only one model detects; "
                            f"qualitative, no ground truth)", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(legend, (grid.shape[1] - 120, 8), (grid.shape[1] - 100, 26), COL_A, -1)
        cv2.rectangle(legend, (grid.shape[1] - 60, 8), (grid.shape[1] - 40, 26), COL_B, -1)
        cv2.imwrite(str(out / f"{fa.stem}.jpg"), np.vstack([legend, grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{fa.stem}: {len(sel)} frames -> {out / (fa.stem + '.jpg')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("detect")
    d.add_argument("--checkpoint", required=True); d.add_argument("--imgsz", type=int, required=True)
    d.add_argument("--conf", type=float, required=True); d.add_argument("--tag", required=True)
    d.add_argument("--recordings", default=str(ROOT / "data" / "recordings")); d.add_argument("--iou", type=float, default=0.7)
    d.add_argument("--batch", type=int, default=8); d.add_argument("--device", default="0")
    c = sp.add_parser("contact")
    c.add_argument("--tags", nargs=2, required=True, metavar=("A", "B")); c.add_argument("--max-frames", type=int, default=12)
    a = ap.parse_args(argv)
    {"detect": cmd_detect, "contact": cmd_contact}[a.cmd](a)


if __name__ == "__main__":
    main()
