"""Three-model qualitative review of the 13 personal recordings (inference only; NO ground truth).

Models: OLD (the original project's YOLOv8s-seg checkpoint), BASELINE@640 and CANDIDATE@1280 (rebuild).
Baseline/candidate detections are REUSED from runs/inference/personal_recordings_comparison (read-only); only the old model is run here.

    python -B scripts/three_model_comparison.py init       # output dir, hashes + Exisiting Implementation snapshot (before)
    python -B scripts/three_model_comparison.py identify   # old_implementation/MODEL_IDENTIFICATION.json
    python -B scripts/three_model_comparison.py run        # old model on every frame of all 13 recordings
    python -B scripts/three_model_comparison.py compare    # summary.csv, frame_agreement.csv, box_overlap.csv, 3-panel videos, verify_outputs.csv
    python -B scripts/three_model_comparison.py timing     # separate timing-only pass of all three models
    python -B scripts/three_model_comparison.py sheets     # seeded review sheets + SELECTION_METHOD.json + index.csv
    python -B scripts/three_model_comparison.py cases      # hand-picked interesting_cases from scripts/three_model_interesting_cases.json
    python -B scripts/three_model_comparison.py finalize   # integrity after + comparison
    python -B scripts/three_model_comparison.py report     # REPORT.md generated from the CSV/JSON files

Nothing here measures precision/recall; counts do not show which model is more accurate.
Run with PYTHONDONTWRITEBYTECODE=1. Never run with the Exisiting Implementation folder as cwd; it is only read (checkpoint + source text).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent            # New Implementation
PROJECT = ROOT.parent
REC_DIR = ROOT / "data" / "recordings"
EXIST = ROOT / "runs" / "inference" / "personal_recordings_comparison"
OLD_ROOT = PROJECT.parent / "Old Project" / "Exisiting Implementation"   # PROJECT = New Project/
OLD_APP = OLD_ROOT / "AI-Powered-Pothole-Detection-and-Intelligent-Route"
OLD_CKPT = OLD_APP / "runs" / "pothole_seg" / "weights" / "best.pt"
OUT_BASE = PROJECT / "Three_Model_Comparison"
CASES_SPEC = Path(__file__).resolve().parent / "three_model_interesting_cases.json"

NMS_IOU_NEW = 0.7
SEED = 0
K = 6
IOU_AGREE = 0.3
MODELS = {  # order = left, middle, right
    "old": dict(label="OLD IMPLEMENTATION", short="OLD", imgsz=640, conf=0.35, nms=0.5, color=(200, 70, 200),
                path=OLD_CKPT, sha=None, desc="yolov8s-seg"),
    "baseline": dict(label="NEW BASELINE", short="BASE", imgsz=640, conf=0.2705, nms=0.7, color=(60, 200, 60),
                     path=ROOT / "runs/train/baseline/weights/best.pt",
                     sha="2aabdd131d1664089e06e3219fabcaed4cc96610fa4603ff165429f92696d798", desc="640"),
    "candidate": dict(label="NEW CANDIDATE", short="CAND", imgsz=1280, conf=0.245, nms=0.7, color=(0, 140, 255),
                      path=ROOT / "runs/train/unified_v1_y8s_1280/weights/epoch25.pt",
                      sha="0a1c19f2586b6b730e9c5bf2bdeff6fcb3ec8f4567d161f1aba0459d09a5adc3", desc="1280"),
}
KEYS = list(MODELS)
CATS = ["only_old", "only_baseline", "only_candidate", "old_baseline_not_candidate", "old_candidate_not_baseline",
        "baseline_candidate_not_old", "all_three", "spatial_disagreement"]
COMBO = {  # (old, base, cand) fired -> category
    (1, 0, 0): "only_old", (0, 1, 0): "only_baseline", (0, 0, 1): "only_candidate", (1, 1, 0): "old_baseline_not_candidate",
    (1, 0, 1): "old_candidate_not_baseline", (0, 1, 1): "baseline_candidate_not_old", (1, 1, 1): "all_three", (0, 0, 0): "none"}
PAIRS = [("old", "baseline"), ("old", "candidate"), ("baseline", "candidate")]


# ----------------------------------------------------------------------------- helpers
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def snapshot(root):
    """file count, total bytes, newest file/dir mtime and a digest of (relpath,size,mtime_ns) over all files and dirs."""
    root = str(root)
    pre = "\\\\?\\" + os.path.abspath(root)
    n_files = n_dirs = total = errors = 0
    newest_f = newest_d = 0
    newest_f_path = ""
    h = hashlib.sha256()
    for dp, dn, fn in os.walk(pre):
        dn.sort()
        try:
            dm = os.stat(dp).st_mtime_ns
            n_dirs += 1
            newest_d = max(newest_d, dm)
            h.update(f"D|{dp[len(pre):]}|{dm}\n".encode("utf-8", "replace"))
        except OSError:
            errors += 1
        for f in sorted(fn):
            p = os.path.join(dp, f)
            try:
                st = os.lstat(p)
            except OSError:
                errors += 1
                continue
            n_files += 1
            total += st.st_size
            if st.st_mtime_ns > newest_f:
                newest_f, newest_f_path = st.st_mtime_ns, p[len(pre):]
            h.update(f"F|{p[len(pre):]}|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8", "replace"))

    def ts(ns):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ns / 1e9)) + f".{ns % 10**9:09d}"
    return dict(root=root, file_count=n_files, dir_count=n_dirs, total_bytes=total, newest_file_mtime=ts(newest_f),
                newest_file_mtime_ns=newest_f, newest_file=newest_f_path, newest_dir_mtime=ts(newest_d), newest_dir_mtime_ns=newest_d,
                stat_errors=errors, digest_sha256=h.hexdigest())


def find_videos():
    return sorted((p for p in REC_DIR.rglob("*.mp4") if "_contact" not in p.parts), key=lambda p: p.stem)


def out_dir(create=False):
    d = OUT_BASE
    if create:
        n = 2
        while d.exists() and any(d.iterdir()):
            d = OUT_BASE.with_name(OUT_BASE.name + f"_v{n}")
            n += 1
        d.mkdir(parents=True, exist_ok=True)
        return d
    cands = sorted([p for p in OUT_BASE.parent.glob(OUT_BASE.name + "*") if p.is_dir()], key=lambda p: (len(p.name), p.name))
    return cands[-1]


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,avg_frame_rate,nb_read_packets,duration:stream_side_data=rotation", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    n, d = s["avg_frame_rate"].split("/")
    return dict(width=int(s["width"]), height=int(s["height"]), fps_frac=s["avg_frame_rate"], fps=int(n) / int(d),
                frames=int(s["nb_read_packets"]), duration=float(s["duration"]), rotation=s.get("side_data_list", [{}])[0].get("rotation", 0))


def start_enc(path, w, h, fps_frac):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-framerate", fps_frac, "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def dump(path, rows, fields=None):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def read_csv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_det(path):
    d = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            d.setdefault(int(r["frame"]), []).append(([float(r[c]) for c in ("x1", "y1", "x2", "y2")], float(r["conf"])))
    return d


def det_path(out, stem, k):
    return (out / "old_implementation" / "detections" / f"{stem}_old.csv") if k == "old" else (EXIST / "detections" / f"{stem}_{k}.csv")


def load_all(out, stem):
    return {k: read_det(det_path(out, stem, k)) for k in KEYS}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def boxes_overlap(da, db, thr=IOU_AGREE):
    return any(iou(a, b) >= thr for a, _ in da for b, _ in db)


def frame_category(dets, i):
    """returns (agreement category, spatial_disagreement flag) for frame i."""
    fired = tuple(int(i in dets[k]) for k in KEYS)
    cat = COMBO[fired]
    spatial = False
    ks = [k for k, f in zip(KEYS, fired) if f]
    if len(ks) >= 2:
        spatial = not any(boxes_overlap(dets[a][i], dets[b][i]) for a, b in PAIRS if a in ks and b in ks)
    return cat, spatial


FULLFRAME = 0.8  # box covering >= 80 % of the frame area


def drop_fullframe(d, w, h, thr=FULLFRAME):
    res = {}
    for i, L in d.items():
        kept = [(b, c) for b, c in L if (b[2] - b[0]) * (b[3] - b[1]) / (w * h) < thr]
        if kept:
            res[i] = kept
    return res


def count_fullframe(d, w, h, thr=FULLFRAME):
    return sum((b[2] - b[0]) * (b[3] - b[1]) / (w * h) >= thr for L in d.values() for b, _ in L)


def frame_labels(out, stem, n, dets=None):
    """category list per frame: list of (cat, spatial)."""
    dets = dets or load_all(out, stem)
    return dets, [frame_category(dets, i) for i in range(n)]


# ----------------------------------------------------------------------------- drawing
FONT = None


def draw_dets(img, boxes, confs, color, label="pothole", fs_min=0.5):
    import cv2
    h, w = img.shape[:2]
    lw = max(2, round(w / 400))
    fs = max(fs_min, w / 1280 * 0.6)
    th = max(1, round(fs * 2))
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, lw)
        txt = f"{label} {c:.2f}"
        (tw, tht), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        y0 = p1[1] - tht - bl - 2
        if y0 < 0:
            y0 = p1[1] + 2
        cv2.rectangle(img, (p1[0], y0), (p1[0] + tw + 4, y0 + tht + bl + 2), color, -1)
        cv2.putText(img, txt, (p1[0] + 2, y0 + tht + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th, cv2.LINE_AA)


def tag_strip(img, lines, color, fs_min=0.55):
    """black strip with coloured text lines drawn over the top of the image."""
    import cv2
    w = img.shape[1]
    fs = max(fs_min, w / 1280 * 0.7)
    th = max(1, round(fs * 2))
    hs = []
    for ln in lines:
        (tw, tht), bl = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        hs.append(tht + bl + 8)
    cv2.rectangle(img, (0, 0), (w, sum(hs) + 4), (0, 0, 0), -1)
    y = 0
    for ln, hh in zip(lines, hs):
        cv2.putText(img, ln, (8, y + hh - 6), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)
        y += hh


def make_strip(w, lines, color, fs_min=0.55):
    """black strip (placed ABOVE the picture so it never hides a box label) with coloured text lines.
    Line height is taken from a fixed reference string so the strip height depends only on width and line count."""
    import cv2
    import numpy as np
    fs = max(fs_min, w / 1280 * 0.7)
    th = max(1, round(fs * 2))
    (_, tht), bl = cv2.getTextSize("Ag(|", cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    hh = tht + bl + 8
    st = np.zeros((hh * len(lines) + 4, w, 3), np.uint8)
    for j, ln in enumerate(lines):
        cv2.putText(st, ln, (8, j * hh + hh - 6), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)
    return st


def model_tag(k):
    m = MODELS[k]
    return f"{m['label']} | imgsz {m['imgsz']} | conf {m['conf']}"


def panel(frame, k, dets_k, i, width, rec_short=None, extra=None):
    import cv2
    sc = width / frame.shape[1]
    img = cv2.resize(frame, (width, int(round(frame.shape[0] * sc))), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
    d = dets_k.get(i, [])
    draw_dets(img, [[x * sc for x in b] for b, _ in d], [c for _, c in d], MODELS[k]["color"], fs_min=0.5)
    m = MODELS[k]
    l1 = f"{m['label']} ({m['imgsz']}, conf {m['conf']})"
    l2 = f"{rec_short or ''} f{i} | {len(d)} box{'es' if len(d) != 1 else ''}"
    if extra:
        l2 += " | " + extra
    import numpy as np
    return np.vstack([make_strip(width, [l1, l2], m["color"]), img])


# ----------------------------------------------------------------------------- init
def cmd_init(_a):
    out = out_dir(create=True)
    for sub in ("old_implementation", "old_implementation/detections", "review_sheets", "review_sheets/interesting_cases", "comparison",
                "comparison/side_by_side_3panel"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    vids = find_videos()
    before = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"),
                  exisiting_implementation_snapshot=snapshot(OLD_ROOT),
                  model_hashes={}, recordings={v.stem: sha256(v) for v in vids})
    before["model_hashes"]["old_best.pt"] = dict(path=str(OLD_CKPT), sha256=sha256(OLD_CKPT), size_bytes=OLD_CKPT.stat().st_size)
    for k in ("baseline", "candidate"):
        before["model_hashes"][k] = dict(path=str(MODELS[k]["path"]), sha256=sha256(MODELS[k]["path"]))
        if before["model_hashes"][k]["sha256"] != MODELS[k]["sha"]:
            sys.exit(f"hash mismatch for {k}")
    (out / "integrity.json").write_text(json.dumps(dict(out_dir=str(out), before=before), indent=2))
    print("out", out)
    print(json.dumps(before["exisiting_implementation_snapshot"], indent=1))
    print("old ckpt sha", before["model_hashes"]["old_best.pt"])


# ----------------------------------------------------------------------------- identify
def cmd_identify(_a):
    import torch
    from ultralytics import YOLO
    out = out_dir()
    net = YOLO(str(OLD_CKPT))
    ck = net.ckpt or {}
    ta = ck.get("train_args", {}) or {}
    other = []
    for p in sorted(OLD_APP.glob("runs/*/weights/*.pt")) + sorted(OLD_APP.glob("*.pt")) + sorted((OLD_APP / "models").glob("*.pt")):
        other.append(dict(path=str(p.relative_to(OLD_ROOT)).replace("\\", "/"), size_bytes=p.stat().st_size, sha256=sha256(p)))
    ident = {
        "conclusion": "The old application's trained model is runs/pothole_seg/weights/best.pt (YOLOv8s-seg, single class 'pothole').",
        "checkpoint": dict(path=str(OLD_CKPT), size_bytes=OLD_CKPT.stat().st_size, sha256=sha256(OLD_CKPT),
                           task=net.task, class_names={int(k): v for k, v in net.names.items()},
                           ckpt_ultralytics_version=ck.get("version"), ckpt_date=ck.get("date"), ckpt_epoch=ck.get("epoch"),
                           ckpt_train_args=dict(model=ta.get("model"), imgsz=ta.get("imgsz"), epochs=ta.get("epochs"), batch=ta.get("batch"),
                                                data=ta.get("data"), seed=ta.get("seed"), optimizer=ta.get("optimizer")),
                           loaded_with=dict(ultralytics=__import__("ultralytics").__version__, torch=torch.__version__)),
        "evidence_that_old_app_uses_it": [
            dict(file="AI-Powered-Pothole-Detection-and-Intelligent-Route/app.py", lines="72-78",
                 quote='found.append({"label": f"trained: {p.parent.parent.name}", "path": str(p)}) for p in sorted(ROOT.glob("runs/*/weights/best.pt"))',
                 note="The backend's model list puts runs/*/weights/best.pt first (sorted: pothole_seg before pothole_seg_smoke), then models/*.pt (the folder is empty), then COCO base weights last. The comment at lines 72-75 says the root *.pt files are untrained COCO base weights that will not find potholes."),
            dict(file="AI-Powered-Pothole-Detection-and-Intelligent-Route/ui/index.html", lines="232-233, 521-541, 566",
                 quote='<select id="modelSelect">; options filled from /api/models; form.append("model_path", modelSelect.value)',
                 note="No default is selected explicitly, so the first option (runs/pothole_seg/weights/best.pt) is what the UI submits."),
            dict(file="AI-Powered-Pothole-Detection-and-Intelligent-Route/README.md", lines="42, 80, 86, 240-248",
                 quote="Trained pothole model: put the checkpoint at `runs/pothole_seg/weights/best.pt` ... YOLOv8s-seg was trained at 640 px for 102 epochs ... best checkpoint is from epoch 72",
                 note="README names this exact path in setup, in the validation example and in the manual inference example."),
            dict(file="AI-Powered-Pothole-Detection-and-Intelligent-Route/tests/test_infer_seg.py", lines="17",
                 quote='WEIGHTS = ROOT / "runs" / "pothole_seg" / "weights" / "best.pt"', note="The end-to-end inference test loads the same file."),
            dict(file="AI-Powered-Pothole-Detection-and-Intelligent-Route/runs/pothole_seg/args.yaml", lines="2-3, 8",
                 quote="model: yolov8s-seg.pt; imgsz: 640; epochs: 150; project .../runs; name: pothole_seg", note="Training run that produced this checkpoint (stopped at epoch 102, patience 30)."),
        ],
        "other_checkpoints_found": other,
        "other_checkpoints_notes": [
            "runs/pothole_seg/weights/last.pt has the same size as best.pt but a different sha256 (best.pt is the epoch-72 checkpoint per README.md:241; last.pt is the final epoch-102 save); the app code names only best.pt.",
            "runs/pothole_seg_smoke/weights/*.pt is a 3-epoch smoke-test run (README step 7 `--smoke-test`); it appears second in the app's dropdown, so it can be chosen manually, but it is not the default.",
            "yolov8s-seg.pt, yolov8m-seg.pt, yolo26n.pt in the app root are COCO base weights (app.py:72-75, 83-84 list them last and label them 'not pothole-trained').",
            "models/ (target of scripts/13_finetune_local.py outputs) is empty; runs/faster_rcnn/pothole_faster_rcnn_smoke holds no weight file."],
        "inference_settings_in_app_pipeline": {
            "framework": "Ultralytics YOLO (requirements.txt:39 pins ultralytics==8.4.86, torch==2.12.1); task segment (instance segmentation)",
            "frames": dict(file="app.py:47, 117; utils/frames.py:30-60; ui/index.html:236-237", note="UI/default frame stride 5 (every 5th decoded frame saved as JPEG quality 95 via config.JPEG_QUALITY=95 and then read back)."),
            "imgsz": dict(file="app.py:150-158 and scripts/09_infer_seg.py:207", value=640, note="app.py passes no --imgsz so config.IMG_SIZE=640 (config.py:64) is used. Ultralytics letterboxes to 640 on the long side (rect, stride-32 padding) for 1280x720 frames."),
            "confidence": dict(file="app.py:48, ui/index.html:241-242, config.py:98", value=0.35, note="UI default 0.35; with --track the model is called with conf=min(conf,0.1)=0.1 (09_infer_seg.py:67, 100) and rows below 0.35 are dropped afterwards (145-146)."),
            "nms_iou": dict(file="config.py:99, 09_infer_seg.py:93, 209", value=0.5),
            "retina_masks": dict(file="09_infer_seg.py:94-98", value=True, note="masks returned at frame resolution."),
            "class_filter_or_mapping": "none; single class 0 = 'pothole' (config.py:43-44, data.yaml:4-5). Severity labels Mild/Moderate/Severe are added later by 11_classify_severity.py and do not change what is a detection.",
            "roi_or_cropping": "none found in app.py, 09_infer_seg.py, utils/frames.py",
            "color": "cv2.imread (BGR), no colour conversion beyond Ultralytics default",
            "mask_post_processing": dict(file="09_infer_seg.py:60-73, 148-159; config.py:100-101",
                                         note="each predicted mask is morphologically closed then opened with a 5x5 ellipse kernel; instances whose smoothed mask has fewer than 150 pixels (MIN_MASK_AREA_PX) or no contour are discarded. This changes what counts as a detection."),
            "tracking": dict(file="app.py:157 (--track); 09_infer_seg.py:99-101, 131-140, 198-199; 12_build_detections.py:19-24, 49-60",
                             note="ByteTrack; only detections belonging to a confirmed track are kept (a pothole must appear in at least two sampled frames, except in the first frame), and 12_build_detections.py collapses tracks to one record per pothole."),
            "temporal_smoothing_or_size_filters_beyond_the_above": "none found",
        },
        "reproduction_in_this_comparison": {
            "applied": ["imgsz 640", "conf 0.35", "NMS IoU 0.5", "retina_masks=True", "single class 'pothole' (class 0) counted as pothole",
                        "5x5 elliptical close+open mask smoothing and the 150 px minimum mask area filter (detections failing it are dropped and counted)"],
            "not_applied_deliberately": ["frame stride 5 (all frames processed so the three models are compared on identical frames)",
                                         "ByteTrack tracking, the 'confirmed track' requirement and the collapse to one record per pothole (per-frame detections wanted)",
                                         "JPEG q95 round trip of extracted frames (decoded video frames are fed directly)", "severity classification"],
            "differences_that_could_not_be_removed": ["Ultralytics version: this run uses the new venv (ultralytics 8.4.210, torch 2.14.1+cu126) while the old repo pinned 8.4.86 / torch 2.12.1; letterbox/NMS internals could differ slightly.",
                                                      "Timing of the old model includes retina-mask generation inside Ultralytics postprocess (and, separately, the 5x5 smoothing filter); the two new models are box-only detectors, so timing is not like-for-like."],
        },
        "documented_old_test_metrics_not_remeasured": "README.md:243-248 reports mask P 0.959 / R 0.907 and box P 0.936 / R 0.896 on the OLD project's own test split at conf 0.35; these are copied from its README, were not re-measured here and say nothing about the personal recordings.",
    }
    (out / "old_implementation" / "MODEL_IDENTIFICATION.json").write_text(json.dumps(ident, indent=2), encoding="utf-8")
    print(json.dumps(ident["checkpoint"], indent=1))
    print(json.dumps(other, indent=1))


# ----------------------------------------------------------------------------- run (old model)
def old_filter(r, w, h, cv2, np, ker):
    """Old pipeline post-processing: smooth each mask, drop if < 150 px or no contour. Returns kept list and dropped count."""
    kept, dropped = [], 0
    if r.masks is None:
        return kept, dropped
    for mask_t, box, conf, cls in zip(r.masks.data, r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
        if float(conf) < MODELS["old"]["conf"]:
            continue
        mask = mask_t.cpu().numpy().astype("uint8") * 255
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
        area = int(np.count_nonzero(mask))
        if area < 150:
            dropped += 1
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            dropped += 1
            continue
        poly = max(cnts, key=cv2.contourArea).reshape(-1, 2)
        kept.append((box.cpu().numpy(), float(conf), int(cls), area, poly))
    return kept, dropped


def cmd_run(_a):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    out = out_dir()
    integ = json.loads((out / "integrity.json").read_text())
    if sha256(OLD_CKPT) != integ["before"]["model_hashes"]["old_best.pt"]["sha256"]:
        sys.exit("old checkpoint hash changed")
    m = MODELS["old"]
    net = YOLO(str(OLD_CKPT))
    names = net.names
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cfg = dict(imgsz=m["imgsz"], conf=m["conf"], iou=m["nms"], retina_masks=True, augment=False, device=0, half=False, batch=1,
               classes="all (single class 'pothole')", min_mask_area_px=150, mask_smooth="5x5 ellipse close then open")

    def infer(frame):
        r = net.predict(frame, imgsz=m["imgsz"], conf=m["conf"], iou=m["nms"], device=0, retina_masks=True, verbose=False)[0]
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        kept, dropped = old_filter(r, w, h, cv2, np, ker)
        return kept, dropped, r.speed, (time.perf_counter() - t0) * 1000

    vids = find_videos()
    cap = cv2.VideoCapture(str(vids[0]))
    for _ in range(20):
        ok, fr = cap.read()
        if not ok:
            break
        infer(fr)
    cap.release()
    torch.cuda.synchronize()

    summary, verify, errors = [], [], []
    tot = dict(frames=0, dets=0, withdet=0, t_inf=0.0, t_all=0.0, t_filt=0.0, dur=0.0, dropped=0)
    allconf = []
    stats_extra = {}
    for v in vids:
        stem = v.stem
        print("==", stem, flush=True)
        enc = None
        try:
            pr = probe(v)
            if pr["rotation"]:
                raise RuntimeError(f"rotation {pr['rotation']} not handled")
            w, h = pr["width"], pr["height"]
            cap = cv2.VideoCapture(str(v))
            vf = out / "old_implementation" / f"{stem}_old640.mp4"
            enc = start_enc(vf, w, h, pr["fps_frac"])
            rows, t_inf, t_all, t_filt, confs = [], [], [], [], []
            dets_n = withdet = dropped_n = idx = 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                kept, dropped, sp, tf = infer(fr)
                t_inf.append(sp["inference"] + sp["postprocess"])
                t_all.append(sp["preprocess"] + sp["inference"] + sp["postprocess"])
                t_filt.append(tf)
                dropped_n += dropped
                dets_n += len(kept)
                withdet += len(kept) > 0
                img = fr.copy()
                for box, conf, cls, area, poly in kept:
                    rows.append([idx, f"{idx / pr['fps']:.4f}", *(f"{x:.1f}" for x in box), f"{conf:.4f}", cls, names[cls], area])
                    confs.append(conf)
                    cv2.polylines(img, [poly.astype("int32")], True, m["color"], 1, cv2.LINE_AA)
                draw_dets(img, [b for b, *_ in kept], [c for _, c, *_ in kept], m["color"], label="pothole")
                tag_strip(img, [f"{m['label']} | yolov8s-seg | imgsz {m['imgsz']} | conf {m['conf']} | NMS {m['nms']}"], m["color"], fs_min=0.6)
                enc.stdin.write(img.tobytes())
                idx += 1
            cap.release()
            enc.stdin.close()
            if enc.wait() != 0:
                raise RuntimeError("ffmpeg failed")
            enc = None
            n = idx
            with open(out / "old_implementation" / "detections" / f"{stem}_old.csv", "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["frame", "time_s", "x1", "y1", "x2", "y2", "conf", "class_id", "class_name", "mask_area_px"])
                wr.writerows(rows)
            summary.append(dict(recording=stem, model="old", weights=str(OLD_CKPT.relative_to(PROJECT)), imgsz=m["imgsz"], conf_threshold=m["conf"],
                                source_resolution=f"{w}x{h}", source_fps=round(pr["fps"], 3), source_duration_s=round(pr["duration"], 3),
                                frames_processed=n, total_detections=dets_n, avg_dets_per_frame=round(dets_n / n, 4),
                                pct_frames_with_detection=round(100 * withdet / n, 2), avg_forward_nms_ms=round(sum(t_inf) / n, 3),
                                avg_incl_preprocess_ms=round(sum(t_all) / n, 3), median_confidence=round(statistics.median(confs), 4) if confs else "",
                                output_file=str(vf.relative_to(out)), status="ok"))
            stats_extra[stem] = dict(dropped_by_min_mask_area=dropped_n, avg_postfilter_ms=round(sum(t_filt) / n, 3))
            tot["frames"] += n; tot["dets"] += dets_n; tot["withdet"] += withdet; tot["t_inf"] += sum(t_inf); tot["t_all"] += sum(t_all)
            tot["t_filt"] += sum(t_filt); tot["dur"] += pr["duration"]; tot["dropped"] += dropped_n
            allconf += confs
            o = probe(vf)
            issues = []
            if o["frames"] != n: issues.append(f"frames {o['frames']} vs {n}")
            if o["frames"] != pr["frames"]: issues.append(f"frames vs source packets {pr['frames']}")
            if abs(o["fps"] - pr["fps"]) > 0.01: issues.append("fps")
            if (o["width"], o["height"]) != (w, h): issues.append("res")
            if abs(o["duration"] - pr["duration"]) > 0.05: issues.append("duration")
            verify.append(dict(recording=stem, output="old", src_frames_ffprobe=pr["frames"], frames_decoded_by_opencv=n, out_frames=o["frames"],
                               src_fps=round(pr["fps"], 4), out_fps=round(o["fps"], 4), src_res=f"{w}x{h}", out_res=f"{o['width']}x{o['height']}",
                               src_duration_s=round(pr["duration"], 3), out_duration_s=round(o["duration"], 3),
                               mismatch="; ".join(issues) or "none", size_mb=round(vf.stat().st_size / 1e6, 2)))
            print(f"   frames={n} dets={dets_n} withdet={withdet} dropped_area={dropped_n}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors.append((stem, repr(e)))
            if enc:
                try:
                    enc.kill()
                except Exception:
                    pass
            summary.append(dict(recording=stem, model="old", status=f"ERROR: {e!r}"))
    n = max(tot["frames"], 1)
    summary.append(dict(recording="TOTAL", model="old", weights=str(OLD_CKPT.relative_to(PROJECT)), imgsz=m["imgsz"], conf_threshold=m["conf"],
                        source_duration_s=round(tot["dur"], 3), frames_processed=tot["frames"], total_detections=tot["dets"],
                        avg_dets_per_frame=round(tot["dets"] / n, 4), pct_frames_with_detection=round(100 * tot["withdet"] / n, 2),
                        avg_forward_nms_ms=round(tot["t_inf"] / n, 3), avg_incl_preprocess_ms=round(tot["t_all"] / n, 3),
                        median_confidence=round(statistics.median(allconf), 4) if allconf else "", status="ok"))
    stats_extra["TOTAL"] = dict(dropped_by_min_mask_area=tot["dropped"], avg_postfilter_ms=round(tot["t_filt"] / n, 3))
    dump(out / "old_implementation" / "old_summary.csv", summary)
    dump(out / "old_implementation" / "verify_old_outputs.csv", verify)
    (out / "old_implementation" / "old_run_config.json").write_text(json.dumps(dict(cfg=cfg, extra=stats_extra, errors=errors, torch=torch.__version__,
                                                                                    ultralytics=__import__("ultralytics").__version__, gpu=torch.cuda.get_device_name(0)), indent=2))
    print("done; errors:", errors, "totals", tot)


# ----------------------------------------------------------------------------- compare
def cmd_compare(_a):
    import cv2
    import numpy as np
    out = out_dir()
    cd = out / "comparison"
    old_s = read_csv(out / "old_implementation" / "old_summary.csv")
    ex_s = read_csv(EXIST / "summary.csv")
    extra = json.loads((out / "old_implementation" / "old_run_config.json").read_text())["extra"]
    fields = ["recording", "model", "weights", "imgsz", "conf_threshold", "source_resolution", "source_fps", "source_duration_s", "frames_processed",
              "total_detections", "avg_dets_per_frame", "pct_frames_with_detection", "avg_forward_nms_ms", "avg_incl_preprocess_ms",
              "median_confidence", "output_file", "status", "classes_counted_as_pothole", "dropped_by_min_mask_area", "avg_postfilter_ms", "boxes_ge_80pct_of_frame_area", "source_of_row"]
    rows = []
    stems = [r["recording"] for r in old_s if r["recording"] != "TOTAL"]
    by = {(r["recording"], r["model"]): r for r in old_s}
    for r in ex_s:
        by[(r["recording"], r["model"])] = r
    ffc = {}
    for stem in stems:
        w_, h_ = (int(x) for x in by[(stem, "old")]["source_resolution"].split("x"))
        for k in KEYS:
            ffc[(stem, k)] = count_fullframe(read_det(det_path(out, stem, k)), w_, h_)
    for k in KEYS:
        ffc[("TOTAL", k)] = sum(ffc[(st, k)] for st in stems)
    for stem in stems + ["TOTAL"]:
        for k in KEYS:
            r = dict(by.get((stem, k), dict(recording=stem, model=k, status="missing")))
            r["boxes_ge_80pct_of_frame_area"] = ffc[(stem, k)]
            if k == "old":
                r["classes_counted_as_pothole"] = "class 0 'pothole' (only class)"
                r["dropped_by_min_mask_area"] = extra.get(stem, {}).get("dropped_by_min_mask_area", "")
                r["avg_postfilter_ms"] = extra.get(stem, {}).get("avg_postfilter_ms", "")
                r["source_of_row"] = "old_implementation/old_summary.csv (this run)"
                r["weights"] = r.get("weights", "")
            else:
                r["classes_counted_as_pothole"] = "single class 'pothole'"
                r["source_of_row"] = "reused: runs/inference/personal_recordings_comparison/summary.csv"
                if r.get("output_file"):
                    r["output_file"] = "runs/inference/personal_recordings_comparison/" + r["output_file"].replace("\\", "/")
            r["weights"] = r.get("weights", "").replace("\\", "/")
            rows.append(r)
    dump(out / "summary.csv", rows, fields)

    # frame agreement + box overlap + 3-panel videos
    agree_rows, sens_rows, ov_acc = [], [], {p: {} for p in PAIRS}
    ex_diff = {r["recording"]: r for r in read_csv(EXIST / "comparison" / "frame_differences.csv")}
    verify = read_csv(out / "old_implementation" / "verify_old_outputs.csv")
    cat_index = {}
    notes = []
    for stem in stems:
        try:
            n = int(by[(stem, "old")]["frames_processed"])
            if int(by[(stem, "baseline")]["frames_processed"]) != n or int(by[(stem, "candidate")]["frames_processed"]) != n:
                notes.append(f"{stem}: frame count mismatch across models")
            dets, labs = frame_labels(out, stem, n)
            cnt = {c: 0 for c in COMBO.values()}
            for c, _s in labs:
                cnt[c] += 1
            spatial = sum(s for _, s in labs)
            row = dict(recording=stem, frames=n, none=cnt["none"], **{c: cnt[c] for c in CATS[:7]}, spatial_disagreement_frames=spatial)
            for a, b in PAIRS:
                fa = [i in dets[a] for i in range(n)]
                fb = [i in dets[b] for i in range(n)]
                agree = sum(x == y for x, y in zip(fa, fb))
                both = sum(x and y for x, y in zip(fa, fb))
                either = sum(x or y for x, y in zip(fa, fb))
                row[f"{a}_{b}_agree_frames"] = agree
                row[f"{a}_{b}_agree_pct"] = round(100 * agree / n, 2)
                row[f"{a}_{b}_both_fire"] = both
                row[f"{a}_{b}_either_fire"] = either
                row[f"{a}_{b}_jaccard_of_firing_frames"] = round(both / either, 4) if either else ""
                # box-level overlap on frames where both fire
                acc = ov_acc[(a, b)].setdefault(stem, dict(frames_both=0, boxes_a=0, boxes_b=0, a03=0, b03=0, a05=0, b05=0, best_a=[], best_b=[], frames_no_overlap=0))
                for i in range(n):
                    if i in dets[a] and i in dets[b]:
                        da, db = dets[a][i], dets[b][i]
                        acc["frames_both"] += 1
                        acc["boxes_a"] += len(da); acc["boxes_b"] += len(db)
                        ba = [max(iou(x, y) for y, _ in db) for x, _ in da]
                        bb = [max(iou(x, y) for x, _ in da) for y, _ in db]
                        acc["a03"] += sum(v >= 0.3 for v in ba); acc["b03"] += sum(v >= 0.3 for v in bb)
                        acc["a05"] += sum(v >= 0.5 for v in ba); acc["b05"] += sum(v >= 0.5 for v in bb)
                        acc["best_a"] += ba; acc["best_b"] += bb
                        acc["frames_no_overlap"] += not any(v >= 0.3 for v in ba)
            e = ex_diff.get(stem)
            if e:  # consistency of the reused baseline/candidate stats with the earlier two-model frame_differences.csv
                bo = cnt["baseline_candidate_not_old"] + cnt["all_three"]
                ob = cnt["only_baseline"] + cnt["old_baseline_not_candidate"]
                oc = cnt["only_candidate"] + cnt["old_candidate_not_baseline"]
                if (int(e["both"]), int(e["only_baseline"]), int(e["only_candidate"])) != (bo, ob, oc):
                    notes.append(f"{stem}: baseline/candidate fire-combinations differ from frame_differences.csv")
            agree_rows.append(row)
            # sensitivity: old model with near-full-frame boxes removed
            w0, h0 = (int(x) for x in by[(stem, "old")]["source_resolution"].split("x"))
            dx = dict(dets); dx["old"] = drop_fullframe(dets["old"], w0, h0)
            _, labx = frame_labels(out, stem, n, dx)
            cx = {c: 0 for c in COMBO.values()}
            for c, _s in labx:
                cx[c] += 1
            rx = dict(recording=stem, frames=n, old_frames_firing_primary=len(dets["old"]), old_frames_firing_excl_fullframe=len(dx["old"]),
                      none=cx["none"], **{c: cx[c] for c in CATS[:7]}, spatial_disagreement_frames=sum(s_ for _, s_ in labx))
            for a, b in PAIRS:
                if "old" in (a, b):
                    fa = [i in dx[a] for i in range(n)]; fb = [i in dx[b] for i in range(n)]
                    rx[f"{a}_{b}_agree_pct"] = round(100 * sum(x == y for x, y in zip(fa, fb)) / n, 2)
                    rx[f"{a}_{b}_both_fire"] = sum(x and y for x, y in zip(fa, fb))
            sens_rows.append(rx)
            cat_index[stem] = n
            # three-panel video
            pr = probe(next(REC_DIR.rglob(stem + ".mp4")))
            w, h = pr["width"], pr["height"]
            pw = min(w, 1280)
            ph = int(round(h * pw / w))
            tw, th = 3 * pw, ph + make_strip(pw, ["x", "y"], (0, 0, 0)).shape[0]
            sp = cd / "side_by_side_3panel" / f"{stem}_old_baseline_candidate.mp4"
            enc = start_enc(sp, tw, th, pr["fps_frac"])
            cap = cv2.VideoCapture(str(next(REC_DIR.rglob(stem + ".mp4"))))
            i = 0
            short = stem.replace("road_", "")
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                ps = [panel(fr, k, dets[k], i, pw, short) for k in KEYS]
                frame_out = np.hstack(ps)
                if frame_out.shape != (th, tw, 3):
                    raise RuntimeError(f"panel shape {frame_out.shape} != {(th, tw, 3)}")
                enc.stdin.write(frame_out.tobytes())
                i += 1
            cap.release()
            enc.stdin.close()
            if enc.wait() != 0:
                raise RuntimeError("ffmpeg failed")
            o = probe(sp)
            issues = []
            if o["frames"] != n: issues.append(f"frames {o['frames']} vs {n}")
            if o["frames"] != pr["frames"]: issues.append(f"frames vs source packets {pr['frames']}")
            if abs(o["fps"] - pr["fps"]) > 0.01: issues.append("fps")
            if (o["width"], o["height"]) != (tw, th): issues.append(f"res {o['width']}x{o['height']} vs {tw}x{th}")
            if tw > 3840: issues.append("width > 3840")
            if abs(o["duration"] - pr["duration"]) > 0.05: issues.append("duration")
            verify.append(dict(recording=stem, output="three_panel", src_frames_ffprobe=pr["frames"], frames_decoded_by_opencv=i, out_frames=o["frames"],
                               src_fps=round(pr["fps"], 4), out_fps=round(o["fps"], 4), src_res=f"{w}x{h}", out_res=f"{o['width']}x{o['height']}",
                               src_duration_s=round(pr["duration"], 3), out_duration_s=round(o["duration"], 3),
                               mismatch="; ".join(issues) or "none", size_mb=round(sp.stat().st_size / 1e6, 2)))
            print(stem, row, flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            notes.append(f"{stem}: {e!r}")
    # totals row for agreement
    tot = dict(recording="ALL", frames=sum(r["frames"] for r in agree_rows))
    for c in ["none"] + CATS[:7] + ["spatial_disagreement_frames"]:
        tot[c] = sum(r[c] for r in agree_rows)
    for a, b in PAIRS:
        ag = sum(r[f"{a}_{b}_agree_frames"] for r in agree_rows)
        bo = sum(r[f"{a}_{b}_both_fire"] for r in agree_rows)
        either = sum(r[f"{a}_{b}_either_fire"] for r in agree_rows)
        tot[f"{a}_{b}_either_fire"] = either
        tot[f"{a}_{b}_agree_frames"] = ag
        tot[f"{a}_{b}_agree_pct"] = round(100 * ag / tot["frames"], 2)
        tot[f"{a}_{b}_both_fire"] = bo
        tot[f"{a}_{b}_jaccard_of_firing_frames"] = round(bo / either, 4) if either else ""
    agree_rows.append(tot)
    dump(cd / "frame_agreement.csv", agree_rows)
    st_ = dict(recording="ALL", frames=sum(r["frames"] for r in sens_rows))
    for c in ["old_frames_firing_primary", "old_frames_firing_excl_fullframe", "none"] + CATS[:7] + ["spatial_disagreement_frames"]:
        st_[c] = sum(r[c] for r in sens_rows)
    for a, b in PAIRS:
        if "old" in (a, b):
            st_[f"{a}_{b}_both_fire"] = sum(r[f"{a}_{b}_both_fire"] for r in sens_rows)
            st_[f"{a}_{b}_agree_pct"] = round(100 * sum(round(r[f"{a}_{b}_agree_pct"] * r["frames"] / 100) for r in sens_rows) / st_["frames"], 2)
    sens_rows.append(st_)
    dump(cd / "frame_agreement_old_excl_fullframe_boxes.csv", sens_rows)
    # box overlap csv
    ovr = []
    for (a, b), per in ov_acc.items():
        agg = dict(frames_both=0, boxes_a=0, boxes_b=0, a03=0, b03=0, a05=0, b05=0, best_a=[], best_b=[], frames_no_overlap=0)
        for stem, acc in list(per.items()) + [("ALL", None)]:
            if acc is None:
                acc = agg
            else:
                for kk in agg:
                    agg[kk] = agg[kk] + acc[kk]
            ovr.append(dict(pair=f"{a} vs {b}", recording=stem, frames_both_fire=acc["frames_both"], boxes_a=acc["boxes_a"], boxes_b=acc["boxes_b"],
                            a_boxes_with_iou_ge_0_3=acc["a03"], b_boxes_with_iou_ge_0_3=acc["b03"], a_boxes_with_iou_ge_0_5=acc["a05"], b_boxes_with_iou_ge_0_5=acc["b05"],
                            pct_a_boxes_iou_ge_0_3=round(100 * acc["a03"] / acc["boxes_a"], 2) if acc["boxes_a"] else "",
                            pct_b_boxes_iou_ge_0_3=round(100 * acc["b03"] / acc["boxes_b"], 2) if acc["boxes_b"] else "",
                            median_best_iou_a=round(statistics.median(acc["best_a"]), 4) if acc["best_a"] else "",
                            median_best_iou_b=round(statistics.median(acc["best_b"]), 4) if acc["best_b"] else "",
                            frames_both_fire_but_no_box_pair_iou_ge_0_3=acc["frames_no_overlap"], a=a, b=b))
    dump(cd / "box_overlap.csv", ovr, ["pair", "recording", "frames_both_fire", "boxes_a", "boxes_b", "a_boxes_with_iou_ge_0_3", "b_boxes_with_iou_ge_0_3",
                                       "a_boxes_with_iou_ge_0_5", "b_boxes_with_iou_ge_0_5", "pct_a_boxes_iou_ge_0_3", "pct_b_boxes_iou_ge_0_3",
                                       "median_best_iou_a", "median_best_iou_b", "frames_both_fire_but_no_box_pair_iou_ge_0_3"])
    dump(out / "verify_outputs.csv", verify)
    (cd / "compare_notes.json").write_text(json.dumps(dict(notes=notes), indent=2))
    print("notes", notes)


# ----------------------------------------------------------------------------- timing-only pass
def cmd_timing(_a):
    import cv2
    import torch
    from ultralytics import YOLO
    out = out_dir()
    nets = {k: YOLO(str(MODELS[k]["path"])) for k in KEYS}
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    import numpy as np

    def run1(k, fr):
        m = MODELS[k]
        kw = dict(retina_masks=True) if k == "old" else {}
        r = nets[k].predict(fr, imgsz=m["imgsz"], conf=m["conf"], iou=m["nms"], device=0, verbose=False, **kw)[0]
        return r

    vids = find_videos()
    cap = cv2.VideoCapture(str(vids[0]))
    for _ in range(30):
        ok, fr = cap.read()
        for k in KEYS:
            run1(k, fr)
    cap.release()
    torch.cuda.synchronize()
    STEP = 32
    res = {k: dict(fwd=[], all=[], filt=[]) for k in KEYS}
    n = 0
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % STEP == 0:
                n += 1
                h, w = fr.shape[:2]
                for k in KEYS:
                    r = run1(k, fr)
                    sp = r.speed
                    res[k]["fwd"].append(sp["inference"] + sp["postprocess"])
                    res[k]["all"].append(sp["preprocess"] + sp["inference"] + sp["postprocess"])
                    if k == "old":
                        t0 = time.perf_counter()
                        old_filter(r, w, h, cv2, np, ker)
                        res[k]["filt"].append((time.perf_counter() - t0) * 1000)
            i += 1
        cap.release()
    rows = []
    for k in KEYS:
        m = MODELS[k]
        rows.append(dict(model=k, imgsz=m["imgsz"], frames_timed=n, mean_forward_nms_ms=round(statistics.mean(res[k]["fwd"]), 3),
                         median_forward_nms_ms=round(statistics.median(res[k]["fwd"]), 3),
                         mean_incl_preprocess_ms=round(statistics.mean(res[k]["all"]), 3), median_incl_preprocess_ms=round(statistics.median(res[k]["all"]), 3),
                         mean_old_postfilter_ms=round(statistics.mean(res[k]["filt"]), 3) if res[k]["filt"] else "",
                         note=("forward+postprocess includes retina-mask generation (segmentation head); box-only for the others" if k == "old" else "box detector")))
    dump(out / "comparison" / "timing_only_pass.csv", rows)
    (out / "comparison" / "timing_only_pass.json").write_text(json.dumps(dict(
        method=f"every {STEP}th decoded frame of each of the 13 recordings ({n} frames), all three models run on each frame in order old, baseline, candidate, batch 1, FP32, "
               "30-frame warm-up, Ultralytics speed counters (preprocess/inference/postprocess, CUDA-synchronised). Same GPU session for all three; one pass.",
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__), indent=2))
    print(rows)


# ----------------------------------------------------------------------------- sheets
def grab_frames(stem, want):
    import cv2
    got = {}
    cap = cv2.VideoCapture(str(next(REC_DIR.rglob(stem + ".mp4"))))
    i = 0
    want = set(want)
    top = max(want) if want else -1
    while i <= top:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = fr
        i += 1
    cap.release()
    return got


def select_frames(frames, stem, cat):
    if len(frames) <= K:
        return list(frames)
    rng = random.Random(f"{SEED}-{stem}-{cat}")
    step = len(frames) / K
    return [frames[rng.randrange(int(j * step), max(int((j + 1) * step), int(j * step) + 1))] for j in range(K)]


def cmd_sheets(_a):
    import cv2
    import numpy as np
    out = out_dir()
    sd = out / "review_sheets"
    per = 640
    summ = {r["recording"]: r for r in read_csv(out / "comparison" / "frame_agreement.csv")}
    idx_rows, counts = [], {}
    for stem, r in summ.items():
        if stem == "ALL":
            continue
        n = int(r["frames"])
        dets, labs = frame_labels(out, stem, n)
        cats = {c: [i for i, (cc, _s) in enumerate(labs) if cc == c] for c in CATS[:7]}
        cats["spatial_disagreement"] = [i for i, (_cc, s) in enumerate(labs) if s]
        sel = {c: select_frames(fr, stem, c) for c, fr in cats.items()}
        grabbed = grab_frames(stem, {i for fr in sel.values() for i in fr})
        short = stem.replace("road_", "")
        for c, fr in sel.items():
            if not fr:
                continue
            rows = []
            for i in fr:
                if i not in grabbed:
                    continue
                rows.append(np.hstack([panel(grabbed[i], k, dets[k], i, per, short) for k in KEYS]))
            hdr = np.zeros((40, rows[0].shape[1], 3), np.uint8)
            cv2.putText(hdr, f"{stem} | category: {c} ({len(cats[c])} of {n} frames in category, {len(fr)} shown) | LEFT old implementation, MIDDLE new baseline 640, RIGHT new candidate 1280 | no ground truth",
                        (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            (sd / c).mkdir(exist_ok=True)
            fn = sd / c / f"{stem}_{c}.jpg"
            cv2.imwrite(str(fn), np.vstack([hdr] + rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
            idx_rows.append(dict(sheet=str(fn.relative_to(out)).replace("\\", "/"), recording=stem, category=c, frames_in_category=len(cats[c]),
                                 frames_shown=len(rows), frame_numbers=";".join(str(i) for i in fr if i in grabbed)))
            counts[c] = counts.get(c, 0) + 1
        print(stem, {c: len(v) for c, v in cats.items()}, flush=True)
    dump(sd / "index.csv", idx_rows)
    (sd / "SELECTION_METHOD.json").write_text(json.dumps(dict(
        seed=SEED, frames_per_sheet_max=K,
        per_frame_firing_definition="a model 'fires' in a frame if it has at least one pothole detection (after that model's own confidence threshold and, for the old model, its 5x5 smoothing + 150 px minimum-mask-area filter) in that frame.",
        categories={"only_old": "old fires; baseline and candidate do not", "only_baseline": "baseline only", "only_candidate": "candidate only",
                    "old_baseline_not_candidate": "old and baseline fire, candidate does not", "old_candidate_not_baseline": "old and candidate fire, baseline does not",
                    "baseline_candidate_not_old": "baseline and candidate fire, old does not", "all_three": "all three fire",
                    "spatial_disagreement": f"at least two models fire in the frame and NO pair of firing models has any pair of boxes with IoU >= {IOU_AGREE} (their boxes do not overlap); independent of the agreement categories, so a frame can appear in both"},
        selection=("For each recording and category, take the sorted list of frame numbers (0-based decoded frame index) in that category. If it has <= 6 frames show all; otherwise split the list into 6 equal slices and pick one frame per slice with "
                   "random.Random(f'{seed}-{recording}-{category}') (seed 0), so the sample is seeded and spread across the clip. Same method as the earlier two-model sheets, except the generator is seeded per category instead of per recording."),
        sheet_layout="one row per selected frame, three panels of 640 px width: LEFT old implementation, MIDDLE new baseline (imgsz 640), RIGHT new candidate (imgsz 1280); each panel is labelled with model name, imgsz, confidence threshold, recording, frame number and box count; boxes carry confidence scores (all models are single-class 'pothole')."),
        indent=2))
    print(counts)


# ----------------------------------------------------------------------------- hand-picked cases
def cmd_cases(_a):
    import cv2
    import numpy as np
    out = out_dir()
    cd = out / "review_sheets" / "interesting_cases"
    spec = json.loads(CASES_SPEC.read_text(encoding="utf-8"))
    rows = []
    pw = 960
    cache = {}
    for n_case, c in enumerate(spec["cases"], 1):
        stem, fr_no = c["recording"], int(c["frame"])
        if stem not in cache:
            cache[stem] = load_all(out, stem)
        dets = cache[stem]
        g = grab_frames(stem, {fr_no})
        if fr_no not in g:
            print("missing frame", stem, fr_no)
            continue
        short = stem.replace("road_", "")
        img = np.hstack([panel(g[fr_no], k, dets[k], fr_no, pw, short) for k in KEYS])
        hdr = np.zeros((44, img.shape[1], 3), np.uint8)
        cv2.putText(hdr, f"HAND-PICKED, NOT REPRESENTATIVE | case {c['id']} | {c['kind']} | {stem} f{fr_no} | no ground truth", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        fn = cd / f"{c['id']}_{c['kind']}_{short}_f{fr_no}.jpg"
        cv2.imwrite(str(fn), np.vstack([hdr, img]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cat, sp = frame_category(dets, fr_no)
        rows.append(dict(file=fn.name, case_id=c["id"], recording=stem, frame=fr_no, kind=c["kind"],
                         fires_old=int(fr_no in dets["old"]), fires_baseline=int(fr_no in dets["baseline"]), fires_candidate=int(fr_no in dets["candidate"]),
                         agreement_category=cat, spatial_disagreement=int(sp), looks_unfavourable_to=c.get("unfavourable_to", ""),
                         looks_favourable_to=c.get("favourable_to", ""), why_chosen=c["why"]))
    dump(cd / "index.csv", rows)
    (cd / "README.txt").write_text("HAND-PICKED cases (manholes, drains, patches, repairs, possible potholes). Chosen by looking, starting from earlier notes and the seeded sheets. "
                                   "NOT a representative sample; includes cases that look unfavourable to each model. No ground truth; all remarks are visual impressions. See index.csv.\n", encoding="utf-8")
    print(len(rows), "cases")


# ----------------------------------------------------------------------------- finalize
def cmd_finalize(_a):
    out = out_dir()
    integ = json.loads((out / "integrity.json").read_text())
    b = integ["before"]
    vids = find_videos()
    after = dict(finished=time.strftime("%Y-%m-%d %H:%M:%S"), exisiting_implementation_snapshot=snapshot(OLD_ROOT),
                 model_hashes={"old_best.pt": dict(path=str(OLD_CKPT), sha256=sha256(OLD_CKPT), size_bytes=OLD_CKPT.stat().st_size)},
                 recordings={v.stem: sha256(v) for v in vids})
    for k in ("baseline", "candidate"):
        after["model_hashes"][k] = dict(path=str(MODELS[k]["path"]), sha256=sha256(MODELS[k]["path"]))
    integ["after"] = after
    s1, s2 = b["exisiting_implementation_snapshot"], after["exisiting_implementation_snapshot"]
    keys = ("file_count", "dir_count", "total_bytes", "newest_file_mtime_ns", "newest_dir_mtime_ns", "digest_sha256")
    integ["exisiting_implementation_unchanged"] = all(s1[k] == s2[k] for k in keys)
    integ["model_hashes_unchanged"] = all(b["model_hashes"][k]["sha256"] == after["model_hashes"][k]["sha256"] for k in b["model_hashes"])
    integ["recordings_unchanged"] = b["recordings"] == after["recordings"]
    integ["expected_hashes_match"] = (after["model_hashes"]["baseline"]["sha256"] == MODELS["baseline"]["sha"]
                                      and after["model_hashes"]["candidate"]["sha256"] == MODELS["candidate"]["sha"])
    (out / "integrity.json").write_text(json.dumps(integ, indent=2))
    print({k: integ[k] for k in ("exisiting_implementation_unchanged", "model_hashes_unchanged", "recordings_unchanged", "expected_hashes_match")})


# ----------------------------------------------------------------------------- report
def md_table(rows, cols):
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return s


def cmd_report(_a):
    out = out_dir()
    summ = read_csv(out / "summary.csv")
    agree = read_csv(out / "comparison" / "frame_agreement.csv")
    ov = read_csv(out / "comparison" / "box_overlap.csv")
    ver = read_csv(out / "verify_outputs.csv")
    integ = json.loads((out / "integrity.json").read_text())
    ident = json.loads((out / "old_implementation" / "MODEL_IDENTIFICATION.json").read_text())
    idx = read_csv(out / "review_sheets" / "index.csv")
    sel = json.loads((out / "review_sheets" / "SELECTION_METHOD.json").read_text())
    cases = read_csv(out / "review_sheets" / "interesting_cases" / "index.csv") if (out / "review_sheets" / "interesting_cases" / "index.csv").exists() else []
    tim = read_csv(out / "comparison" / "timing_only_pass.csv") if (out / "comparison" / "timing_only_pass.csv").exists() else []
    tim_meta = json.loads((out / "comparison" / "timing_only_pass.json").read_text()) if tim else {}
    obs_p, ov_p = out / "observations.csv", out / "overall_observations.csv"
    obs = read_csv(obs_p) if obs_p.exists() else []
    ovo = read_csv(ov_p) if ov_p.exists() else []
    cfg = json.loads((out / "old_implementation" / "old_run_config.json").read_text())
    ck = ident["checkpoint"]
    L = []
    L.append("# Three-model review of the personal recordings: old implementation vs new baseline vs new candidate\n")
    L.append("**There is no ground truth for these recordings. This is a qualitative visual comparison only. Detection counts, detection rates and confidence scores do not show which model is more accurate. "
             "No precision, recall, accuracy or false-positive rate is computed, and no model is declared better.** Visual notes are impressions from looking at sampled frames and may be wrong.\n")
    L.append("## Which old model was used, and why\n")
    L.append(f"- Checkpoint: `{Path(ck['path']).relative_to(PROJECT).as_posix()}` (sha256 `{ck['sha256']}`, {ck['size_bytes']} bytes). Architecture: {ck['task']} model, `{ck['ckpt_train_args']['model']}` "
             f"(YOLOv8s-seg), trained by Ultralytics {ck['ckpt_ultralytics_version']} at imgsz {ck['ckpt_train_args']['imgsz']}. Stored class names: {ck['class_names']} (single class).")
    for e in ident["evidence_that_old_app_uses_it"]:
        L.append(f"- Evidence: `{e['file']}` lines {e['lines']}: {e['note']}")
    L.append("- Other checkpoints found in the old project (not used): " + "; ".join(f"`{o['path']}`" for o in ident["other_checkpoints_found"] if o["sha256"] != ck["sha256"]) + ". " + " ".join(ident["other_checkpoints_notes"]))
    L.append("- Full detail with line numbers: `old_implementation/MODEL_IDENTIFICATION.json`.\n")
    L.append("## Settings of the three models (they differ, so counts and confidences are not comparable)\n")
    L.append(md_table([dict(model=k, weights=Path(m["path"]).relative_to(PROJECT).as_posix(), imgsz=m["imgsz"], conf=m["conf"], nms_iou=m["nms"],
                            classes=("1 (pothole); segmentation model, boxes used" if k == "old" else "1 (pothole); box detector")) for k, m in MODELS.items()],
                      ["model", "weights", "imgsz", "conf", "nms_iou", "classes"]))
    L.append("The old model reproduces its own application settings per frame: imgsz 640, conf 0.35, NMS IoU 0.5, retina masks, 5x5 elliptical close+open smoothing of each mask and the 150 px minimum mask area filter (detections failing it are dropped). "
             "The only class is `pothole` (class 0), which is what counts as a pothole for every comparison here. "
             "**Not reproduced on purpose:** the old app processes every 5th frame, tracks with ByteTrack, keeps only confirmed tracks and collapses them to one record per pothole; here every frame is processed with no tracking, so the three models are compared on identical frames. "
             "Also not applied: the JPEG q95 round trip of extracted frames. Different library versions: this run used ultralytics "
             f"{cfg['ultralytics']} / torch {cfg['torch']}, the old repo pinned 8.4.86 / torch 2.12.1. The new baseline and candidate thresholds are validation-selected operating points of those models; the old threshold is the old app's default.\n")
    L.append("Baseline and candidate detections were NOT rerun: they are reused from `New Implementation/runs/inference/personal_recordings_comparison/` (summary.csv, detections/*.csv).\n")
    L.append("## Totals per model\n")
    tot = [r for r in summ if r["recording"] == "TOTAL"]
    L.append(md_table(tot, ["model", "imgsz", "conf_threshold", "frames_processed", "total_detections", "avg_dets_per_frame", "pct_frames_with_detection", "median_confidence",
                            "avg_forward_nms_ms", "avg_incl_preprocess_ms"]))
    old_t = next(r for r in tot if r["model"] == "old")
    L.append(f"Old model: {old_t['dropped_by_min_mask_area']} instances passed the confidence threshold but were dropped by the 150 px minimum-mask-area filter.\n")
    L.append("## Per recording (detections / % of frames with at least one detection / median confidence)\n")
    per = {}
    for r in summ:
        if r["recording"] != "TOTAL":
            per.setdefault(r["recording"], {})[r["model"]] = r
    rows = []
    for rec, d in per.items():
        row = dict(recording=rec, frames=d["old"].get("frames_processed", ""), dur_s=d["old"].get("source_duration_s", ""))
        for k in KEYS:
            x = d.get(k, {})
            row[f"{k}_dets"] = x.get("total_detections", x.get("status", ""))
            row[f"{k}_pct_frames"] = x.get("pct_frames_with_detection", "")
            row[f"{k}_med_conf"] = x.get("median_confidence", "")
        rows.append(row)
    L.append(md_table(rows, list(rows[0])))
    L.append("## Agreement and disagreement between the models (behaviour only, not correctness)\n")
    L.append("A model 'fires' in a frame if it has at least one pothole detection there. Counts of frames per combination:\n")
    L.append(md_table(agree, ["recording", "frames", "none", "only_old", "only_baseline", "only_candidate", "old_baseline_not_candidate", "old_candidate_not_baseline",
                              "baseline_candidate_not_old", "all_three", "spatial_disagreement_frames"]))
    ff = {r["model"]: r["boxes_ge_80pct_of_frame_area"] for r in tot}
    sens = read_csv(out / "comparison" / "frame_agreement_old_excl_fullframe_boxes.csv")[-1]
    L.append(f"**Important caveat about the old model:** many of its boxes cover almost the whole frame. Boxes covering >= 80% of the frame area (column `boxes_ge_80pct_of_frame_area` in `summary.csv`): old {ff['old']} of {old_t['total_detections']}, baseline {ff['baseline']}, candidate {ff['candidate']}. "
             f"This is reproduced faithfully (the old project's own sample `results/detections.json` also has a first entry whose mask polygon runs along the frame edges), not removed. As a sensitivity check only, `comparison/frame_agreement_old_excl_fullframe_boxes.csv` recomputes the old model's firing frames without those boxes: old fires in {sens['old_frames_firing_excl_fullframe']} of {sens['frames']} frames instead of {sens['old_frames_firing_primary']} (categories there: only_old {sens['only_old']}, all_three {sens['all_three']}, none {sens['none']}). The sheets and primary tables use the faithful version.\n")
    L.append("Pairwise frame-level agreement (both fire or both silent), as % of frames:\n")
    L.append(md_table(agree, ["recording", "old_baseline_agree_pct", "old_candidate_agree_pct", "baseline_candidate_agree_pct", "old_baseline_both_fire", "old_candidate_both_fire", "baseline_candidate_both_fire"]))
    L.append("Box-level overlap on frames where both models of a pair fire (all recordings; per recording in `comparison/box_overlap.csv`):\n")
    L.append(md_table([r for r in ov if r["recording"] == "ALL"], ["pair", "frames_both_fire", "boxes_a", "boxes_b", "pct_a_boxes_iou_ge_0_3", "pct_b_boxes_iou_ge_0_3", "median_best_iou_a", "median_best_iou_b", "frames_both_fire_but_no_box_pair_iou_ge_0_3"]))
    L.append("a = first model of the pair, b = second. Percentages are the share of that model's boxes (on frames where both fire) that overlap some box of the other with IoU >= 0.3.\n")
    L.append("## Review sheets and how frames were selected\n")
    L.append(f"Seed {sel['seed']}, at most {sel['frames_per_sheet_max']} frames per category per recording. {sel['selection']}\n")
    L.append("Category definitions: " + "; ".join(f"`{k}`: {v}" for k, v in sel["categories"].items()) + "\n")
    cnt = {}
    for r in idx:
        cnt[r["category"]] = cnt.get(r["category"], 0) + 1
    L.append(md_table([dict(category=c, sheets_on_disk=cnt.get(c, 0)) for c in CATS], ["category", "sheets_on_disk"]))
    L.append(f"Sheets: `review_sheets/<category>/<recording>_<category>.jpg`; index: `review_sheets/index.csv`; method: `review_sheets/SELECTION_METHOD.json`. "
             f"Hand-picked set (not representative): `review_sheets/interesting_cases/` ({len(cases)} images, `index.csv` records why each was chosen).\n")
    L.append("## Per-recording observations (impressions from the review sheets)\n")
    if obs:
        for r in obs:
            L.append(f"- **{r['recording']}**: old: {r['old_fires_on']} | baseline: {r['baseline_fires_on']} | candidate: {r['candidate_fires_on']} | apparent misses: {r['apparent_misses']} | other: {r['other']}")
    else:
        L.append("(observations.csv not provided)")
    for sec in ("Apparent false-alarm patterns", "Apparent misses", "Differences in behaviour"):
        L.append(f"\n## {sec}\n")
        t = [r["text"] for r in ovo if r["section"] == sec]
        L.append("\n".join(f"- {x}" for x in t) if t else "(not provided)")
    L.append("\n## Timing (indicative only)\n")
    L.append("Per-recording `avg_forward_nms_ms` / `avg_incl_preprocess_ms` in `summary.csv` mix two runs: the old model was timed in this run, baseline and candidate were timed in an earlier run under different GPU load. "
             "The old model is a segmentation model whose forward+postprocess figure includes retina-mask generation (its additional 5x5 smoothing/area filter time is in `avg_postfilter_ms`), so it is not like-for-like with the two box detectors.\n")
    if tim:
        L.append("Clean separate timing-only pass (all three models on the same frames in one session; " + tim_meta.get("method", "") + "):\n")
        L.append(md_table(tim, ["model", "imgsz", "frames_timed", "mean_forward_nms_ms", "median_forward_nms_ms", "mean_incl_preprocess_ms", "median_incl_preprocess_ms", "mean_old_postfilter_ms"]))
    L.append("## Failures and mismatches\n")
    fails = [r for r in summ if r.get("status", "ok") != "ok"]
    L.append("Failed recordings: " + (", ".join(f"{r['recording']}/{r['model']}: {r['status']}" for r in fails) if fails else "none") + "\n")
    mm = [r for r in ver if r["mismatch"] != "none"]
    L.append("Output vs source mismatches in the files produced here (old annotated videos, 3-panel videos; ffprobe checks in `verify_outputs.csv`, " + f"{len(ver)} checks): " + ("none" if not mm else "") + "\n")
    if mm:
        L.append(md_table(mm, ["recording", "output", "mismatch"]))
    L.append("## Integrity\n")
    L.append(f"Exisiting Implementation folder unchanged (file count, bytes, newest mtimes, digest of paths/sizes/mtimes before vs after): {integ.get('exisiting_implementation_unchanged')}. "
             f"Checkpoint hashes unchanged: {integ.get('model_hashes_unchanged')}. Recordings unchanged: {integ.get('recordings_unchanged')}. Details: `integrity.json`.\n")
    L.append("## Where things are\n")
    L.append(f"Everything is under `{out.relative_to(PROJECT).as_posix()}/`: `old_implementation/` (annotated videos, `detections/*.csv`, `MODEL_IDENTIFICATION.json`, `old_summary.csv`), "
             "`review_sheets/` (+ `interesting_cases/`), `comparison/` (`frame_agreement.csv`, `box_overlap.csv`, `side_by_side_3panel/*.mp4` old | baseline | candidate, `timing_only_pass.csv`), "
             "`summary.csv`, `integrity.json`, `verify_outputs.csv`, `observations.csv`, `overall_observations.csv`. "
             "Baseline/candidate annotated videos and detections are in `New Implementation/runs/inference/personal_recordings_comparison/`. "
             "Code: `New Implementation/scripts/three_model_comparison.py` (hand-picked case list: `New Implementation/scripts/three_model_interesting_cases.json`); the two-model script `compare_personal_recordings.py` is untouched.\n")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out / "REPORT.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "identify", "run", "compare", "timing", "sheets", "cases", "finalize", "report"])
    a = ap.parse_args()
    {"init": cmd_init, "identify": cmd_identify, "run": cmd_run, "compare": cmd_compare, "timing": cmd_timing, "sheets": cmd_sheets,
     "cases": cmd_cases, "finalize": cmd_finalize, "report": cmd_report}[a.cmd](a)
