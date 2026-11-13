"""Annotated inference videos of two models on the 13 personal recordings (inference only; NO ground truth).

    run:    python scripts/compare_personal_recordings.py run      # videos, detections, summary, verify, differences
    sheets: python scripts/compare_personal_recordings.py sheets   # review contact sheets (seeded sample)
    report: python scripts/compare_personal_recordings.py report   # REPORT.md from CSVs + observations.csv + overall_observations.csv

Nothing here measures precision/recall; detection counts do not show which model is more accurate.
Recordings are opened read-only. Output dir: runs/inference/personal_recordings_comparison (or _v2, _v3 ... if non-empty).
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
ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "data" / "recordings"
BASE_OUT = ROOT / "runs" / "inference" / "personal_recordings_comparison"
MODELS = {
    "baseline": dict(path=ROOT / "runs/train/baseline/weights/best.pt", imgsz=640, conf=0.2705, suffix="baseline640", label="BASELINE",
                     sha="2aabdd131d1664089e06e3219fabcaed4cc96610fa4603ff165429f92696d798", color=(60, 200, 60)),
    "candidate": dict(path=ROOT / "runs/train/unified_v1_y8s_1280/weights/epoch25.pt", imgsz=1280, conf=0.245, suffix="candidate1280", label="CANDIDATE",
                      sha="0a1c19f2586b6b730e9c5bf2bdeff6fcb3ec8f4567d161f1aba0459d09a5adc3", color=(0, 140, 255)),
}
NMS_IOU = 0.7
SEED = 0


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_videos():
    return sorted((p for p in REC_DIR.rglob("*.mp4") if "_contact" not in p.parts), key=lambda p: p.stem)


def out_dir(create=False):
    d = BASE_OUT
    if create:
        n = 2
        while d.exists() and any(d.iterdir()):
            d = BASE_OUT.with_name(BASE_OUT.name + f"_v{n}"); n += 1
        d.mkdir(parents=True, exist_ok=True)
        return d
    cands = sorted([p for p in BASE_OUT.parent.glob(BASE_OUT.name + "*") if p.is_dir()], key=lambda p: (len(p.name), p.name))
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


def draw(img, boxes, confs, color, tag):
    import cv2
    h, w = img.shape[:2]
    lw = max(2, round(w / 400))
    fs = max(0.5, w / 1280 * 0.6)
    th = max(1, round(fs * 2))
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, lw)
        txt = f"pothole {c:.2f}"
        (tw, tht), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        y0 = p1[1] - tht - bl - 2
        if y0 < 0:
            y0 = p1[1] + 2
        cv2.rectangle(img, (p1[0], y0), (p1[0] + tw + 4, y0 + tht + bl + 2), color, -1)
        cv2.putText(img, txt, (p1[0] + 2, y0 + tht + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th, cv2.LINE_AA)
    fs2 = max(0.6, w / 1280 * 0.7)
    th2 = max(1, round(fs2 * 2))
    (tw, tht), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs2, th2)
    cv2.rectangle(img, (0, 0), (tw + 16, tht + bl + 14), (0, 0, 0), -1)
    cv2.putText(img, tag, (8, tht + 8), cv2.FONT_HERSHEY_SIMPLEX, fs2, color, th2, cv2.LINE_AA)


def cmd_run(_a):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    t_start = time.time()
    out = out_dir(create=True)
    for sub in ("baseline", "candidate", "comparison", "detections"):
        (out / sub).mkdir(exist_ok=True)
    vids = find_videos()
    errors = []
    integ = dict(models_before={}, recordings_before={}, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                 started=time.strftime("%Y-%m-%d %H:%M:%S"), out_dir=str(out))
    for k, m in MODELS.items():
        integ["models_before"][k] = sha256(m["path"])
        if integ["models_before"][k] != m["sha"]:
            sys.exit(f"hash mismatch for {k}")
    for v in vids:
        integ["recordings_before"][v.stem] = sha256(v)
    (out / "integrity.json").write_text(json.dumps(integ, indent=2))
    cfg = dict(nms_iou=NMS_IOU, precision="FP32", batch=1, seed=SEED, models={k: dict(path=str(m["path"]), imgsz=m["imgsz"], conf=m["conf"], sha256=m["sha"]) for k, m in MODELS.items()},
               n_recordings=len(vids), gpu=integ["gpu"], audio="dropped", codec="libx264 crf20 yuv420p")
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    nets = {k: YOLO(str(m["path"])) for k, m in MODELS.items()}

    def infer(k, frame):
        m = MODELS[k]
        r = nets[k].predict(frame, imgsz=m["imgsz"], conf=m["conf"], iou=NMS_IOU, device=0, verbose=False)[0]
        b = r.boxes
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), r.speed

    cap = cv2.VideoCapture(str(vids[0]))
    for _ in range(20):
        ok, fr = cap.read()
        if not ok:
            break
        for k in MODELS:
            infer(k, fr)
    cap.release()
    torch.cuda.synchronize()

    summary, verify, diffs = [], [], []
    all_conf = {k: [] for k in MODELS}
    totals = {k: dict(frames=0, dets=0, withdet=0, t_inf=0.0, t_all=0.0, dur=0.0) for k in MODELS}
    for v in vids:
        stem = v.stem
        print(f"== {stem}", flush=True)
        procs = []
        try:
            pr = probe(v)
            if pr["rotation"]:
                raise RuntimeError(f"rotation metadata {pr['rotation']} not handled")
            w, h = pr["width"], pr["height"]
            cap = cv2.VideoCapture(str(v))
            files = {k: out / k / f"{stem}_{MODELS[k]['suffix']}.mp4" for k in MODELS}
            sbs = out / "comparison" / f"{stem}_side_by_side.mp4"
            sbs_w, sbs_h = 2 * w, h
            scale = 1.0
            if sbs_w > 3840:
                scale = 3840 / sbs_w
                sbs_w, sbs_h = int(w * scale) * 2, int(h * scale)
            enc = {k: start_enc(files[k], w, h, pr["fps_frac"]) for k in MODELS}
            enc["sbs"] = start_enc(sbs, sbs_w, sbs_h, pr["fps_frac"])
            procs = list(enc.values())
            st = {k: dict(dets=0, withdet=0, t_inf=[], t_all=[], conf=[]) for k in MODELS}
            rows = {k: [] for k in MODELS}
            fired = {k: [] for k in MODELS}
            idx = 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                tsec = idx / pr["fps"]
                ann = {}
                for k in MODELS:
                    boxes, confs, sp = infer(k, fr)
                    s = st[k]
                    s["t_inf"].append(sp["inference"] + sp["postprocess"])
                    s["t_all"].append(sp["preprocess"] + sp["inference"] + sp["postprocess"])
                    n = len(confs)
                    s["dets"] += n; s["withdet"] += n > 0; s["conf"] += [float(c) for c in confs]
                    fired[k].append(n > 0)
                    for b, c in zip(boxes, confs):
                        rows[k].append([idx, f"{tsec:.4f}", *(f"{x:.1f}" for x in b), f"{float(c):.4f}"])
                    img = fr.copy()
                    m = MODELS[k]
                    draw(img, boxes, confs, m["color"], f"{m['label']} | imgsz {m['imgsz']} | conf {m['conf']}")
                    ann[k] = img
                    enc[k].stdin.write(img.tobytes())
                both = np.hstack([ann["baseline"], ann["candidate"]])
                if scale != 1.0:
                    both = cv2.resize(both, (sbs_w, sbs_h), interpolation=cv2.INTER_AREA)
                enc["sbs"].stdin.write(both.tobytes())
                idx += 1
            cap.release()
            for p in procs:
                p.stdin.close()
            for p in procs:
                if p.wait() != 0:
                    raise RuntimeError("ffmpeg failed")
            procs = []
            n = idx
            for k, m in MODELS.items():
                with open(out / "detections" / f"{stem}_{k}.csv", "w", newline="") as f:
                    wr = csv.writer(f); wr.writerow(["frame", "time_s", "x1", "y1", "x2", "y2", "conf"]); wr.writerows(rows[k])
                s = st[k]
                med = round(statistics.median(s["conf"]), 4) if s["conf"] else ""
                summary.append(dict(recording=stem, model=k, weights=str(m["path"].relative_to(ROOT)), imgsz=m["imgsz"], conf_threshold=m["conf"],
                                    source_resolution=f"{w}x{h}", source_fps=round(pr["fps"], 3), source_duration_s=round(pr["duration"], 3),
                                    frames_processed=n, total_detections=s["dets"], avg_dets_per_frame=round(s["dets"] / n, 4),
                                    pct_frames_with_detection=round(100 * s["withdet"] / n, 2),
                                    avg_forward_nms_ms=round(sum(s["t_inf"]) / n, 3), avg_incl_preprocess_ms=round(sum(s["t_all"]) / n, 3),
                                    median_confidence=med, output_file=str(files[k].relative_to(out)), status="ok"))
                t = totals[k]; t["frames"] += n; t["dets"] += s["dets"]; t["withdet"] += s["withdet"]
                t["t_inf"] += sum(s["t_inf"]); t["t_all"] += sum(s["t_all"]); t["dur"] += pr["duration"]
                all_conf[k] += s["conf"]
            fb, fc = fired["baseline"], fired["candidate"]
            diffs.append(dict(recording=stem, frames=n, only_baseline=sum(a and not b for a, b in zip(fb, fc)),
                              only_candidate=sum(b and not a for a, b in zip(fb, fc)), both=sum(a and b for a, b in zip(fb, fc)),
                              neither=sum((not a) and (not b) for a, b in zip(fb, fc))))
            for name, path in [("baseline", files["baseline"]), ("candidate", files["candidate"]), ("side_by_side", sbs)]:
                o = probe(path)
                ew, eh = (w, h) if name != "side_by_side" else (sbs_w, sbs_h)
                issues = []
                if o["frames"] != n: issues.append(f"frames {o['frames']} vs {n}")
                if o["frames"] != pr["frames"]: issues.append(f"frames vs source packets {pr['frames']}")
                if abs(o["fps"] - pr["fps"]) > 0.01: issues.append(f"fps {o['fps']:.4f} vs {pr['fps']:.4f}")
                if (o["width"], o["height"]) != (ew, eh): issues.append(f"res {o['width']}x{o['height']} vs {ew}x{eh}")
                if abs(o["duration"] - pr["duration"]) > 0.05: issues.append(f"duration {o['duration']:.3f} vs {pr['duration']:.3f}")
                verify.append(dict(recording=stem, output=name, src_frames_ffprobe=pr["frames"], frames_decoded_by_opencv=n, out_frames=o["frames"],
                                   src_fps=round(pr["fps"], 4), out_fps=round(o["fps"], 4), src_res=f"{w}x{h}", out_res=f"{o['width']}x{o['height']}",
                                   src_duration_s=round(pr["duration"], 3), out_duration_s=round(o["duration"], 3),
                                   mismatch="; ".join(issues) or "none", size_mb=round(path.stat().st_size / 1e6, 2)))
            print(f"   frames={n} {diffs[-1]}", flush=True)
        except Exception as e:  # log and continue
            import traceback; traceback.print_exc()
            errors.append((stem, repr(e)))
            for p in procs:
                try: p.kill()
                except Exception: pass
            for k in MODELS:
                summary.append(dict(recording=stem, model=k, status=f"ERROR: {e!r}"))
    for k, m in MODELS.items():
        t = totals[k]; n = max(t["frames"], 1)
        summary.append(dict(recording="TOTAL", model=k, weights=str(m["path"].relative_to(ROOT)), imgsz=m["imgsz"], conf_threshold=m["conf"],
                            source_duration_s=round(t["dur"], 3), frames_processed=t["frames"], total_detections=t["dets"],
                            avg_dets_per_frame=round(t["dets"] / n, 4), pct_frames_with_detection=round(100 * t["withdet"] / n, 2),
                            avg_forward_nms_ms=round(t["t_inf"] / n, 3), avg_incl_preprocess_ms=round(t["t_all"] / n, 3),
                            median_confidence=round(statistics.median(all_conf[k]), 4) if all_conf[k] else "", status="ok"))
    fields = ["recording", "model", "weights", "imgsz", "conf_threshold", "source_resolution", "source_fps", "source_duration_s", "frames_processed",
              "total_detections", "avg_dets_per_frame", "pct_frames_with_detection", "avg_forward_nms_ms", "avg_incl_preprocess_ms",
              "median_confidence", "output_file", "status"]

    def dump(path, rows, fields=None):
        if not rows: return
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields or list(rows[0])); wr.writeheader(); wr.writerows(rows)
    dump(out / "summary.csv", summary, fields)
    dump(out / "comparison" / "frame_differences.csv", diffs)
    dump(out / "verify_outputs.csv", verify)
    integ["models_after"] = {k: sha256(m["path"]) for k, m in MODELS.items()}
    integ["recordings_after"] = {v.stem: sha256(v) for v in vids}
    integ["models_unchanged"] = integ["models_before"] == integ["models_after"]
    integ["recordings_unchanged"] = integ["recordings_before"] == integ["recordings_after"]
    integ["errors"] = errors
    integ["run_seconds"] = round(time.time() - t_start, 1)
    (out / "integrity.json").write_text(json.dumps(integ, indent=2))
    print("done", out, integ["run_seconds"], "s; errors:", errors)


def read_det(out, stem, k):
    d = {}
    with open(out / "detections" / f"{stem}_{k}.csv") as f:
        for r in csv.DictReader(f):
            d.setdefault(int(r["frame"]), []).append(([float(r[c]) for c in ("x1", "y1", "x2", "y2")], float(r["conf"])))
    return d


def cmd_sheets(_a):
    import cv2
    import numpy as np
    out = out_dir()
    sd = out / "comparison" / "review_sheets"; sd.mkdir(parents=True, exist_ok=True)
    diffs = list(csv.DictReader(open(out / "comparison" / "frame_differences.csv")))
    per, cols, K = 640, 2, 6
    for dr in diffs:
        stem, n = dr["recording"], int(dr["frames"])
        db, dc = read_det(out, stem, "baseline"), read_det(out, stem, "candidate")
        cats = {"only_baseline": [i for i in range(n) if i in db and i not in dc],
                "only_candidate": [i for i in range(n) if i in dc and i not in db],
                "both": [i for i in range(n) if i in db and i in dc]}
        rng = random.Random(f"{SEED}-{stem}")
        sel = {}
        for c, fr in cats.items():
            if len(fr) <= K:
                sel[c] = fr
            else:  # spread across clip: one random frame per equal slice of the category's frame list
                step = len(fr) / K
                sel[c] = [fr[rng.randrange(int(j * step), max(int((j + 1) * step), int(j * step) + 1))] for j in range(K)]
        want = {i for fr in sel.values() for i in fr}
        grabbed = {}
        cap = cv2.VideoCapture(str(next(REC_DIR.rglob(stem + ".mp4"))))
        i = 0
        while want:
            ok, fr = cap.read()
            if not ok:
                break
            if i in want:
                grabbed[i] = fr
            i += 1
            if i > max(want):
                break
        cap.release()
        for c, fr in sel.items():
            if not fr:
                continue
            tiles = []
            for i in fr:
                if i not in grabbed:
                    continue
                pair = []
                for k, d in (("baseline", db), ("candidate", dc)):
                    src = grabbed[i]
                    sc = per / src.shape[1]
                    img = cv2.resize(src, (per, int(src.shape[0] * sc)))
                    bx = [[x * sc for x in b] for b, _ in d.get(i, [])]
                    draw(img, bx, [cf for _, cf in d.get(i, [])], MODELS[k]["color"], f"{MODELS[k]['label']} f{i}")
                    pair.append(img)
                tiles.append(np.hstack(pair))
            if not tiles:
                continue
            while len(tiles) % cols:
                tiles.append(np.zeros_like(tiles[0]))
            grid = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
            hdr = np.zeros((36, grid.shape[1], 3), np.uint8)
            cv2.putText(hdr, f"{stem} | {c} ({len(cats[c])} frames in category, {len(fr)} shown) | left baseline 640, right candidate 1280 | no ground truth",
                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(sd / f"{stem}_{c}.jpg"), np.vstack([hdr, grid]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(stem, {c: len(v) for c, v in cats.items()}, flush=True)


def md_table(rows, cols):
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return s


def cmd_report(_a):
    out = out_dir()
    summ = list(csv.DictReader(open(out / "summary.csv")))
    diffs = list(csv.DictReader(open(out / "comparison" / "frame_differences.csv")))
    ver = list(csv.DictReader(open(out / "verify_outputs.csv")))
    integ = json.loads((out / "integrity.json").read_text())
    cfg = json.loads((out / "config.json").read_text())
    obs_p, ov_p = out / "observations.csv", out / "overall_observations.csv"
    obs = list(csv.DictReader(open(obs_p, encoding="utf-8"))) if obs_p.exists() else []
    ov = list(csv.DictReader(open(ov_p, encoding="utf-8"))) if ov_p.exists() else []
    L = []
    L.append("# Personal recordings: baseline vs candidate (qualitative)\n")
    L.append("Two models were run on the owner's 13 personal road recordings and the results drawn onto annotated videos. "
             "**There is no ground truth for these recordings, so this is a qualitative visual comparison. Detection counts do not show which model is more accurate.** "
             "No precision, recall or error rate is computed. Visual notes are impressions from looking at sampled frames, not measurements.\n")
    L.append("## Settings\n")
    for k, m in cfg["models"].items():
        L.append(f"- {k}: `{Path(m['path']).relative_to(ROOT).as_posix()}`, imgsz {m['imgsz']}, confidence threshold {m['conf']}, sha256 `{m['sha256']}`")
    L.append(f"- NMS IoU {cfg['nms_iou']}, FP32, batch 1, GPU: {cfg['gpu']}. Thresholds are each model's validation-selected operating point, not tuned on these recordings.")
    L.append(f"- Hashes before and after the run: models unchanged = {integ.get('models_unchanged')}, recordings unchanged = {integ.get('recordings_unchanged')}.")
    L.append("- Output videos are H.264 (libx264, crf 20), audio dropped, original resolution and frame rate. Timing is model forward + NMS (GPU synchronised), excluding decode, drawing and encode; model warmed up first. Both models ran alternately on the same GPU.\n")
    L.append("## Totals per model\n")
    tot = [r for r in summ if r["recording"] == "TOTAL"]
    L.append(md_table(tot, ["model", "imgsz", "conf_threshold", "frames_processed", "total_detections", "avg_dets_per_frame",
                            "pct_frames_with_detection", "median_confidence", "avg_forward_nms_ms", "avg_incl_preprocess_ms"]))
    L.append("## Per recording\n")
    per = {}
    for r in summ:
        if r["recording"] != "TOTAL":
            per.setdefault(r["recording"], {})[r["model"]] = r
    rows = []
    for rec, d in per.items():
        b, c = d.get("baseline", {}), d.get("candidate", {})
        rows.append(dict(recording=rec, frames=b.get("frames_processed", ""), dur_s=b.get("source_duration_s", ""),
                         base_dets=b.get("total_detections") or b.get("status", ""), cand_dets=c.get("total_detections") or c.get("status", ""),
                         base_pct_frames=b.get("pct_frames_with_detection", ""), cand_pct_frames=c.get("pct_frames_with_detection", ""),
                         base_med_conf=b.get("median_confidence", ""), cand_med_conf=c.get("median_confidence", ""),
                         base_ms=b.get("avg_forward_nms_ms", ""), cand_ms=c.get("avg_forward_nms_ms", "")))
    L.append(md_table(rows, list(rows[0])))
    L.append("## Frame-level agreement (behaviour only, not correctness)\n")
    L.append(md_table(diffs, ["recording", "frames", "only_baseline", "only_candidate", "both", "neither"]))
    L.append("## Per-recording observations (impressions from review sheets)\n")
    if obs:
        for r in obs:
            L.append(f"- **{r['recording']}**: baseline: {r['baseline_fires_on']} | candidate: {r['candidate_fires_on']} | possible misses: {r['possible_misses']} | other: {r['other']}")
    else:
        L.append("(observations.csv not provided)")
    for sec in ("Apparent false-positive patterns", "Apparent misses", "Differences in behaviour"):
        L.append(f"\n## {sec}\n")
        t = [r["text"] for r in ov if r["section"] == sec]
        L.append("\n".join(f"- {x}" for x in t) if t else "(not provided)")
    L.append("\n## Speed\n")
    L.append("See the totals table: `avg_forward_nms_ms` is per-frame model forward + NMS; `avg_incl_preprocess_ms` adds pre-processing. Measured on this GPU only, with both models sharing it.\n")
    L.append("## Failures and mismatches\n")
    fails = [r for r in summ if r.get("status", "ok") != "ok"]
    L.append("Failed recordings: " + (", ".join(f"{r['recording']}/{r['model']}: {r['status']}" for r in fails) if fails else "none") + "\n")
    mm = [r for r in ver if r["mismatch"] != "none"]
    L.append("Output vs source mismatches (frames, fps, resolution, duration; see `verify_outputs.csv`): " + ("none" if not mm else "") + "\n")
    if mm:
        L.append(md_table(mm, ["recording", "output", "mismatch"]))
    L.append("## Where things are\n")
    L.append(f"All under `{out.relative_to(ROOT).as_posix()}/`: `baseline/`, `candidate/` (annotated videos), `comparison/<stem>_side_by_side.mp4`, "
             "`comparison/frame_differences.csv`, `comparison/review_sheets/`, `detections/<stem>_<model>.csv`, `summary.csv`, `verify_outputs.csv`, "
             "`integrity.json`, `config.json`, `observations.csv`, `overall_observations.csv`.")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out / "REPORT.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "sheets", "report"])
    a = ap.parse_args()
    {"run": cmd_run, "sheets": cmd_sheets, "report": cmd_report}[a.cmd](a)
