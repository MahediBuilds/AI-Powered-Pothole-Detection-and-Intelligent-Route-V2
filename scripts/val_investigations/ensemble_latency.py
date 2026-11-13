"""Latency of the baseline@640 + candidate@960 ensemble versus single models (batch 1, this machine's GPU).

Same approach as scripts/latency_bench.py (letterbox once per imgsz, CUDA-synchronised timing, warm-up then timed
iterations, forward + NMS, preprocessing excluded) but the frames are 8 HRP4K VALIDATION images resized to 1280x720
(no owner recordings are touched). The ensemble = two sequential forward passes + NMS each + copy of the detections to
the CPU + Weighted Boxes Fusion (``ensemble-boxes``) on the CPU. NMS confidence is timed at 0.25 (deployment-like,
what latency_bench.py uses) and at 0.001 (what the validation evaluation fuses); NMS IoU 0.7.

    .venv/Scripts/python.exe scripts/val_investigations/ensemble_latency.py [--iters 200]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import fusion as F  # noqa: E402

OUTD = C.OUT / "ensemble"


def gpu_snapshot():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        return dict(gpu_util_mem=q, compute_apps=p or "none listed")
    except Exception as e:  # pragma: no cover
        return dict(error=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    a = ap.parse_args()
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression

    C.verify_checkpoint("baseline"); C.verify_checkpoint("candidate")
    hv = C.ValSet("hrp4k_val")
    frames = []
    for f in hv.files[:8]:
        im = cv2.imread(str(C.guard(f)))
        frames.append(cv2.resize(im, (1280, 720)))
    head = json.loads((OUTD / "headline_seed0.json").read_text())
    used = head["systems"]["ensemble_WBF"]["settings_used"]["evaluated_on_fold0"]
    st = next(s for s in __import__("ensemble").settings() if s["name"] == used)
    dev = torch.device("cuda:0")
    before = gpu_snapshot()

    def prep(sz, half):
        lb = LetterBox((sz, sz), auto=False)
        out = []
        for f in frames:
            x = torch.from_numpy(np.ascontiguousarray(lb(image=f)[:, :, ::-1].transpose(2, 0, 1))).to(dev)[None]
            out.append((x.half() if half else x.float()) / 255)
        return out

    def timeit(fn, n):
        ts = []
        for i in range(a.warmup + n):
            torch.cuda.synchronize(); t = time.perf_counter()
            fn(i)
            torch.cuda.synchronize()
            if i >= a.warmup:
                ts.append((time.perf_counter() - t) * 1e3)
        return dict(mean_ms=float(np.mean(ts)), p95_ms=float(np.percentile(ts, 95)))

    rows = []
    for half in (False, True):
        nets = {}
        for key, (mname, sz) in {"baseline@640": ("baseline", 640), "candidate@960": ("candidate", 960),
                                 "candidate@1280": ("candidate", 1280), "candidate@640": ("candidate", 640)}.items():
            net = YOLO(str(C.CKPT[mname])).model.to(dev).eval()
            nets[key] = (net.half() if half else net.float(), sz, prep(sz, half))
        for conf in (0.25, 0.001):
            def single(key):
                net, sz, xs = nets[key]

                def fn(i):
                    with torch.inference_mode():
                        y = net(xs[i % len(xs)])
                        y = y[0] if isinstance(y, (list, tuple)) else y
                        non_max_suppression(y, conf, 0.7)
                return fn

            def ens(i):
                outs = []
                with torch.inference_mode():
                    for key in ("baseline@640", "candidate@960"):
                        net, sz, xs = nets[key]
                        y = net(xs[i % len(xs)])
                        y = y[0] if isinstance(y, (list, tuple)) else y
                        d = non_max_suppression(y, conf, 0.7)[0].float().cpu().numpy()
                        outs.append((d[:, :4], d[:, 4], sz))
                (ba, ca, sa), (bb, cb, sb) = outs
                F.wbf_fuse(ba / sa, ca, bb / sb, cb, (1, 1), st["iou"], st["skip"], st["weights"], st["conf"])

            def fusion_only_factory():
                # measure the CPU fusion step alone on one fixed pair of detection sets
                with torch.inference_mode():
                    ys = []
                    for key in ("baseline@640", "candidate@960"):
                        net, sz, xs = nets[key]
                        y = net(xs[0]); y = y[0] if isinstance(y, (list, tuple)) else y
                        d = non_max_suppression(y, conf, 0.7)[0].float().cpu().numpy()
                        ys.append((d[:, :4] / sz, d[:, 4]))
                n_a, n_b = len(ys[0][1]), len(ys[1][1])

                def fn(i):
                    F.wbf_fuse(ys[0][0], ys[0][1], ys[1][0], ys[1][1], (1, 1), st["iou"], st["skip"], st["weights"], st["conf"])
                return fn, n_a, n_b
            for key in ("baseline@640", "candidate@640", "candidate@960", "candidate@1280"):
                torch.cuda.reset_peak_memory_stats()
                r = timeit(single(key), a.iters)
                rows.append(dict(precision="fp16" if half else "fp32", nms_conf=conf, system=key, **r,
                                 peak_gpu_mem_mib=torch.cuda.max_memory_allocated() / 2**20))
                print(rows[-1], flush=True)
            torch.cuda.reset_peak_memory_stats()
            r = timeit(ens, a.iters)
            rows.append(dict(precision="fp16" if half else "fp32", nms_conf=conf, system="ensemble(baseline@640+candidate@960)+WBF", **r,
                             peak_gpu_mem_mib=torch.cuda.max_memory_allocated() / 2**20))
            print(rows[-1], flush=True)
            fn, na, nb = fusion_only_factory()
            ts = []
            for i in range(a.iters):
                t = time.perf_counter(); fn(i); ts.append((time.perf_counter() - t) * 1e3)
            rows.append(dict(precision="fp16" if half else "fp32", nms_conf=conf, system="WBF fusion step only (CPU)", mean_ms=float(np.mean(ts)),
                             p95_ms=float(np.percentile(ts, 95)), n_boxes_model_a=na, n_boxes_model_b=nb))
            print(rows[-1], flush=True)
        del nets
        torch.cuda.empty_cache()
    after = gpu_snapshot()
    res = dict(gpu=torch.cuda.get_device_name(0), frames="8 HRP4K validation images resized to 1280x720", batch=1, iters=a.iters,
               warmup=a.warmup, nms_iou=0.7, wbf_setting=st, gpu_before=before, gpu_after=after,
               note="forward + NMS (+ CPU copy and WBF for the ensemble); preprocessing excluded; letterbox once per imgsz. "
                    "Models run sequentially. Timings can be disturbed by other GPU users (see gpu_before/gpu_after).",
               results=rows)
    C.write_json(OUTD / "latency.json", res)
    L = ["# Ensemble latency (batch 1)", "", f"GPU {res['gpu']}; {res['frames']}; WBF setting `{st['name']}`; NMS IoU 0.7.",
         f"GPU state before: {before}; after: {after}", "",
         "| precision | NMS conf | system | mean ms | p95 ms | peak GPU MiB |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['precision']} | {r['nms_conf']} | {r['system']} | {r['mean_ms']:.2f} | {r['p95_ms']:.2f} | {r.get('peak_gpu_mem_mib', float('nan')):.0f} |")
    (OUTD / "LATENCY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    C.progress("ensemble latency measured (LATENCY.md)")


if __name__ == "__main__":
    main()
