"""Batch-1 GPU latency of a YOLO checkpoint: model forward + NMS (preprocessing excluded).

    python scripts/latency_bench.py --checkpoint W.pt --video V.mp4 --out runs/eval/NAME/latency.json
Frames are decoded read-only from the video (first frames, resized to 1280x720), letterboxed once per
imgsz. Per config: warm-up then timed iterations, CUDA-synchronised, mean / p95 / peak GPU memory.
"""
import argparse, json, os, time
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
    ap.add_argument("--warmup", type=int, default=30); ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--n-frames", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    for bad in ("runs/train", "model_archive"):
        if (Path.cwd() / bad).resolve() in [out, *out.parents]:
            raise SystemExit(f"refusing to write inside {bad}")
    import cv2, numpy as np, torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression
    cap = cv2.VideoCapture(a.video); frames = []
    while len(frames) < a.n_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (1280, 720)))
    cap.release()
    assert frames, "no frames decoded"
    dev = torch.device("cuda:0")
    rows = []
    for half in (False, True):
        for sz in a.imgsz:
            net = YOLO(a.checkpoint).model.to(dev).eval()
            net = net.half() if half else net.float()
            lb = LetterBox((sz, sz), auto=False)
            xs = []
            for f in frames:
                x = torch.from_numpy(np.ascontiguousarray(lb(image=f)[:, :, ::-1].transpose(2, 0, 1))).to(dev)[None]
                xs.append((x.half() if half else x.float()) / 255)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            ts = []
            with torch.inference_mode():
                for i in range(a.warmup + a.iters):
                    x = xs[i % len(xs)]
                    torch.cuda.synchronize(); t = time.perf_counter()
                    y = net(x); y = y[0] if isinstance(y, (list, tuple)) else y
                    non_max_suppression(y, a.conf, a.iou)
                    torch.cuda.synchronize()
                    if i >= a.warmup:
                        ts.append((time.perf_counter() - t) * 1e3)
            rows.append(dict(precision="fp16" if half else "fp32", imgsz=sz, mean_ms=float(np.mean(ts)),
                             p95_ms=float(np.percentile(ts, 95)), peak_gpu_mem_mib=torch.cuda.max_memory_allocated() / 2**20))
            print(rows[-1], flush=True)
            del net
    res = dict(checkpoint=str(Path(a.checkpoint).resolve()), video=a.video, frames=len(frames), frame_size="1280x720",
               iters=a.iters, warmup=a.warmup, conf=a.conf, nms_iou=a.iou, batch=1, gpu=torch.cuda.get_device_name(0),
               note="forward + NMS, preprocessing excluded; peak = torch.cuda.max_memory_allocated (excludes CUDA context)",
               results=rows)
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
