"""Cache raw predictions (conf >= 0.001, after NMS IoU 0.7) per model + imgsz + validation set.

Uses Ultralytics ``val()`` exactly as scripts/eval_checkpoint.py does (batch 8, conf 0.001, NMS IoU 0.7,
seed 0, deterministic) and keeps the boxes it saves (``save_json``), so letterboxing matches the project
protocol. One run per (model, imgsz, set); all three investigations read the cache.

    .venv/Scripts/python.exe scripts/val_investigations/predict_cache.py

Validation lists only: ``guard()`` refuses any path containing 'test' or 'kaggle_probe'. Idempotent: existing
cache entries are kept (delete the .npz to redo). Writes only under runs/eval/val_investigations_v1/cache.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    a = ap.parse_args(argv)
    import torch, ultralytics
    from ultralytics import YOLO

    manifest_path = C.CACHE / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    yaml_dir = C.CACHE / "yaml"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    run_dir = C.CACHE / "_val_runs"
    for name in C.SET_NAMES:
        lst = C.guard(C.VAL_SETS[name])
        (yaml_dir / f"{name}.yaml").write_text(
            f"train: {Path(lst).as_posix()}\nval: {Path(lst).as_posix()}\nnames:\n  0: pothole\n")

    for model_name, imgsz in C.PRED_CONFIGS:
        todo = [s for s in C.SET_NAMES if not C.pred_path(model_name, imgsz, s).exists()]
        if not todo:
            continue
        sha = C.verify_checkpoint(model_name)
        model = YOLO(str(C.CKPT[model_name]))
        for sn in todo:
            vs = C.ValSet(sn)  # also applies the guard to every image/label path
            tag = f"{model_name}_{imgsz}_{sn}"
            m = model.val(data=str(yaml_dir / f"{sn}.yaml"), split="val", imgsz=imgsz, batch=8, conf=0.001, iou=0.7,
                          device=a.device, plots=False, save_json=True, project=str(run_dir), name=tag, exist_ok=True,
                          workers=2, verbose=False, seed=0, deterministic=True)
            pj = Path(m.save_dir) / "predictions.json"
            preds = C.ec.load_preds(pj, set(vs.stems))
            C.save_preds(C.pred_path(model_name, imgsz, sn), preds)
            n = int(sum(len(v[1]) for v in preds.values()))
            manifest[tag] = dict(model=model_name, imgsz=imgsz, set=sn, checkpoint_sha256=sha, n_images=len(vs.stems),
                                 n_images_with_preds=len(preds), n_preds=n,
                                 list_sha256=C.sha256(C.VAL_SETS[sn]),
                                 ultralytics_mAP50=float(m.box.map50), ultralytics_mAP50_95=float(m.box.map),
                                 ultralytics=ultralytics.__version__, torch=torch.__version__,
                                 date=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
            manifest_path.write_text(json.dumps(manifest, indent=2))
            shutil.rmtree(m.save_dir, ignore_errors=True)  # predictions are in the npz now
            C.progress(f"cached predictions {tag}: {n} boxes, Ultralytics mAP50 {m.box.map50:.4f}")
            print(tag, n, round(float(m.box.map50), 4), flush=True)
        del model
        torch.cuda.empty_cache()
    C.verify_checkpoint("baseline"); C.verify_checkpoint("candidate")


if __name__ == "__main__":
    main()
