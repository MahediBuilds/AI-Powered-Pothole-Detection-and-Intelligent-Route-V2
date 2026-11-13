"""Print the headline table from runs/eval/baseline_yolov8s_rdd2022_v3 (read-only)."""
import json, sys
from pathlib import Path
R = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/eval/baseline_yolov8s_rdd2022_v3")
f = lambda x: "nan" if x is None or x != x else f"{x:.3f}"
for sz in sorted(p.name for p in R.iterdir() if p.is_dir() and p.name.isdigit()):
    thr = json.loads((R / sz / "pooled_val" / "operating_points.json").read_text())["thresholds"]["max_f1"]
    print(f"== imgsz {sz} pooled-val max-F1 thr {thr:.4f}")
    for name, d in [("rdd2022 val", "rdd2022_val_at_pooled"), ("hrp4k val", "hrp4k_val_at_pooled"),
                    ("rdd2022 test", "rdd2022_test"), ("hrp4k test", "hrp4k_test"), ("kaggle probe", "kaggle_probe")]:
        m = json.loads((R / sz / d / "metrics.json").read_text())
        tf, o = m["threshold_free"], m["operating_points"]["max_f1"]
        ci = tf["mAP50_ci95"]
        print(f"{name:13s} mAP50 {f(tf['mAP50'])} [{f(ci[0])},{f(ci[1])}] mAP50-95 {f(tf['mAP50_95'])} | P {f(o['precision'])} R {f(o['recall'])} "
              f"FP/img {f(o['fp_per_image'])} n={m['n_images']} gt={m['n_instances']} neg={m['n_negative_images']} "
              f"FP/neg {f(o['fp_per_negative_image'])} negdet {f(o['negative_images_with_detection'])} in_sample={m['in_sample']}")
    for s in ("rdd2022", "hrp4k"):
        sel = json.loads((R / sz / f"{s}_val" / "metrics.json").read_text())["threshold_free"]
        print(f"  select run {s} val: mAP50 {f(sel['mAP50'])} mAP50-95 {f(sel['mAP50_95'])}")
    pm = json.loads((R / sz / "kaggle_probe" / "metrics.json").read_text())
    print(json.dumps(pm["manhole_probe"], indent=None)); print(json.dumps(pm["easy_subset_positive_frames"]))
