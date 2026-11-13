"""Write SUMMARY.md / SUMMARY.json for runs/eval/baseline_yolov8s_rdd2022_v4 (read-only on the run outputs)."""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/eval/baseline_yolov8s_rdd2022_v4")
f = lambda x, n=3: "nan" if x is None or x != x else f"{x:.{n}f}"
ci = lambda c, n=3: f"[{f(c[0], n)}, {f(c[1], n)}]"
SETS = [("rdd2022 val", "rdd2022_val_at_pooled"), ("hrp4k val", "hrp4k_val_at_pooled"),
        ("rdd2022 test", "rdd2022_test"), ("hrp4k test", "hrp4k_test"), ("kaggle probe", "kaggle_probe")]
J = lambda p: json.loads(Path(p).read_text())
out, md = {}, []
max_diff = 0.0; max_where = None
md.append("# Baseline yolov8s_rdd2022, evaluation v4\n")
md.append("Fixed protocol: eval batch 8, conf 0.001, NMS IoU 0.7, match IoU 0.5; threshold = max-F1 confidence selected on pooled val "
          "(unified_v1) at the same imgsz, applied unchanged elsewhere; group bootstrap, 1000 resamples, seed 0, 95% percentile CIs. "
          "mAP50 / mAP50-95 are the script's own recomputation (CI around it); Ultralytics value shown for agreement. "
          "'val' rows use the pooled-val threshold on a subset of the selection list (in-sample for the threshold).\n")
for sz in sorted(p.name for p in R.iterdir() if p.is_dir() and p.name.isdigit()):
    ops = J(R / sz / "pooled_val" / "operating_points.json")
    thr = ops["thresholds"]["max_f1"]
    o = out[sz] = dict(pooled_val_threshold=thr, sets={}, selects={})
    md.append(f"## imgsz {sz}  (pooled-val max-F1 conf = {thr:.4f}; pooled-val max F1 = {f(ops['max_f1'], 4)})\n")
    md.append("| set | n img / n GT / n empty | mAP50 [CI] (Ultralytics) | mAP50-95 [CI] (Ultralytics) | P [CI] | R [CI] | FP/img [CI] | FP/empty img | empty imgs with detection | in-sample |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, d in SETS:
        m = J(R / sz / d / "metrics.json")
        a = m["ap_agreement"]
        diff = a["abs_diff_mAP50"]
        if diff > max_diff:
            max_diff, max_where = diff, f"{sz}/{d}"
        tf, op = m["threshold_free"], m["operating_points"]["max_f1"]
        c = op["ci95"]
        o["sets"][name] = dict(dir=d, n_images=m["n_images"], n_gt=m["n_instances"], n_empty=m["n_negative_images"],
                               mAP50_own=a["own"]["mAP50"], mAP50_ultralytics=a["ultralytics"]["mAP50"], mAP50_ci95=tf["mAP50_ci95"],
                               mAP50_95_own=a["own"]["mAP50_95"], mAP50_95_ultralytics=a["ultralytics"]["mAP50_95"],
                               mAP50_95_ci95=tf["mAP50_95_ci95_own"], abs_diff_mAP50=diff, abs_diff_mAP50_95=a["abs_diff_mAP50_95"],
                               operating_point={k: op[k] for k in ("conf_threshold", "TP", "FP", "FN", "precision", "recall", "f1",
                                                                   "fp_per_image", "fp_per_negative_image", "negative_images_with_detection")},
                               operating_point_ci95=c, in_sample=m["in_sample"])
        md.append(f"| {name} | {m['n_images']} / {m['n_instances']} / {m['n_negative_images']} | "
                  f"{f(a['own']['mAP50'])} {ci(tf['mAP50_ci95'])} ({f(a['ultralytics']['mAP50'])}) | "
                  f"{f(a['own']['mAP50_95'])} {ci(tf['mAP50_95_ci95_own'])} ({f(a['ultralytics']['mAP50_95'])}) | "
                  f"{f(op['precision'])} {ci(c['precision'])} | {f(op['recall'])} {ci(c['recall'])} | "
                  f"{f(op['fp_per_image'])} {ci(c['fp_per_image'])} | {f(op['fp_per_negative_image'])} | "
                  f"{f(op['negative_images_with_detection'])} | {m['in_sample']} |")
    md.append("")
    md.append("Per-source select runs (threshold = that source's own max-F1; in-sample):\n")
    for s in ("rdd2022", "hrp4k"):
        m = J(R / sz / f"{s}_val" / "metrics.json"); tf = m["threshold_free"]; a = m["ap_agreement"]
        diff = a["abs_diff_mAP50"]
        if diff > max_diff:
            max_diff, max_where = diff, f"{sz}/{s}_val"
        sel = J(R / sz / f"{s}_val" / "operating_points.json")
        o["selects"][s] = dict(mAP50_own=a["own"]["mAP50"], mAP50_ultralytics=a["ultralytics"]["mAP50"], mAP50_ci95=tf["mAP50_ci95"],
                               mAP50_95_own=a["own"]["mAP50_95"], max_f1_conf=sel["max_f1_conf"], max_f1=sel["max_f1"],
                               recall_0_8_conf=sel.get("target_recall_conf"))
        md.append(f"- {s} val: mAP50 {f(a['own']['mAP50'])} {ci(tf['mAP50_ci95'])} (Ultralytics {f(a['ultralytics']['mAP50'])}), "
                  f"max-F1 conf {f(sel['max_f1_conf'], 4)} (F1 {f(sel['max_f1'])}), conf at recall 0.8: {f(sel.get('target_recall_conf'), 4)}")
    # rdd2022 test breakdown by country
    m = J(R / sz / "rdd2022_test" / "metrics.json")
    bk = m.get("breakdown_by_key")
    if bk:
        md.append("\nrdd2022 test, breakdown by country key (pooled-val threshold):\n")
        md.append("| key | n img | n GT | P | R | FP/img |"); md.append("|---|---|---|---|---|---|")
        o["rdd2022_test_by_key"] = bk["max_f1"]
        for k, v in bk["max_f1"].items():
            md.append(f"| {k} | {int(v['n_images'])} | {int(v['n_gt'])} | {f(v['precision'])} | {f(v['recall'])} | {f(v['fp_per_image'])} |")
    pm = J(R / sz / "kaggle_probe" / "metrics.json")
    mp, es = pm["manhole_probe"], pm["easy_subset_positive_frames"]
    o["manhole_probe"] = mp; o["probe_easy_subset"] = es
    c = mp["ci95"]
    md.append("\nManhole probe (Kaggle probe, threshold from pooled val):\n")
    md.append(f"- intact-manhole boxes: {mp['n_manhole_boxes']}; hit share by IoU>=0.3: {f(mp['manhole_hit_share_iou03'])} {ci(c['manhole_hit_share_iou03'])} "
              f"({mp['hits_iou03']} hits); hit share by prediction centre inside box: {f(mp['manhole_hit_share_centre_inside'])} "
              f"{ci(c['manhole_hit_share_centre_inside'])} ({mp['hits_centre_inside']} hits)")
    md.append(f"- predictions per pothole-free frame: {f(mp['preds_per_pothole_free_image'])} {ci(c['preds_per_pothole_free_image'])} "
              f"({mp['total_preds_on_pothole_free_images']} preds on {mp['n_pothole_free_images']} frames); "
              f"pothole-free frames with a prediction: {f(mp['pothole_free_images_with_pred'])} {ci(c['pothole_free_images_with_pred'])}")
    md.append(f"- easy subset (frames with reviewed pothole boxes; not a real-world recall estimate): P {f(es['precision'])} {ci(es['ci95']['precision'])}, "
              f"R {f(es['recall'])} {ci(es['ci95']['recall'])}, n={int(es['n_images'])} images, {int(es['n_gt'])} GT\n")
out["max_abs_diff_own_vs_ultralytics_mAP50"] = dict(value=max_diff, where=max_where)
md.insert(2, f"Maximum |own AP50 - Ultralytics mAP50| over all evaluated sets: {max_diff:.6f} ({max_where}).\n")
(R / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(R / "SUMMARY.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\n".join(md))
