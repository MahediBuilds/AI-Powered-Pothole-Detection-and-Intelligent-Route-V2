"""Compare a candidate against the baseline from two run_model_eval.sh output roots.

    python scripts/compare_eval_runs.py BASELINE_ROOT CANDIDATE_ROOT --baseline-size 640 --candidate-size 1280 \
        --out-dir runs/eval/compare_X [--baseline-tag B] [--candidate-tag C] [--note TEXT]

For rdd2022 test, hrp4k test and the Kaggle probe it reports both models' mAP50, mAP50-95, precision, recall,
F1, FP/image (point + 95% group-bootstrap CI), the PAIRED difference candidate minus baseline with its 95%
interval (same group resamples; refuses if the lists/groups/seeds differ), HRP4K FP per empty image, the
manhole hit shares (both definitions) and predictions per pothole-free frame, recall by box-size bin and the
rdd2022 per-country table.

Pre-registered matched-recall comparison: for each source, the baseline's recall at its own pooled-val
threshold is measured on that source's VALIDATION list (rdd2022_val_at_pooled / hrp4k_val_at_pooled; the Kaggle
probe has no validation list, so the pooled-val list is used). The candidate confidence with the same recall on
the same VALIDATION sweep is found (linear interpolation, eval_checkpoint.select_thresholds) and the candidate is
then run once on the test list at that explicit confidence (eval_checkpoint.py --explicit-conf; this needs the
GPU unless a matching run already exists under OUT_DIR/matched). Both models' test precision / FP per image are
reported at their respective thresholds. Neither evaluation root is written to.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


cm = _load("compare_models")
ec = _load("eval_checkpoint")

U = "data/processed/unified_v1"; K = "data/processed/kaggle_probe_v1"
TEST_SETS = {
    "rdd2022 test": dict(dir="rdd2022_test", val="rdd2022_val", val_at_pooled="rdd2022_val_at_pooled",
                         args=["--data", f"{U}/eval/rdd2022_test.yaml", "--split", "test", "--groups", f"{U}/groups_rdd2022.csv"]),
    "hrp4k test": dict(dir="hrp4k_test", val="hrp4k_val", val_at_pooled="hrp4k_val_at_pooled",
                       args=["--data", f"{U}/eval/hrp4k_test.yaml", "--split", "test", "--groups", f"{U}/groups_hrp4k.csv"]),
    "kaggle probe": dict(dir="kaggle_probe", val="pooled_val", val_at_pooled="pooled_val",
                         args=["--data", f"{K}/eval/probe.yaml", "--split", "test", "--groups", f"{K}/groups_v2.csv",
                               "--manhole-boxes", f"{K}/manhole_boxes.csv"]),
}
PRIMARY = "max_f1"
MATCHED = ec.EXPLICIT_NAME
J = lambda p: json.loads(Path(p).read_text())


def f3(x, n=3):
    return "n/a" if x is None or x != x else f"{x:.{n}f}"


def ci_s(c, n=3):
    return "n/a" if c is None else f"[{f3(c[0], n)}, {f3(c[1], n)}]"


def f1_samples(samples, name=PRIMARY):
    """Per-resample F1 from the stored precision/recall resamples (exact: F1 is a function of the two)."""
    p = np.asarray(samples[f"{name}__precision"], float); r = np.asarray(samples[f"{name}__recall"], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = 2 * p * r / (p + r)
    f[~np.isfinite(f)] = np.nan
    return f


def pct_ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None


def model_block(metrics, samples):
    a = metrics["ap_agreement"]["own"]; tf = metrics["threshold_free"]
    op = metrics["operating_points"][PRIMARY]; c = op["ci95"]
    return dict(
        n_images=metrics["n_images"], n_gt=metrics["n_instances"], n_empty=metrics["n_negative_images"],
        conf_threshold=op["conf_threshold"],
        mAP50=a["mAP50"], mAP50_ci95=tf["mAP50_ci95"], mAP50_95=a["mAP50_95"], mAP50_95_ci95=tf["mAP50_95_ci95_own"],
        precision=op["precision"], precision_ci95=c["precision"], recall=op["recall"], recall_ci95=c["recall"],
        f1=op["f1"], f1_ci95=pct_ci(f1_samples(samples)), fp_per_image=op["fp_per_image"], fp_per_image_ci95=c["fp_per_image"],
        fp_per_negative_image=op["fp_per_negative_image"], negative_images_with_detection=op["negative_images_with_detection"],
        recall_by_size=dict(metrics["breakdown_by_gt_size"]["bins"][PRIMARY]),
        manhole_probe=metrics.get("manhole_probe"))


def paired_block(sb, sc, pb, pc):
    sb = dict(sb); sc = dict(sc); pb = dict(pb); pc = dict(pc)
    sb["f1"] = f1_samples(sb); sc["f1"] = f1_samples(sc)
    pb["f1"] = pb.pop(f"{PRIMARY}__f1"); pc["f1"] = pc.pop(f"{PRIMARY}__f1")
    ren = {"mAP50": "mAP50", "mAP50_95": "mAP50_95", f"{PRIMARY}__precision": "precision", f"{PRIMARY}__recall": "recall",
           "f1": "f1", f"{PRIMARY}__fp_per_image": "fp_per_image"}
    out = {}
    for k, v in cm.paired_diff(sb, sc, pb, pc).items():
        if k in ren:
            out[ren[k]] = v
        elif k.startswith("probe_"):
            out[k] = v
    return out


def load_run(d):
    """(eval_config, groups, bootstrap samples, point metrics, metrics.json) of one eval_checkpoint output dir."""
    d = Path(d)
    cfg = J(d / "eval_config.json")
    with open(d / "bootstrap_groups.csv", newline="") as fh:
        groups = [tuple(r) for r in list(csv.reader(fh))[1:]]
    m = J(d / "metrics.json")
    pts = cm.points(m)
    for n, op in m["operating_points"].items():
        pts[f"{n}__f1"] = op["f1"]
    return cfg, groups, cm.read_samples(d / "bootstrap_samples.csv"), pts, m


def matched_threshold(broot, bsize, croot, csize, ts):
    """Target recall = baseline recall at its pooled threshold on the source's val; candidate conf at that recall."""
    bm = J(Path(broot) / bsize / ts["val_at_pooled"] / "metrics.json")
    tgt = bm["operating_points"][PRIMARY]["recall"]
    sw = J(Path(croot) / csize / ts["val"] / "operating_points.json")["sweep"]
    sel = ec.select_thresholds(sw, tgt)
    return dict(target_recall=tgt, baseline_val_dir=ts["val_at_pooled"], candidate_val_dir=ts["val"],
                candidate_conf=sel["target_recall_conf"],
                candidate_val_max_recall=float(max(sw["recall"])) if sw["recall"] else None, note=sel.get("note_target"))


def run_matched(ck, conf, ts, out, imgsz, country_csv, device):
    out = Path(out)
    if (out / "eval_config.json").exists() and (out / "metrics.json").exists():
        c = J(out / "eval_config.json")
        if c.get("explicit_conf") is not None and abs(c["explicit_conf"] - conf) < 1e-12 and c["checkpoint_sha256"] == ec.sha256(ck) \
                and c["imgsz"] == imgsz:
            return "reused"
    cmd = [sys.executable, str(HERE / "eval_checkpoint.py"), "--mode", "apply", "--checkpoint", str(ck), "--imgsz", str(imgsz),
           "--batch", "8", "--device", device, "--explicit-conf", repr(conf), *ts["args"], "--out", str(out)]
    if ts["dir"] == "rdd2022_test":
        cmd += ["--breakdown-csv", str(country_csv)]
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    so, se = r.stdout or "", r.stderr or ""
    (out.parent / f"{out.name}.log").write_text(so[-4000:] + "\n--- stderr ---\n" + se[-4000:], encoding="utf-8")
    if r.returncode:
        raise RuntimeError(f"matched-recall run failed for {out.name}: {se[-800:]}")
    return "ran"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("baseline_root"); ap.add_argument("candidate_root")
    ap.add_argument("--baseline-size", required=True); ap.add_argument("--candidate-size", required=True)
    ap.add_argument("--baseline-tag"); ap.add_argument("--candidate-tag")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--note", default="", help="text placed at the top of the report (e.g. a DRY RUN marker)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-matched-run", action="store_true", help="only plan the matched thresholds; skip the GPU runs")
    a = ap.parse_args(argv)
    broot, croot = Path(a.baseline_root), Path(a.candidate_root)
    bs, cs = str(a.baseline_size), str(a.candidate_size)
    btag, ctag = a.baseline_tag or broot.name, a.candidate_tag or croot.name
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    ro = out.resolve()
    for bad in ("runs/train", "model_archive"):
        if (ROOT / bad).resolve() in [ro, *ro.parents]:
            sys.exit(f"refusing to write inside {bad}")
    if ro in [broot.resolve(), croot.resolve()] or broot.resolve() in ro.parents or croot.resolve() in ro.parents:
        sys.exit("out-dir must be outside both evaluation roots")

    res = dict(note=a.note, baseline=dict(tag=btag, root=str(broot), imgsz=int(bs)), candidate=dict(tag=ctag, root=str(croot), imgsz=int(cs)),
               difference="candidate minus baseline; CI = 95% percentile interval of paired group-bootstrap differences", sets={})
    for name, ts in TEST_SETS.items():
        try:
            cb, gb, sb, pb, mb = load_run(broot / bs / ts["dir"])
            cc, gc, sc, pc, mc = load_run(croot / cs / ts["dir"])
            cm.check_pairable(cb, cc, gb, gc)
        except (cm.NotPairable, FileNotFoundError) as e:
            sys.exit(f"REFUSED ({name}): {e}")
        res["sets"][name] = dict(baseline=model_block(mb, sb), candidate=model_block(mc, sc), paired=paired_block(sb, sc, pb, pc),
                                 n_boot=cb["boot_n"], seed=cb["boot_seed"], image_list_sha256=cb["image_list_sha256"])
        if name == "rdd2022 test":
            g = lambda m: (m.get("breakdown_by_key") or {}).get(PRIMARY)
            res["rdd2022_by_country"] = dict(baseline=g(mb), candidate=g(mc))

    ck = Path(J(croot / cs / "pooled_val" / "eval_config.json")["checkpoint"])
    country_csv = next((p for p in (croot / "aux" / "rdd2022_country.csv", broot / "aux" / "rdd2022_country.csv") if p.exists()), None)
    matched = {}
    for name, ts in TEST_SETS.items():
        mt = matched_threshold(broot, bs, croot, cs, ts)
        entry = dict(match=mt, status="planned only (--no-matched-run)")
        if mt["candidate_conf"] is None:
            entry["status"] = "candidate never reaches the target recall on validation"
        elif not a.no_matched_run:
            od = out / "matched" / ts["dir"]
            entry["status"] = run_matched(ck, mt["candidate_conf"], ts, od, int(cs), country_csv, a.device)
            cx, gx, sx, _, mx = load_run(od)
            cb, gb, sb, _, mb = load_run(broot / bs / ts["dir"])
            cm.check_pairable(cb, cx, gb, gx)
            op = mx["operating_points"][MATCHED]; c = op["ci95"]; bop = mb["operating_points"][PRIMARY]; bc = bop["ci95"]
            keys = ("conf_threshold", "precision", "recall", "fp_per_image", "fp_per_negative_image")
            entry["candidate_at_matched_conf"] = dict({k: op[k] for k in keys}, precision_ci95=c["precision"], recall_ci95=c["recall"],
                                                      fp_per_image_ci95=c["fp_per_image"], run_dir=str(od))
            entry["baseline_at_own_threshold"] = dict({k: bop[k] for k in keys}, precision_ci95=bc["precision"], recall_ci95=bc["recall"],
                                                      fp_per_image_ci95=bc["fp_per_image"])
            col = lambda n, k: f"{n}__{k}"
            sx2 = {k: sx[col(MATCHED, k)] for k in ("precision", "recall", "fp_per_image")}
            sb2 = {k: sb[col(PRIMARY, k)] for k in ("precision", "recall", "fp_per_image")}
            entry["paired"] = cm.paired_diff(sb2, sx2, {k: bop[k] for k in sb2}, {k: op[k] for k in sx2})
        matched[name] = entry
    res["matched_recall"] = matched
    md = render_md(res)
    (out / "COMPARE.json").write_text(json.dumps(res, indent=2))
    (out / "COMPARE.md").write_text(md, encoding="utf-8")
    print(md)


# ----------------------------------------------------------------------------- markdown
def _d(p):
    return "n/a" if p is None else f"{f3(p['point_diff'], 4)} {ci_s(p['ci95'], 4)}{' *' if p['ci_excludes_zero'] else ''}"


def render_md(r):
    b, c = r["baseline"], r["candidate"]
    md = []
    if r["note"]:
        md.append(f"> **{r['note']}**\n")
    md.append(f"# Comparison: candidate `{c['tag']}` (imgsz {c['imgsz']}) vs baseline `{b['tag']}` (imgsz {b['imgsz']})\n")
    md.append("Each model uses its own pooled-validation max-F1 confidence (selected at its own imgsz), applied unchanged. "
              "CIs: 95% percentile, group bootstrap. Difference = candidate minus baseline, from paired resamples (* = interval excludes 0). "
              "mAP values are the script's own recomputation (agrees with Ultralytics). F1 intervals are derived per resample "
              "from the stored precision and recall resamples.\n")
    metrics = [("mAP50", "mAP50"), ("mAP50_95", "mAP50-95"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
               ("fp_per_image", "FP / image")]
    for name, s in r["sets"].items():
        B, C, P = s["baseline"], s["candidate"], s["paired"]
        md.append(f"## {name}  ({B['n_images']} images, {B['n_gt']} GT boxes, {B['n_empty']} empty; thresholds baseline "
                  f"{B['conf_threshold']:.4f} / candidate {C['conf_threshold']:.4f})\n")
        md.append("| metric | baseline [CI] | candidate [CI] | candidate - baseline [95% CI] |"); md.append("|---|---|---|---|")
        for k, lab in metrics:
            md.append(f"| {lab} | {f3(B[k])} {ci_s(B.get(k + '_ci95'))} | {f3(C[k])} {ci_s(C.get(k + '_ci95'))} | {_d(P.get(k))} |")
        md.append("")
        if name == "hrp4k test":
            md.append(f"FP per empty image: baseline {f3(B['fp_per_negative_image'], 4)}, candidate {f3(C['fp_per_negative_image'], 4)} "
                      f"({B['n_empty']} empty images; share of empty images with a detection: baseline "
                      f"{f3(B['negative_images_with_detection'], 4)}, candidate {f3(C['negative_images_with_detection'], 4)}).\n")
        md.append("Recall by box size (sqrt(area)/image width):\n")
        md.append("| bin | n GT | baseline recall | candidate recall |"); md.append("|---|---|---|---|")
        for k, v in B["recall_by_size"].items():
            md.append(f"| {k} | {int(v['n_gt'])} | {f3(v['recall'])} | {f3(C['recall_by_size'][k]['recall'])} |")
        md.append("")
        if name == "kaggle probe" and B.get("manhole_probe"):
            mb, mc = B["manhole_probe"], C["manhole_probe"]
            md.append(f"Manhole probe ({mb['n_manhole_boxes']} intact-manhole boxes, {mb['n_pothole_free_images']} pothole-free frames):\n")
            md.append("| quantity | baseline [CI] | candidate [CI] | candidate - baseline [95% CI] |"); md.append("|---|---|---|---|")
            for k, lab in [("manhole_hit_share_iou03", "manhole hit share, IoU>=0.3"),
                           ("manhole_hit_share_centre_inside", "manhole hit share, centre inside box"),
                           ("preds_per_pothole_free_image", "predictions per pothole-free frame"),
                           ("pothole_free_images_with_pred", "pothole-free frames with a prediction")]:
                md.append(f"| {lab} | {f3(mb[k])} {ci_s(mb['ci95'][k])} | {f3(mc[k])} {ci_s(mc['ci95'][k])} | {_d(P.get('probe_' + k))} |")
            md.append("")
    bk = r.get("rdd2022_by_country") or {}
    if bk.get("baseline") and bk.get("candidate"):
        md.append("## rdd2022 test per country (each model at its own threshold)\n")
        md.append("| country | n img | n GT | P base | P cand | R base | R cand | FP/img base | FP/img cand |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for k, v in bk["baseline"].items():
            w = bk["candidate"].get(k)
            if w:
                md.append(f"| {k} | {int(v['n_images'])} | {int(v['n_gt'])} | {f3(v['precision'])} | {f3(w['precision'])} | {f3(v['recall'])} | "
                          f"{f3(w['recall'])} | {f3(v['fp_per_image'])} | {f3(w['fp_per_image'])} |")
        md.append("")
    md.append("## Pre-registered matched-recall comparison\n")
    md.append("Target recall = baseline recall at its pooled-val threshold, measured on the source's validation list (Kaggle probe: pooled val, "
              "no probe val exists). Candidate confidence = where its recall on the same validation list equals that target. Test precision and "
              "FP per image are then reported for both models.\n")
    md.append("| set | target recall (val) | baseline conf | candidate conf | baseline test P [CI] | candidate test P [CI] | dP [95% CI] | "
              "baseline FP/img [CI] | candidate FP/img [CI] | dFP/img [95% CI] | test recall base / cand |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, e in r["matched_recall"].items():
        mt = e["match"]
        if "candidate_at_matched_conf" not in e:
            md.append(f"| {name} | {f3(mt['target_recall'])} | | {f3(mt['candidate_conf'], 4)} | | | | | | | {e['status']} |")
            continue
        cb, cc, p = e["baseline_at_own_threshold"], e["candidate_at_matched_conf"], e["paired"]
        md.append(f"| {name} | {f3(mt['target_recall'])} | {f3(cb['conf_threshold'], 4)} | {f3(cc['conf_threshold'], 4)} | "
                  f"{f3(cb['precision'])} {ci_s(cb['precision_ci95'])} | {f3(cc['precision'])} {ci_s(cc['precision_ci95'])} | {_d(p['precision'])} | "
                  f"{f3(cb['fp_per_image'])} {ci_s(cb['fp_per_image_ci95'])} | {f3(cc['fp_per_image'])} {ci_s(cc['fp_per_image_ci95'])} | "
                  f"{_d(p['fp_per_image'])} | {f3(cb['recall'])} / {f3(cc['recall'])} |")
    md.append("\nCaveat: the baseline's validation recall at its pooled threshold is in-sample for that threshold (the pooled val list contains each "
              "source's val list); the candidate's matched confidence uses validation data only, no test data.\n")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    main()
