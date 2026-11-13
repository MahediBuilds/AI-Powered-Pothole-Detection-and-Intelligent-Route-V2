"""Investigation 3: validation-only ensemble of baseline@640 + candidate@960 (WBF and concat-NMS comparator).

Grid, selection rule and cross-fit are declared in runs/eval/val_investigations_v1/ensemble/GRID.md (written before
the first run). Reads cached predictions only.

    .venv/Scripts/python.exe scripts/val_investigations/ensemble.py

Outputs under runs/eval/val_investigations_v1/ensemble/: fused/*.npz (fused predictions per setting), grid_insample.csv,
selection_by_seed.csv, oof_comparison.csv, paired_differences.json, RESULTS.md.
"""
from __future__ import annotations

import csv
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import fusion as F  # noqa: E402

OUTD = C.OUT / "ensemble"
A_MODEL, A_SZ = "baseline", 640
B_MODEL, B_SZ = "candidate", 960
SINGLES = [("baseline", 640, "baseline@640"), ("candidate", 640, "candidate@640"),
           ("candidate", 960, "candidate@960"), ("candidate", 1280, "candidate@1280")]
SETS = ["rdd2022_val", "hrp4k_val"]
WBF_IOU = [0.5, 0.55, 0.6, 0.7]
WBF_W = [(1, 1), (1, 2), (2, 1)]
WBF_SKIP = [0.001, 0.05]
WBF_CONF = ["avg", "max", "box_and_model_avg"]
NMS_IOU = [0.5, 0.7]
FOLD_SEEDS = list(range(20))
N_BOOT = 1000


def settings():
    s = []
    for iou, w, sk, ct in itertools.product(WBF_IOU, WBF_W, WBF_SKIP, WBF_CONF):
        s.append(dict(kind="wbf", name=f"wbf_i{iou:.2f}_w{w[0]}-{w[1]}_s{sk}_{ct}", iou=iou, weights=w, skip=sk, conf=ct))
    for iou in NMS_IOU:
        s.append(dict(kind="nms", name=f"nms_i{iou:.2f}", iou=iou))
    return s


def fuse_setting(pool, pa, pb, st):
    out = {}
    for stem in pool.stems:
        ba, ca = pa[stem]
        bb, cb = pb[stem]
        if st["kind"] == "wbf":
            out[stem] = F.wbf_fuse(ba, ca, bb, cb, pool.sizes[stem], st["iou"], st["skip"], st["weights"], st["conf"])
        else:
            out[stem] = F.concat_nms(ba, ca, bb, cb, st["iou"])
    return out


# ----------------------------------------------------------------------------- metrics on Preds
def maxf1_threshold(P, w):
    """Max-F1 confidence over every distinct confidence on the images weighted by ``w`` (IoU 0.5), tie blocks
    handled as in eval_checkpoint.sweep. Returns (threshold, f1)."""
    wp = w[P.p_img]
    nz = wp > 0
    n_gt = float(w @ P.gt_count)
    if not nz.any() or n_gt <= 0:
        return 1.0, 0.0
    conf = P.p_conf[nz]
    tp0 = P.p_tp[nz, 0]
    tpc = np.cumsum(wp[nz] * tp0)
    fpc = np.cumsum(wp[nz] * ~tp0)
    last = np.r_[conf[1:] != conf[:-1], True]
    tp, fp, c = tpc[last], fpc[last], conf[last]
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / n_gt
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    i = int(f1.argmax())
    return float(c[i]), float(f1[i])


def compose(Ps, choice):
    """Preds over the whole pool where image i takes its detections (and TP flags) from Ps[choice[i]]."""
    choice = np.asarray(choice)
    p_img, p_conf, p_tp = [], [], []
    for k, P in enumerate(Ps):
        m = choice[P.p_img] == k
        p_img.append(P.p_img[m]); p_conf.append(P.p_conf[m]); p_tp.append(P.p_tp[m])
    P0 = Ps[0]
    return C.ec.Preds(P0.n_images, P0.gt_count, P0.gt_img, P0.gt_size_frac, P0.gt_match_conf,
                      np.concatenate(p_img), np.concatenate(p_conf), np.concatenate(p_tp), P0.iou_thrs)


def per_image_counts(Q, thr_img):
    """Per-image TP/FP counts keeping detections with conf >= the threshold of THEIR image."""
    keep = Q.p_conf >= thr_img[Q.p_img]
    tp0 = Q.p_tp[:, 0]
    n = Q.n_images
    tpi = np.bincount(Q.p_img[keep & tp0], minlength=n).astype(float)
    fpi = np.bincount(Q.p_img[keep & ~tp0], minlength=n).astype(float)
    return tpi, fpi


def op_rates(tpi, fpi, gt, w):
    TP, FP = float(w @ tpi), float(w @ fpi)
    ngt, nimg = float(w @ gt), float(w.sum())
    neg = (gt == 0).astype(float)
    nneg = float(w @ neg)
    p = TP / (TP + FP) if TP + FP > 0 else float("nan")
    r = TP / ngt if ngt > 0 else float("nan")
    f1 = 2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else 0.0
    return dict(precision=p, recall=r, f1=f1, fp_per_image=FP / nimg, fp_per_empty=float((w * neg) @ fpi) / nneg if nneg > 0 else float("nan"))


METRICS = ["mAP50", "mAP50_95", "precision", "recall", "f1", "fp_per_image", "fp_per_empty"]


def system_metrics(Q, thr_img, w, gt):
    a = C.ec.ap_all(Q, w)
    m = dict(mAP50=float(a[0]), mAP50_95=float(a.mean()))
    tpi, fpi = per_image_counts(Q, thr_img)
    m.update(op_rates(tpi, fpi, gt, w))
    return m


# ----------------------------------------------------------------------------- main
def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "fused").mkdir(exist_ok=True)
    C.verify_checkpoint("baseline"); C.verify_checkpoint("candidate")
    pool = C.load_pool()
    n = len(pool.stems)
    pa = C.load_model_preds(A_MODEL, A_SZ, pool)
    pb = C.load_model_preds(B_MODEL, B_SZ, pool)
    sts = settings()
    assert sum(s["kind"] == "wbf" for s in sts) == 72
    P = {}
    for st in sts:
        f = OUTD / "fused" / f"{st['name']}.npz"
        if f.exists():
            fused = C.load_preds_npz(f)
            empty = (np.zeros((0, 4)), np.zeros(0))
            fused = {s: fused.get(s, empty) for s in pool.stems}
        else:
            fused = fuse_setting(pool, pa, pb, st)
            C.save_preds(f, fused)
        P[st["name"]] = C.build_P(pool, fused)
    C.progress(f"ensemble: fused + matched {len(sts)} settings")
    single_P = {}
    for model, sz, tag in SINGLES:
        single_P[tag] = C.build_P(pool, C.load_model_preds(model, sz, pool))

    gt = pool.n_gt.astype(float)
    src_w = {s: (pool.source == C.SOURCE_OF_SET[s]).astype(float) for s in SETS}
    wbf_names = [s["name"] for s in sts if s["kind"] == "wbf"]
    nms_names = [s["name"] for s in sts if s["kind"] == "nms"]

    def criterion(Pm, fold, f):
        vals = [C.ap50(Pm, src_w[s] * ((fold == f) if f is not None else 1)) for s in SETS]
        return float(np.mean(vals)), vals

    # in-sample grid table (all validation images) -- reference only
    rows = []
    for name in wbf_names + nms_names:
        crit, vals = criterion(P[name], None, None)
        rows.append([name, crit, vals[0], vals[1], int(len(P[name].p_conf))])
    with open(OUTD / "grid_insample.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["setting", "mean_per_source_mAP50", "mAP50_rdd2022_val", "mAP50_hrp4k_val", "n_fused_boxes"])
        wr.writerows(rows)
    best_in = {k: max((r for r in rows if r[0] in names), key=lambda r: r[1])[0]
               for k, names in (("wbf", wbf_names), ("nms", nms_names))}

    # selection by fold seed
    sel_rows, per_seed = [], {}
    for sd in FOLD_SEEDS:
        fold = C.make_folds(pool, sd)
        res = {}
        for kind, names in (("wbf", wbf_names), ("nms", nms_names)):
            chosen = {}
            for f in (0, 1):  # selection fold
                crits = [criterion(P[nm], fold, f)[0] for nm in names]
                nm = names[int(np.argmax(crits))]
                thr, f1 = maxf1_threshold(P[nm], (fold == f).astype(float))
                chosen[f] = dict(setting=nm, criterion=float(max(crits)), threshold=thr, sel_pooled_f1=f1)
            res[kind] = chosen
        # single-model thresholds (pooled max-F1 on the selection fold)
        res["singles"] = {tag: {f: maxf1_threshold(single_P[tag], (fold == f).astype(float))[0] for f in (0, 1)}
                          for tag in single_P}
        per_seed[sd] = (fold, res)
        for kind in ("wbf", "nms"):
            for f in (0, 1):
                c = res[kind][f]
                sel_rows.append([sd, kind, f, c["setting"], c["criterion"], c["threshold"], c["sel_pooled_f1"]])
    with open(OUTD / "selection_by_seed.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["fold_seed", "family", "selection_fold", "chosen_setting", "criterion_on_selection_fold",
                     "threshold_pooled_maxF1_on_selection_fold", "selection_fold_pooled_F1"])
        wr.writerows(sel_rows)
    C.progress("ensemble: selection by fold seed done")

    # OOF composites for every fold seed: per-source mAP50 + F1 at OOF thresholds (single-IoU quick table)
    robust = []
    for sd, (fold, res) in per_seed.items():
        for kind in ("wbf", "nms"):
            names = [res[kind][1 - f]["setting"] for f in (0, 1)]  # evaluation fold f uses the setting chosen on the other fold
            uniq = sorted(set(names))
            Ps = [P[u] for u in uniq]
            choice = np.array([uniq.index(names[f]) for f in fold])
            Q = compose(Ps, choice)
            thr_img = np.array([res[kind][1 - f]["threshold"] for f in fold])
            for s in SETS:
                m = system_metrics_light(Q, thr_img, src_w[s], gt)
                robust.append(dict(seed=sd, system=f"ensemble_{kind}", set=s, **m))
        for tag, Pm in single_P.items():
            thr_img = np.array([res["singles"][tag][1 - f] for f in fold])
            for s in SETS:
                robust.append(dict(seed=sd, system=tag, set=s, **system_metrics_light(Pm, thr_img, src_w[s], gt)))
    import pandas as pd
    rdf = pd.DataFrame(robust)
    rdf.to_csv(OUTD / "oof_all_fold_seeds.csv", index=False)
    C.progress("ensemble: 20-seed OOF robustness table written")

    # ---- headline: fold seed 0 with full metrics (mAP50-95 too) and paired bootstrap
    fold, res = per_seed[0]
    systems = {}  # name -> (Q, thr_img, info)
    for kind, label in (("wbf", "ensemble_WBF"), ("nms", "ensemble_concat_NMS")):
        names = [res[kind][1 - f]["setting"] for f in (0, 1)]
        uniq = sorted(set(names))
        Ps = [P[u] for u in uniq]
        choice = np.array([uniq.index(names[f]) for f in fold])
        Q = compose(Ps, choice)
        thr_img = np.array([res[kind][1 - f]["threshold"] for f in fold])
        systems[label] = (Q, thr_img, dict(settings_used={f"evaluated_on_fold{f}": names[f] for f in (0, 1)},
                                           thresholds={f"fold{f}": res[kind][1 - f]["threshold"] for f in (0, 1)}))
    for tag, Pm in single_P.items():
        thr_img = np.array([res["singles"][tag][1 - f] for f in fold])
        systems[tag] = (Pm, thr_img, dict(thresholds={f"fold{f}": res["singles"][tag][1 - f] for f in (0, 1)}))

    point = {}
    for name, (Q, thr_img, info) in systems.items():
        for s in SETS:
            point[(name, s)] = system_metrics(Q, thr_img, src_w[s], gt)
    # in-sample reference (best setting on all val images; pooled max-F1 threshold on all images; optimistic)
    insample = {}
    for kind, label in (("wbf", "ensemble_WBF"), ("nms", "ensemble_concat_NMS")):
        nm = best_in[kind]
        thr, f1 = maxf1_threshold(P[nm], np.ones(n))
        thr_img = np.full(n, thr)
        for s in SETS:
            insample[(label, s)] = dict(setting=nm, threshold=thr, **system_metrics(P[nm], thr_img, src_w[s], gt))
    for tag, Pm in single_P.items():
        thr, f1 = maxf1_threshold(Pm, np.ones(n))
        for s in SETS:
            insample[(tag, s)] = dict(setting="-", threshold=thr, **system_metrics(Pm, np.full(n, thr), src_w[s], gt))
    C.progress("ensemble: point estimates done")

    fams = {s: C.boot_weights(pool, pool.idx(s), N_BOOT, 0) for s in SETS}
    boot = {(name, s): {k: np.zeros(N_BOOT) for k in METRICS} for name in systems for s in SETS}
    tpfp = {name: per_image_counts(Q, thr_img) for name, (Q, thr_img, _) in systems.items()}
    for b in range(N_BOOT):
        for s in SETS:
            w = np.zeros(n)
            w[pool.idx(s)] = fams[s][b]
            for name, (Q, thr_img, _) in systems.items():
                a = C.ec.ap_all(Q, w)
                bm = boot[(name, s)]
                bm["mAP50"][b] = a[0]; bm["mAP50_95"][b] = a.mean()
                r = op_rates(*tpfp[name], gt, w)
                for k in ("precision", "recall", "f1", "fp_per_image", "fp_per_empty"):
                    bm[k][b] = r[k]
    C.progress("ensemble: bootstrap done")

    diffs = []
    for s in SETS:
        for other in [x for x in systems if x != "ensemble_WBF"]:
            rec = dict(set=s, ensemble="ensemble_WBF", minus=other)
            for k in METRICS:
                d = boot[("ensemble_WBF", s)][k] - boot[(other, s)][k]
                rec[k] = dict(point=float(point[("ensemble_WBF", s)][k] - point[(other, s)][k]), ci95=C.pctl(d),
                              excludes_zero=bool(np.nanpercentile(d, 2.5) > 0 or np.nanpercentile(d, 97.5) < 0))
            diffs.append(rec)
    C.write_json(OUTD / "paired_differences.json", diffs)
    comp = []
    for (name, s), m in point.items():
        comp.append(dict(system=name, set=s, **{k: m[k] for k in METRICS},
                         **{k + "_ci95": C.pctl(boot[(name, s)][k]) for k in METRICS}))
    pd.DataFrame([{k: (str(v) if isinstance(v, list) else v) for k, v in r.items()} for r in comp]).to_csv(OUTD / "oof_comparison.csv", index=False)
    C.write_json(OUTD / "headline_seed0.json", dict(
        systems={k: v[2] for k, v in systems.items()}, point={f"{a}|{b}": v for (a, b), v in point.items()},
        insample={f"{a}|{b}": v for (a, b), v in insample.items()}, best_insample_setting=best_in))
    write_results_md(sts, rows, best_in, per_seed, rdf, point, insample, systems, diffs, boot)
    C.progress("ensemble: DONE (RESULTS.md written)")
    C.verify_checkpoint("baseline"); C.verify_checkpoint("candidate")


def system_metrics_light(Q, thr_img, w, gt):
    m = dict(mAP50=C.ap50(Q, w))
    tpi, fpi = per_image_counts(Q, thr_img)
    m.update(op_rates(tpi, fpi, gt, w))
    return m


# ----------------------------------------------------------------------------- report
def ci(v, n=3):
    return f"[{v[0]:+.{n}f}, {v[1]:+.{n}f}]"


def write_results_md(sts, rows, best_in, per_seed, rdf, point, insample, systems, diffs, boot):
    L = ["# Ensemble results (validation only)", "",
         "Ensemble = archived baseline@640 + candidate@960; WBF = Weighted Boxes Fusion. Selection of the grid setting and the "
         "operating threshold is 2-fold cross-fitted by group (see GRID.md); seed-0 folds are the headline. "
         "HRP4K val is in-domain for the candidate and out-of-domain for the baseline.", "",
         "## 1. Out-of-fold comparison (fold seed 0; threshold = pooled max-F1 on the selection fold, applied to the other fold)", "",
         "| set | system | mAP50 [95% CI] | mAP50-95 [95% CI] | P | R | F1 | FP/img | FP/empty img |", "|---|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        for name in systems:
            m = point[(name, s)]
            cis = {k: C.pctl(boot[(name, s)][k]) for k in ("mAP50", "mAP50_95")}
            L.append(f"| {s} | {name} | {m['mAP50']:.3f} [{cis['mAP50'][0]:.3f}, {cis['mAP50'][1]:.3f}] | "
                     f"{m['mAP50_95']:.3f} [{cis['mAP50_95'][0]:.3f}, {cis['mAP50_95'][1]:.3f}] | {m['precision']:.3f} | "
                     f"{m['recall']:.3f} | {m['f1']:.3f} | {m['fp_per_image']:.3f} | {m['fp_per_empty']:.3f} |")
    L += ["", "Settings and thresholds used for the ensembles (evaluation fold <- setting chosen on the other fold):", ""]
    for name, (_, _, info) in systems.items():
        L.append(f"- {name}: {info}")
    L += ["", "## 2. Paired differences, ensemble_WBF minus other system (group bootstrap 95% interval; * = interval excludes zero)", "",
          "| set | minus | dmAP50 | dmAP50-95 | dP | dR | dF1 | dFP/img | dFP/empty img |", "|---|---|---|---|---|---|---|---|---|"]
    for d in diffs:
        cells = []
        for k in METRICS:
            x = d[k]
            cells.append(f"{x['point']:+.3f} {ci(x['ci95'])}{'*' if x['excludes_zero'] else ''}")
        L.append(f"| {d['set']} | {d['minus']} | " + " | ".join(cells) + " |")
    L += ["", "## 3. In-sample reference (best setting and pooled max-F1 threshold chosen on ALL validation images; optimistic)", "",
          "| set | system | setting | thr | mAP50 | mAP50-95 | P | R | F1 | FP/img |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (name, s), m in insample.items():
        L.append(f"| {s} | {name} | {m['setting']} | {m['threshold']:.3f} | {m['mAP50']:.3f} | {m['mAP50_95']:.3f} | {m['precision']:.3f} | "
                 f"{m['recall']:.3f} | {m['f1']:.3f} | {m['fp_per_image']:.3f} |")
    top = sorted((r for r in rows if r[0].startswith("wbf")), key=lambda r: -r[1])[:10]
    L += ["", "## 4. Grid, in-sample criterion (mean of per-source mAP50 on all validation images): top 10 WBF settings and spread", "",
          "| setting | mean per-source mAP50 | RDD2022 val | HRP4K val | fused boxes |", "|---|---|---|---|---|"]
    for r in top:
        L.append(f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]} |")
    wb = [r[1] for r in rows if r[0].startswith("wbf")]
    L.append(f"\nAll 72 WBF settings: criterion min {min(wb):.4f}, median {float(np.median(wb)):.4f}, max {max(wb):.4f}. "
             f"Concat-NMS settings: " + ", ".join(f"{r[0]} {r[1]:.4f}" for r in rows if r[0].startswith("nms")) + ".")
    cnt = Counter()
    for sd, (fold, res) in per_seed.items():
        for f in (0, 1):
            cnt[res["wbf"][f]["setting"]] += 1
    L += ["", "## 5. Selection stability over 20 fold seeds (40 selections)", "",
          "| WBF setting | times selected |", "|---|---|"] + [f"| {k} | {v} |" for k, v in cnt.most_common()]
    L += ["", "Out-of-fold results over 20 fold seeds (mean, min, max); single-IoU quick metrics:", "",
          "| set | system | mAP50 mean (min-max) | F1 mean (min-max) | P mean | R mean | FP/img mean |", "|---|---|---|---|---|---|---|"]
    g = rdf.groupby(["set", "system"])
    for (s, sysn), d in g:
        L.append(f"| {s} | {sysn} | {d.mAP50.mean():.3f} ({d.mAP50.min():.3f}-{d.mAP50.max():.3f}) | "
                 f"{d.f1.mean():.3f} ({d.f1.min():.3f}-{d.f1.max():.3f}) | {d.precision.mean():.3f} | {d.recall.mean():.3f} | "
                 f"{d.fp_per_image.mean():.3f} |")
    (OUTD / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
