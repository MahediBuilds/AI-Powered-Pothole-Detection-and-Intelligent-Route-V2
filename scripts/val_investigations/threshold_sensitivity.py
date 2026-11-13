"""Investigation 2: confidence-threshold sensitivity on VALIDATION data only.

Models: baseline@640, candidate@1280, candidate@960. Sets: rdd2022_val, hrp4k_val, pooled (= both, concatenated).
Reads the cached predictions (predict_cache.py). IoU-0.5 greedy matching (same matcher as eval_checkpoint.py).

    .venv/Scripts/python.exe scripts/val_investigations/threshold_sensitivity.py

Writes runs/eval/val_investigations_v1/threshold_sensitivity/ (CSV, JSON, PNG, RESULTS.md).
Group bootstrap: 1000 resamples, seed 0, 95% percentile; groups are resampled within each set (pooled: over
all groups); resamples are identical for every model. Thresholds that are 'chosen on the full set' are held
fixed inside the bootstrap, so the intervals describe evaluation noise, not selection noise (the cross-fit
numbers address the selection optimism). 2-fold cross-fit folds: C.make_folds (group-level, stratified by source).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODELS = [("baseline", 640, "baseline@640"), ("candidate", 1280, "candidate@1280"), ("candidate", 960, "candidate@960")]
SETS = ["rdd2022_val", "hrp4k_val", "pooled"]
MILLI = np.arange(1, 951)
T_FINE = MILLI / 1000.0
SW_COLS = np.arange(49, 800, 10)            # thresholds 0.05 .. 0.80 step 0.01 (index = milli - 1)
T_SW = T_FINE[SW_COLS]
BUDGETS = [0.05, 0.10, 0.20]
RECALLS = [0.3, 0.4, 0.5]
N_BOOT = 1000
FOLD_SEEDS = list(range(20))
OUTD = C.OUT / "threshold_sensitivity"


# ----------------------------------------------------------------------------- helpers
def set_weights(pool, sname):
    w = np.zeros(len(pool.stems))
    if sname == "pooled":
        w[:] = 1
    else:
        w[pool.idx(sname)] = 1
    return w


def plateau(thr, f1, delta):
    """Contiguous threshold range around the argmax where F1 >= max - delta -> (lo, hi)."""
    i = C.argmax_first(f1)
    lo = hi = i
    while lo > 0 and f1[lo - 1] >= f1[i] - delta:
        lo -= 1
    while hi < len(f1) - 1 and f1[hi + 1] >= f1[i] - delta:
        hi += 1
    return float(thr[lo]), float(thr[hi])


def budget_stats(m):
    """m: metrics on the FINE grid. Max recall within each FP/image budget; min FP/image at each recall."""
    fpi, rec, prec = m["fp_per_image"], m["recall"], m["precision"]
    out = {}
    for b in BUDGETS:
        ok = np.where(fpi <= b)[0]
        j = ok[0] if len(ok) else None  # fpi falls with threshold: lowest threshold meeting the budget
        out[f"fpbudget_{b:.2f}"] = (dict(threshold=float(T_FINE[j]), recall=float(rec[j]), precision=float(prec[j]),
                                         fp_per_image=float(fpi[j])) if j is not None else None)
    for r in RECALLS:
        ok = np.where(rec >= r)[0]
        j = ok[-1] if len(ok) else None  # recall falls with threshold: highest threshold reaching the recall
        out[f"recall_{r:.1f}"] = (dict(threshold=float(T_FINE[j]), recall=float(rec[j]), precision=float(prec[j]),
                                       fp_per_image=float(fpi[j])) if j is not None else None)
    return out


def budget_arrays(m):
    """(recall at each FP budget (3,), fp/image at each recall target (3,)) with NaN when unattainable."""
    fpi, rec = m["fp_per_image"], m["recall"]
    a = []
    for b in BUDGETS:
        ok = np.where(fpi <= b)[0]
        a.append(rec[ok[0]] if len(ok) else np.nan)
    c = []
    for r in RECALLS:
        ok = np.where(rec >= r)[0]
        c.append(fpi[ok[-1]] if len(ok) else np.nan)
    return np.array(a), np.array(c)


def counts(tbl, w, cols):
    """Raw weighted counts at the given threshold columns."""
    return dict(tp=w @ tbl.tp[:, cols], fp=w @ tbl.fp[:, cols], fpneg=(w * tbl.neg) @ tbl.fp[:, cols],
                ngt=float(w @ tbl.gt), nimg=float(w.sum()), nneg=float(w @ tbl.neg))


def rates(c):
    tp, fp = np.asarray(c["tp"], float), np.asarray(c["fp"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        r = tp / c["ngt"]
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return dict(precision=p, recall=r, f1=f1, fp_per_image=fp / c["nimg"],
                fp_per_empty=np.asarray(c["fpneg"], float) / c["nneg"])


def add_counts(a, b):
    return dict(tp=a["tp"] + b["tp"], fp=a["fp"] + b["fp"], fpneg=a["fpneg"] + b["fpneg"],
                ngt=a["ngt"] + b["ngt"], nimg=a["nimg"] + b["nimg"], nneg=a["nneg"] + b["nneg"])


# ----------------------------------------------------------------------------- cross-fit
def crossfit(tbl, pool, seed):
    """2-fold group cross-fit. Returns {(set, rule): {metric: float, thresholds_by_fold}} out-of-fold (every
    image scored with a threshold chosen on the OTHER fold).
    rules: own (the scored set's own max-F1; rdd/hrp only), pooled (pooled micro max-F1; all sets),
    macro (max of mean of the two per-source F1; rdd/hrp), maximin (max of the lower of the two; rdd/hrp)."""
    fold = C.make_folds(pool, seed)
    sw = {s: set_weights(pool, s) for s in SETS}
    res = {}
    plan = [("own", s) for s in SETS[:2]] + [("pooled", s) for s in SETS] +            [(r, s) for r in ("macro", "maximin") for s in SETS[:2]]
    for rule, sname in plan:
        acc, thrs = None, []
        for fsel in (0, 1):
            wsel = {s: sw[s] * (fold == fsel) for s in SETS}
            if rule == "own":
                j = C.argmax_first(tbl.metrics(wsel[sname])["f1"][SW_COLS])
            elif rule == "pooled":
                j = C.argmax_first(tbl.metrics(wsel["pooled"])["f1"][SW_COLS])
            else:
                f_r = tbl.metrics(wsel["rdd2022_val"])["f1"][SW_COLS]
                f_h = tbl.metrics(wsel["hrp4k_val"])["f1"][SW_COLS]
                j = C.argmax_first((f_r + f_h) / 2 if rule == "macro" else np.minimum(f_r, f_h))
            thrs.append(float(T_SW[j]))
            c = counts(tbl, sw[sname] * (fold == 1 - fsel), [SW_COLS[j]])
            acc = c if acc is None else add_counts(acc, c)
        r = rates(acc)
        res[(sname, rule)] = dict(thresholds_by_fold=thrs, **{k: float(v[0]) for k, v in r.items()})
    return res


# ----------------------------------------------------------------------------- main
def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    C.verify_checkpoint("baseline"); C.verify_checkpoint("candidate")
    pool = C.load_pool()
    sw = {s: set_weights(pool, s) for s in SETS}
    tbl, P_cont = {}, {}
    for model, imgsz, tag in MODELS:
        preds = C.load_model_preds(model, imgsz, pool)
        P = C.build_P(pool, preds, iou_thrs=np.array([0.5]))
        tbl[tag] = C.ThrTable(P, T_FINE)
        P_cont[tag] = (preds, P)
    C.progress("threshold_sensitivity: tables built")

    # --- full-data curves, max-F1, plateaus, continuous-sweep sanity values, budget points
    sweep_rows, summary, cont_thr = [], {}, {}
    full = {}
    for tag in tbl:
        for sname in SETS:
            m = tbl[tag].metrics(sw[sname])
            full[(tag, sname)] = m
            f1 = m["f1"][SW_COLS]
            j = C.argmax_first(f1)
            lo1, hi1 = plateau(T_SW, f1, 0.01)
            lo2, hi2 = plateau(T_SW, f1, 0.02)
            # continuous max-F1 (every distinct confidence) for comparison with earlier audit values
            idx = pool.idx(sname) if sname != "pooled" else np.arange(len(pool.stems))
            Psub = C.build_P(pool, P_cont[tag][0], idx, iou_thrs=np.array([0.5]))
            sel = C.ec.select_thresholds(C.ec.sweep(Psub), 0.8)
            cont_thr[(tag, sname)] = sel["max_f1_conf"]
            ba = budget_stats(m)
            summary[(tag, sname)] = dict(
                max_f1_threshold=float(T_SW[j]), max_f1=float(f1[j]), precision=float(m["precision"][SW_COLS][j]),
                recall=float(m["recall"][SW_COLS][j]), fp_per_image=float(m["fp_per_image"][SW_COLS][j]),
                fp_per_empty=float(m["fp_per_empty"][SW_COLS][j]),
                plateau_0p01=[lo1, hi1, hi1 - lo1], plateau_0p02=[lo2, hi2, hi2 - lo2],
                max_f1_threshold_continuous=float(sel["max_f1_conf"]), max_f1_continuous=float(sel["max_f1"]),
                budget_points=ba, n_images=int(sw[sname].sum()), n_gt=int(sw[sname] @ pool.n_gt),
                n_empty=int(sw[sname] @ (pool.n_gt == 0)))
            for k, col in enumerate(SW_COLS):
                sweep_rows.append([tag, sname, f"{T_FINE[col]:.2f}"] + [f"{m[x][col]:.6g}" for x in
                                   ("precision", "recall", "f1", "fp_per_image", "fp_per_empty", "tp", "fp")])
    import csv
    with open(OUTD / "sweep_all.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "set", "threshold", "precision", "recall", "f1", "fp_per_image", "fp_per_empty_image", "tp", "fp"])
        wr.writerows(sweep_rows)
    C.progress("threshold_sensitivity: sweeps written (sweep_all.csv)")

    # --- bootstrap (shared resamples for all models)
    fams = {"rdd2022_val": C.boot_weights(pool, pool.idx("rdd2022_val"), N_BOOT, 0),
            "hrp4k_val": C.boot_weights(pool, pool.idx("hrp4k_val"), N_BOOT, 0),
            "pooled": C.boot_weights(pool, None, N_BOOT, 0)}
    B = {}
    for tag in tbl:
        for sname in SETS:
            B[(tag, sname)] = dict(f1=np.zeros((N_BOOT, len(SW_COLS))), precision=np.zeros((N_BOOT, len(SW_COLS))),
                                   recall=np.zeros((N_BOOT, len(SW_COLS))), fp_per_image=np.zeros((N_BOOT, len(SW_COLS))),
                                   fp_per_empty=np.zeros((N_BOOT, len(SW_COLS))), budget_recall=np.zeros((N_BOOT, 3)),
                                   recall_fpi=np.zeros((N_BOOT, 3)))
    for b in range(N_BOOT):
        for sname in SETS:
            if sname == "pooled":
                w = fams["pooled"][b]
            else:
                w = np.zeros(len(pool.stems)); w[pool.idx(sname)] = fams[sname][b]
            for tag in tbl:
                m = tbl[tag].metrics(w)
                for k in ("f1", "precision", "recall", "fp_per_image", "fp_per_empty"):
                    B[(tag, sname)][k][b] = m[k][SW_COLS]
                a, c = budget_arrays(m)
                B[(tag, sname)]["budget_recall"][b] = a; B[(tag, sname)]["recall_fpi"][b] = c
    C.progress("threshold_sensitivity: bootstrap done")

    col_of = {round(float(t), 2): i for i, t in enumerate(T_SW)}

    def at(tag, sname, thr, key):
        return B[(tag, sname)][key][:, col_of[round(thr, 2)]]

    # --- max-F1 location CI
    for (tag, sname), s in summary.items():
        locs = T_SW[[C.argmax_first(B[(tag, sname)]["f1"][b]) for b in range(N_BOOT)]]
        s["max_f1_threshold_ci95"] = C.pctl(locs)
        s["max_f1_ci95"] = C.pctl(np.nanmax(B[(tag, sname)]["f1"], axis=1))

    # --- cross-application
    cross = []
    for tag in tbl:
        thr = {s: summary[(tag, s)]["max_f1_threshold"] for s in SETS}
        for X, Y in (("rdd2022_val", "hrp4k_val"), ("hrp4k_val", "rdd2022_val")):
            rows = {"own": thr[X], "other": thr[Y], "pooled": thr["pooled"]}
            ref = {k: at(tag, X, rows["own"], k) for k in ("f1",)}
            for rule, t in rows.items():
                m = tbl[tag].metrics(sw[X])
                col = int(round(t * 1000)) - 1
                rec = dict(model=tag, set=X, rule=rule, threshold=t)
                for k in ("f1", "precision", "recall", "fp_per_image", "fp_per_empty"):
                    rec[k] = float(m[k][col])
                    rec[k + "_ci95"] = C.pctl(at(tag, X, t, k))
                d = at(tag, X, t, "f1") - ref["f1"]
                rec["f1_minus_own"] = float(rec["f1"] - full[(tag, X)]["f1"][int(round(rows["own"] * 1000)) - 1])
                rec["f1_minus_own_ci95"] = C.pctl(d)
                cross.append(rec)
    C.write_json(OUTD / "cross_application.json", cross)

    # --- cross-fit (seed 0 detail + 20-seed summary)
    cf0, cf_all = {}, []
    for tag in tbl:
        cf0[tag] = crossfit(tbl[tag], pool, 0)
        for sd in FOLD_SEEDS:
            r = crossfit(tbl[tag], pool, sd)
            for (sname, rule), v in r.items():
                cf_all.append(dict(model=tag, seed=sd, set=sname, rule=rule, **{k: v[k] for k in v if k != "thresholds_by_fold"}))
    import pandas as pd
    cfdf = pd.DataFrame(cf_all)
    cfdf.to_csv(OUTD / "crossfit_all_seeds.csv", index=False)
    cfsum = (cfdf.groupby(["model", "set", "rule"])[["f1", "precision", "recall", "fp_per_image", "fp_per_empty"]]
             .agg(["mean", "std", "min", "max"]))
    cfsum.columns = ["_".join(c) for c in cfsum.columns]
    cfsum.reset_index().to_csv(OUTD / "crossfit_summary_20_fold_seeds.csv", index=False)
    cf0_json = {tag: {f"{s}|{r}": v for (s, r), v in d.items()} for tag, d in cf0.items()}
    C.write_json(OUTD / "crossfit_seed0.json", cf0_json)
    C.progress("threshold_sensitivity: cross-fit done")

    # --- single-threshold balance analysis (in-sample, bootstrap paired vs pooled-max-F1 threshold)
    balance = []
    for tag in tbl:
        fr, fh = full[(tag, "rdd2022_val")]["f1"][SW_COLS], full[(tag, "hrp4k_val")]["f1"][SW_COLS]
        macro, mm = (fr + fh) / 2, np.minimum(fr, fh)
        cand = {"pooled_maxF1": summary[(tag, "pooled")]["max_f1_threshold"],
                "rdd_maxF1": summary[(tag, "rdd2022_val")]["max_f1_threshold"],
                "hrp_maxF1": summary[(tag, "hrp4k_val")]["max_f1_threshold"],
                "macro_maxF1": float(T_SW[C.argmax_first(macro)]), "maximin": float(T_SW[C.argmax_first(mm)]),
                "midpoint_rdd_hrp": round((summary[(tag, "rdd2022_val")]["max_f1_threshold"] +
                                           summary[(tag, "hrp4k_val")]["max_f1_threshold"]) / 2, 2)}
        Br, Bh = B[(tag, "rdd2022_val")]["f1"], B[(tag, "hrp4k_val")]["f1"]
        base_t = cand["pooled_maxF1"]; cb = col_of[round(base_t, 2)]
        macro_b = (Br[:, cb] + Bh[:, cb]) / 2
        for name, t in cand.items():
            c = col_of[round(t, 2)]
            mb = (Br[:, c] + Bh[:, c]) / 2
            balance.append(dict(model=tag, rule=name, threshold=t, f1_rdd=float(fr[c]), f1_hrp=float(fh[c]),
                                macro_f1=float(macro[c]), min_f1=float(mm[c]),
                                macro_f1_minus_pooledthr=float(macro[c] - macro[cb]),
                                macro_f1_minus_pooledthr_ci95=C.pctl(mb - macro_b),
                                f1_rdd_minus_pooledthr_ci95=C.pctl(Br[:, c] - Br[:, cb]),
                                f1_hrp_minus_pooledthr_ci95=C.pctl(Bh[:, c] - Bh[:, cb]),
                                f1_rdd_minus_pooledthr=float(fr[c] - fr[cb]), f1_hrp_minus_pooledthr=float(fh[c] - fh[cb])))
    C.write_json(OUTD / "balance_single_threshold.json", balance)

    # --- fixed-budget points with bootstrap CIs and paired differences vs baseline
    bud = []
    for sname in SETS:
        for k, name in enumerate([f"FP/img<={b:.2f}" for b in BUDGETS] + [f"recall>={r:.1f}" for r in RECALLS]):
            for tag in tbl:
                key, kk = ("budget_recall", k) if k < 3 else ("recall_fpi", k - 3)
                pt = summary[(tag, sname)]["budget_points"][(f"fpbudget_{BUDGETS[k]:.2f}" if k < 3 else f"recall_{RECALLS[k-3]:.1f}")]
                arr = B[(tag, sname)][key][:, kk]
                rec = dict(set=sname, target=name, model=tag, point=pt, ci95=C.pctl(arr) if not np.all(np.isnan(arr)) else None,
                           n_boot_unattainable=int(np.isnan(arr).sum()))
                if tag != "baseline@640":
                    d = arr - B[("baseline@640", sname)][key][:, kk]
                    rec["minus_baseline_ci95"] = C.pctl(d) if not np.all(np.isnan(d)) else None
                    rec["minus_baseline_n_valid"] = int((~np.isnan(d)).sum())
                bud.append(rec)
    C.write_json(OUTD / "fixed_budget_points.json", bud)

    C.write_json(OUTD / "summary_max_f1.json", {f"{t}|{s}": v for (t, s), v in summary.items()})
    plot(full)
    write_results_md(summary, cross, cf0, cfsum, balance, bud, pool)
    C.progress("threshold_sensitivity: DONE (RESULTS.md, plots written)")
    C.verify_checkpoint("baseline"); C.verify_checkpoint("candidate")


# ----------------------------------------------------------------------------- plots / report
def plot(full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"baseline@640": "#0072B2", "candidate@1280": "#D55E00", "candidate@960": "#009E73"}
    labels = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"), ("fp_per_image", "False positives per image")]
    for sname in SETS:
        fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
        for ax, (k, lab) in zip(axs.ravel(), labels):
            for tag, c in colors.items():
                ax.plot(T_SW, full[(tag, sname)][k][SW_COLS], color=c, label=tag, lw=1.8)
                if k == "f1":
                    j = C.argmax_first(full[(tag, sname)]["f1"][SW_COLS])
                    ax.plot(T_SW[j], full[(tag, sname)]["f1"][SW_COLS][j], "o", color=c, ms=5)
            ax.set_ylabel(lab); ax.grid(alpha=.3)
            if k == "fp_per_image":
                ax.set_yscale("log")
        for ax in axs[1]:
            ax.set_xlabel("confidence threshold")
        axs[0, 0].legend(frameon=False)
        fig.suptitle(f"{sname}: metrics vs confidence threshold (IoU 0.5; dots = max-F1 on the 0.01 grid)")
        fig.tight_layout()
        fig.savefig(OUTD / f"threshold_curves_{sname}.png", dpi=130)
        plt.close(fig)


def fmt(x, n=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def ci(v, n=3):
    return "n/a" if v is None else f"[{v[0]:.{n}f}, {v[1]:.{n}f}]"


def write_results_md(summary, cross, cf0, cfsum, balance, bud, pool):
    L = ["# Threshold sensitivity (validation only)", "",
         "Generated by scripts/val_investigations/threshold_sensitivity.py. IoU 0.5 matching; thresholds on a 0.01 grid "
         "from 0.05 to 0.80; 'continuous' = max-F1 over every distinct confidence (the project's own rule).",
         "HRP4K val is in-domain for the candidate and out-of-domain for the baseline.", "",
         "## 1. Max-F1 threshold, plateau width, bootstrap interval of its location", "",
         "| model | set | max-F1 thr (0.01 grid) [95% CI of location] | continuous max-F1 thr | max F1 [CI] | P | R | FP/img | FP/empty img | width F1 within 0.01 (lo-hi) | width within 0.02 (lo-hi) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (tag, s), v in summary.items():
        L.append(f"| {tag} | {s} | {v['max_f1_threshold']:.2f} {ci(v['max_f1_threshold_ci95'], 2)} | "
                 f"{v['max_f1_threshold_continuous']:.4f} | {v['max_f1']:.3f} {ci(v['max_f1_ci95'])} | {v['precision']:.3f} | "
                 f"{v['recall']:.3f} | {v['fp_per_image']:.3f} | {v['fp_per_empty']:.3f} | "
                 f"{v['plateau_0p01'][2]:.2f} ({v['plateau_0p01'][0]:.2f}-{v['plateau_0p01'][1]:.2f}) | "
                 f"{v['plateau_0p02'][2]:.2f} ({v['plateau_0p02'][0]:.2f}-{v['plateau_0p02'][1]:.2f}) |")
    L += ["", "Width = highest minus lowest threshold of the contiguous range around the maximum where F1 stays within the stated amount.", "",
          "## 2. Cross-application of thresholds (in-sample: each threshold was chosen on the full validation images)", "",
          "own = this set's max-F1 threshold; other = the other set's; pooled = pooled max-F1 threshold. "
          "F1 difference is versus 'own', with a paired group-bootstrap 95% interval (thresholds held fixed).", "",
          "| model | scored on | rule | thr | F1 [CI] | P | R | FP/img | FP/empty img | F1 minus own [CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in cross:
        L.append(f"| {r['model']} | {r['set']} | {r['rule']} | {r['threshold']:.2f} | {r['f1']:.3f} {ci(r['f1_ci95'])} | "
                 f"{r['precision']:.3f} | {r['recall']:.3f} | {r['fp_per_image']:.3f} | {r['fp_per_empty']:.3f} | "
                 f"{r['f1_minus_own']:+.3f} {ci(r['f1_minus_own_ci95'])} |")
    L += ["", "## 3. In-sample versus 2-fold cross-fit (threshold chosen on one fold, scored on the other)", "",
          "Cross-fit: group-level folds stratified by source; every image is scored with the threshold chosen on the other fold. "
          "Seed-0 numbers, then mean (min-max) over 20 fold seeds.", "",
          "| model | scored on | rule | in-sample F1 | cross-fit F1 (seed 0) | cross-fit F1 mean over 20 seeds (min-max) | cross-fit P / R / FP per img (seed 0) | thresholds chosen on fold 0 / fold 1 (seed 0) |",
          "|---|---|---|---|---|---|---|---|"]
    for tag, d in cf0.items():
        for (s, rule), v in d.items():
            if rule == "maximin":
                continue
            insample = None
            if rule == "own":
                insample = summary[(tag, s)]["max_f1"]
            elif rule == "pooled":
                insample = _f1_at(cross, tag, s, "pooled") if s != "pooled" else summary[(tag, "pooled")]["max_f1"]
            row = cfsum.loc[(tag, s, rule)]
            L.append(f"| {tag} | {s} | {rule} | {fmt(insample)} | {v['f1']:.3f} | {row['f1_mean']:.3f} ({row['f1_min']:.3f}-{row['f1_max']:.3f}) | "
                     f"{v['precision']:.3f} / {v['recall']:.3f} / {v['fp_per_image']:.3f} | {v['thresholds_by_fold'][0]:.2f} / {v['thresholds_by_fold'][1]:.2f} |")
    L += ["", "Rule 'macro' and 'maximin' appear below in the balance analysis (section 4).", "",
          "## 4. Would a single per-model threshold other than the pooled max-F1 one balance the two sets better?", "",
          "Balance measure: macro F1 = mean of the RDD2022-val F1 and the HRP4K-val F1 (pooled F1 is dominated by HRP4K, "
          "which has more boxes), and the lower of the two F1. Differences are versus the pooled-max-F1 threshold with paired "
          "group-bootstrap 95% intervals (thresholds fixed; in-sample).", "",
          "| model | single threshold rule | thr | F1 RDD | F1 HRP4K | macro F1 | min F1 | macro F1 minus pooled-thr [CI] | F1 RDD minus pooled-thr [CI] | F1 HRP4K minus pooled-thr [CI] |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in balance:
        L.append(f"| {r['model']} | {r['rule']} | {r['threshold']:.2f} | {r['f1_rdd']:.3f} | {r['f1_hrp']:.3f} | {r['macro_f1']:.3f} | {r['min_f1']:.3f} | "
                 f"{r['macro_f1_minus_pooledthr']:+.3f} {ci(r['macro_f1_minus_pooledthr_ci95'])} | "
                 f"{r['f1_rdd_minus_pooledthr']:+.3f} {ci(r['f1_rdd_minus_pooledthr_ci95'])} | "
                 f"{r['f1_hrp_minus_pooledthr']:+.3f} {ci(r['f1_hrp_minus_pooledthr_ci95'])} |")
    L += ["", "Out-of-fold macro-F1 (cross-fit, 20 fold seeds, mean) for rules selected on one fold and scored on the other:", "",
          "| model | rule | RDD F1 (mean over seeds) | HRP4K F1 (mean) | macro F1 (mean) |", "|---|---|---|---|---|"]
    for tag in dict.fromkeys(r["model"] for r in balance):
        for rule in ("pooled", "macro", "maximin"):
            fr = cfsum.loc[(tag, "rdd2022_val", rule), "f1_mean"]; fh = cfsum.loc[(tag, "hrp4k_val", rule), "f1_mean"]
            L.append(f"| {tag} | {rule} | {fr:.3f} | {fh:.3f} | {(fr + fh) / 2:.3f} |")
    L += ["", "## 5. Operating points at fixed false-alarm budgets and fixed recall (threshold grid 0.001)", "",
          "Budget rows: highest recall with FP/img <= budget. Recall rows: lowest FP/img that reaches the recall. "
          "'minus baseline' is a paired group-bootstrap 95% interval (candidate minus baseline@640) on the same resamples. "
          "Intervals can be 'n/a' when the target is unreachable in the point estimate.", "",
          "| set | target | model | threshold | recall | precision | FP/img | metric 95% CI | minus baseline [CI] |", "|---|---|---|---|---|---|---|---|---|"]
    for r in bud:
        p = r["point"]
        if p is None:
            L.append(f"| {r['set']} | {r['target']} | {r['model']} | n/a (not reached) | | | | | |"); continue
        key = "recall" if r["target"].startswith("FP") else "fp_per_image"
        L.append(f"| {r['set']} | {r['target']} | {r['model']} | {p['threshold']:.3f} | {p['recall']:.3f} | {p['precision']:.3f} | {p['fp_per_image']:.3f} | "
                 f"{key}: {ci(r['ci95'])} | {ci(r.get('minus_baseline_ci95')) if 'minus_baseline_ci95' in r else '-'} |")
    (OUTD / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _f1_at(cross, tag, s, rule):
    for r in cross:
        if r["model"] == tag and r["set"] == s and r["rule"] == rule:
            return r["f1"]
    return None


if __name__ == "__main__":
    main()
