"""Paired comparison of two evaluation output directories (scripts/eval_checkpoint.py) on the same image list.

    python scripts/compare_models.py DIR_A DIR_B [--out FILE.json]

Reports, per metric, the point difference (B minus A) and a 95% percentile interval of the PAIRED
bootstrap difference (resample r of A is subtracted from resample r of B; both used identical group
resamples because the resampling is determined by (seed, sorted group list)). Refuses if the image
lists, the image->group assignment, the seed or the number of resamples differ.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


class NotPairable(Exception):
    pass


def check_pairable(cfg_a, cfg_b, groups_a, groups_b):
    """cfg_*: eval_config.json dicts; groups_*: list of (image, group) rows. Raises NotPairable."""
    for k, msg in (("image_list_sha256", "image lists differ"),
                   ("boot_seed", "bootstrap seeds differ"), ("boot_n", "number of resamples differ")):
        if cfg_a.get(k) is None or cfg_a.get(k) != cfg_b.get(k):
            raise NotPairable(f"{msg} ({k}: {cfg_a.get(k)} vs {cfg_b.get(k)})")
    if sorted(map(tuple, groups_a)) != sorted(map(tuple, groups_b)):
        raise NotPairable("image->group files differ")
    if cfg_a.get("boot_image_group_sha256") != cfg_b.get("boot_image_group_sha256"):
        raise NotPairable("image->group digests differ")


def paired_diff(samples_a, samples_b, point_a=None, point_b=None, level=0.95):
    """samples_*: {column: array (n_boot,)}. Returns {column: dict(point_diff, mean_boot_diff, ci95, n_valid)}
    for the columns present in both. Pairing is by resample index (row-aligned)."""
    out = {}
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    for c in samples_a:
        if c not in samples_b:
            continue
        a = np.asarray(samples_a[c], float); b = np.asarray(samples_b[c], float)
        if a.shape != b.shape:
            raise NotPairable(f"resample counts differ for {c}")
        d = b - a
        ok = ~np.isnan(d)
        pd_ = None
        if point_a and point_b and point_a.get(c) is not None and point_b.get(c) is not None:
            pd_ = float(point_b[c] - point_a[c])
        out[c] = dict(point_diff=pd_, mean_boot_diff=float(d[ok].mean()) if ok.any() else float("nan"),
                      ci95=[float(np.percentile(d[ok], lo_q)), float(np.percentile(d[ok], hi_q))] if ok.any() else [float("nan")] * 2,
                      n_valid=int(ok.sum()), ci_excludes_zero=bool(ok.any() and (np.percentile(d[ok], lo_q) > 0 or np.percentile(d[ok], hi_q) < 0)))
    return out


def read_samples(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = [c for c in rows[0] if c != "resample"]
    idx = [int(r["resample"]) for r in rows]
    if idx != list(range(len(rows))):
        raise NotPairable("resample indices are not 0..n-1")
    return {c: np.array([float(r[c]) for r in rows]) for c in cols}


def points(metrics):
    pt = {"mAP50": metrics["ap_agreement"]["own"]["mAP50"], "mAP50_95": metrics["ap_agreement"]["own"]["mAP50_95"]}
    for n, d in metrics["operating_points"].items():
        for k in ("precision", "recall", "fp_per_image"):
            pt[f"{n}__{k}"] = d[k]
    mp = metrics.get("manhole_probe")
    if mp:
        for k, v in mp.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                pt[f"probe_{k}"] = v
    return pt


def load_dir(d):
    d = Path(d)
    cfg = json.loads((d / "eval_config.json").read_text())
    with open(d / "bootstrap_groups.csv", newline="") as f:
        groups = [tuple(r) for r in list(csv.reader(f))[1:]]
    return cfg, groups, read_samples(d / "bootstrap_samples.csv"), points(json.loads((d / "metrics.json").read_text()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir_a"); ap.add_argument("dir_b"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        ca, ga, sa, pa = load_dir(a.dir_a)
        cb, gb, sb, pb = load_dir(a.dir_b)
        check_pairable(ca, cb, ga, gb)
        res = paired_diff(sa, sb, pa, pb)
    except (NotPairable, FileNotFoundError) as e:
        sys.exit(f"REFUSED: {e}")
    doc = dict(a=str(a.dir_a), b=str(a.dir_b), difference="B minus A", ci="95% percentile interval of paired bootstrap differences",
               n_boot=ca["boot_n"], seed=ca["boot_seed"], image_list_sha256=ca["image_list_sha256"], metrics=res)
    for k, v in res.items():
        pd_ = "n/a" if v["point_diff"] is None else f"{v['point_diff']:+.4f}"
        print(f"{k:45s} diff {pd_}  95% [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  excludes0={v['ci_excludes_zero']}")
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
