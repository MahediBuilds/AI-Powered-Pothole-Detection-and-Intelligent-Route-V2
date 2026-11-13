"""Synthetic tests for the evaluation protocol in scripts/eval_checkpoint.py."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("eval_checkpoint", Path(__file__).resolve().parents[1] / "scripts" / "eval_checkpoint.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

B = lambda x: [x, 0, x + 10, 10]  # 10x10 box at x


def test_matcher_one_pred_per_gt_and_confidence_order():
    gt = [B(0)]
    # two preds on the same GT: the higher-confidence one matches, the other is FP
    order, tp, gm = ev.match_image([B(1), B(0)], [0.5, 0.9], gt, [0.5])
    assert list(order) == [1, 0]
    assert tp[:, 0].tolist() == [True, False]
    assert gm[0, 0] == pytest.approx(0.9)


def test_matcher_iou_threshold_and_misses():
    _, tp, gm = ev.match_image([B(8)], [0.9], [B(0)], [0.5])  # IoU 2*10/(200-20)=0.11
    assert not tp.any() and gm[0, 0] == -1
    _, tp, _ = ev.match_image([], [], [B(0)], [0.5])
    assert tp.shape == (0, 1)


def _toy():
    # img0: 2 GT, preds hit GT0 (0.9) + FP (0.6); img1: negative, FP 0.8; img2: 1 GT, hit 0.3
    images = ["a", "b", "c"]
    gts = {"a": (np.array([B(0), B(50)], float), np.array([0.05, 0.005])), "b": (np.zeros((0, 4)), np.zeros(0)),
           "c": (np.array([B(0)], float), np.array([0.03]))}
    preds = {"a": (np.array([B(0), B(200)], float), np.array([0.9, 0.6])), "b": (np.array([B(0)], float), np.array([0.8])),
             "c": (np.array([B(0)], float), np.array([0.3]))}
    return ev.build_preds(images, gts, preds, np.array([0.5]))


def test_operating_metrics():
    P = _toy()
    m = ev.op_metrics(P, 0.5)  # keeps 0.9(TP) 0.8(FP, neg) 0.6(FP)
    assert (m["TP"], m["FP"], m["FN"]) == (1, 2, 2)
    assert m["precision"] == pytest.approx(1 / 3) and m["recall"] == pytest.approx(1 / 3)
    assert m["fp_per_image"] == pytest.approx(2 / 3)
    assert m["fp_per_negative_image"] == 1.0 and m["negative_images_with_detection"] == 1.0
    m = ev.op_metrics(P, 0.95)
    assert m["TP"] == 0 and m["FN"] == 3 and m["fp_per_negative_image"] == 0.0


def test_size_breakdown():
    P = _toy()
    s = ev.size_breakdown(P, 0.5)  # only GT 0.05 detected at >=0.5; 0.03 hit has conf .3
    assert s[">4%"]["recall"] == 1.0 and s["2-4%"]["recall"] == 0.0 and s["<1%"]["recall"] == 0.0


def test_threshold_selection():
    P = _toy()
    sw = ev.sweep(P)
    assert sw["conf"] == [0.9, 0.8, 0.6, 0.3]
    sel = ev.select_thresholds(sw, 2 / 3)  # recall 1/3 at conf .9..0.6, 2/3 at 0.3
    assert sel["max_f1_conf"] == 0.3  # F1: .5, .4, .33, .571
    assert sel["target_recall_conf"] == pytest.approx(0.3)
    sel = ev.select_thresholds(sw, 0.5)  # between rec 1/3 @0.6 and 2/3 @0.3 -> halfway
    assert sel["target_recall_conf"] == pytest.approx(0.45)
    assert ev.select_thresholds(sw, 0.99)["target_recall_conf"] is None


def test_ap_perfect_and_weighted():
    images = ["a", "b"]
    gts = {i: (np.array([B(0)], float), np.array([0.05])) for i in images}
    preds = {i: (np.array([B(0)], float), np.array([0.9])) for i in images}
    P = ev.build_preds(images, gts, preds, np.array([0.5]))
    assert ev.ap_all(P)[0] == pytest.approx(0.995)  # 101-point interpolation caps a perfect curve at 0.995
    assert ev.ap_all(P, np.array([2.0, 0.0]))[0] == pytest.approx(0.995)


def test_group_bootstrap_resamples_groups_and_is_deterministic():
    P = _toy()
    # all three images in one group: every resample is the full set -> degenerate CI == point value
    ci = ev.group_bootstrap(P, ["g", "g", "g"], {"t": 0.5}, n_boot=50)
    pt = ev.op_metrics(P, 0.5)
    lo, hi = ci["operating_point"]["t"]["precision"]
    assert lo == pytest.approx(pt["precision"]) and hi == pytest.approx(pt["precision"])
    assert ci["n_groups"] == 1
    # image-level groups give a genuinely wider interval, and the run is reproducible
    c1 = ev.group_bootstrap(P, [0, 1, 2], {"t": 0.5}, n_boot=200)
    c2 = ev.group_bootstrap(P, [0, 1, 2], {"t": 0.5}, n_boot=200)
    drop = lambda c: {k: v for k, v in c.items() if k != "samples"}  # samples may hold NaN (nan != nan)
    assert drop(c1) == drop(c2) and c1["n_groups"] == 3
    r = c1["operating_point"]["t"]["fp_per_image"]
    assert r[0] < r[1]


def test_manhole_hits_iou_and_centre_are_separate():
    m = [[0, 0, 100, 100]]
    # IoU 0.25 (<0.3) but centre (30,30) inside -> centre hit only
    hi, hc = ev.manhole_hits([[10, 10, 60, 60]], [0.9], m, 0.5)
    assert not hi[0] and hc[0]
    # IoU 100*100/22000 = 0.45 but centre (-10,50) outside -> IoU hit only
    hi, hc = ev.manhole_hits([[-120, 0, 100, 100]], [0.9], m, 0.5)
    assert hi[0] and not hc[0]
    # neither: IoU 0.25, centre (110,50) outside
    hi, hc = ev.manhole_hits([[60, 0, 160, 100]], [0.9], m, 0.5)
    assert not hi[0] and not hc[0]
    # both
    hi, hc = ev.manhole_hits([[20, 0, 120, 100]], [0.9], m, 0.5)
    assert hi[0] and hc[0]


def test_manhole_hits_threshold_edges_and_multiple():
    m = [[0, 0, 100, 100], [200, 200, 300, 300]]
    # prediction below threshold is ignored; at exactly the threshold it counts (>=)
    hi, hc = ev.manhole_hits([[0, 0, 100, 100]], [0.3], m, 0.31)
    assert not hi.any() and not hc.any()
    hi, hc = ev.manhole_hits([[0, 0, 100, 100]], [0.3], m, 0.3)
    assert hi.tolist() == [True, False] and hc.tolist() == [True, False]
    # centre exactly on the edge counts as inside; empty predictions -> no hits
    _, hc = ev.manhole_hits([[100, 40, 120, 60]], [0.9], m, 0.5)  # centre (110,50) outside
    assert not hc[0]
    _, hc = ev.manhole_hits([[90, 40, 110, 60]], [0.9], m, 0.5)   # centre (100,50) on edge
    assert hc[0]
    hi, hc = ev.manhole_hits([], [], m, 0.5)
    assert hi.shape == (2,) and not hi.any() and not hc.any()


def test_manhole_probe_aggregates_and_groups():
    stems = ["a", "b", "c"]; sizes = {s: (100, 100) for s in stems}
    # manhole centred at (50,50), 20x20 -> box [40,40,60,60] on images a and b
    mrows = [("a", .5, .5, .2, .2), ("b", .5, .5, .2, .2)]
    preds = {"a": (np.array([[42, 42, 58, 58], [0, 0, 5, 5]], float), np.array([0.9, 0.8])),
             "b": (np.array([[0, 0, 5, 5]], float), np.array([0.2]))}
    r = ev.manhole_probe(stems, sizes, preds, [0, 0, 0], mrows, 0.5, ["g1", "g1", "g2"], n_boot=50)
    assert r["hits_iou03"] == 1 and r["hits_centre_inside"] == 1
    assert r["manhole_hit_share_iou03"] == pytest.approx(0.5)
    assert r["preds_per_pothole_free_image"] == pytest.approx(2 / 3)
    assert r["pothole_free_images_with_pred"] == pytest.approx(1 / 3)
    assert r["n_groups"] == 2


# ---------------------------------------------------------------- AP closure / paired bootstrap
def _ultralytics_ap_in_subprocess(cases):
    """Ultralytics' compute_ap for each (rec, prec) case, computed in a child process so that
    ultralytics is never imported into the pytest process (other tests assert it is absent from sys.modules)."""
    code = chr(10).join(["import json, sys, numpy as np", "from ultralytics.utils.metrics import compute_ap",
                       "cases = json.load(sys.stdin)",
                       "print(json.dumps([float(compute_ap(np.array(r), np.array(p))[0]) for r, p in cases]))"])
    r = subprocess.run([sys.executable, "-c", code], input=json.dumps(cases), capture_output=True, text=True,
                       env={**os.environ, "YOLO_OFFLINE": "1"}, timeout=300)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_compute_ap_matches_ultralytics_including_low_recall():
    rng = np.random.default_rng(1)
    cases = []
    for n_gt, n_pred in [(1000, 100), (50, 200), (20, 20), (500, 800), (10, 3)]:
        tpc = np.cumsum(rng.random(n_pred) < 0.3).astype(float); fpc = np.arange(1, n_pred + 1) - tpc
        cases.append((tpc / n_gt, tpc / np.maximum(tpc + fpc, 1e-12)))
    # low-recall case: max recall about 0.1, precision 1 throughout -> area ~0.1, not ~1
    low_rec = np.linspace(0.01, 0.1, 10); low_prec = np.ones(10)
    assert low_rec.max() == pytest.approx(0.1)
    cases.append((low_rec, low_prec))
    uly = _ultralytics_ap_in_subprocess([(r.tolist(), p.tolist()) for r, p in cases])
    for (rec, prec), u in zip(cases, uly):
        assert ev.compute_ap(rec, prec) == pytest.approx(u, abs=1e-6)
    own = ev.compute_ap(low_rec, low_prec)
    assert 0.09 < own < 0.12


def test_resample_weights_determined_by_seed_and_sorted_groups():
    g1 = ["b", "a", "c", "a", "b"]
    g2 = ["a", "a", "b", "b", "c"]  # same groups, different image order
    w1 = list(ev.resample_weights(g1, 20, 7)); w2 = list(ev.resample_weights(g2, 20, 7))
    # group counts per resample are identical (map back to group labels)
    lab1 = [{g: w[i] for i, g in enumerate(g1)} for w in w1]; lab2 = [{g: w[i] for i, g in enumerate(g2)} for w in w2]
    assert lab1 == lab2
    assert [w.tolist() for w in w1] == [w.tolist() for w in ev.resample_weights(g1, 20, 7)]
    assert any((x != y).any() for x, y in zip(w1, ev.resample_weights(g1, 20, 8)))


def _load_compare():
    spec = importlib.util.spec_from_file_location("compare_models", Path(__file__).resolve().parents[1] / "scripts" / "compare_models.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_compare_models_pairing_and_refusals():
    cm = _load_compare()
    cfg = dict(image_list_sha256="x", boot_seed=0, boot_n=4, boot_image_group_sha256="h")
    grp = [("i1", "g1"), ("i2", "g2")]
    cm.check_pairable(cfg, dict(cfg), grp, list(reversed(grp)))
    for k, v in [("image_list_sha256", "y"), ("boot_seed", 1), ("boot_n", 5), ("boot_image_group_sha256", "z")]:
        with pytest.raises(cm.NotPairable):
            cm.check_pairable(cfg, dict(cfg, **{k: v}), grp, grp)
    with pytest.raises(cm.NotPairable):
        cm.check_pairable(cfg, dict(cfg), grp, [("i1", "g1"), ("i2", "gX")])
    a = {"m": np.array([0.1, 0.2, 0.3, 0.4]), "only_a": np.zeros(4)}
    b = {"m": np.array([0.2, 0.2, 0.5, 0.4]), "nanm": np.ones(4)}
    r = cm.paired_diff(a, b, {"m": 0.25}, {"m": 0.35})
    assert set(r) == {"m"}
    assert r["m"]["point_diff"] == pytest.approx(0.1)
    assert r["m"]["mean_boot_diff"] == pytest.approx(0.075)
    assert r["m"]["ci95"][0] >= 0.0 and r["m"]["ci_excludes_zero"] is False  # lowest diff is exactly 0
    # pairing is by index: shuffling B would change the interval of differences
    r2 = cm.paired_diff({"m": np.array([0., 1.])}, {"m": np.array([1., 0.])})
    assert r2["m"]["mean_boot_diff"] == 0.0
    with pytest.raises(cm.NotPairable):
        cm.paired_diff({"m": np.zeros(3)}, {"m": np.zeros(4)})


def test_explicit_conf_argument_validation():
    base = ["--checkpoint", "x.pt", "--data", "d.yaml", "--out", "unused_out"]
    for argv in (["--mode", "apply"],  # neither operating points nor explicit conf
                 ["--mode", "apply", "--explicit-conf", "0.3", "--operating-points", "o.json"],
                 ["--mode", "select", "--explicit-conf", "0.3"],
                 ["--mode", "apply", "--explicit-conf", "1.5"],
                 ["--mode", "apply", "--explicit-conf", "0"]):
        with pytest.raises(SystemExit) as e:  # all rejected before ultralytics is imported
            ev.main(argv + base)
        assert e.value.code != 0
    assert ev.EXPLICIT_NAME == "matched_recall"
