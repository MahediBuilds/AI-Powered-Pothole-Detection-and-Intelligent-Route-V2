"""Tests for scripts/val_investigations (guard, fusion, mechanical FP split, folds, threshold tables)."""
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "val_investigations"
sys.path.insert(0, str(SCRIPTS))
import common as C  # noqa: E402
import fusion as F  # noqa: E402
import fp_mech as M  # noqa: E402


# ----------------------------------------------------------------------------- guard
@pytest.mark.parametrize("p", [
    "data/processed/unified_v1/lists/rdd2022_test.txt", "data/processed/unified_v1/lists/hrp4k_test.txt",
    "data/processed/unified_v1/images/hrp4k/test/hrp4k_test_30000.jpg", "data/processed/kaggle_probe_v1/x.jpg",
    "C:\\data\\Test\\a.jpg", "runs/eval/foo/KAGGLE_PROBE/metrics.json"])
def test_guard_refuses_held_out_paths(p):
    with pytest.raises(C.ForbiddenPath):
        C.guard(p)


def test_guard_allows_validation_paths():
    for s in C.VAL_SETS.values():
        assert C.guard(s) == s
    assert C.guard("data/processed/unified_v1/images/hrp4k/val/hrp4k_val_20000.jpg")


def test_read_list_rejects_a_list_with_held_out_images(tmp_path, monkeypatch):
    bad = tmp_path / "lst.txt"
    bad.write_text("images/hrp4k/test/a.jpg\n")
    monkeypatch.setitem(C.VAL_SETS, "x", bad)
    with pytest.raises(C.ForbiddenPath):
        C.read_list("x")


def test_read_list_rejects_list_file_whose_name_is_held_out(tmp_path, monkeypatch):
    bad = tmp_path / "hrp4k_test.txt"
    bad.write_text("images/hrp4k/val/a.jpg\n")
    monkeypatch.setitem(C.VAL_SETS, "x", bad)
    with pytest.raises(C.ForbiddenPath):
        C.read_list("x")


# ----------------------------------------------------------------------------- WBF (hand-worked)
def test_wbf_matches_hand_calculation():
    # image 100x100. model A box (10,10,50,50) conf .8, model B box (12,10,52,50) conf .6, equal weights, avg.
    # fused coords = conf-weighted mean: x1 = (.8*.10+.6*.12)/1.4, x2 = (.8*.50+.6*.52)/1.4, y unchanged;
    # fused conf = mean(.8,.6) * min(sum w=2, n boxes=2)/2 = .7
    b, s = F.wbf_fuse([[10, 10, 50, 50]], [0.8], [[12, 10, 52, 50]], [0.6], (100, 100), iou_thr=0.55,
                      skip_box_thr=0.0, weights=(1, 1), conf_type="avg")
    assert len(s) == 1
    assert s[0] == pytest.approx(0.7)
    assert b[0] == pytest.approx([100 * (0.8 * 0.10 + 0.6 * 0.12) / 1.4, 10, 100 * (0.8 * 0.50 + 0.6 * 0.52) / 1.4, 50])


def test_wbf_single_model_box_conf_is_scaled_and_weights_matter():
    # ensemble-boxes multiplies each score by its model weight first, then scales the cluster mean by
    # min(#models, #boxes in cluster) / sum(weights). Box seen only by A, conf .9:
    #   weights (1,1): .9*1  -> mean .9  * min(2,1)/2 = .45
    #   weights (2,1): .9*2=1.8 -> mean 1.8 * min(2,1)/3 = .6
    for w, expect in (((1, 1), 0.45), ((2, 1), 0.6)):
        b, s = F.wbf_fuse([[10, 10, 50, 50]], [0.9], np.zeros((0, 4)), [], (100, 100), 0.55, 0.0, w, "avg")
        assert s == pytest.approx([expect]) and b[0] == pytest.approx([10, 10, 50, 50])


def test_wbf_weighted_pair_by_hand():
    # A (.8, weight 1) -> .8 ; B (.6, weight 2) -> 1.2. Coordinates weighted by .8 and 1.2:
    # x1 = (.8*.10 + 1.2*.12)/2.0 = .112 ; conf = mean(.8,1.2)=1.0 * min(2,2)/3 = 2/3
    b, s = F.wbf_fuse([[10, 10, 50, 50]], [0.8], [[12, 10, 52, 50]], [0.6], (100, 100), 0.55, 0.0, (1, 2), "avg")
    assert s == pytest.approx([2 / 3])
    assert b[0] == pytest.approx([11.2, 10, 100 * (0.8 * 0.50 + 1.2 * 0.52) / 2.0, 50])


def test_wbf_conf_types_and_clusters():
    A = [[10, 10, 50, 50]]
    Bx = [[12, 10, 52, 50]]
    _, smax = F.wbf_fuse(A, [0.8], Bx, [0.6], (100, 100), 0.55, 0.0, (1, 1), "max")
    assert smax == pytest.approx([0.8])
    _, sbm = F.wbf_fuse(A, [0.8], Bx, [0.6], (100, 100), 0.55, 0.0, (1, 1), "box_and_model_avg")
    assert sbm == pytest.approx([0.7])
    b, s = F.wbf_fuse(A, [0.8], [[60, 60, 90, 90]], [0.6], (100, 100), 0.55, 0.0, (1, 1), "avg")
    assert len(s) == 2 and s[0] > s[1]
    # IoU of the pair is ~0.905; a threshold above it must not cluster them
    _, s2 = F.wbf_fuse(A, [0.8], Bx, [0.6], (100, 100), 0.95, 0.0, (1, 1), "avg")
    assert len(s2) == 2


def test_wbf_skip_box_thr_and_empty():
    _, s = F.wbf_fuse([[10, 10, 50, 50]], [0.04], [[12, 10, 52, 50]], [0.6], (100, 100), 0.55, 0.05, (1, 1), "avg")
    assert len(s) == 1  # the 0.04 box is skipped
    b, s = F.wbf_fuse(np.zeros((0, 4)), [], np.zeros((0, 4)), [], (100, 100), 0.55, 0.0)
    assert len(s) == 0 and b.shape == (0, 4)


# ----------------------------------------------------------------------------- concat NMS
def test_nms_and_concat():
    k = F.nms([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], [0.5, 0.9, 0.4], 0.5)
    assert k.tolist() == [1, 2]  # IoU(0,1)=0.82>0.5 suppressed, far box kept
    b, s = F.concat_nms([[0, 0, 10, 10]], [0.5], [[1, 0, 11, 10]], [0.9], 0.7)
    assert s.tolist() == [0.9] and b[0].tolist() == [1, 0, 11, 10]
    # IoU exactly at the threshold survives (torchvision convention): IoU = 50/150 = 1/3
    k = F.nms([[0, 0, 10, 10], [5, 0, 15, 10]], [0.9, 0.8], 1 / 3)
    assert len(k) == 2


# ----------------------------------------------------------------------------- mechanical FP split
def test_mechanical_split_classes():
    gt = [[0, 0, 100, 100]]
    pb = [[0, 0, 100, 100],      # TP, conf .9
          [2, 2, 98, 98],        # duplicate of matched GT (IoU .92), conf .8
          [60, 0, 160, 100],     # IoU = 4000/16000 = .25 -> loc, conf .7
          [300, 300, 340, 340],  # bg, conf .6
          [0, 0, 10, 10]]        # IoU .01 -> bg, conf .5
    pc = [0.9, 0.8, 0.7, 0.6, 0.5]
    r = M.mechanical_split(pb, pc, gt, 0.25)
    assert [x["cls"] for x in r] == ["tp", "duplicate", "loc", "bg", "bg"]
    assert [x["idx"] for x in M.mechanical_split(pb, pc, gt, 0.65)] == [0, 1, 2]
    assert {x["cls"] for x in M.mechanical_split(pb, pc, np.zeros((0, 4)), 0.0)} == {"bg"}
    assert M.mechanical_split(np.zeros((0, 4)), [], gt, 0.1) == []


def test_mechanical_split_greedy_by_confidence():
    # the higher-confidence box (IoU .7) takes the GT even though a lower-confidence box fits better
    gt = [[0, 0, 100, 100]]
    pb = [[0, 0, 70, 100], [0, 0, 100, 100]]
    r = M.mechanical_split(pb, [0.9, 0.8], gt, 0.1)
    assert [x["cls"] for x in r] == ["tp", "duplicate"]


# ----------------------------------------------------------------------------- folds
class FakePool:
    def __init__(self):
        rng = np.random.default_rng(1)
        self.groups, self.source = [], []
        for src, ng in (("rdd2022", 40), ("hrp4k", 25)):
            for g in range(ng):
                for _ in range(int(rng.integers(1, 6))):
                    self.groups.append(f"{src}:g{g}")
                    self.source.append(src)
        self.source = np.array(self.source)
        self.stems = [f"i{i}" for i in range(len(self.groups))]


def test_folds_group_level_stratified_deterministic():
    p = FakePool()
    f = C.make_folds(p, 0)
    assert (f == C.make_folds(p, 0)).all() and not (f == C.make_folds(p, 1)).all()
    gf = {}
    for g, k in zip(p.groups, f):
        assert gf.setdefault(g, k) == k           # no group split across folds
    for src in ("rdd2022", "hrp4k"):
        sel = p.source == src
        assert set(f[sel]) == {0, 1}
        ng = len({g for g, s in zip(p.groups, p.source) if s == src})
        n0 = len({g for g, s, k in zip(p.groups, p.source, f) if s == src and k == 0})
        assert abs(n0 - ng / 2) <= 1               # groups dealt alternately


# ----------------------------------------------------------------------------- threshold tables vs project metrics
def test_thrtable_matches_op_metrics_and_weights():
    rng = np.random.default_rng(3)
    stems = [f"s{i}" for i in range(30)]
    gts, preds = {}, {}
    for s in stems:
        ng = int(rng.integers(0, 3))
        g = rng.uniform(0, 200, (ng, 2))
        gb = np.c_[g, g + rng.uniform(20, 60, (ng, 2))]
        gts[s] = (gb, np.full(ng, 0.1))
        npred = int(rng.integers(0, 8))
        q = rng.uniform(0, 200, (npred, 2))
        extra = np.c_[q, q + rng.uniform(10, 50, (npred, 2))]
        pbox = np.r_[gb + rng.normal(0, 4, gb.shape), extra]
        preds[s] = (pbox, np.round(rng.uniform(0.001, 0.99, len(pbox)), 5))

    class PP:
        pass
    pool = PP()
    pool.stems = stems
    pool.gts = gts
    P = C.build_P(pool, preds, iou_thrs=np.array([0.5]))
    T = np.arange(1, 951) / 1000
    tab = C.ThrTable(P, T)
    w = rng.integers(0, 3, len(stems)).astype(float)
    m = tab.metrics(w)
    for thr in (0.001, 0.2, 0.245, 0.5, 0.9):
        k = int(round(thr * 1000)) - 1
        ref = C.ec.op_metrics(P, thr, w)
        assert m["tp"][k] == pytest.approx(ref["TP"]) and m["fp"][k] == pytest.approx(ref["FP"])
        assert m["fp_per_image"][k] == pytest.approx(ref["fp_per_image"])
        assert m["fp_per_empty"][k] == pytest.approx(ref["fp_per_negative_image"], nan_ok=True)
        if ref["precision"] == ref["precision"]:
            assert m["precision"][k] == pytest.approx(ref["precision"])
            assert m["recall"][k] == pytest.approx(ref["recall"])
    assert C.ap50(P, w) == pytest.approx(C.ec.ap_all(P, w)[0])


# ----------------------------------------------------------------------------- ensemble.py / threshold_sensitivity.py helpers
def _synthetic(seed=5, n=40):
    rng = np.random.default_rng(seed)
    stems = [f"s{i}" for i in range(n)]
    gts, preds = {}, {}
    for s in stems:
        ng = int(rng.integers(0, 3))
        g = rng.uniform(0, 200, (ng, 2))
        gb = np.c_[g, g + rng.uniform(20, 60, (ng, 2))]
        gts[s] = (gb, np.full(ng, 0.1))
        npred = int(rng.integers(0, 6))
        q = rng.uniform(0, 200, (npred, 2))
        pbox = np.r_[gb + rng.normal(0, 3, gb.shape), np.c_[q, q + rng.uniform(10, 50, (npred, 2))]]
        preds[s] = (pbox, np.round(rng.uniform(0.01, 0.99, len(pbox)), 5))

    class PP:
        pass
    pool = PP()
    pool.stems = stems
    pool.gts = gts
    return pool, preds


def test_maxf1_threshold_matches_project_sweep_and_respects_weights():
    import ensemble as E
    pool, preds = _synthetic()
    P = C.build_P(pool, preds, iou_thrs=np.array([0.5]))
    thr, f1 = E.maxf1_threshold(P, np.ones(len(pool.stems)))
    ref = C.ec.select_thresholds(C.ec.sweep(P), 0.8)
    assert thr == pytest.approx(ref["max_f1_conf"]) and f1 == pytest.approx(ref["max_f1"])
    # 0/1 weights equal building Preds on that subset only
    w = np.zeros(len(pool.stems))
    w[::2] = 1
    sub = C.build_P(pool, preds, idx=np.arange(0, len(pool.stems), 2), iou_thrs=np.array([0.5]))
    thr2, f12 = E.maxf1_threshold(P, w)
    ref2 = C.ec.select_thresholds(C.ec.sweep(sub), 0.8)
    assert thr2 == pytest.approx(ref2["max_f1_conf"]) and f12 == pytest.approx(ref2["max_f1"])


def test_compose_and_per_image_thresholds_match_op_metrics():
    import ensemble as E
    pool, preds = _synthetic(7)
    other = {s: (b[: max(len(b) - 1, 0)], c[: max(len(b) - 1, 0)]) for s, (b, c) in preds.items()}
    P1 = C.build_P(pool, preds, iou_thrs=np.array([0.5]))
    P2 = C.build_P(pool, other, iou_thrs=np.array([0.5]))
    choice = np.arange(len(pool.stems)) % 2
    Q = E.compose([P1, P2], choice)
    # composite == building from the per-image chosen predictions
    mixed = {s: (preds if choice[i] == 0 else other)[s] for i, s in enumerate(pool.stems)}
    Qref = C.build_P(pool, mixed, iou_thrs=np.array([0.5]))
    assert E.maxf1_threshold(Q, np.ones(len(pool.stems)))[1] == pytest.approx(E.maxf1_threshold(Qref, np.ones(len(pool.stems)))[1])
    # one threshold everywhere == op_metrics
    tpi, fpi = E.per_image_counts(Q, np.full(len(pool.stems), 0.3))
    gt = np.array([len(pool.gts[s][0]) for s in pool.stems], float)
    r = E.op_rates(tpi, fpi, gt, np.ones(len(pool.stems)))
    ref = C.ec.op_metrics(Qref, 0.3)
    assert r["precision"] == pytest.approx(ref["precision"]) and r["recall"] == pytest.approx(ref["recall"])
    assert r["fp_per_image"] == pytest.approx(ref["fp_per_image"])
    # per-image thresholds differ by fold: counts are the sum of the two single-threshold counts on their images
    thr_img = np.where(choice == 0, 0.2, 0.6)
    t, f = E.per_image_counts(Qref, thr_img)
    t_lo, f_lo = E.per_image_counts(Qref, np.full(len(choice), 0.2))
    t_hi, f_hi = E.per_image_counts(Qref, np.full(len(choice), 0.6))
    assert (t == np.where(choice == 0, t_lo, t_hi)).all() and (f == np.where(choice == 0, f_lo, f_hi)).all()


def test_plateau_and_budget_helpers():
    import threshold_sensitivity as T
    thr = np.arange(0.05, 0.80, 0.01)
    f1 = 0.5 - np.abs(thr - 0.3)          # peak 0.5 at 0.30
    lo, hi = T.plateau(thr, f1, 0.02)
    assert lo == pytest.approx(0.28) and hi == pytest.approx(0.32)
    m = dict(fp_per_image=np.linspace(2.0, 0.0, 950), recall=np.linspace(0.9, 0.0, 950), precision=np.full(950, 0.5))
    out = T.budget_stats(m)
    p = out["fpbudget_0.10"]
    assert p["fp_per_image"] <= 0.10
    prev = m["fp_per_image"][int(round(p["threshold"] * 1000)) - 2]
    assert prev > 0.10                                                 # the next-lower threshold breaks the budget
    q = out["recall_0.5"]
    assert q["recall"] >= 0.5 and T.T_FINE[0] < q["threshold"]
    m["recall"] = np.linspace(0.2, 0.0, 950)
    assert T.budget_stats(m)["recall_0.3"] is None                     # unreachable recall
