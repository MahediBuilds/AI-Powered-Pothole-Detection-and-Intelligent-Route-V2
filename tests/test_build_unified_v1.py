"""Tests for the pure logic in scripts/build_unified_v1.py."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "build_unified_v1", Path(__file__).resolve().parents[1] / "scripts" / "build_unified_v1.py")
bu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bu)


def test_popcount_and_pairs():
    a = np.array([0, 0xFF, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    assert bu.popcount64(a).tolist() == [0, 8, 64]
    pairs = bu.pairs_within(a, None, 8)
    assert pairs == [(0, 1, 8)]
    assert bu.pairs_within(a, np.array([0], dtype=np.uint64), 0) == [(0, 0, 0)]


def test_components_are_transitive_and_deterministic():
    lab = bu.connected_components(5, [(0, 1, 1), (3, 1, 2)])
    assert lab.tolist() == [0, 0, 1, 0, 2]
    assert bu.cluster_group_ids(lab, "g")[4] == "g00002"


def test_leakage_guard_only_drops_val_test_with_train_in_cluster():
    splits = ["train", "val", "test", "val", "test"]
    labels = np.array([0, 0, 0, 1, 2])
    assert bu.leakage_guard(splits, labels) == {1, 2}


def test_valtest_spanning_drops_test_side_only():
    splits = ["val", "test", "test", "val"]
    labels = np.array([0, 0, 1, 2])
    assert bu.valtest_spanning(splits, labels, {0, 1, 2, 3}) == {1}
    assert bu.valtest_spanning(splits, labels, {1, 2, 3}) == set()  # val side already gone


IDMAP = [dict(tile_id=f"T{i}", split="train", image_file=f"{i // 2}.jpg") for i in range(4)]


def test_verdicts_complete_and_reject_drops_whole_image():
    v = [dict(tile_id="T0", verdict="ok"), dict(tile_id="T1", verdict="reject"),
         dict(tile_id="T2", verdict="unclear"), dict(tile_id="T3", verdict="ok")]
    assert bu.check_verdicts(v, IDMAP, expected=4) == {"ok": 2, "reject": 1, "unclear": 1}
    assert bu.rejected_images(v, IDMAP) == {("train", "0.jpg")}  # T1 belongs to image 0


def test_verdicts_missing_or_invalid_raise():
    ok = [dict(tile_id=f"T{i}", verdict="ok") for i in range(4)]
    with pytest.raises(ValueError):
        bu.check_verdicts(ok[:3], IDMAP, expected=4)
    with pytest.raises(ValueError):
        bu.check_verdicts(ok[:3] + [dict(tile_id="T3", verdict="maybe")], IDMAP, expected=4)
    with pytest.raises(ValueError):
        bu.check_verdicts(ok, IDMAP, expected=5)


def test_timestamp_parse():
    assert bu.parse_timestamp("20250216_164325.jpg").minute == 43
    assert bu.parse_timestamp("IMG_0001.jpg") is None


def test_time_groups_60s_gap_and_untimestamped():
    names = ["20250216_164300.jpg", "20250216_164400.jpg", "20250216_164501.jpg", "x.jpg", "y.jpg"]
    g = bu.time_groups(names)
    assert g["20250216_164300.jpg"][0] == g["20250216_164400.jpg"][0]  # exactly 60 s apart: same group
    assert g["20250216_164501.jpg"][0] != g["20250216_164400.jpg"][0]  # 61 s: new group
    assert g["x.jpg"][1] and g["y.jpg"][1] and g["x.jpg"][0] != g["y.jpg"][0]
    assert not g["20250216_164300.jpg"][1]


def test_xywh_to_yolo_and_clipping():
    box, clipped = bu.xywh_px_to_yolo(100, 50, 200, 100, 640, 360)
    assert not clipped
    assert box == pytest.approx((0.3125, 100 / 360, 0.3125, 100 / 360))
    box, clipped = bu.xywh_px_to_yolo(600, 300, 100, 100, 640, 360)
    assert clipped and bu.label_ok(bu.yolo_line(box))
    with pytest.raises(ValueError):
        bu.xywh_px_to_yolo(700, 0, 10, 10, 640, 360)


def test_label_ok():
    assert bu.label_ok("0 0.5 0.5 0.2 0.2")
    assert not bu.label_ok("1 0.5 0.5 0.2 0.2")
    assert not bu.label_ok("0 0.95 0.5 0.2 0.2")
    assert not bu.label_ok("0 0.5 0.5 0 0.2")
    assert not bu.label_ok("0 0.5 0.5 0.2")


def test_list_entry_format_is_relative_and_forward_slash():
    e = bu.list_entry("hrp4k", "val", "hrp4k_val_1.jpg")
    assert e == "./../images/hrp4k/val/hrp4k_val_1.jpg" and "\\" not in e
