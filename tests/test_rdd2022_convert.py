import hashlib
import io
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import PIL
import pytest
import yaml
from PIL import Image
from rdd_fixtures import (
    BOX,
    Obj,
    RawTree,
    add_val_test_positives,
    dhash_of_file,
    hamming,
    make_cfg,
    noise_pixels,
    read_manifest,
    variant_pixels,
)

from roadsense.config import ConfigError, load_dataset_config, project_root
from roadsense.data import convert_rdd2022 as conv
from roadsense.data import splits
from roadsense.data.rdd2022 import ConversionError, dhash64
from roadsense.training import train

SPLITS = ("train", "val", "test")


def run(cfg, root, **kw):
    return conv.convert(cfg, root=root, out=lambda m: None, **kw)


def raw(tmp_path):
    return RawTree(tmp_path / "data" / "raw")


def out_dir(tmp_path):
    return tmp_path / "data" / "processed" / "ds"


def label_text(tmp_path, split, sid):
    return (out_dir(tmp_path) / "labels" / split / f"{sid}.txt").read_text(encoding="utf-8")


def manifest_rows(tmp_path):
    return {r["source_id"]: r for r in read_manifest(out_dir(tmp_path) / "manifest.csv")}


def jpeg_bytes(size=(100, 50)):
    buf = io.BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(buf, "JPEG")
    return buf.getvalue()


# ------------------------------------------------------------------ labels and normalisation


def test_known_boxes_become_exact_yolo_lines(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX)])  # 100x50 (non-square)
    tree.add("India", 1, [Obj("D40", ("12.5", "5.25", "32.5", "25.25"))], style="norway")
    tree.add(
        "India", 2,
        [Obj("D40", BOX, full=True), Obj("D00", (1, 1, 9, 9)), Obj("D40", (50, 10, 70, 30))],
        style="japan",
    )  # fmt: skip
    tree.add("India", 3, [Obj("D40", (4, 8, 24, 48))], size=(40, 80))  # portrait
    add_val_test_positives(tree)
    result = run(make_cfg(tmp_path), tmp_path)
    assert result.exit_code == 0
    assert label_text(tmp_path, "train", "India_000000") == "0 0.200000 0.300000 0.200000 0.400000\n"
    assert label_text(tmp_path, "train", "India_000001") == "0 0.225000 0.305000 0.200000 0.400000\n"
    assert label_text(tmp_path, "train", "India_000002") == (
        "0 0.200000 0.300000 0.200000 0.400000\n0 0.600000 0.400000 0.200000 0.400000\n"
    )  # the D00 box is not written
    assert label_text(tmp_path, "train", "India_000003") == "0 0.350000 0.350000 0.500000 0.500000\n"
    assert manifest_rows(tmp_path)["India_000002"]["n_pothole_boxes"] == "2"
    assert manifest_rows(tmp_path)["India_000002"]["labels_seen"] == "D00|D40"


def test_real_image_size_is_authoritative_when_xml_has_no_size(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", (20, 10, 60, 50))], size=(200, 100), xml_size=None)
    add_val_test_positives(tree)
    run(make_cfg(tmp_path), tmp_path)
    assert label_text(tmp_path, "train", "India_000000") == "0 0.200000 0.300000 0.200000 0.400000\n"
    row = manifest_rows(tmp_path)["India_000000"]
    assert (row["width"], row["height"], row["xml_width"], row["xml_height"]) == ("200", "100", "", "")


def test_size_mismatch_excludes_positive_but_keeps_negative(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX)], xml_size=(50, 25))
    tree.add("India", 1, [Obj("D44", BOX)], xml_size=(50, 25))
    add_val_test_positives(tree)
    result = run(make_cfg(tmp_path), tmp_path)
    rows = manifest_rows(tmp_path)
    assert rows["India_000000"]["status"] == "excluded:size_mismatch"
    assert rows["India_000000"]["xml_width"] == "50" and rows["India_000000"]["width"] == "100"
    assert rows["India_000001"]["status"] == "included"
    assert rows["India_000001"]["negative_kind"] == "other_labels"
    assert not (out_dir(tmp_path) / "labels" / "train" / "India_000000.txt").exists()
    mism = {m["source_id"]: m for m in result.report["size_mismatches"]}
    assert mism["India_000000"]["resolution"] == "excluded"
    assert mism["India_000000"]["xml_size"] == [50, 25] and mism["India_000000"]["real_size"] == [100, 50]
    assert mism["India_000001"]["resolution"] == "kept_as_negative_candidate"


def test_partly_outside_box_is_clipped_and_recorded(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", (90, 5, 120, 25))])
    add_val_test_positives(tree)
    result = run(make_cfg(tmp_path), tmp_path)
    assert label_text(tmp_path, "train", "India_000000") == "0 0.950000 0.300000 0.100000 0.400000\n"
    row = manifest_rows(tmp_path)["India_000000"]
    assert (row["status"], row["n_pothole_boxes"], row["n_pothole_clipped"]) == ("included", "1", "1")
    (clip,) = result.report["clipped_boxes"]
    assert clip["source_id"] == "India_000000" and clip["clipped_to"] == [90.0, 5.0, 100.0, 25.0]
    assert clip["raw"]["xmax"] == "120"


@pytest.mark.parametrize(
    ("bad_box", "reason"),
    [
        ((150, 5, 180, 25), "outside_image"),
        ((30, 5, 10, 25), "degenerate_box"),  # swapped
        ((20, 5, 20, 25), "degenerate_box"),  # zero width
        ((10, 5, 11, 25), "below_min_side"),
        (None, "missing_bndbox"),
        (("abc", 5, 30, 25), "non_numeric_coordinate"),
        (("", 5, 30, 25), "missing_coordinate"),
    ],
)
def test_invalid_pothole_box_excludes_whole_image_and_is_not_a_negative(tmp_path, bad_box, reason):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX), Obj("D40", bad_box)])  # one good, one bad
    add_val_test_positives(tree)
    result = run(make_cfg(tmp_path), tmp_path)
    row = manifest_rows(tmp_path)["India_000000"]
    assert row["status"] == "excluded:invalid_pothole_box"
    assert row["is_positive"] == "" and row["negative_kind"] == ""
    assert not list((out_dir(tmp_path) / "labels").glob("*/India_000000.txt"))
    assert not list((out_dir(tmp_path) / "images").glob("*/India_000000.jpg"))
    (bad,) = result.report["invalid_annotations"]
    assert (bad["source_id"], bad["object_index"], bad["label"], bad["reason"]) == (
        "India_000000", 1, "D40", reason,
    )  # fmt: skip
    if bad_box is not None:
        assert bad["raw"]["xmin"] == (None if bad_box[0] == "" else str(bad_box[0]))
    assert result.report["by_country"]["India"]["negatives"] == {
        "crack_only": 0, "other_labels": 0, "no_objects": 0,
    }  # fmt: skip
    assert any(e["source_id"] == "India_000000" for e in result.report["excluded_items"])


def test_min_box_side_is_configurable(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", (10, 5, 12, 25))])  # 2 px wide
    add_val_test_positives(tree)
    run(make_cfg(tmp_path), tmp_path)
    assert manifest_rows(tmp_path)["India_000000"]["status"] == "included"  # side == min is fine
    run(make_cfg(tmp_path, min_box_side_px=3), tmp_path, overwrite=True)
    assert manifest_rows(tmp_path)["India_000000"]["status"] == "excluded:invalid_pothole_box"


def test_invalid_non_pothole_box_is_reported_but_image_stays(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX), Obj("D00", ("x", 1, 2, 3)), Obj("D10", None)])
    add_val_test_positives(tree)
    result = run(make_cfg(tmp_path), tmp_path)
    assert manifest_rows(tmp_path)["India_000000"]["status"] == "included"
    assert result.report["non_pothole_object_problems"]["India"] == 2
    assert result.report["invalid_annotations"] == []


# ------------------------------------------------------------------ negatives


def test_negative_kinds_and_empty_label_files(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D00", (1, 1, 9, 9)), Obj("D10", (2, 2, 9, 9))])  # crack only
    tree.add("India", 1, [Obj("D44", BOX)])
    tree.add("India", 2, [Obj("Repair", BOX)])
    tree.add("India", 3, [])  # zero objects
    tree.add("India", 4, [Obj("D00", BOX), Obj("D44", BOX)])  # crack + other
    tree.add("India", 5, [Obj("D0w0", BOX)])  # unknown label
    tree.add_unlabelled("India", 100)
    add_val_test_positives(tree)
    result = run(make_cfg(tmp_path), tmp_path)
    kinds = {0: "crack_only", 1: "other_labels", 2: "other_labels", 3: "no_objects",
             4: "other_labels", 5: "other_labels"}  # fmt: skip
    rows = manifest_rows(tmp_path)
    for num, kind in kinds.items():
        sid = f"India_{num:06d}"
        assert (rows[sid]["status"], rows[sid]["is_positive"], rows[sid]["negative_kind"]) == ("included", "0", kind)
        assert rows[sid]["n_pothole_boxes"] == "0"
        assert (out_dir(tmp_path) / "labels" / "train" / f"{sid}.txt").read_bytes() == b""
        assert (out_dir(tmp_path) / "images" / "train" / f"{sid}.jpg").is_file()
    assert result.report["by_country"]["India"]["negatives"] == {
        "crack_only": 1, "other_labels": 4, "no_objects": 1,
    }  # fmt: skip
    counts = result.report["label_counts"]["India"]
    assert counts["D0w0"] == 1 and counts["Repair"] == 1 and counts["D44"] == 2
    # unlabelled official test images are never read or used
    assert "India_000100" not in rows
    assert not list(out_dir(tmp_path).rglob("India_000100*"))
    assert result.report["unlabelled_official_test_images"] == {"Czech": 0, "India": 1, "Japan": 0}


def test_negative_from_country_not_in_negative_countries_is_excluded(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX)])
    tree.add("Japan", 0, [Obj("D40", BOX)])
    tree.add("Japan", 1, [])
    tree.add("Czech", 0, [Obj("D40", BOX)])
    run(make_cfg(tmp_path, negative_countries=["India"]), tmp_path)
    rows = manifest_rows(tmp_path)
    assert rows["Japan_000001"]["status"] == "excluded:negatives_not_enabled"
    assert rows["Japan_000001"]["negative_kind"] == "no_objects"
    assert not (out_dir(tmp_path) / "labels" / "val" / "Japan_000001.txt").exists()


# ------------------------------------------------------------------ bad inputs


def test_broken_items_get_statuses_and_run_completes(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX)], image=False)
    tree.add("India", 1, [Obj("D40", BOX)], xml=False)
    good = jpeg_bytes()
    tree.add("India", 2, [Obj("D40", BOX)], image_bytes=good[: len(good) // 3])
    tree.add("India", 3, xml_text="<annotation><size><width>1</width>")
    tree.add("India", 4, xml_text="<notannotation/>")
    tree.add("India", 5, [Obj("D40", BOX)])
    add_val_test_positives(tree)
    result = run(make_cfg(tmp_path), tmp_path)
    assert result.exit_code == 0
    rows = manifest_rows(tmp_path)
    assert {n: rows[f"India_{n:06d}"]["status"] for n in range(6)} == {
        0: "excluded:missing_image",
        1: "excluded:missing_annotation",
        2: "excluded:corrupt_image",
        3: "excluded:malformed_xml",
        4: "excluded:malformed_xml",
        5: "included",
    }
    assert rows["India_000000"]["image_relpath"] == ""
    assert rows["India_000000"]["xml_relpath"] == "India/train/annotations/xmls/India_000000.xml"
    assert rows["India_000001"]["xml_relpath"] == ""
    assert rows["India_000001"]["image_relpath"] == "India/train/images/India_000001.jpg"
    assert result.report["status_counts"]["excluded:corrupt_image"] == 1
    reasons = {e["source_id"]: e["detail"] for e in result.report["excluded_items"]}
    assert "truncated" in reasons["India_000002"].lower() or "OSError" in reasons["India_000002"]


def test_exif_orientation_is_reported_as_warning(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX)], exif_orientation=6)
    add_val_test_positives(tree)
    result = run(make_cfg(tmp_path), tmp_path)
    assert result.report["exif_oriented_images"] == ["India_000000"]
    assert any("EXIF" in w for w in result.report["warnings"])


def test_missing_country_folder_is_a_hard_error_naming_fetch_command(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX)])
    tree.add("Japan", 0, [Obj("D40", BOX)])
    with pytest.raises(ConversionError, match=r"fetch_rdd2022 --countries Czech"):
        run(make_cfg(tmp_path), tmp_path)
    assert not out_dir(tmp_path).exists()


# ------------------------------------------------------------------ negative cap


def test_train_negative_cap_exact_count_deterministic_choice_val_test_untouched(tmp_path):
    tree = raw(tmp_path)
    for n in range(3):
        tree.add("India", n, [Obj("D40", BOX)])
    negatives = [tree.add("India", 10 + n, []) for n in range(10)]
    tree.add("Japan", 0, [Obj("D40", BOX)])
    tree.add("Czech", 0, [Obj("D40", BOX)])
    for n in range(1, 6):
        tree.add("Japan", n, [])
        tree.add("Czech", n, [])
    result = run(make_cfg(tmp_path, train_negative_ratio=1.5), tmp_path)  # floor(1.5 * 3) = 4
    rows = manifest_rows(tmp_path)
    kept = {s for s in negatives if rows[s]["status"] == "included"}
    expected = sorted(negatives, key=lambda s: hashlib.sha256(f"42:{s}".encode()).hexdigest())[:4]
    assert kept == set(expected)
    assert sum(rows[s]["status"] == "excluded:negative_cap" for s in negatives) == 6
    assert result.report["by_split"]["train"]["negatives"]["no_objects"] == 4
    assert result.report["by_split"]["val"]["negatives"]["no_objects"] == 5  # untouched
    assert result.report["by_split"]["test"]["negatives"]["no_objects"] == 5
    assert result.report["status_counts"]["excluded:negative_cap"] == 6
    # same choice on a rerun
    run(make_cfg(tmp_path, train_negative_ratio=1.5), tmp_path, overwrite=True)
    assert {s for s in negatives if manifest_rows(tmp_path)[s]["status"] == "included"} == kept


def test_null_ratio_keeps_all_negatives(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX)])
    for n in range(1, 8):
        tree.add("India", n, [])
    add_val_test_positives(tree)
    result = run(make_cfg(tmp_path, train_negative_ratio=None), tmp_path)
    assert result.report["by_split"]["train"]["negatives"]["no_objects"] == 7


# ------------------------------------------------------------------ splits through the converter


def test_block_strategy_has_no_group_or_source_leakage(tmp_path):
    tree = raw(tmp_path)
    for n in range(40):
        tree.add("India", n, [Obj("D40", BOX)])
    cfg = make_cfg(
        tmp_path,
        countries=["India"],
        negative_countries=["India"],
        split={
            "strategy": "blocks", "block_size": 5,
            "fractions": {"train": 0.5, "val": 0.25, "test": 0.25},
        },
    )  # fmt: skip
    result = run(cfg, tmp_path)
    assert result.exit_code == 0
    rows = list(manifest_rows(tmp_path).values())
    by_group: dict[str, set[str]] = {}
    for r in rows:
        assert r["group_id"] == f"India:b{int(r['numeric_id']) // 5}"
        by_group.setdefault(r["group_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_group.values())  # a block never straddles splits
    assert Counter(next(iter(s)) for s in by_group.values()) == {"train": 4, "val": 2, "test": 2}
    on_disk = {s: {p.stem for p in (out_dir(tmp_path) / "images" / s).iterdir()} for s in SPLITS}
    assert not (on_disk["train"] & on_disk["val"]) and not (on_disk["train"] & on_disk["test"])
    assert not (on_disk["val"] & on_disk["test"])
    assert sum(len(v) for v in on_disk.values()) == 40
    groups = result.report["groups_per_split"]
    assert not (set(groups["train"]) & set(groups["val"])) and len(groups["train"]) == 4


def test_block_strategy_same_seed_same_assignment_different_seed_differs(tmp_path):
    tree = raw(tmp_path)
    for n in range(80):
        tree.add("India", n, [Obj("D40", BOX)])

    def assignment(seed):
        cfg = make_cfg(
            tmp_path, countries=["India"], negative_countries=["India"],
            split={"strategy": "blocks", "block_size": 2, "seed": seed},
        )  # fmt: skip
        run(cfg, tmp_path, scan_only=True)
        scan = tmp_path / "data" / "processed" / "ds_scan" / "manifest.csv"
        return {r["source_id"]: r["split"] for r in read_manifest(scan)}

    assert assignment(42) == assignment(42)
    assert assignment(42) != assignment(43)


def test_block_strategy_with_few_groups_warns(tmp_path):
    tree = raw(tmp_path)
    for n in range(4):
        tree.add("Czech", n, [Obj("D40", BOX)])
    cfg = make_cfg(
        tmp_path, countries=["Czech"], negative_countries=["Czech"],
        split={"strategy": "blocks", "block_size": 5},
    )  # fmt: skip
    result = run(cfg, tmp_path)
    assert result.exit_code == 1  # one block: val and test stay empty
    assert any(w.startswith("Czech: only 1 block") for w in result.report["warnings"])
    assert any("val split has zero positive" in w for w in result.report["warnings"])


def test_country_strategy_places_whole_countries(tmp_path):
    tree = raw(tmp_path)
    for country in ("India", "Japan", "Czech"):
        for n in range(3):
            tree.add(country, n, [Obj("D40", BOX)])
    run(make_cfg(tmp_path), tmp_path)
    for r in manifest_rows(tmp_path).values():
        assert r["split"] == {"India": "train", "Japan": "val", "Czech": "test"}[r["country"]]
        assert r["group_id"] == r["country"]


def test_empty_split_gives_nonzero_exit_code(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX)])
    tree.add("Japan", 0, [Obj("D40", BOX)])
    tree.add("Czech", 0, [])  # the test split holds only a negative, which still counts as non-empty
    cfg = make_cfg(tmp_path, split={"val_countries": ["Japan"], "test_countries": ["Czech"]})
    assert run(cfg, tmp_path).exit_code == 0
    (tmp_path / "data" / "raw" / "Japan" / "train" / "images" / "Japan_000000.jpg").unlink()
    result = run(cfg, tmp_path, overwrite=True)
    assert result.exit_code == 1


def test_leakage_check_raises():
    rows = [
        conv.Row("India_000000", "India", 0, group_id="India:b0", split="train"),
        conv.Row("India_000001", "India", 1, group_id="India:b0", split="val"),
    ]
    with pytest.raises(conv.LeakageError, match="India:b0"):
        conv.check_leakage(rows)
    rows[1].split = "train"
    conv.check_leakage(rows)
    rows.append(conv.Row("India_000000", "India", 0, group_id="India:b9", split="test"))
    with pytest.raises(conv.LeakageError, match="more than once"):
        conv.check_leakage(rows)


def test_exact_duplicates_across_splits_are_reported(tmp_path):
    tree = raw(tmp_path)
    same = dict(size=(100, 50), color=(5, 5, 5))
    tree.add("India", 0, [Obj("D40", BOX)], **same)
    tree.add("Japan", 0, [Obj("D40", BOX)], **same)
    tree.add("Czech", 0, [Obj("D40", BOX)])
    result = run(make_cfg(tmp_path), tmp_path)
    (dup,) = result.report["duplicate_images_across_splits"]
    assert {(i["source_id"], i["split"]) for i in dup["items"]} == {("India_000000", "train"), ("Japan_000000", "val")}
    assert any("exact-duplicate" in w for w in result.report["warnings"])


# ------------------------------------------------------------------ manifest and report


def mixed_tree(raw_dir, reverse=False):
    """A scenario touching every status, in India (train), Japan (val), Czech (test)."""
    tree = RawTree(raw_dir)
    good = jpeg_bytes()
    specs = [
        ("India", 0, [Obj("D40", BOX)], {}),
        ("India", 1, [Obj("D40", BOX), Obj("D40", (50, 10, 70, 30))], {"style": "japan"}),
        ("India", 2, [Obj("D00", (1, 1, 9, 9))], {}),
        ("India", 3, [], {}),
        ("India", 4, [Obj("D40", (30, 5, 10, 25))], {}),
        ("India", 5, [Obj("D40", BOX)], {"image": False}),
        ("India", 6, [Obj("D40", BOX)], {"xml": False}),
        ("India", 7, [Obj("D40", BOX)], {"image_bytes": good[:50]}),
        ("India", 8, None, {"xml_text": "<annotation>"}),
        ("India", 9, [Obj("D40", BOX)], {"xml_size": (7, 7)}),
        ("India", 10, [Obj("D40", (90, 5, 120, 25))], {}),
        ("India", 11, [Obj("D44", BOX)], {}),
        ("Japan", 0, [Obj("D40", BOX)], {}),
        ("Japan", 1, [], {}),
        ("Japan", 2, [Obj("D40", ("1.5", "2.5", "30.5", "20.5"))], {"style": "norway"}),
        ("Czech", 0, [Obj("D40", BOX)], {}),
        ("Czech", 1, [Obj("D10", (1, 1, 9, 9))], {}),
    ]
    for country, num, objs, kw in reversed(specs) if reverse else specs:
        tree.add(country, num, objs, **kw)
    tree.add_unlabelled("India", 100)
    return tree, {(c, n) for c, n, _, _ in specs}


def test_manifest_has_one_sorted_row_per_labelled_item_and_report_matches(tmp_path):
    tree, expected_items = mixed_tree(tmp_path / "data" / "raw")
    cfg = make_cfg(tmp_path, train_negative_ratio=0.67)  # India: 3 positives, 3 negatives -> 2 kept
    result = run(cfg, tmp_path)
    assert result.exit_code == 0
    text = (out_dir(tmp_path) / "manifest.csv").read_bytes().decode("utf-8")
    assert not text.startswith("﻿") and "\r" not in text
    assert text.splitlines()[0] == (
        "source_id,country,numeric_id,group_id,split,status,is_positive,negative_kind,"
        "n_pothole_boxes,n_pothole_clipped,n_other_objects,labels_seen,width,height,"
        "xml_width,xml_height,image_relpath,xml_relpath,image_sha256,dhash"
    )
    rows = read_manifest(out_dir(tmp_path) / "manifest.csv")
    assert {(r["country"], int(r["numeric_id"])) for r in rows} == expected_items
    assert len(rows) == len(expected_items)  # exactly one row per labelled source item
    keys = [(r["country"], int(r["numeric_id"]), r["source_id"]) for r in rows]
    assert keys == sorted(keys)
    assert all(r["status"] in conv.STATUSES for r in rows)
    status_counts = Counter(r["status"] for r in rows)
    assert sum(status_counts.values()) == len(rows) == result.report["total_labelled_items"]
    assert result.report["status_counts"] == {s: status_counts.get(s, 0) for s in conv.STATUSES}
    assert result.report["manifest_sha256"] == hashlib.sha256((out_dir(tmp_path) / "manifest.csv").read_bytes()).hexdigest()
    # hashes only for included rows
    for r in rows:
        assert bool(r["image_sha256"]) == (r["status"] == "included")
    # dhash: filled for every row whose image decodes (whatever its status), recomputable
    by_id_all = {r["source_id"]: r for r in rows}
    for r in rows:
        if r["source_id"] in ("India_000005", "India_000007"):  # missing / truncated image
            assert r["dhash"] == ""
            continue
        assert re.fullmatch(r"[0-9a-f]{16}", r["dhash"]), r["source_id"]
        assert int(r["dhash"], 16) == dhash_of_file(tmp_path / "data" / "raw" / r["image_relpath"])
    assert by_id_all["India_000006"]["status"] == "excluded:missing_annotation"
    assert by_id_all["India_000006"]["dhash"] != ""  # excluded before inspection, still hashed
    assert by_id_all["India_000008"]["status"] == "excluded:malformed_xml"
    assert by_id_all["India_000008"]["dhash"] != ""
    assert by_id_all["India_000009"]["status"] == "excluded:size_mismatch"
    assert by_id_all["India_000009"]["dhash"] != ""

    # report counts == counts recomputed from the manifest and from the label files on disk
    for split in SPLITS:
        included = [r for r in rows if r["status"] == "included" and r["split"] == split]
        label_files = sorted((out_dir(tmp_path) / "labels" / split).glob("*.txt"))
        image_files = sorted((out_dir(tmp_path) / "images" / split).glob("*.jpg"))
        assert [p.stem for p in label_files] == [p.stem for p in image_files] == sorted(r["source_id"] for r in included)
        positives_on_disk = sum(1 for p in label_files if p.stat().st_size > 0)
        boxes_on_disk = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in label_files)
        rep = result.report["by_split"][split]
        assert rep["images"] == len(included)
        assert rep["positives"] == positives_on_disk == sum(r["is_positive"] == "1" for r in included)
        assert sum(rep["negatives"].values()) == len(included) - positives_on_disk
        assert rep["pothole_boxes"] == boxes_on_disk == sum(int(r["n_pothole_boxes"]) for r in included)
    for country in ("India", "Japan", "Czech"):
        included = [r for r in rows if r["status"] == "included" and r["country"] == country]
        assert result.report["by_country"][country]["images"] == len(included)
        for split in SPLITS:
            n = sum(r["split"] == split for r in included)
            assert result.report["by_country_split"][country][split]["images"] == n
    # a few specific rows
    by_id = {r["source_id"]: r for r in rows}
    assert by_id["India_000004"]["status"] == "excluded:invalid_pothole_box"
    assert by_id["India_000009"]["status"] == "excluded:size_mismatch"
    assert by_id["India_000010"]["n_pothole_clipped"] == "1"
    assert status_counts["excluded:negative_cap"] == 1
    assert (out_dir(tmp_path) / "dataset.yaml").read_text(encoding="utf-8") == (
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: pothole\n"
    )


def test_report_has_no_absolute_paths_or_timestamps_and_records_provenance(tmp_path):
    mixed_tree(tmp_path / "data" / "raw")
    result = run(make_cfg(tmp_path), tmp_path)
    text = (out_dir(tmp_path) / "conversion_report.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in text and tmp_path.as_posix() not in text
    assert text.endswith("}\n") and "\r" not in text
    report = json.loads(text)
    assert list(report) == sorted(report)
    assert report["dataset_source"]["doi"] == "10.6084/m9.figshare.21431547.v1"
    assert report["dataset_source"]["outer_zip_md5"] == "b62bd51d2ffcfaa76c60f234f0cc2bb3"
    assert report["config"]["raw_dir"] == "data/raw" and report["config"]["out_dir"] == "data/processed/ds"
    assert len(report["config_sha256"]) == 64 and report["tool_version"]
    assert report["link_method"] == "hardlink"
    assert result.report == report


def test_warning_for_country_without_pothole_boxes(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX)])
    tree.add("Japan", 0, [Obj("D40", BOX)])
    tree.add("Czech", 0, [Obj("D00", BOX)])
    tree.add("Czech", 1, [])
    result = run(make_cfg(tmp_path), tmp_path)
    warnings = result.report["warnings"]
    assert any(w.startswith("Czech: no pothole boxes found") and "unreliable" in w for w in warnings)
    assert any("test split has zero positive images" in w for w in warnings)


# ------------------------------------------------------------------ determinism, scan-only, dirs


def snapshot(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def test_two_runs_with_different_creation_order_are_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    mixed_tree(a / "data" / "raw", reverse=False)
    mixed_tree(b / "data" / "raw", reverse=True)
    ra = run(make_cfg(a, train_negative_ratio=0.67), a)
    rb = run(make_cfg(b, train_negative_ratio=0.67), b)
    snap_a = snapshot(a / "data" / "processed" / "ds")
    snap_b = snapshot(b / "data" / "processed" / "ds")
    assert snap_a.keys() == snap_b.keys()
    assert "manifest.csv" in snap_a and "dataset.yaml" in snap_a and "conversion_report.json" in snap_a
    for name in snap_a:
        assert snap_a[name] == snap_b[name], name
    assert ra.report == rb.report
    # and a second run into the same location (after --overwrite) reproduces the first
    run(make_cfg(a, train_negative_ratio=0.67), a, overwrite=True)
    assert snapshot(a / "data" / "processed" / "ds") == snap_a


def test_scan_only_writes_no_images_or_labels(tmp_path):
    mixed_tree(tmp_path / "data" / "raw")
    cfg = make_cfg(tmp_path)
    result = run(cfg, tmp_path, scan_only=True)
    scan = tmp_path / "data" / "processed" / "ds_scan"
    assert sorted(p.name for p in scan.iterdir()) == ["conversion_report.json", "manifest.csv"]
    assert not out_dir(tmp_path).exists()
    assert result.report["scan_only"] is True and result.report["link_method"] == "none"
    run(cfg, tmp_path)  # a full run afterwards gives the same manifest
    assert (scan / "manifest.csv").read_bytes() == (out_dir(tmp_path) / "manifest.csv").read_bytes()
    run(cfg, tmp_path, scan_only=True)  # scan dir may be rewritten freely


def test_non_empty_out_dir_refused_without_overwrite(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX)])
    add_val_test_positives(tree)
    cfg = make_cfg(tmp_path)
    run(cfg, tmp_path)
    stale = out_dir(tmp_path) / "labels" / "train.cache"
    stale.write_bytes(b"stale")
    with pytest.raises(ConversionError, match="not empty"):
        run(cfg, tmp_path)
    assert stale.exists()  # refusal changes nothing
    run(cfg, tmp_path, overwrite=True)
    assert not stale.exists()
    # the raw data is untouched by removal of hard links
    assert (tmp_path / "data" / "raw" / "India" / "train" / "images" / "India_000000.jpg").is_file()
    assert Image.open(tmp_path / "data" / "raw" / "India" / "train" / "images" / "India_000000.jpg").size == (100, 50)


def test_empty_existing_out_dir_is_fine(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX)])
    add_val_test_positives(tree)
    out_dir(tmp_path).mkdir(parents=True)
    assert run(make_cfg(tmp_path), tmp_path).exit_code == 0


@pytest.mark.parametrize(
    ("raw_dir", "out", "message"),
    [
        ("data/raw", "data/raw", "overlap"),
        ("data/raw", "data/raw/out", "overlap"),
        ("data/proc/raw", "data/proc", "overlap"),
        ("data/raw", "data", "must be inside"),
        ("data/raw", "outside", "must be inside"),
    ],
)
def test_out_dir_must_be_inside_data_and_not_overlap_raw(tmp_path, raw_dir, out, message):
    cfg = make_cfg(tmp_path, raw_dir=raw_dir, out_dir=out)
    with pytest.raises(ConversionError, match=message):
        run(cfg, tmp_path)
    with pytest.raises(ConversionError, match=message):
        run(cfg, tmp_path, overwrite=True)  # --overwrite must never widen what may be deleted


def test_hardlink_link_mode_links_and_copy_mode_copies(tmp_path):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX)])
    add_val_test_positives(tree)
    src = tmp_path / "data" / "raw" / "India" / "train" / "images" / "India_000000.jpg"
    dst = out_dir(tmp_path) / "images" / "train" / "India_000000.jpg"
    assert run(make_cfg(tmp_path, link_mode="hardlink"), tmp_path).report["link_method"] == "hardlink"
    assert os.path.samefile(src, dst)
    result = run(make_cfg(tmp_path, link_mode="copy"), tmp_path, overwrite=True)
    assert result.report["link_method"] == "copy" and not os.path.samefile(src, dst)
    assert src.read_bytes() == dst.read_bytes()


def test_hardlink_failure_falls_back_to_copy_with_warning(tmp_path, monkeypatch):
    tree = raw(tmp_path)
    tree.add("India", 0, [Obj("D40", BOX)])
    add_val_test_positives(tree)

    def no_link(src, dst):
        raise OSError(18, "cross-device")

    monkeypatch.setattr(conv.os, "link", no_link)
    result = run(make_cfg(tmp_path, link_mode="hardlink"), tmp_path)
    assert result.report["link_method"] == "copy"
    assert any("hardlink failed for 3 file(s)" in w and "errno 18" in w for w in result.report["warnings"])
    assert (out_dir(tmp_path) / "images" / "train" / "India_000000.jpg").is_file()


# ------------------------------------------------------------------ CLI and training integration


def test_cli_scan_only_and_error_codes(tmp_path, capsys):
    mixed_tree(tmp_path / "data" / "raw")
    make_cfg(tmp_path)
    cfg_file = tmp_path / "dataset.yaml.cfg"
    assert conv.main(["--config", str(cfg_file), "--scan-only"], root=tmp_path) == 0
    assert "India" not in capsys.readouterr().err
    assert (tmp_path / "data" / "processed" / "ds_scan" / "manifest.csv").is_file()
    assert conv.main(["--config", str(cfg_file)], root=tmp_path) == 0
    assert conv.main(["--config", str(cfg_file)], root=tmp_path) == 2  # non-empty out_dir
    assert "--overwrite" in capsys.readouterr().err
    assert conv.main(["--config", str(cfg_file), "--overwrite"], root=tmp_path) == 0
    assert conv.main(["--config", str(tmp_path / "missing.cfg")], root=tmp_path) == 2


def test_converted_dataset_passes_training_validation_and_dry_run(tmp_path, write_yaml):
    mixed_tree(tmp_path / "data" / "raw")
    assert run(make_cfg(tmp_path), tmp_path).exit_code == 0
    dataset_yaml = out_dir(tmp_path) / "dataset.yaml"
    ds = train.load_dataset_yaml(dataset_yaml)  # validator incl. non-empty train/val dirs
    assert ds["names"] == {0: "pothole"} and "path" not in ds
    cfg_path = write_yaml(
        "train.yaml",
        {"data": str(dataset_yaml), "project": str(tmp_path / "runs"), "device": "cpu",
         "imgsz": 320, "batch": 2, "epochs": 1, "patience": 1},
    )  # fmt: skip
    assert "ultralytics" not in sys.modules
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 0
    assert "ultralytics" not in sys.modules
    run_dir = tmp_path / "runs" / "baseline"
    record = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert record["dataset_manifest_sha256"] == hashlib.sha256((out_dir(tmp_path) / "manifest.csv").read_bytes()).hexdigest()
    resolved = yaml.safe_load((run_dir / "dataset_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["path"] == out_dir(tmp_path).resolve().as_posix()
    assert (Path(resolved["path"]) / resolved["train"]).is_dir()


# ------------------------------------------------------------------ DatasetConfig


def test_shipped_dataset_config_loads():
    cfg = load_dataset_config(project_root() / "configs" / "dataset_rdd2022.yaml")
    assert cfg.countries == ["India", "Japan", "Czech", "United_States", "China_MotorBike"]
    assert cfg.pothole_labels == ["D40"]
    assert cfg.split.strategy == "similarity" and cfg.split.max_hash_distance_bits == 8
    assert cfg.split.block_size == 200 and cfg.split.seed == 42
    assert (cfg.split.fractions.train, cfg.split.fractions.val, cfg.split.fractions.test) == (0.8, 0.1, 0.1)
    assert cfg.train_negative_ratio == 2.0 and cfg.min_box_side_px == 2 and cfg.link_mode == "hardlink"
    assert cfg.raw_dir == project_root() / "data" / "raw" / "rdd2022"
    assert cfg.out_dir == project_root() / "data" / "processed" / "rdd2022_pothole"


BASE = {
    "countries": ["India", "Japan", "Czech"],
    "negative_countries": ["India"],
}


def ds_cfg(tmp_path, **changes):
    data = {**BASE, **changes}
    path = tmp_path / "ds.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_dataset_config(path, root=tmp_path)


def test_dataset_config_defaults_and_relative_paths(tmp_path):
    cfg = ds_cfg(tmp_path, raw_dir="x/raw")
    assert cfg.raw_dir == tmp_path / "x" / "raw"
    assert cfg.out_dir == tmp_path / "data" / "processed" / "rdd2022_pothole"
    assert cfg.pothole_labels == ["D40"] and cfg.train_negative_ratio == 2.0


def test_dataset_config_null_ratio_allowed(tmp_path):
    assert ds_cfg(tmp_path, train_negative_ratio=None).train_negative_ratio is None


COUNTRY_SPLIT = {"strategy": "country", "val_countries": ["Japan"], "test_countries": ["Czech"]}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"countries": []}, "countries"),
        ({"countries": ["India", "India"]}, "duplicates"),
        ({"countries": ["../India"]}, "countries"),
        ({"negative_countries": ["India", "Norway"]}, "subset of countries"),
        ({"pothole_labels": []}, "pothole_labels"),
        ({"pothole_labels": [" D40"]}, "whitespace"),
        ({"bogus": 1}, "bogus"),
        ({"link_mode": "symlink"}, "link_mode"),
        ({"min_box_side_px": -1}, "min_box_side_px"),
        ({"train_negative_ratio": -0.5}, "train_negative_ratio"),
        ({"split": {"fractions": {"train": 0.5, "val": 0.1, "test": 0.1}}}, "sum to 1"),
        ({"split": {"fractions": {"train": 0.0, "val": 0.5, "test": 0.5}}}, "train fraction"),
        ({"split": {"fractions": {"train": 1.2, "val": -0.1, "test": -0.1}}}, "fractions"),
        ({"split": {"block_size": 0}}, "block_size"),
        ({"split": {"strategy": "similarity", "max_hash_distance_bits": -1}}, "max_hash_distance_bits"),
        ({"split": {"strategy": "similarity", "max_hash_distance_bits": 17}}, "max_hash_distance_bits"),
        ({"split": {"max_hash_distance_bits": 8.0}}, "max_hash_distance_bits"),
        ({"split": {"max_hash_distance_bits": "8"}}, "max_hash_distance_bits"),
        ({"split": {"strategy": "random"}}, "strategy"),
        ({"split": {"seed": -1}}, "seed"),
        ({"split": {"cutoff": 1}}, "cutoff"),
        ({"split": {**COUNTRY_SPLIT, "val_countries": []}}, "val_countries must not be empty"),
        ({"split": {**COUNTRY_SPLIT, "val_countries": ["Japan"], "test_countries": ["Japan"]}}, "disjoint"),
        ({"split": {**COUNTRY_SPLIT, "val_countries": ["Norway"]}}, "subset of countries"),
        ({"split": {**COUNTRY_SPLIT, "val_countries": ["India", "Japan"], "test_countries": ["Czech"]}}, "left for the train"),
    ],
)
def test_dataset_config_rejects_bad_files(tmp_path, changes, message):
    with pytest.raises(ConfigError, match=message):
        ds_cfg(tmp_path, **changes)


def test_dataset_config_country_strategy_valid(tmp_path):
    cfg = ds_cfg(tmp_path, split=COUNTRY_SPLIT)
    assert cfg.split.val_countries == ["Japan"]


def test_dataset_config_relative_path_escape_rejected(tmp_path):
    with pytest.raises(ConfigError, match="escapes the project root"):
        ds_cfg(tmp_path, out_dir="../elsewhere")


def test_cap_choice_matches_documented_rule():
    ids = ["a", "b", "c", "d"]
    expect = sorted(ids, key=lambda s: hashlib.sha256(f"7:{s}".encode()).hexdigest())[:2]
    assert splits.select_capped_negatives(ids, 2, 7) == set(expect)


# ------------------------------------------------------------------ dhash64


def reference_dhash(im):
    """The recipe from the task brief, verbatim; the audit script reproduces this."""
    import numpy as np

    g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    v = 0
    for b in (g[:, 1:] > g[:, :-1]).flatten():
        v = (v << 1) | int(b)
    return v


def gray_9x8(rows):
    import numpy as np

    return Image.fromarray(np.array(rows, dtype=np.uint8))


def test_dhash64_hand_computed_gradients():
    increasing = [list(range(0, 90, 10))] * 8
    assert dhash64(gray_9x8(increasing)) == (1 << 64) - 1  # every comparison true
    assert dhash64(gray_9x8([r[::-1] for r in increasing])) == 0  # strictly decreasing
    assert dhash64(gray_9x8([[7] * 9] * 8)) == 0  # equal neighbours are not "greater"
    rows = [
        list(range(0, 90, 10)),  # 11111111 -> 0xFF
        [0, 1, 0, 1, 0, 1, 0, 1, 0],  # 10101010 -> 0xAA
        [5] * 9,  # 00000000
        [0, 5, 5, 5, 5, 5, 5, 5, 5],  # 10000000 -> 0x80 (first comparison is the MSB of its byte)
        [5, 5, 5, 5, 5, 5, 5, 5, 9],  # 00000001 -> 0x01
        list(range(80, -1, -10)),  # 0
        list(range(80, -1, -10)),
        list(range(80, -1, -10)),
    ]
    assert dhash64(gray_9x8(rows)) == 0xFFAA008001000000  # first row = most significant byte
    assert dhash64(gray_9x8(rows[::-1])) != dhash64(gray_9x8(rows))  # row order matters


def test_dhash64_identical_pixels_same_hash_and_fits_64_bits():
    a, b = noise_pixels(3), noise_pixels(3)
    ha = dhash64(Image.fromarray(a))
    assert ha == dhash64(Image.fromarray(b))
    for seed in range(20):
        h = dhash64(Image.fromarray(noise_pixels(seed)))
        assert 0 <= h < 1 << 64
    assert dhash64(Image.new("RGB", (100, 50), (9, 9, 9))) == 0


@pytest.mark.parametrize("size", [(100, 50), (37, 91), (9, 8), (640, 480)])
def test_dhash64_matches_reference_recipe(size):
    rng = random.Random(size[0])
    im = Image.fromarray(noise_pixels(rng.randrange(1000), size))
    assert dhash64(im) == reference_dhash(im)
    assert dhash64(im.convert("L")) == reference_dhash(im)
    assert dhash64(im.convert("P")) == reference_dhash(im.convert("P"))


def test_inspect_image_hash_uses_stored_pixels_ignoring_exif_orientation(tmp_path):
    from PIL import ImageOps

    from roadsense.data.rdd2022 import inspect_image

    pixels = noise_pixels(3)
    path = tmp_path / "x.jpg"
    exif = Image.Exif()
    exif[274] = 6  # would rotate the picture if applied
    Image.fromarray(pixels).save(path, "JPEG", quality=95, exif=exif)
    info = inspect_image(path)
    assert (info.width, info.height, info.exif_orientation) == (100, 50, 6)
    with Image.open(path) as im:
        im.load()
        assert info.dhash == reference_dhash(im)
        assert info.dhash != reference_dhash(ImageOps.exif_transpose(im))


# ------------------------------------------------------------------ similarity strategy

THRESH = 8


def sim_cfg(tmp_path, seed=42, countries=("India",), **kw):
    return make_cfg(
        tmp_path,
        countries=list(countries),
        negative_countries=list(countries),
        split={"strategy": "similarity", "seed": seed, "max_hash_distance_bits": THRESH},
        **kw,
    )


def sim_tree(raw_dir, reverse=False, negatives=False, extras=False):
    """India with planted near-duplicates and shuffled ids (ids carry no capture order).

    40 distinct base images; bases 0..11 also get a brightness variant (pairs); bases 0..3
    a 1 px shifted variant (chains of three); 3 more images exist twice as identical files.
    Returns (tree, groups) where groups lists the source ids that must stay together.
    """
    tree = RawTree(raw_dir)
    rng = random.Random(7)
    ids = list(range(1000, 1000 + 40 + 12 + 4 + 6))
    rng.shuffle(ids)
    take = iter(ids)
    specs = []  # (num, pixels, is_negative)
    groups: list[list[int]] = []
    for b in range(40):
        base = noise_pixels(b)
        neg = negatives and b % 2 == 1
        members = [next(take)]
        specs.append((members[0], base, neg))
        if b < 12:
            members.append(next(take))
            specs.append((members[-1], variant_pixels(base, brightness=3), neg))
        if b < 4:
            members.append(next(take))
            specs.append((members[-1], variant_pixels(base, shift=1), neg))
        groups.append(members)
    for k in range(3):  # exact duplicate files under different ids
        dup = noise_pixels(100 + k)
        members = [next(take), next(take)]
        specs += [(m, dup, False) for m in members]
        groups.append(members)
    for num, pixels, neg in reversed(specs) if reverse else specs:
        tree.add("India", num, [] if neg else [Obj("D40", BOX)], pixels=pixels)
    if extras:
        good = jpeg_bytes()
        tree.add("India", 5000, [Obj("D40", BOX)], image_bytes=good[:50])  # corrupt: no hash
        tree.add("India", 5001, [Obj("D40", BOX)], image=False)  # no image: no hash
        tree.add("India", 5002, [Obj("D40", BOX)], xml=False, pixels=noise_pixels(77))
    named = [[tree.source_id("India", m) for m in g] for g in groups]
    return tree, named


def hash_of(raw_dir, sid):
    return dhash_of_file(raw_dir / "India" / "train" / "images" / f"{sid}.jpg")


def scan_manifest(tmp_path):
    return read_manifest(tmp_path / "data" / "processed" / "ds_scan" / "manifest.csv")


def test_similarity_fixture_pairs_are_really_near_and_far(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    _, groups = sim_tree(raw_dir)
    hashes = {sid: hash_of(raw_dir, sid) for g in groups for sid in g}
    for g in groups:
        for sid in g[1:]:  # every variant is within the threshold of its base ...
            assert hamming(hashes[g[0]], hashes[sid]) <= THRESH, (g[0], sid)
    for i, g in enumerate(groups):  # ... and no member of a group is near any other group
        for h in groups[i + 1 :]:
            for a in g:
                for b in h:
                    assert hamming(hashes[a], hashes[b]) > THRESH, (a, b)
    dups = groups[-3:]
    assert all(hashes[g[0]] == hashes[g[1]] for g in dups)  # identical files
    assert all(
        (raw_dir / "India" / "train" / "images" / f"{g[0]}.jpg").read_bytes()
        == (raw_dir / "India" / "train" / "images" / f"{g[1]}.jpg").read_bytes()
        for g in dups
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5, 8, 42, 123])
def test_similarity_keeps_planted_near_duplicates_in_one_split_for_any_seed(tmp_path, seed):
    raw_dir = tmp_path / "data" / "raw"
    _, groups = sim_tree(raw_dir)
    result = run(sim_cfg(tmp_path, seed=seed), tmp_path, scan_only=True)
    assert result.exit_code == 0
    rows = scan_manifest(tmp_path)
    by_id = {r["source_id"]: r for r in rows}
    for g in groups:
        assert len({by_id[s]["split"] for s in g}) == 1, g
        assert len({by_id[s]["group_id"] for s in g}) == 1
    # components are exactly the planted groups (all other pairs are far apart)
    assert len({r["group_id"] for r in rows}) == len(groups)
    for g in groups:
        smallest = min(int(by_id[s]["numeric_id"]) for s in g)
        assert by_id[g[0]]["group_id"] == f"India:c{smallest:06d}"
    # the split is not vacuous
    counts = Counter(r["split"] for r in rows)
    assert counts["val"] > 0 and counts["test"] > 0 and counts["train"] > counts["val"]
    # independent recomputation from the manifest columns: nothing near across splits
    hashes = [(int(r["dhash"], 16), r["split"]) for r in rows]
    for i, (hi, si) in enumerate(hashes):
        for hj, sj in hashes[i + 1 :]:
            if si != sj:
                assert hamming(hi, hj) > THRESH


def test_similarity_dhash_column_matches_recomputation(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    sim_tree(raw_dir)
    run(sim_cfg(tmp_path), tmp_path, scan_only=True)
    for r in scan_manifest(tmp_path):
        assert re.fullmatch(r"[0-9a-f]{16}", r["dhash"])
        assert int(r["dhash"], 16) == hash_of(raw_dir, r["source_id"])


def test_similarity_self_check_fires_when_components_are_wrong(tmp_path, monkeypatch):
    sim_tree(tmp_path / "data" / "raw")
    cfg = sim_cfg(tmp_path)
    run(cfg, tmp_path, scan_only=True)  # fine with the real union-find
    monkeypatch.setattr(splits, "similarity_components", lambda hashes, max_bits: list(range(len(hashes))))
    with pytest.raises(conv.LeakageError, match=r"near-duplicate leakage: India_\d{6} \(\w+\) and India_\d{6}"):
        run(cfg, tmp_path, scan_only=True)
    with pytest.raises(conv.LeakageError):
        run(cfg, tmp_path)  # nothing is written when it fires
    assert not out_dir(tmp_path).exists()


def test_check_leakage_near_duplicate_pair_is_reported():
    def row(num, split, dhash):
        return conv.Row(f"India_{num:06d}", "India", num, group_id=f"India:c{num:06d}", split=split, dhash=dhash)

    rows = [row(0, "train", "0" * 16), row(1, "val", "0000000000000007"), row(2, "test", "f" * 16)]
    conv.check_leakage(rows)  # without the threshold only ids and groups are checked
    conv.check_leakage(rows, 2)  # 3 bits apart is beyond 2
    with pytest.raises(conv.LeakageError, match=r"India_000000 \(train\) and India_000001 \(val\).*3 bit"):
        conv.check_leakage(rows, 3)
    rows[1].split = "train"
    conv.check_leakage(rows, 3)
    rows[1].dhash = ""  # unhashed items are never matched
    rows[1].split = "val"
    conv.check_leakage(rows, 16)


def test_similarity_exact_duplicates_share_a_split_and_report_is_empty(tmp_path):
    _, groups = sim_tree(tmp_path / "data" / "raw")
    for seed in (1, 2, 3):
        result = run(sim_cfg(tmp_path, seed=seed), tmp_path, overwrite=True)
        assert result.report["duplicate_images_across_splits"] == []
        assert not any("exact-duplicate" in w for w in result.report["warnings"])
        rows = manifest_rows(tmp_path)
        for first, second in groups[-3:]:
            assert rows[first]["split"] == rows[second]["split"]
            assert rows[first]["image_sha256"] == rows[second]["image_sha256"] != ""


def test_similarity_negative_cap_keeps_group_split_and_hash_on_excluded_rows(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    sim_tree(raw_dir, negatives=True)
    capped = run(sim_cfg(tmp_path, train_negative_ratio=0.3), tmp_path, scan_only=True)
    cap_rows = scan_manifest(tmp_path)
    run(sim_cfg(tmp_path, train_negative_ratio=None), tmp_path, scan_only=True)
    free_rows = scan_manifest(tmp_path)
    excluded = [r for r in cap_rows if r["status"] == "excluded:negative_cap"]
    assert excluded and capped.report["status_counts"]["excluded:negative_cap"] == len(excluded)
    for r in excluded:
        assert r["split"] == "train" and r["group_id"].startswith("India:c")
        assert re.fullmatch(r"[0-9a-f]{16}", r["dhash"]) and r["image_sha256"] == ""
        assert int(r["dhash"], 16) == hash_of(raw_dir, r["source_id"])
    # the cap runs after the split: the assignment does not depend on it
    assert [(r["source_id"], r["group_id"], r["split"], r["dhash"]) for r in cap_rows] == [
        (r["source_id"], r["group_id"], r["split"], r["dhash"]) for r in free_rows
    ]


def test_similarity_unhashable_items_are_singletons_and_counted(tmp_path):
    sim_tree(tmp_path / "data" / "raw", extras=True)
    result = run(sim_cfg(tmp_path), tmp_path, scan_only=True)
    rows = {r["source_id"]: r for r in scan_manifest(tmp_path)}
    assert rows["India_005000"]["dhash"] == "" and rows["India_005000"]["status"] == "excluded:corrupt_image"
    assert rows["India_005001"]["dhash"] == "" and rows["India_005001"]["status"] == "excluded:missing_image"
    assert rows["India_005002"]["status"] == "excluded:missing_annotation"
    assert rows["India_005002"]["dhash"] != ""
    for sid in ("India_005000", "India_005001", "India_005002"):
        assert rows[sid]["group_id"] == f"India:c{int(sid[-6:]):06d}"  # each is its own component
        assert rows[sid]["split"] in SPLITS
    assert result.report["near_duplicate_grouping"]["by_country"]["India"]["items_without_hash"] == 2


def test_similarity_two_runs_and_creation_orders_are_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    sim_tree(a / "data" / "raw", reverse=False, negatives=True, extras=True)
    sim_tree(b / "data" / "raw", reverse=True, negatives=True, extras=True)
    ra = run(sim_cfg(a, train_negative_ratio=0.5), a)
    rb = run(sim_cfg(b, train_negative_ratio=0.5), b)
    snap_a = snapshot(a / "data" / "processed" / "ds")
    snap_b = snapshot(b / "data" / "processed" / "ds")
    assert snap_a.keys() == snap_b.keys()
    for name in ("manifest.csv", "dataset.yaml", "conversion_report.json"):
        assert name in snap_a
    for name in snap_a:
        assert snap_a[name] == snap_b[name], name
    assert ra.report == rb.report
    run(sim_cfg(a, train_negative_ratio=0.5), a, overwrite=True)
    assert snapshot(a / "data" / "processed" / "ds") == snap_a
    assert str(tmp_path) not in snap_a["conversion_report.json"].decode("utf-8")


def test_similarity_report_matches_manifest(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    sim_tree(raw_dir, extras=True)
    result = run(sim_cfg(tmp_path), tmp_path)
    rows = read_manifest(out_dir(tmp_path) / "manifest.csv")
    grouping = json.loads((out_dir(tmp_path) / "conversion_report.json").read_text(encoding="utf-8"))[
        "near_duplicate_grouping"
    ]
    assert grouping == result.report["near_duplicate_grouping"]
    assert grouping["strategy"] == "similarity" and grouping["max_hash_distance_bits"] == THRESH
    assert grouping["hash_recipe"] == "dhash64: L, 9x8 bilinear, row-major, msb first"
    assert grouping["pillow_version"] == PIL.__version__
    assert list(grouping["by_country"]) == ["India"]
    g = grouping["by_country"]["India"]
    n = len(rows)
    sizes = Counter(r["group_id"] for r in rows)
    split_of = {r["group_id"]: r["split"] for r in rows}
    assert g["labelled_items"] == n
    assert g["hashed_items"] == sum(1 for r in rows if r["dhash"]) and g["items_without_hash"] == 2
    assert g["hashed_items"] + g["items_without_hash"] == n
    assert g["components"] == len(sizes)
    assert g["singletons"] == sum(1 for v in sizes.values() if v == 1)
    expected_largest = sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    assert g["largest_components"] == [
        {"group_id": gid, "size": size, "split": split_of[gid]} for gid, size in expected_largest
    ]
    assert len(g["largest_components"]) == 5
    for split in ("val", "test"):
        assert g[split]["target_images"] == int(n * 0.1 + 0.5)
        assert g[split]["achieved_images"] == sum(1 for r in rows if r["split"] == split)
        assert g[split]["achieved_images"] <= g[split]["target_images"]
    # the existing included-image counts are unchanged in meaning
    for split in SPLITS:
        included = [r for r in rows if r["status"] == "included" and r["split"] == split]
        assert result.report["by_country_split"]["India"][split]["images"] == len(included)
        assert result.report["by_country_split"]["India"][split]["positives"] == sum(
            r["is_positive"] == "1" for r in included
        )


def test_similarity_summary_prints_strategy_threshold_and_country_lines(tmp_path):
    sim_tree(tmp_path / "data" / "raw")
    lines: list[str] = []
    result = conv.convert(sim_cfg(tmp_path), root=tmp_path, out=lines.append)
    text = "\n".join(lines)
    g = result.report["near_duplicate_grouping"]["by_country"]["India"]
    top = g["largest_components"][0]
    assert "split strategy: similarity" in text
    assert f"max hash distance: {THRESH} bit(s)" in text
    assert (
        f"India: val {g['val']['achieved_images']}/{g['val']['target_images']}, "
        f"test {g['test']['achieved_images']}/{g['test']['target_images']}, "
        f"largest component {top['size']} -> {top['split']}"
    ) in text


def test_similarity_with_several_countries_groups_each_country_separately(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    tree = RawTree(raw_dir)
    same = noise_pixels(1)
    for country in ("India", "Japan", "Czech"):  # identical image in every country
        for n in range(12):
            tree.add(country, n, [Obj("D40", BOX)], pixels=noise_pixels(10 * n + len(country)))
        tree.add(country, 99, [Obj("D40", BOX)], pixels=same)
    cfg = sim_cfg(tmp_path, countries=("India", "Japan", "Czech"))
    result = run(cfg, tmp_path)
    rows = manifest_rows(tmp_path)
    assert result.exit_code == 0
    # groups never span countries, and each country has its own val/test share
    assert all(r["group_id"].startswith(r["country"] + ":c") for r in rows.values())
    for country in ("India", "Japan", "Czech"):
        n = sum(r["country"] == country for r in rows.values())
        assert n == 13
        assert Counter(r["split"] for r in rows.values() if r["country"] == country)["val"] == 1


def test_similarity_threshold_is_used(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    tree = RawTree(raw_dir)
    for n in range(30):
        tree.add("India", n, [Obj("D40", BOX)], pixels=noise_pixels(n))
    tree.add("India", 100, [Obj("D40", BOX)], pixels=variant_pixels(noise_pixels(0), shift=1))
    path = raw_dir / "India" / "train" / "images"
    distance = hamming(dhash_of_file(path / "India_000000.jpg"), dhash_of_file(path / "India_000100.jpg"))
    assert 0 < distance <= THRESH

    def groups_of(bits):
        cfg = make_cfg(
            tmp_path, countries=["India"], negative_countries=["India"],
            split={"strategy": "similarity", "max_hash_distance_bits": bits},
        )  # fmt: skip
        run(cfg, tmp_path, scan_only=True)
        by = {r["source_id"]: r["group_id"] for r in scan_manifest(tmp_path)}
        return by["India_000000"], by["India_000100"]

    a, b = groups_of(THRESH)
    assert a == b
    a, b = groups_of(distance - 1)  # one bit too strict: the pair is no longer linked
    assert a != b


@pytest.mark.parametrize("strategy", ["blocks", "country"])
def test_report_names_strategy_only_when_not_similarity(tmp_path, strategy):
    tree = raw(tmp_path)
    for country in ("India", "Japan", "Czech"):
        for n in range(6):
            tree.add(country, n, [Obj("D40", BOX)])
    split = {"strategy": strategy, "block_size": 2}
    result = run(make_cfg(tmp_path, split=split), tmp_path)
    assert result.report["near_duplicate_grouping"] == {
        "strategy": strategy,
        "max_hash_distance_bits": None,
        "hash_recipe": None,
        "pillow_version": None,
        "by_country": {},
    }
    rows = manifest_rows(tmp_path)
    assert all(re.fullmatch(r"[0-9a-f]{16}", r["dhash"]) for r in rows.values())  # hashed anyway


def test_dataset_config_similarity_valid_and_bits_default(tmp_path):
    cfg = ds_cfg(tmp_path, split={"strategy": "similarity"})
    assert cfg.split.strategy == "similarity" and cfg.split.max_hash_distance_bits == 8
    for bits in (0, 16):
        assert ds_cfg(tmp_path, split={"strategy": "similarity", "max_hash_distance_bits": bits}).split.max_hash_distance_bits == bits
    assert ds_cfg(tmp_path).split.strategy == "blocks"  # the model default is unchanged
