"""Independent verification of a converted RDD2022 dataset. Read-only.

Does not import the converter: it re-derives everything from the raw XML files, the output tree and
the manifest, so a converter bug shows up as a mismatch.

Usage:
    python scripts/verify_converted_dataset.py <out_dir> <raw_dir> <report.json>
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

SPLITS = ("train", "val", "test")
POTHOLE = "D40"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    out_dir, raw_dir, result_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    checks: dict[str, object] = {}
    failures: list[str] = []

    def check(name: str, ok: bool, detail: object = None) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            failures.append(name)

    rows = list(csv.DictReader((out_dir / "manifest.csv").open(encoding="utf-8", newline="")))
    report = json.loads((out_dir / "conversion_report.json").read_text(encoding="utf-8"))
    included = [r for r in rows if r["status"] == "included"]

    # 1. manifest integrity
    ids = [r["source_id"] for r in rows]
    check("manifest_one_row_per_source_id", len(ids) == len(set(ids)), len(ids))
    check("manifest_sha256_matches_report", sha256(out_dir / "manifest.csv") == report["manifest_sha256"])
    check("status_counts_match_report", dict(Counter(r["status"] for r in rows)) ==
          {k: v for k, v in report["status_counts"].items() if v}, dict(Counter(r["status"] for r in rows)))

    # 2. files on disk vs manifest
    disk = {}
    for split in SPLITS:
        imgs = {p.stem: p for p in (out_dir / "images" / split).iterdir() if p.is_file()}
        labs = {p.stem: p for p in (out_dir / "labels" / split).iterdir() if p.suffix == ".txt"}
        disk[split] = (imgs, labs)
        want = {r["source_id"] for r in included if r["split"] == split}
        check("%s_image_label_stems_identical" % split, set(imgs) == set(labs), [len(imgs), len(labs)])
        check("%s_files_equal_manifest_included" % split, set(imgs) == want, [len(imgs), len(want)])
    all_stems = [s for split in SPLITS for s in disk[split][0]]
    check("no_source_id_in_two_splits", len(all_stems) == len(set(all_stems)), len(all_stems))
    groups = defaultdict(set)
    for r in rows:
        groups[r["group_id"]].add(r["split"])
    check("no_group_in_two_splits", all(len(s) == 1 for s in groups.values()), len(groups))

    # 3. only labelled training images were used (never the unlabelled official test images)
    official_test = set()
    for country_dir in raw_dir.iterdir():
        t = country_dir / "test" / "images"
        if t.is_dir():
            official_test |= {p.stem for p in t.iterdir()}
    check("no_official_test_image_used", not (set(all_stems) & official_test), len(official_test))
    check("all_sources_under_train", all("/train/" in r["image_relpath"] for r in rows if r["image_relpath"]))

    # 4. labels re-derived independently from the raw XML and the real image size
    n_pos = n_neg = n_boxes = 0
    mismatches = []
    bad_lines = []
    by_split = {s: Counter() for s in SPLITS}
    for r in included:
        split = r["split"]
        lines = [ln for ln in disk[split][1][r["source_id"]].read_text(encoding="utf-8").splitlines() if ln.strip()]
        root = ET.parse(raw_dir / r["xml_relpath"]).getroot()
        with Image.open(raw_dir / r["image_relpath"]) as im:
            W, H = im.size
        expected = []
        for obj in root.findall("object"):
            if (obj.findtext("name") or "").strip() != POTHOLE:
                continue
            bb = obj.find("bndbox")
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, x2 = max(0.0, x1), min(float(W), x2)
            y1, y2 = max(0.0, y1), min(float(H), y2)
            expected.append(((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
        got = []
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5 or parts[0] != "0":
                bad_lines.append([r["source_id"], ln])
                continue
            vals = tuple(float(v) for v in parts[1:])
            if not all(0.0 <= v <= 1.0 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                bad_lines.append([r["source_id"], ln])
            got.append(vals)
        if len(got) != len(expected) or any(abs(a - b) > 1e-6 for g, e in zip(sorted(got), sorted(expected)) for a, b in zip(g, e)):
            mismatches.append(r["source_id"])
        if expected:
            n_pos += 1
            by_split[split]["positives"] += 1
        else:
            n_neg += 1
            by_split[split]["negatives"] += 1
        n_boxes += len(expected)
        by_split[split]["boxes"] += len(expected)
        by_split[split]["images"] += 1
        if (r["is_positive"] == "1") != bool(expected) or int(r["n_pothole_boxes"] or 0) != len(expected):
            mismatches.append(r["source_id"] + ":manifest")
    check("labels_match_independent_recomputation", not mismatches, mismatches[:10])
    check("label_lines_well_formed", not bad_lines, bad_lines[:10])
    rep_split = {s: {"images": v["images"], "positives": v["positives"], "boxes": v["pothole_boxes"],
                     "negatives": sum(v["negatives"].values())} for s, v in report["by_split"].items()}
    check("split_counts_match_report", {s: dict(c) for s, c in by_split.items()} == rep_split,
          {s: dict(c) for s, c in by_split.items()})

    # 5. negatives are empty label files; positives are non-empty
    empties = sum(1 for split in SPLITS for p in disk[split][1].values() if p.stat().st_size == 0)
    check("empty_label_files_equal_negatives", empties == n_neg, [empties, n_neg])

    # 6. output image bytes equal the raw source (hash of a random sample against the manifest)
    rng = random.Random(0)
    sample = rng.sample(included, min(600, len(included)))
    bad_hash = [r["source_id"] for r in sample
                if sha256(disk[r["split"]][0][r["source_id"]]) != r["image_sha256"]
                or sha256(raw_dir / r["image_relpath"]) != r["image_sha256"]]
    check("image_hashes_match_manifest_sample_%d" % len(sample), not bad_hash, bad_hash[:10])

    # 7. dataset.yaml
    yaml_text = (out_dir / "dataset.yaml").read_text(encoding="utf-8")
    check("dataset_yaml_single_class_pothole", "pothole" in yaml_text and "images/train" in yaml_text
          and "images/val" in yaml_text, yaml_text)

    totals = {"included": len(included), "positives": n_pos, "negatives": n_neg, "pothole_boxes": n_boxes,
              "by_split": {s: dict(c) for s, c in by_split.items()},
              "by_country_positive": dict(Counter(r["country"] for r in included if r["is_positive"] == "1"))}
    result = {"ok": not failures, "failures": failures, "totals": totals, "checks": checks}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"ok": result["ok"], "failures": failures, "totals": totals}, indent=1))
    for name, c in checks.items():
        print("  %-48s %s" % (name, "OK" if c["ok"] else "FAIL  " + json.dumps(c["detail"])[:300]))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
