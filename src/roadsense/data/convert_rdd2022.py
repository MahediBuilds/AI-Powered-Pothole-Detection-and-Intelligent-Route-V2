"""Convert RDD2022 into a single-class (pothole) Ultralytics dataset.

    python -m roadsense.data.convert_rdd2022 --config configs/dataset_rdd2022.yaml [--scan-only] [--overwrite]

Only ``<raw_dir>/<Country>/train/`` is read (the official test images are
unlabelled and only counted). Every labelled source item gets exactly one row in
``manifest.csv`` with a status; nothing is dropped silently. Output under
``out_dir``: ``images/<split>/``, ``labels/<split>/``, ``dataset.yaml``,
``manifest.csv`` and ``conversion_report.json``. ``--scan-only`` writes just the
manifest and report to ``<out_dir>_scan/`` so the numbers can be reviewed first.

Raw data is only ever read (or hard-linked); it is never modified.

Manifest conventions: ``split`` and ``group_id`` are filled for every row (the
block, near-duplicate component or country the item belongs to), including
excluded rows; counts of images, positives and negatives always refer to
``included`` rows. ``dhash`` (16 hex digits, perceptual hash of the stored pixels)
is filled for every row whose image can be decoded, whatever its status, while
``image_sha256`` is filled only for included rows. With ``split.strategy:
similarity`` (RDD2022 ids are not in capture order) items of one country whose
dhash values are within ``max_hash_distance_bits`` are grouped (single linkage) and
whole groups are assigned to a split by image-count quota. Negative
kinds: ``no_objects`` (no objects), ``crack_only`` (every object is D00/D10/D20),
``other_labels`` (any other non-pothole label present).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import PIL

from roadsense import __version__
from roadsense.config import ConfigError, DatasetConfig, load_dataset_config, project_root
from roadsense.data import splits
from roadsense.data.fetch_rdd2022 import DOI, OUTER_MD5, OUTER_NAME
from roadsense.data.rdd2022 import (
    CRACK_LABELS,
    DHASH_RECIPE,
    ConversionError,
    CorruptImage,
    MalformedAnnotation,
    RawItem,
    check_box,
    format_dhash,
    inspect_image,
    parse_annotation,
    scan_country,
    yolo_line,
)

STATUSES = (
    "included",
    "excluded:missing_image",
    "excluded:missing_annotation",
    "excluded:corrupt_image",
    "excluded:malformed_xml",
    "excluded:invalid_pothole_box",
    "excluded:size_mismatch",
    "excluded:negatives_not_enabled",
    "excluded:negative_cap",
)
NEGATIVE_KINDS = ("crack_only", "other_labels", "no_objects")
MANIFEST_COLUMNS = (
    "source_id", "country", "numeric_id", "group_id", "split", "status", "is_positive",
    "negative_kind", "n_pothole_boxes", "n_pothole_clipped", "n_other_objects", "labels_seen",
    "width", "height", "xml_width", "xml_height", "image_relpath", "xml_relpath", "image_sha256",
    "dhash",
)  # fmt: skip
DATASET_YAML = (
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    "  0: pothole\n"
)


class LeakageError(ConversionError):
    pass


@dataclass
class Row:
    source_id: str
    country: str
    numeric_id: int
    group_id: str = ""
    split: str = ""
    status: str = ""
    is_positive: bool | None = None
    negative_kind: str = ""
    n_pothole_boxes: int | None = None
    n_pothole_clipped: int | None = None
    n_other_objects: int | None = None
    labels_seen: str = ""
    width: int | None = None
    height: int | None = None
    xml_width: int | None = None
    xml_height: int | None = None
    image_relpath: str = ""
    xml_relpath: str = ""
    image_sha256: str = ""
    dhash: str = ""
    # Not in the manifest:
    detail: str = ""
    lines: list[str] = field(default_factory=list)
    image_path: Path | None = None


@dataclass
class Findings:
    label_counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    invalid_annotations: list[dict] = field(default_factory=list)
    size_mismatches: list[dict] = field(default_factory=list)
    clipped_boxes: list[dict] = field(default_factory=list)
    other_object_problems: Counter = field(default_factory=Counter)
    exif_oriented: list[str] = field(default_factory=list)
    unlabelled_test_images: dict[str, int] = field(default_factory=dict)


@dataclass
class Plan:
    rows: list[Row]
    findings: Findings
    warnings: list[str]


@dataclass
class Result:
    exit_code: int
    report: dict[str, Any]
    output_dir: Path


# --------------------------------------------------------------------------- per item


def _relpath(path: Path | None, raw_dir: Path) -> str:
    return "" if path is None else path.relative_to(raw_dir).as_posix()


def _hash_only(path: Path) -> str:
    """dhash of an image that is excluded before the normal inspection; '' if unreadable."""
    try:
        info = inspect_image(path)
    except CorruptImage:
        return ""
    return "" if info.dhash is None else format_dhash(info.dhash)


def process_item(item: RawItem, cfg: DatasetConfig, findings: Findings) -> Row:
    """Classify one labelled source item. Never raises for bad data; sets a status instead."""
    row = Row(item.source_id, item.country, item.numeric_id, image_path=item.image_path)
    row.image_relpath = _relpath(item.image_path, cfg.raw_dir)
    row.xml_relpath = _relpath(item.xml_path, cfg.raw_dir)
    if item.xml_path is None:
        row.status = "excluded:missing_annotation"
        row.detail = "no annotation xml for this image"
        if item.image_path is not None:
            row.dhash = _hash_only(item.image_path)
        return row

    ann = None
    parse_error = ""
    try:
        ann = parse_annotation(item.xml_path)
    except MalformedAnnotation as exc:
        parse_error = str(exc)
    pothole_labels = set(cfg.pothole_labels)
    if ann is not None:
        for obj in ann.objects:
            findings.label_counts[item.country][obj.label] += 1
        row.labels_seen = "|".join(sorted({o.label for o in ann.objects}))
        row.xml_width, row.xml_height = ann.xml_width, ann.xml_height
        row.n_other_objects = sum(1 for o in ann.objects if o.label not in pothole_labels)

    if item.image_path is None:
        row.status = "excluded:missing_image"
        row.detail = "no image for this annotation"
        return row
    if ann is None:
        row.status = "excluded:malformed_xml"
        row.detail = parse_error
        row.dhash = _hash_only(item.image_path)
        return row
    try:
        info = inspect_image(item.image_path)
    except CorruptImage as exc:
        row.status = "excluded:corrupt_image"
        row.detail = str(exc)
        return row
    row.width, row.height = info.width, info.height
    if info.dhash is not None:
        row.dhash = format_dhash(info.dhash)
    if info.exif_orientation not in (None, 1):
        findings.exif_oriented.append(item.source_id)

    pothole_objs = [o for o in ann.objects if o.label in pothole_labels]
    other_objs = [o for o in ann.objects if o.label not in pothole_labels]

    declared = (ann.xml_width, ann.xml_height)
    mismatch = None not in declared and declared != (info.width, info.height)
    if mismatch:
        findings.size_mismatches.append(
            {
                "source_id": item.source_id,
                "xml_size": list(declared),
                "real_size": [info.width, info.height],
                "n_pothole_objects": len(pothole_objs),
                "resolution": "excluded" if pothole_objs else "kept_as_negative_candidate",
            }
        )
        if pothole_objs:
            row.status = "excluded:size_mismatch"
            row.detail = (
                f"xml declares {declared[0]}x{declared[1]}, image is {info.width}x{info.height}"
            )
            return row

    for obj in other_objs:
        if not check_box(obj, info.width, info.height, cfg.min_box_side_px).valid:
            findings.other_object_problems[item.country] += 1

    invalid = 0
    clipped = 0
    for obj in pothole_objs:
        check = check_box(obj, info.width, info.height, cfg.min_box_side_px)
        if not check.valid:
            invalid += 1
            findings.invalid_annotations.append(
                {
                    "source_id": item.source_id,
                    "object_index": obj.index,
                    "label": obj.label,
                    "raw": dict(obj.raw),
                    "reason": check.reason,
                }
            )
            continue
        if check.clipped:
            clipped += 1
            findings.clipped_boxes.append(
                {
                    "source_id": item.source_id,
                    "object_index": obj.index,
                    "raw": dict(obj.raw),
                    "clipped_to": [round(v, 6) for v in check.box],  # type: ignore[union-attr]
                }
            )
        row.lines.append(yolo_line(check.box, info.width, info.height))  # type: ignore[arg-type]
    if invalid:
        row.lines = []
        row.status = "excluded:invalid_pothole_box"
        row.detail = f"{invalid} invalid pothole box(es)"
        return row

    if row.lines:
        row.is_positive = True
        row.n_pothole_boxes = len(row.lines)
        row.n_pothole_clipped = clipped
        row.status = "included"
        return row

    row.is_positive = False
    row.n_pothole_boxes = 0
    row.n_pothole_clipped = 0
    if not ann.objects:
        row.negative_kind = "no_objects"
    elif all(o.label in CRACK_LABELS for o in ann.objects):
        row.negative_kind = "crack_only"
    else:
        row.negative_kind = "other_labels"
    if item.country in cfg.negative_countries:
        row.status = "included"
    else:
        row.status = "excluded:negatives_not_enabled"
        row.detail = f"{item.country} is not in negative_countries"
    return row


# --------------------------------------------------------------------------- plan


def check_leakage(rows: list[Row], max_hash_distance_bits: int | None = None) -> None:
    """Raise LeakageError if a source_id or group_id appears in more than one split.

    With ``max_hash_distance_bits`` (the similarity strategy) it also verifies, per
    country and independently of the component labels, that no two items assigned to
    different splits have dhash values within that Hamming distance.
    """
    seen_ids: dict[str, str] = {}
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.source_id in seen_ids:
            raise LeakageError(f"source_id {row.source_id} appears more than once")
        seen_ids[row.source_id] = row.split
        group_splits[row.group_id].add(row.split)
    leaked = sorted(g for g, s in group_splits.items() if len(s) > 1)
    if leaked:
        raise LeakageError(f"group(s) span more than one split: {leaked}")
    if max_hash_distance_bits is None:
        return
    for country in sorted({r.country for r in rows}):
        members = [r for r in rows if r.country == country]
        pair = splits.find_cross_split_near_pair(
            [int(r.dhash, 16) if r.dhash else None for r in members],
            [r.split for r in members],
            max_hash_distance_bits,
        )
        if pair is not None:
            a, b, dist = members[pair[0]], members[pair[1]], pair[2]
            raise LeakageError(
                f"near-duplicate leakage: {a.source_id} ({a.split}) and {b.source_id} "
                f"({b.split}) differ by {dist} bit(s) of dhash, within the threshold of "
                f"{max_hash_distance_bits}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(cfg: DatasetConfig) -> Plan:
    """Scan, classify, split and cap. Reads the raw data; writes nothing."""
    findings = Findings()
    warnings: list[str] = []
    rows: list[Row] = []
    countries = sorted(cfg.countries)
    for country in countries:
        scan = scan_country(cfg.raw_dir, country)
        findings.unlabelled_test_images[country] = scan.n_unlabelled_test_images
        rows.extend(process_item(item, cfg, findings) for item in scan.items)

    # Split: every labelled item belongs to a group, so assignment does not
    # depend on which items were later excluded.
    sp = cfg.split
    if sp.strategy == "blocks":
        for country in countries:
            members = [r for r in rows if r.country == country]
            blocks = [splits.block_index(r.numeric_id, sp.block_size) for r in members]
            mapping, extra = splits.assign_block_splits(
                country, blocks, sp.seed, float(sp.fractions.val), float(sp.fractions.test)
            )
            warnings.extend(extra)
            for r, b in zip(members, blocks):
                r.group_id = splits.block_group_id(country, b)
                r.split = mapping[b]
    elif sp.strategy == "similarity":
        for country in countries:
            members = [r for r in rows if r.country == country]
            hashes = [int(r.dhash, 16) if r.dhash else None for r in members]
            labels = splits.similarity_components(hashes, sp.max_hash_distance_bits)
            keys: dict[int, tuple[int, str]] = {}
            for r, label in zip(members, labels):
                key = (r.numeric_id, r.source_id)
                if label not in keys or key < keys[label]:
                    keys[label] = key
            sizes = Counter(keys[label] for label in labels)
            mapping_s, extra = splits.assign_component_splits(
                country, sizes, sp.seed, float(sp.fractions.val), float(sp.fractions.test)
            )
            warnings.extend(extra)
            for r, label in zip(members, labels):
                key = keys[label]
                r.group_id = splits.similarity_group_id(country, key[0])
                r.split = mapping_s[key]
    else:
        mapping_c = splits.assign_country_splits(countries, sp.val_countries, sp.test_countries)
        for r in rows:
            r.group_id = r.country
            r.split = mapping_c[r.country]

    if cfg.train_negative_ratio is not None:
        n_pos = sum(1 for r in rows if r.status == "included" and r.split == "train" and r.is_positive)
        negatives = [
            r for r in rows if r.status == "included" and r.split == "train" and not r.is_positive
        ]
        keep = splits.select_capped_negatives(
            [r.source_id for r in negatives],
            splits.negative_cap(n_pos, float(cfg.train_negative_ratio)),
            sp.seed,
        )
        for r in negatives:
            if r.source_id not in keep:
                r.status = "excluded:negative_cap"
                r.detail = f"train negatives capped at {cfg.train_negative_ratio} x positives"

    rows.sort(key=lambda r: (r.country, r.numeric_id, r.source_id))
    check_leakage(rows, sp.max_hash_distance_bits if sp.strategy == "similarity" else None)

    for r in rows:
        if r.status == "included" and r.image_path is not None:
            r.image_sha256 = _sha256_file(r.image_path)

    included = [r for r in rows if r.status == "included"]
    pothole_label_set = set(cfg.pothole_labels)
    for country in countries:
        n_instances = sum(
            n for label, n in findings.label_counts[country].items() if label in pothole_label_set
        )
        if n_instances == 0:
            warnings.append(
                f"{country}: no pothole boxes found; pothole class may not be annotated "
                "in this subset; its negatives are unreliable"
            )
    for split in splits.SPLITS:
        if not any(r.split == split and r.is_positive for r in included):
            warnings.append(f"{split} split has zero positive images")
    if findings.exif_oriented:
        warnings.append(
            f"{len(findings.exif_oriented)} image(s) carry an EXIF orientation other than 1; "
            "box coordinates were normalised by the stored pixel size"
        )
    return Plan(rows, findings, warnings)


# --------------------------------------------------------------------------- report


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def manifest_bytes(rows: list[Row]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(MANIFEST_COLUMNS)
    for r in rows:
        writer.writerow([_cell(getattr(r, col)) for col in MANIFEST_COLUMNS])
    return buf.getvalue().encode("utf-8")


def _summary(rows: list[Row]) -> dict[str, Any]:
    included = [r for r in rows if r.status == "included"]
    return {
        "images": len(included),
        "positives": sum(1 for r in included if r.is_positive),
        "negatives": {k: sum(1 for r in included if r.negative_kind == k) for k in NEGATIVE_KINDS},
        "pothole_boxes": sum(r.n_pothole_boxes or 0 for r in included),
    }


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _grouping_report(cfg: DatasetConfig, rows: list[Row]) -> dict[str, Any]:
    """The ``near_duplicate_grouping`` section; only the strategy is filled unless 'similarity'."""
    sp = cfg.split
    if sp.strategy != "similarity":
        return {
            "strategy": sp.strategy,
            "max_hash_distance_bits": None,
            "hash_recipe": None,
            "pillow_version": None,
            "by_country": {},
        }
    by_country: dict[str, Any] = {}
    for country in sorted(cfg.countries):
        members = [r for r in rows if r.country == country]
        groups: dict[str, list[Row]] = defaultdict(list)
        for r in members:
            groups[r.group_id].append(r)
        largest = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:5]
        val_target, test_target = splits.component_targets(
            len(members), float(sp.fractions.val), float(sp.fractions.test)
        )
        hashed = sum(1 for r in members if r.dhash)
        by_country[country] = {
            "labelled_items": len(members),
            "hashed_items": hashed,
            "items_without_hash": len(members) - hashed,
            "components": len(groups),
            "singletons": sum(1 for g in groups.values() if len(g) == 1),
            "largest_components": [
                {"group_id": gid, "size": len(g), "split": g[0].split} for gid, g in largest
            ],
            "val": {
                "target_images": val_target,
                "achieved_images": sum(1 for r in members if r.split == "val"),
            },
            "test": {
                "target_images": test_target,
                "achieved_images": sum(1 for r in members if r.split == "test"),
            },
        }
    return {
        "strategy": sp.strategy,
        "max_hash_distance_bits": sp.max_hash_distance_bits,
        "hash_recipe": DHASH_RECIPE,
        "pillow_version": PIL.__version__,
        "by_country": by_country,
    }


def build_report(
    cfg: DatasetConfig, root: Path, plan: Plan, manifest: bytes, link_method: str,
    link_warnings: list[str], scan_only: bool,
) -> dict[str, Any]:  # fmt: skip
    rows, f = plan.rows, plan.findings
    countries = sorted(cfg.countries)
    cfg_dict = cfg.model_dump(mode="json")
    cfg_dict["raw_dir"] = _rel(cfg.raw_dir, root)
    cfg_dict["out_dir"] = _rel(cfg.out_dir, root)
    cfg_json = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":"))

    included = [r for r in rows if r.status == "included"]
    groups = {
        s: sorted({r.group_id for r in included if r.split == s}) for s in splits.SPLITS
    }
    by_hash: dict[str, list[Row]] = defaultdict(list)
    for r in included:
        by_hash[r.image_sha256].append(r)
    duplicates = [
        {
            "sha256": sha,
            "items": [{"source_id": r.source_id, "split": r.split} for r in members],
        }
        for sha, members in sorted(by_hash.items())
        if len({r.split for r in members}) > 1
    ]
    warnings = plan.warnings + link_warnings
    if duplicates:
        warnings.append(
            f"{len(duplicates)} exact-duplicate image(s) fall in different splits (reported only)"
        )
    status_counts = Counter(r.status for r in rows)
    return {
        "tool_version": __version__,
        "scan_only": scan_only,
        "config": cfg_dict,
        "config_sha256": hashlib.sha256(cfg_json.encode("utf-8")).hexdigest(),
        "dataset_source": {"doi": DOI, "outer_zip_name": OUTER_NAME, "outer_zip_md5": OUTER_MD5},
        "total_labelled_items": len(rows),
        "status_counts": {s: status_counts.get(s, 0) for s in STATUSES},
        "overall": _summary(rows),
        "by_country": {c: _summary([r for r in rows if r.country == c]) for c in countries},
        "by_split": {s: _summary([r for r in rows if r.split == s]) for s in splits.SPLITS},
        "by_country_split": {
            c: {s: _summary([r for r in rows if r.country == c and r.split == s]) for s in splits.SPLITS}
            for c in countries
        },
        "groups_per_split": groups,
        "near_duplicate_grouping": _grouping_report(cfg, rows),
        "label_counts": {c: dict(sorted(f.label_counts[c].items())) for c in countries},
        "non_pothole_object_problems": {c: f.other_object_problems.get(c, 0) for c in countries},
        "unlabelled_official_test_images": dict(sorted(f.unlabelled_test_images.items())),
        "invalid_annotations": sorted(
            f.invalid_annotations, key=lambda d: (d["source_id"], d["object_index"])
        ),
        "excluded_items": [
            {"source_id": r.source_id, "status": r.status, "detail": r.detail}
            for r in rows
            if r.status != "included"
        ],
        "size_mismatches": sorted(f.size_mismatches, key=lambda d: d["source_id"]),
        "clipped_boxes": sorted(f.clipped_boxes, key=lambda d: (d["source_id"], d["object_index"])),
        "exif_oriented_images": sorted(f.exif_oriented),
        "duplicate_images_across_splits": duplicates,
        "link_method": link_method,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "warnings": warnings,
    }


def _report_bytes(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- writing


def check_dirs(cfg: DatasetConfig, root: Path, out_dir: Path) -> None:
    """Refuse output locations outside <root>/data or overlapping the raw data."""
    data_root = (root / "data").resolve()
    out = out_dir.resolve()
    raw = cfg.raw_dir.resolve()
    if out == data_root or not out.is_relative_to(data_root):
        raise ConversionError(f"out_dir must be inside {data_root}, got {out}")
    if out == raw or raw.is_relative_to(out) or out.is_relative_to(raw):
        raise ConversionError(f"out_dir {out} and raw_dir {raw} must not overlap")


def write_dataset(rows: list[Row], out_dir: Path, link_mode: str) -> tuple[str, list[str]]:
    """Write images (linked or copied) and label files. Returns (link_method, warnings)."""
    for kind in ("images", "labels"):
        for split in splits.SPLITS:
            (out_dir / kind / split).mkdir(parents=True, exist_ok=True)
    linked = copied = 0
    errors: Counter = Counter()
    for r in rows:
        if r.status != "included":
            continue
        assert r.image_path is not None
        dst = out_dir / "images" / r.split / f"{r.source_id}.jpg"
        done = False
        if link_mode == "hardlink":
            try:
                os.link(r.image_path, dst)
                linked += 1
                done = True
            except OSError as exc:
                errors[f"{type(exc).__name__} (errno {exc.errno})"] += 1
        if not done:
            shutil.copyfile(r.image_path, dst)
            copied += 1
        label = out_dir / "labels" / r.split / f"{r.source_id}.txt"
        label.write_bytes("".join(line + "\n" for line in r.lines).encode("utf-8"))
    (out_dir / "dataset.yaml").write_bytes(DATASET_YAML.encode("utf-8"))
    method = "copy" if not linked else "hardlink" if not copied else "mixed"
    warnings = []
    if link_mode == "hardlink" and copied:
        warnings.append(
            f"hardlink failed for {copied} file(s) ({dict(sorted(errors.items()))}); fell back to copy"
        )
    return method, warnings


def _print_summary(report: dict[str, Any], where: str, out=print) -> None:
    out(f"RDD2022 conversion {'scan' if report['scan_only'] else 'result'} -> {where}")
    out(f"  labelled items: {report['total_labelled_items']}")
    for status, n in report["status_counts"].items():
        if n:
            out(f"    {status}: {n}")
    for split, s in report["by_split"].items():
        neg = sum(s["negatives"].values())
        out(f"  {split}: {s['images']} images, {s['positives']} positive, {neg} negative, "
            f"{s['pothole_boxes']} boxes")  # fmt: skip
    grouping = report["near_duplicate_grouping"]
    out(f"  split strategy: {grouping['strategy']}")
    if grouping["strategy"] == "similarity":
        out(f"  max hash distance: {grouping['max_hash_distance_bits']} bit(s) ({grouping['hash_recipe']})")
        for country, g in grouping["by_country"].items():
            top = g["largest_components"][0] if g["largest_components"] else None
            largest = f"{top['size']} -> {top['split']}" if top else "none"
            out(
                f"    {country}: val {g['val']['achieved_images']}/{g['val']['target_images']}, "
                f"test {g['test']['achieved_images']}/{g['test']['target_images']}, "
                f"largest component {largest}"
            )
    for w in report["warnings"]:
        out(f"  warning: {w}")


def convert(
    cfg: DatasetConfig,
    *,
    root: Path | None = None,
    scan_only: bool = False,
    overwrite: bool = False,
    out=print,
) -> Result:
    root = Path(os.path.normpath(root or project_root()))
    target = cfg.out_dir.with_name(cfg.out_dir.name + "_scan") if scan_only else cfg.out_dir
    check_dirs(cfg, root, target)
    if not scan_only and target.exists():
        if not target.is_dir():
            raise ConversionError(f"{target} exists and is not a directory")
        if any(target.iterdir()) and not overwrite:
            raise ConversionError(f"{target} is not empty; use --overwrite to replace it")

    plan = build_plan(cfg)
    manifest = manifest_bytes(plan.rows)

    if scan_only:
        report = build_report(cfg, root, plan, manifest, "none", [], True)
        target.mkdir(parents=True, exist_ok=True)
        (target / "manifest.csv").write_bytes(manifest)
        (target / "conversion_report.json").write_bytes(_report_bytes(report))
    else:
        if target.exists():
            shutil.rmtree(target)
        method, link_warnings = write_dataset(plan.rows, target, cfg.link_mode)
        report = build_report(cfg, root, plan, manifest, method, link_warnings, False)
        (target / "manifest.csv").write_bytes(manifest)
        (target / "conversion_report.json").write_bytes(_report_bytes(report))

    _print_summary(report, _rel(target, root), out)
    empty = [s for s, v in report["by_split"].items() if v["images"] == 0]
    if empty:
        out(f"error: empty split(s): {', '.join(empty)}")
    return Result(1 if empty else 0, report, target)


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert RDD2022 to a single-class pothole dataset.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scan-only", action="store_true", help="write only manifest and report to <out_dir>_scan")
    parser.add_argument("--overwrite", action="store_true", help="replace a non-empty out_dir")
    ns = parser.parse_args(argv)
    try:
        cfg = load_dataset_config(ns.config, root=root)
        result = convert(cfg, root=root, scan_only=ns.scan_only, overwrite=ns.overwrite)
    except (ConfigError, ConversionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
