"""Build a small STAND-IN dataset in the unified_v1 contract layout, for smoke tests only.

    python scripts/make_smoke_unified.py [--out data/processed/_smoke_unified]
        [--rdd-train 160] [--rdd-val 48] [--hrp-train 64] [--hrp-val 24] [--rdd-test 8]

Layout (the same as the real dataset): images/<source>/<split>/<file>.jpg,
labels/<source>/<split>/<file>.txt, lists/{train,val_pooled,rdd2022_val,hrp4k_val,
rdd2022_test,hrp4k_test}.txt (entries in the real dataset's "./../images/..." form), eval/*.yaml, dataset.yaml,
manifest.csv. Sources are read-only: RDD2022 images come from
data/processed/rdd2022_pothole, HRP4K images from data/external/hrp4k/extracted
(resized to 1280 px wide, as in the real dataset). A SMOKE_ONLY marker file is
written. This is not a research dataset: never train a real model on it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RDD = ROOT / "data" / "processed" / "rdd2022_pothole"
HRP = ROOT / "data" / "external" / "hrp4k" / "extracted" / "HRP4K"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pick(paths: list[Path], n: int, seed: int) -> list[Path]:
    paths = sorted(paths)
    random.Random(seed).shuffle(paths)
    return sorted(paths[:n])


def add(out: Path, rows: list[dict], lists: dict[str, list[str]], source: str, split: str,
        images: list[tuple[Path, Path]], list_names: list[str]) -> None:
    for img, lab in images:
        name = f"{source}_{img.name}"
        dst_img = out / "images" / source / split / name
        dst_lab = out / "labels" / source / split / (Path(name).stem + ".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lab.parent.mkdir(parents=True, exist_ok=True)
        if source == "hrp4k":
            with Image.open(img) as im:
                im = im.convert("RGB")
                im = im.resize((1280, round(im.height * 1280 / im.width)), Image.LANCZOS)
                im.save(dst_img, quality=90)
        else:
            shutil.copyfile(img, dst_img)
        shutil.copyfile(lab, dst_lab)
        rel = dst_img.relative_to(out).as_posix()
        for lname in list_names:
            lists.setdefault(lname, []).append(rel)
        rows.append({
            "source": source, "original_path": str(img), "split": split, "sha256": sha256(dst_img),
            "group_id": f"{source}:{img.stem}",
            "box_count": sum(1 for ln in dst_lab.read_text().splitlines() if ln.strip()),
            "path": rel,
        })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "_smoke_unified")
    ap.add_argument("--rdd-train", type=int, default=160)
    ap.add_argument("--rdd-val", type=int, default=48)
    ap.add_argument("--rdd-test", type=int, default=8)
    ap.add_argument("--hrp-train", type=int, default=64)
    ap.add_argument("--hrp-val", type=int, default=24)
    ap.add_argument("--hrp-test", type=int, default=8)
    ns = ap.parse_args(argv)
    out = ns.out.resolve()
    if "processed" not in out.parts or "unified_v1" in out.name:
        print("refusing: output must be under a 'processed' directory and not be unified_v1", file=sys.stderr)
        return 2
    if out.exists():
        print(f"refusing: {out} exists", file=sys.stderr)
        return 2
    rows: list[dict] = []
    lists: dict[str, list[str]] = {}
    for split, n, rsplit, lnames in (
        ("train", ns.rdd_train, "train", ["train"]),
        ("val", ns.rdd_val, "val", ["val_pooled", "rdd2022_val"]),
        ("test", ns.rdd_test, "test", ["rdd2022_test"]),
    ):
        imgs = pick(list((RDD / "images" / rsplit).glob("*.jpg")), n, 1)
        add(out, rows, lists, "rdd2022", split, [(i, RDD / "labels" / rsplit / (i.stem + ".txt")) for i in imgs], lnames)
    for split, n, hsplit, lnames in (
        ("train", ns.hrp_train, "train", ["train"]),
        ("val", ns.hrp_val, "valid", ["val_pooled", "hrp4k_val"]),
        ("test", ns.hrp_test, "test", ["hrp4k_test"]),
    ):
        imgs = pick(list((HRP / hsplit / "images").glob("*.jpg")), n, 2)
        add(out, rows, lists, "hrp4k", split, [(i, HRP / hsplit / "labels" / (i.stem + ".txt")) for i in imgs], lnames)
    (out / "lists").mkdir(parents=True, exist_ok=True)
    for lname in ("train", "val_pooled", "rdd2022_val", "hrp4k_val", "rdd2022_test", "hrp4k_test"):
        # Same entry format as the real unified_v1 lists: "./../images/..." (Ultralytics resolves the
        # leading "./" against the list file's directory, lists/).
        entries = ["./../" + rel for rel in lists.get(lname, [])]
        (out / "lists" / f"{lname}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    (out / "eval").mkdir()
    for lname in ("rdd2022_val", "hrp4k_val", "rdd2022_test", "hrp4k_test"):
        body = {"train": f"../lists/{lname}.txt", "val": f"../lists/{lname}.txt",
                "test": f"../lists/{lname}.txt", "names": {0: "pothole"}}
        (out / "eval" / f"{lname}.yaml").write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    (out / "dataset.yaml").write_text(
        yaml.safe_dump({"train": "lists/train.txt", "val": "lists/val_pooled.txt", "names": {0: "pothole"}},
                       sort_keys=False), encoding="utf-8")
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "SMOKE_ONLY").write_text("Stand-in dataset for smoke tests. Not for research use.\n", encoding="utf-8")
    print(f"wrote {len(rows)} images to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
