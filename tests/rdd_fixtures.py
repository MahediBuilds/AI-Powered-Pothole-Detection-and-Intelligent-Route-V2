"""Synthetic RDD2022-shaped fixtures: tiny real JPEGs and XML modelled on the real variants."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from roadsense.config import DatasetConfig, load_dataset_config


class Obj:
    """An annotation object. ``box`` is (xmin, ymin, xmax, ymax) or None for no <bndbox>."""

    def __init__(self, name: str, box: tuple[Any, Any, Any, Any] | None, full: bool = False):
        self.name, self.box, self.full = name, box, full


def annotation_xml(
    filename: str,
    size: tuple[int, int] | None,
    objects: list[Obj],
    style: str = "india",
) -> str:
    """Render PASCAL VOC xml. style: india (spaces, <depth> first), japan (tabs), norway
    (empty <depth />, optional <occluded>/<attributes>)."""
    ind = "\t" if style == "japan" else "  "
    lines = ["<annotation>", f"{ind}<folder>images</folder>", f"{ind}<filename>{filename}</filename>"]
    if size is not None:
        w, h = size
        lines.append(f"{ind}<size>")
        if style == "india":
            lines += [f"{ind * 2}<depth>3</depth>", f"{ind * 2}<width>{w}</width>", f"{ind * 2}<height>{h}</height>"]
        elif style == "norway":
            lines += [f"{ind * 2}<width>{w}</width>", f"{ind * 2}<height>{h}</height>", f"{ind * 2}<depth />"]
        else:
            lines += [f"{ind * 2}<width>{w}</width>", f"{ind * 2}<height>{h}</height>", f"{ind * 2}<depth>3</depth>"]
        lines.append(f"{ind}</size>")
    lines.append(f"{ind}<segmented>0</segmented>")
    for obj in objects:
        lines.append(f"{ind}<object>")
        lines.append(f"{ind * 2}<name>{obj.name}</name>")
        if obj.full:
            lines += [
                f"{ind * 2}<pose>Unspecified</pose>",
                f"{ind * 2}<truncated>0</truncated>",
                f"{ind * 2}<difficult>0</difficult>",
                f"{ind * 2}<occluded>0</occluded>",
                f"{ind * 2}<attributes />",
            ]
        elif style == "norway":
            lines += [f"{ind * 2}<occluded>0</occluded>", f"{ind * 2}<attributes />"]
        if obj.box is not None:
            xmin, ymin, xmax, ymax = obj.box
            lines += [
                f"{ind * 2}<bndbox>",
                f"{ind * 3}<xmin>{xmin}</xmin>",
                f"{ind * 3}<ymin>{ymin}</ymin>",
                f"{ind * 3}<xmax>{xmax}</xmax>",
                f"{ind * 3}<ymax>{ymax}</ymax>",
                f"{ind * 2}</bndbox>",
            ]
        lines.append(f"{ind}</object>")
    lines.append("</annotation>")
    return "\n".join(lines) + "\n"


def write_jpeg(path: Path, size: tuple[int, int] = (100, 50), color=(10, 20, 30), exif_orientation=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    if exif_orientation is None:
        im.save(path, "JPEG")
    else:
        exif = Image.Exif()
        exif[274] = exif_orientation
        im.save(path, "JPEG", exif=exif)


def noise_pixels(seed: int, size: tuple[int, int] = (100, 50)) -> np.ndarray:
    """Seeded random image (height x width x 3): coarse random noise smoothly upscaled, so
    small edits barely move its hash. Distinct seeds give hashes ~32 bits apart."""
    rng = np.random.default_rng(seed)
    low = rng.integers(0, 256, (6, 12, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(low).resize(size, Image.BICUBIC))


def variant_pixels(base: np.ndarray, shift: int = 0, brightness: int = 0) -> np.ndarray:
    """A near-duplicate of ``base``: shifted sideways by ``shift`` px (edge-padded) and/or brightened."""
    out = base.copy()
    if shift:
        out = np.pad(base, ((0, 0), (shift, 0), (0, 0)), mode="edge")[:, : base.shape[1]]
    return np.clip(out.astype(np.int16) + brightness, 0, 255).astype(np.uint8)


def write_pixels_jpeg(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, "JPEG", quality=95)


def dhash_of_file(path: Path) -> int:
    from roadsense.data.rdd2022 import dhash64

    with Image.open(path) as im:
        im.load()
        return dhash64(im)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class RawTree:
    """Builds ``<raw_dir>/<Country>/{train,test}/...`` in the real RDD2022 layout."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir

    def source_id(self, country: str, num: int) -> str:
        return f"{country}_{num:06d}"

    def add(
        self,
        country: str,
        num: int,
        objects: list[Obj] | None = None,
        *,
        size: tuple[int, int] = (100, 50),
        xml_size: Any = "same",
        style: str = "india",
        image: bool = True,
        xml: bool = True,
        image_bytes: bytes | None = None,
        xml_text: str | None = None,
        color=None,
        exif_orientation=None,
        pixels: np.ndarray | None = None,
    ) -> str:
        sid = self.source_id(country, num)
        train = self.raw_dir / country / "train"
        if image:
            path = train / "images" / f"{sid}.jpg"
            if image_bytes is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(image_bytes)
            elif pixels is not None:
                write_pixels_jpeg(path, pixels)
                size = (pixels.shape[1], pixels.shape[0])
            else:
                write_jpeg(
                    path, size, color or ((num * 7) % 256, (num * 13 + len(country) * 17) % 256, 40),
                    exif_orientation,
                )
        if xml:
            path = train / "annotations" / "xmls" / f"{sid}.xml"
            path.parent.mkdir(parents=True, exist_ok=True)
            declared = size if xml_size == "same" else xml_size
            text = xml_text if xml_text is not None else annotation_xml(
                f"{sid}.jpg", declared, objects or [], style
            )
            path.write_text(text, encoding="utf-8")
        return sid

    def add_unlabelled(self, country: str, num: int) -> None:
        write_jpeg(self.raw_dir / country / "test" / "images" / f"{self.source_id(country, num)}.jpg")


BOX = (10, 5, 30, 25)  # on a 100x50 image: cx .2, cy .3, w .2, h .4


def make_cfg(root: Path, **overrides: Any) -> DatasetConfig:
    """A DatasetConfig for a project rooted at ``root`` (country-strategy split by default:
    India train, Japan val, Czech test)."""
    split = {
        "strategy": "country",
        "seed": 42,
        "block_size": 1000,
        "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "val_countries": ["Japan"],
        "test_countries": ["Czech"],
    }
    split.update(overrides.pop("split", {}))
    data: dict[str, Any] = {
        "raw_dir": "data/raw",
        "out_dir": "data/processed/ds",
        "countries": ["India", "Japan", "Czech"],
        "negative_countries": ["India", "Japan", "Czech"],
        "pothole_labels": ["D40"],
        "train_negative_ratio": None,
        "min_box_side_px": 2,
        "link_mode": "hardlink",
        "split": split,
    }
    data.update(overrides)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "dataset.yaml.cfg"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_dataset_config(path, root=root)


def add_val_test_positives(tree: RawTree) -> None:
    """One positive each in Japan (val) and Czech (test) so no split is empty."""
    tree.add("Japan", 0, [Obj("D40", BOX)])
    tree.add("Czech", 0, [Obj("D40", BOX)])


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
