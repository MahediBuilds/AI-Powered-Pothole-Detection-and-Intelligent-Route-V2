"""RDD2022 on-disk format: scanning, annotation parsing, box validation, image checks.

Layout under ``<raw_dir>/<Country>/train/``: ``images/<Country>_NNNNNN.jpg`` and
``annotations/xmls/<Country>_NNNNNN.xml`` (PASCAL VOC). Only ``train/`` is
labelled; the official ``test/images`` are unlabelled and are only counted here.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Crack classes of the official label map; used only to name the kind of negative.
CRACK_LABELS = frozenset({"D00", "D10", "D20"})
COORD_TAGS = ("xmin", "ymin", "xmax", "ymax")


class ConversionError(Exception):
    """The raw data cannot be processed as configured."""


class MalformedAnnotation(Exception):
    pass


class CorruptImage(Exception):
    pass


# --------------------------------------------------------------------------- scanning


@dataclass(frozen=True)
class RawItem:
    country: str
    numeric_id: int
    source_id: str
    image_path: Path | None
    xml_path: Path | None


@dataclass(frozen=True)
class CountryScan:
    items: list[RawItem]
    n_unlabelled_test_images: int


def scan_country(raw_dir: Path, country: str) -> CountryScan:
    """List labelled source items (union of image and XML stems) of one country."""
    train = raw_dir / country / "train"
    if not train.is_dir():
        raise ConversionError(
            f"{train} not found. Fetch the data first, e.g. "
            f"python -m roadsense.data.fetch_rdd2022 --countries {country}"
        )
    images = {
        p.stem: p
        for p in (train / "images").glob("*")
        if p.is_file() and p.suffix.lower() == ".jpg"
    }
    xmls = {p.stem: p for p in (train / "annotations" / "xmls").glob("*.xml") if p.is_file()}
    pattern = re.compile(rf"^{re.escape(country)}_(\d+)$")
    items = []
    for stem in sorted(set(images) | set(xmls)):
        match = pattern.match(stem)
        if not match:
            raise ConversionError(
                f"{country}: file stem {stem!r} does not match {country}_<digits>; "
                "cannot assign it a numeric id"
            )
        items.append(RawItem(country, int(match[1]), stem, images.get(stem), xmls.get(stem)))
    items.sort(key=lambda i: (i.numeric_id, i.source_id))
    test_dir = raw_dir / country / "test" / "images"
    n_test = sum(1 for p in test_dir.glob("*") if p.suffix.lower() == ".jpg") if test_dir.is_dir() else 0
    return CountryScan(items, n_test)


# --------------------------------------------------------------------------- annotations


@dataclass
class AnnotationObject:
    index: int
    label: str
    raw: dict[str, str | None]  # raw coordinate strings; all None when <bndbox> is missing
    has_bndbox: bool


@dataclass
class Annotation:
    xml_width: int | None
    xml_height: int | None
    objects: list[AnnotationObject] = field(default_factory=list)


def _int_or_none(text: str | None) -> int | None:
    try:
        return int((text or "").strip())
    except ValueError:
        return None


def _text(parent: ET.Element, tag: str) -> str | None:
    el = parent.find(tag)
    return None if el is None or el.text is None else el.text.strip()


def parse_annotation(path: Path) -> Annotation:
    """Parse a PASCAL VOC xml. Raises MalformedAnnotation if it cannot be read as one."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise MalformedAnnotation(f"unreadable xml: {exc}") from exc
    if root.tag != "annotation":
        raise MalformedAnnotation(f"root element is <{root.tag}>, expected <annotation>")
    size = root.find("size")
    width = _int_or_none(size.findtext("width")) if size is not None else None
    height = _int_or_none(size.findtext("height")) if size is not None else None
    objects = []
    for index, obj in enumerate(root.findall("object")):
        label = (obj.findtext("name") or "").strip() or "(missing)"
        box = obj.find("bndbox")
        if box is None:
            raw: dict[str, str | None] = {tag: None for tag in COORD_TAGS}
        else:
            raw = {tag: _text(box, tag) for tag in COORD_TAGS}
        objects.append(AnnotationObject(index, label, raw, box is not None))
    return Annotation(width, height, objects)


# --------------------------------------------------------------------------- boxes


@dataclass(frozen=True)
class BoxCheck:
    box: tuple[float, float, float, float] | None  # clipped, pixel coordinates
    clipped: bool
    reason: str | None  # None when valid

    @property
    def valid(self) -> bool:
        return self.reason is None


def check_box(
    obj: AnnotationObject, width: int, height: int, min_side: float
) -> BoxCheck:
    """Validate a box against the real image size; clip it to the image if partly outside."""
    if not obj.has_bndbox:
        return BoxCheck(None, False, "missing_bndbox")
    if any(v is None or v == "" for v in obj.raw.values()):
        return BoxCheck(None, False, "missing_coordinate")
    try:
        xmin, ymin, xmax, ymax = (float(obj.raw[tag]) for tag in COORD_TAGS)  # type: ignore[arg-type]
    except ValueError:
        return BoxCheck(None, False, "non_numeric_coordinate")
    if not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax)):
        return BoxCheck(None, False, "non_numeric_coordinate")
    if xmax <= xmin or ymax <= ymin:
        return BoxCheck(None, False, "degenerate_box")
    if xmax <= 0 or ymax <= 0 or xmin >= width or ymin >= height:
        return BoxCheck(None, False, "outside_image")
    cx1, cy1 = max(xmin, 0.0), max(ymin, 0.0)
    cx2, cy2 = min(xmax, float(width)), min(ymax, float(height))
    clipped = (cx1, cy1, cx2, cy2) != (xmin, ymin, xmax, ymax)
    if cx2 - cx1 < min_side or cy2 - cy1 < min_side:
        return BoxCheck(None, clipped, "below_min_side")
    return BoxCheck((cx1, cy1, cx2, cy2), clipped, None)


def yolo_line(box: tuple[float, float, float, float], width: int, height: int) -> str:
    """Format a class-0 YOLO label line, normalised by the real image size."""
    x1, y1, x2, y2 = box

    def unit(v: float) -> float:
        return min(1.0, max(0.0, v))

    cx, cy = unit((x1 + x2) / 2 / width), unit((y1 + y2) / 2 / height)
    w, h = unit((x2 - x1) / width), unit((y2 - y1) / height)
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# --------------------------------------------------------------------------- images


DHASH_RECIPE = "dhash64: L, 9x8 bilinear, row-major, msb first"


def dhash64(image: Any) -> int:
    """64-bit difference hash of a Pillow image (see ``DHASH_RECIPE``).

    The image is converted to mode "L" and resized to 9 columns x 8 rows with
    bilinear resampling; each of the 8 rows yields 8 comparisons "pixel > its left
    neighbour", i.e. ``g[:, 1:] > g[:, :-1]``, packed row-major with the first
    comparison as the most significant bit. EXIF orientation is NOT applied: the
    hash describes the stored pixels, like the box normalisation does.
    """
    import numpy as np
    from PIL import Image

    g = np.asarray(image.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    value = 0
    for bit in (g[:, 1:] > g[:, :-1]).flatten():
        value = (value << 1) | int(bit)
    return value


def format_dhash(value: int) -> str:
    return f"{value:016x}"


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    exif_orientation: int | None
    dhash: int | None = None


def inspect_image(path: Path) -> ImageInfo:
    """Fully decode the image with Pillow; return its real size and perceptual hash.

    ``verify()`` alone does not detect truncated JPEG data, so the image is also
    re-opened and decoded. The hash is computed from the already decoded pixels.
    """
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            orientation = im.getexif().get(274)
            return ImageInfo(
                im.width,
                im.height,
                orientation if isinstance(orientation, int) else None,
                dhash64(im),
            )
    except Exception as exc:  # Pillow raises many types for damaged files
        raise CorruptImage(f"{type(exc).__name__}: {exc}") from exc
