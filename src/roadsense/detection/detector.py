"""Pothole detector: a trained Ultralytics YOLO checkpoint behind a small API.

``Detector`` loads the checkpoint once and turns a BGR frame into a list of
``Detection`` boxes in pixel coordinates of that frame. The checkpoint must be a
``detect`` model whose only class is ``{0: 'pothole'}``; anything else (for
example generic COCO weights) is refused at load time so that a wrong file can
never silently produce "potholes".

Bad input and inference failures raise ``DetectorError`` so the caller can count
and skip a frame. ``DetectorLike`` is the protocol the pipeline relies on, which
lets tests substitute a fake detector.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from roadsense.config import DetectionConfig

EXPECTED_CLASS_NAMES = {0: "pothole"}


class DetectorError(Exception):
    """The detector cannot be built, or cannot process the given image."""


@dataclass(frozen=True)
class Detection:
    """A box in pixels of the input image (x1, y1 top-left; x2, y2 bottom-right)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@runtime_checkable
class DetectorLike(Protocol):
    """What the pipeline needs from a detector."""

    weights_sha256: str

    def detect(self, image: Any) -> list[Detection]: ...

    def info(self) -> dict[str, Any]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Detector:
    """Single-class pothole detector. Not thread-safe: use one per worker thread."""

    def __init__(self, cfg: DetectionConfig) -> None:
        self.warnings: list[str] = []
        self._conf = float(cfg.conf)
        self._iou = float(cfg.iou)
        self._imgsz = int(cfg.imgsz)
        self._weights_path = Path(cfg.weights)
        if not self._weights_path.is_file():
            raise DetectorError(f"{self._weights_path}: weights file not found")
        self._device = self._resolve_device(cfg.device)
        self._sha256 = _sha256(self._weights_path)

        try:
            from ultralytics import YOLO  # heavy import, kept lazy
        except ImportError as exc:
            raise DetectorError(f"ultralytics is not installed: {exc}") from exc
        try:
            self._model = YOLO(str(self._weights_path))
        except Exception as exc:  # corrupt file, wrong format, ...
            raise DetectorError(
                f"{self._weights_path}: cannot load as an Ultralytics model: {exc}"
            ) from exc
        task = getattr(self._model, "task", None)
        if task != "detect":
            raise DetectorError(
                f"{self._weights_path}: model task is {task!r}, expected 'detect'"
            )
        names = {int(k): str(v) for k, v in dict(self._model.names).items()}
        if names != EXPECTED_CLASS_NAMES:
            shown = dict(list(names.items())[:5])
            raise DetectorError(
                f"{self._weights_path}: class names are {shown}"
                f"{' ...' if len(names) > 5 else ''}, expected exactly {EXPECTED_CLASS_NAMES}; "
                "these are not pothole weights"
            )
        self._names = names

    # -- properties -------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    @property
    def weights_path(self) -> Path:
        return self._weights_path

    @property
    def weights_sha256(self) -> str:
        return self._sha256

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._names)

    def info(self) -> dict[str, Any]:
        """JSON-serialisable description of the detector settings."""
        import ultralytics

        return {
            "weights": str(self._weights_path),
            "weights_sha256": self._sha256,
            "device": self._device,
            "conf": self._conf,
            "iou": self._iou,
            "imgsz": self._imgsz,
            "ultralytics_version": ultralytics.__version__,
        }

    # -- device -----------------------------------------------------------

    def _resolve_device(self, requested: str) -> str:
        import torch

        cuda = torch.cuda.is_available()
        if requested == "auto":
            return "0" if cuda else "cpu"
        if requested == "cpu":
            return "cpu"
        if cuda:
            return requested
        self.warnings.append(
            f"device {requested!r} requested but CUDA is not available; using cpu"
        )
        return "cpu"

    # -- inference --------------------------------------------------------

    def detect(self, image: Any) -> list[Detection]:
        """Detect potholes in one BGR uint8 HxWx3 image."""
        import numpy as np

        if not isinstance(image, np.ndarray):
            raise DetectorError(f"image must be a numpy array, got {type(image).__name__}")
        if image.dtype != np.uint8:
            raise DetectorError(f"image must be uint8, got {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise DetectorError(f"image must have shape HxWx3, got {image.shape}")
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            raise DetectorError(f"image is empty (shape {image.shape})")

        try:
            results = self._model.predict(
                image,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._imgsz,
                device=self._device,
                verbose=False,
            )
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
        except Exception as exc:
            raise DetectorError(f"inference failed: {exc}") from exc

        out: list[Detection] = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            x1, x2 = (min(max(float(v), 0.0), float(w)) for v in (x1, x2))
            y1, y2 = (min(max(float(v), 0.0), float(h)) for v in (y1, y2))
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                continue  # degenerate after clipping
            class_id = int(k)
            out.append(
                Detection(x1, y1, x2, y2, float(c), class_id, self._names.get(class_id, str(class_id)))
            )
        return out

    def detect_file(self, path: str | os.PathLike[str]) -> list[Detection]:
        """Detect potholes in one image file."""
        import cv2

        path = Path(path)
        image = cv2.imread(str(path))
        if image is None:
            raise DetectorError(f"{path}: cannot read as an image")
        return self.detect(image)
