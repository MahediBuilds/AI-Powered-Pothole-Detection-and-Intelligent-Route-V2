import importlib.util
import json

import numpy as np
import pytest

from roadsense.config import DetectionConfig, project_root
from roadsense.detection.detector import Detection, Detector, DetectorError, DetectorLike

ROOT = project_root()
WEIGHTS = ROOT / "runs" / "train" / "baseline" / "weights" / "best.pt"
GENERIC = ROOT / "yolov8s.pt"
VAL_IMAGES = ROOT / "data" / "processed" / "rdd2022_pothole" / "images" / "val"
VAL_LABELS = ROOT / "data" / "processed" / "rdd2022_pothole" / "labels" / "val"

# find_spec, not importorskip: importing ultralytics here would happen at collection
# time, before older tests that assert it is not imported yet (see conftest).
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None, reason="ultralytics is not installed"
)
needs_weights = pytest.mark.skipif(not WEIGHTS.is_file(), reason=f"{WEIGHTS} not found")
needs_val = pytest.mark.skipif(
    not (VAL_IMAGES.is_dir() and VAL_LABELS.is_dir()), reason="RDD2022 validation data not found"
)
slow = pytest.mark.slow


def cfg(**kw):
    return DetectionConfig(weights=kw.pop("weights", WEIGHTS), **kw)


@pytest.fixture(scope="module")
def detector():
    if not WEIGHTS.is_file():
        pytest.skip(f"{WEIGHTS} not found")
    return Detector(cfg())


def positive_val_images(n):
    """The first n validation images (by file name) whose label file is not empty."""
    out = []
    for label in sorted(VAL_LABELS.glob("*.txt")):
        if label.stat().st_size > 0:
            image = VAL_IMAGES / f"{label.stem}.jpg"
            if image.is_file():
                out.append(image)
        if len(out) == n:
            break
    return out


# --------------------------------------------------------------------------- real model


@slow
@needs_weights
def test_loads_and_describes_itself(detector):
    assert detector.class_names == {0: "pothole"}
    assert detector.weights_path == WEIGHTS
    assert detector.weights_sha256 == detector.weights_sha256
    assert len(detector.weights_sha256) == 64
    info = detector.info()
    assert info["weights_sha256"] == detector.weights_sha256
    assert info["conf"] == 0.25 and info["iou"] == 0.5 and info["imgsz"] == 640
    assert info["device"] == detector.device and info["weights"] == str(WEIGHTS)
    assert info["ultralytics_version"]
    json.dumps(info)
    assert isinstance(detector, DetectorLike)


@slow
@needs_weights
def test_device_resolution(detector, monkeypatch):
    import torch

    assert detector.device == ("0" if torch.cuda.is_available() else "cpu")
    cpu = Detector(cfg(device="cpu"))
    assert cpu.device == "cpu" and cpu.warnings == []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fallback = Detector(cfg(device="0"))
    assert fallback.device == "cpu"
    assert any("CUDA" in w for w in fallback.warnings)
    assert Detector(cfg(device="auto")).device == "cpu"


@slow
@needs_weights
def test_blank_image_returns_a_list(detector):
    result = detector.detect(np.full((480, 640, 3), 114, np.uint8))
    assert isinstance(result, list)


@slow
@needs_weights
@needs_val
def test_validation_images_give_detections_inside_the_image(detector):
    import cv2

    images = positive_val_images(8)
    assert images, "no validation image with a non-empty label"
    total = 0
    for path in images:
        h, w = cv2.imread(str(path)).shape[:2]
        for det in detector.detect_file(path):
            total += 1
            assert isinstance(det, Detection) and det.class_name == "pothole" and det.class_id == 0
            assert 0 <= det.x1 < det.x2 <= w and 0 <= det.y1 < det.y2 <= h
            assert 0.25 <= det.confidence <= 1.0
    assert total >= 1


@slow
@needs_weights
def test_boxes_are_clipped_to_a_small_image(detector):
    image = np.random.default_rng(0).integers(0, 255, (64, 80, 3), dtype=np.uint8)
    for det in detector.detect(image):
        assert 0 <= det.x1 < det.x2 <= 80 and 0 <= det.y1 < det.y2 <= 64


@slow
@needs_weights
@pytest.mark.parametrize(
    "bad",
    [
        None,
        "image.jpg",
        np.zeros((10, 10), np.uint8),
        np.zeros((10, 10, 4), np.uint8),
        np.zeros((10, 10, 3), np.float32),
        np.zeros((0, 10, 3), np.uint8),
        np.zeros((10, 0, 3), np.uint8),
    ],
)
def test_invalid_input_raises_detector_error(detector, bad):
    with pytest.raises(DetectorError):
        detector.detect(bad)


@slow
@needs_weights
def test_detect_file_unreadable_is_detector_error(detector, tmp_path):
    with pytest.raises(DetectorError, match="cannot read"):
        detector.detect_file(tmp_path / "missing.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(DetectorError, match="cannot read"):
        detector.detect_file(bad)


@slow
@pytest.mark.skipif(not GENERIC.is_file(), reason=f"{GENERIC} not found")
def test_generic_pretrained_weights_are_rejected():
    with pytest.raises(DetectorError, match="not pothole weights"):
        Detector(cfg(weights=GENERIC))


# --------------------------------------------------------------------------- cheap failures


def test_missing_weights_raise_detector_error(tmp_path):
    with pytest.raises(DetectorError, match="not found"):
        Detector(cfg(weights=tmp_path / "nope.pt"))


def test_corrupt_weights_raise_detector_error(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"this is not a checkpoint")
    with pytest.raises(DetectorError, match="cannot load"):
        Detector(cfg(weights=bad))


def test_wrong_class_names_rejected_with_simulated_model(tmp_path, monkeypatch):
    """A loadable detect model with other classes is refused (no real weights needed)."""
    import ultralytics

    class FakeYOLO:
        task = "detect"
        names = {0: "person", 1: "bicycle"}

        def __init__(self, path):
            pass

    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    with pytest.raises(DetectorError, match="not pothole weights"):
        Detector(cfg(weights=weights))
    FakeYOLO.names = {0: "pothole"}
    FakeYOLO.task = "segment"
    with pytest.raises(DetectorError, match="task"):
        Detector(cfg(weights=weights))
    FakeYOLO.task = "detect"
    assert Detector(cfg(weights=weights)).class_names == {0: "pothole"}


def test_inference_exception_is_wrapped(tmp_path, monkeypatch):
    import ultralytics

    class FakeYOLO:
        task = "detect"
        names = {0: "pothole"}

        def __init__(self, path):
            pass

        def predict(self, *a, **k):
            raise RuntimeError("cuda out of memory")

    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    det = Detector(cfg(weights=weights, device="cpu"))
    with pytest.raises(DetectorError, match="cuda out of memory"):
        det.detect(np.zeros((32, 32, 3), np.uint8))
