"""Synthetic videos, a fake detector and database helpers for the pipeline tests."""

import itertools
import uuid

import numpy as np
import pytest

from roadsense.config import AppConfig
from roadsense.detection.detector import Detection, DetectorError

W, H = 160, 120
FPS = 10.0
_counter = itertools.count()


def object_box(x, y=40, w=24, h=18):
    return (x, y, x + w, y + h)


def moving_object(first, last, x0=10, y=40, speed=2):
    """Frame-number range [first, last] in which a bright box moves right by ``speed`` px/frame."""
    return {
        k: object_box(x0 + speed * (k - first), y) for k in range(first, last + 1)
    }


def make_frame(boxes):
    image = np.full((H, W, 3), 20, np.uint8)
    for x1, y1, x2, y2 in boxes:
        image[y1:y2, x1:x2] = 255
    return image


def write_object_video(directory, n_frames, objects, fps=FPS):
    """Write a video of dark frames with bright boxes; ``objects`` is a list of {frame: box}."""
    import cv2

    for fourcc, ext in (("mp4v", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi")):
        path = directory / f"objects_{next(_counter)}{ext}"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (W, H))
        if not writer.isOpened():
            writer.release()
            continue
        for k in range(n_frames):
            writer.write(make_frame([o[k] for o in objects if k in o]))
        writer.release()
        if path.is_file() and path.stat().st_size > 0:
            cap = cv2.VideoCapture(str(path))
            ok, image = cap.read()
            cap.release()
            if ok:
                return path
        path.unlink(missing_ok=True)
    pytest.skip("no OpenCV VideoWriter codec works on this machine")


class FakeDetector:
    """Finds bright rectangles by thresholding; can be made to fail.

    ``fail_on`` is a set of 0-based call numbers that raise ``DetectorError``;
    ``always_fail`` makes every call raise ``DetectorError``; ``crash_on`` is a
    call number that raises a plain RuntimeError.
    """

    weights_sha256 = "f" * 64

    def __init__(self, fail_on=(), always_fail=False, crash_on=None):
        self.fail_on = set(fail_on)
        self.always_fail = always_fail
        self.crash_on = crash_on
        self.calls = 0
        self.warnings = []

    def info(self):
        return {"weights": "fake.pt", "weights_sha256": self.weights_sha256, "device": "cpu"}

    def detect(self, image):
        n = self.calls
        self.calls += 1
        if self.always_fail or n in self.fail_on:
            raise DetectorError("fake failure")
        if self.crash_on == n:
            raise RuntimeError("fake crash")
        ys, xs = np.where(image[:, :, 0] > 200)
        if len(xs) == 0:
            return []
        return [
            Detection(float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1,
                      0.9, 0, "pothole")
        ]  # fmt: skip


def make_cfg(tmp_path, **sections):
    data = {"uploads": {"artifacts_dir": str(tmp_path / "artifacts"), "dir": str(tmp_path / "up")}}
    data.update(sections)
    return AppConfig.model_validate(data)


def add_session(conn, video_path, fixes=(), gps_source=None, name="clip"):
    """Insert a session row and GPS fixes ``(t, lat, lon[, accuracy])``; return the session id."""
    sid = uuid.uuid4().hex
    if gps_source is None:
        gps_source = "recorded" if fixes else "none"
    conn.execute(
        "INSERT INTO sessions (id, name, video_path, gps_source, created_at) VALUES (?,?,?,?,'t')",
        (sid, name, str(video_path), gps_source),
    )
    for seq, fix in enumerate(fixes):
        t, lat, lon = fix[:3]
        acc = fix[3] if len(fix) > 3 else None
        conn.execute(
            "INSERT INTO gps_fixes (session_id, seq, t_video_s, lat, lon, accuracy_m, used) "
            "VALUES (?,?,?,?,?,?,1)",
            (sid, seq, t, lat, lon, acc),
        )
    conn.commit()
    return sid
