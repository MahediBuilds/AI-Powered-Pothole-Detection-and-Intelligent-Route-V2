"""Synthetic video helpers shared by the ingestion tests."""

import itertools

import numpy as np
import pytest

W, H = 96, 64
STEP = 12  # grey-level step between consecutive frame numbers; far above codec noise
_counter = itertools.count()


def frame_value(k):
    return 10 + STEP * k


def decode_frame_number(image):
    return round((float(image.mean()) - 10) / STEP)


def _write(path, n_frames, fps, fourcc):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (W, H))
    if not writer.isOpened():
        writer.release()
        return False
    for k in range(n_frames):
        writer.write(np.full((H, W, 3), frame_value(k), np.uint8))
    writer.release()
    return path.is_file() and path.stat().st_size > 0


def write_synthetic_video(directory, n_frames=20, fps=20.0):
    """Write a video whose frame k is a solid grey of level 10 + 12 k (so k < 20).

    Tries several codecs and returns the first whose output decodes back to the
    right first frame; skips the test if none works on this machine.
    """
    import cv2

    assert n_frames <= 20, "frame number is encoded in a uint8 grey level"
    for fourcc, ext in (("mp4v", ".mp4"), ("avc1", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi")):
        path = directory / f"synthetic_{next(_counter)}{ext}"
        if _write(path, n_frames, fps, fourcc):
            cap = cv2.VideoCapture(str(path))
            ok, image = cap.read()
            cap.release()
            if ok and decode_frame_number(image) == 0:
                return path
        path.unlink(missing_ok=True)
    pytest.skip("no OpenCV VideoWriter codec (mp4v, avc1, MJPG, XVID) works on this machine")
