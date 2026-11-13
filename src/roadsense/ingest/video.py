"""Video probing and frame decoding with trustworthy timestamps.

Frame timestamps decide where on the road each detection is placed, so they
come from the decoder's presentation timestamps (PTS) and not from
``index / fps``, which is wrong for variable-frame-rate phone video.

What OpenCV's ``CAP_PROP_POS_MSEC`` does (OpenCV 5.0.0 with the FFmpeg backend,
checked on this machine against ``ffprobe`` packet timestamps):

* Read right after ``grab()`` it is the PTS of the frame that was just grabbed,
  not of the next one. The first frame reads 0.0, and ``retrieve()`` does not
  change it. So the value can be read after every ``grab()`` without decoding.
* On a constant-rate synthetic file and on the real phone recordings it matched
  the container PTS to under a microsecond. On a synthetic file with real gaps
  (every fifth frame dropped, 24.3 fps average) it also matched, so it is the
  actual PTS and not ``index / average_fps``.
* The real recordings have uniform timestamps (about 1/30.01 s apart), so on
  them PTS and ``index / fps`` agree to within a microsecond.

Other builds or backends may return 0 for every frame or non-monotonic values,
and whether a whole file is usable cannot be known from its first frames. So
``read_timeline`` makes one cheap ``grab()``-only pass (no pixel conversion;
about 0.5 s for 600 frames at 1280x720), and uses the PTS only if they are
finite and strictly increasing for the whole file. Otherwise it uses
``index / fps`` for the whole file and records ``source = "index_fps"``. The
two sources are never mixed. Timestamps are shifted so the first frame is
exactly 0.0, matching the GPS clock (0 is the first frame).
"""

from __future__ import annotations

import math
import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roadsense.ingest import IngestError


class VideoError(IngestError):
    """The video is missing, unopenable, or has no decodable frames."""


@dataclass(frozen=True)
class VideoInfo:
    """What the container reports. ``frame_count`` and ``duration_s`` are None
    when the container does not give a sane value."""

    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_s: float | None
    codec: str | None


@dataclass(frozen=True)
class Timeline:
    """Timestamp of every frame in decode order, and where they came from."""

    source: str  # "pts" or "index_fps"
    times_s: tuple[float, ...]


@dataclass(frozen=True, eq=False)
class Frame:
    index: int  # 0-based position in decode order, counting skipped frames
    t_video_s: float
    image: Any  # BGR uint8 ndarray, as decoded (no resizing or colour conversion)


def _cv2() -> Any:
    import cv2  # heavy import, kept lazy

    return cv2


def _open(path: Path) -> Any:
    if not path.is_file():
        raise VideoError(f"{path}: video file not found")
    cap = _cv2().VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoError(f"{path}: cannot open as a video (unsupported or corrupt file)")
    return cap


def _sane(value: float) -> bool:
    return math.isfinite(value) and value > 0


def _fourcc(value: float) -> str | None:
    code = int(value) if math.isfinite(value) else 0
    if code <= 0:
        return None
    text = struct.pack("<I", code & 0xFFFFFFFF).decode("latin-1")
    return text if text.isprintable() and text.strip() else None


def probe_video(path: str | os.PathLike[str]) -> VideoInfo:
    """Read container metadata and confirm that at least one frame decodes."""
    path = Path(path)
    cv2 = _cv2()
    cap = _open(path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = _fourcc(float(cap.get(cv2.CAP_PROP_FOURCC)))
        ok, image = cap.read()
    finally:
        cap.release()
    if not ok or image is None:
        raise VideoError(f"{path}: no decodable frames")
    if width <= 0 or height <= 0:
        height, width = image.shape[:2]
    if not _sane(fps):
        raise VideoError(f"{path}: container reports no usable frame rate ({fps!r})")
    frame_count = int(count) if _sane(count) else None
    duration = frame_count / fps if frame_count is not None else None
    return VideoInfo(width, height, fps, frame_count, duration, codec)


def _choose_timeline(
    pts_ms: list[float], fps: float, path: Path
) -> Timeline:
    """Use PTS if they are finite and strictly increasing, else ``index / fps``."""
    if not pts_ms:
        raise VideoError(f"{path}: no decodable frames")
    usable = all(math.isfinite(p) for p in pts_ms) and all(
        b > a for a, b in zip(pts_ms, pts_ms[1:])
    )
    if usable:
        first = pts_ms[0]
        return Timeline("pts", tuple((p - first) / 1000.0 for p in pts_ms))
    if not _sane(fps):
        raise VideoError(
            f"{path}: timestamps are unusable and the container reports no frame rate"
        )
    return Timeline("index_fps", tuple(i / fps for i in range(len(pts_ms))))


def read_timeline(path: str | os.PathLike[str]) -> Timeline:
    """Timestamp of every frame, from one ``grab()``-only pass over the file."""
    path = Path(path)
    cv2 = _cv2()
    cap = _open(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        pts_ms: list[float] = []
        while cap.grab():
            pts_ms.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
    finally:
        cap.release()
    return _choose_timeline(pts_ms, fps, path)


def iter_frames(
    path: str | os.PathLike[str], stride: int = 1, timeline: Timeline | None = None
) -> Iterator[Frame]:
    """Yield every ``stride``-th frame (BGR) with its video timestamp.

    Skipped frames are only grabbed, not retrieved. The capture is released when
    the iterator is exhausted, closed, garbage collected, or an error occurs, so
    stopping early is safe. Pass the ``Timeline`` from ``read_timeline`` to avoid
    the extra timestamp pass when it is already known (for example to report
    ``timeline.source``). ``stride < 1`` raises immediately.
    """
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"stride must be an integer >= 1, got {stride!r}")
    return _frames(Path(path), stride, timeline)


def _frames(path: Path, stride: int, timeline: Timeline | None) -> Iterator[Frame]:
    if timeline is None:
        timeline = read_timeline(path)
    times = timeline.times_s
    cap = _open(path)
    try:
        index = 0
        while cap.grab():
            if index >= len(times):
                raise VideoError(
                    f"{path}: decoded more frames than the timeline holds ({len(times)})"
                )
            if index % stride == 0:
                ok, image = cap.retrieve()
                if not ok or image is None:
                    raise VideoError(f"{path}: frame {index} was grabbed but cannot be decoded")
                yield Frame(index, times[index], image)
            index += 1
        if index != len(times):
            raise VideoError(
                f"{path}: decoded {index} frames but the timeline holds {len(times)}"
            )
    finally:
        cap.release()
