"""Load one recording session: video, GPS CSV and optional sidecar.

A session is three sibling files that share a stem::

    <stem>.mp4          video (required)
    <stem>_gps.csv      GPS trace on the video clock (optional)
    <stem>.json         recorder sidecar, format "road-recorder/1" (optional)

``load_session`` probes the video, loads the GPS log and sidecar when present,
and cross-checks them. Problems that leave the session usable (no GPS, a
malformed sidecar, short GPS coverage, mismatched durations) become strings in
``Session.warnings``; a missing or undecodable video, an unreadable GPS file, or
an explicitly named file that does not exist is an error.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roadsense.ingest import IngestError
from roadsense.ingest.gps import GpsLog, load_gps_csv
from roadsense.ingest.video import VideoInfo, probe_video

SIDECAR_FORMAT = "road-recorder/1"
MIN_GPS_COVERAGE = 0.9
MAX_DURATION_MISMATCH_S = 0.5


class SessionError(IngestError):
    """A named session file does not exist."""


@dataclass(frozen=True)
class Session:
    name: str  # the video file's stem
    video_path: Path
    gps_path: Path | None
    sidecar_path: Path | None
    video: VideoInfo
    gps: GpsLog | None
    gps_coverage: float | None  # fraction of the video duration spanned by usable GPS
    started_utc: datetime | None  # timezone-aware UTC, from the sidecar
    device: str | None
    sidecar_median_accuracy_m: float | None
    warnings: list[str]


_ISO_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})$"
)


def _parse_utc(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp with an explicit offset into aware UTC."""
    if not isinstance(value, str):
        return None
    m = _ISO_RE.match(value.strip())
    if not m:
        return None
    day, clock, frac, zone = m.groups()
    micro = (frac or "0")[:6].ljust(6, "0")
    if zone == "Z":
        offset = "+00:00"
    else:
        offset = zone if ":" in zone else f"{zone[:3]}:{zone[3:]}"
    try:
        parsed = datetime.fromisoformat(f"{day}T{clock}.{micro}{offset}")
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


@dataclass(frozen=True)
class _Sidecar:
    started_utc: datetime | None = None
    device: str | None = None
    median_accuracy_m: float | None = None
    duration_sec: float | None = None
    gps_fixes: int | None = None


def _read_sidecar(path: Path, warnings: list[str]) -> _Sidecar:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        warnings.append(f"{path}: sidecar ignored, cannot read as JSON: {exc}")
        return _Sidecar()
    if not isinstance(data, dict):
        warnings.append(f"{path}: sidecar ignored, top level is not a JSON object")
        return _Sidecar()
    if data.get("format") != SIDECAR_FORMAT:
        warnings.append(
            f"{path}: sidecar ignored, format is {data.get('format')!r}, "
            f"expected {SIDECAR_FORMAT!r}"
        )
        return _Sidecar()

    started = None
    if data.get("started_utc") is not None:
        started = _parse_utc(data["started_utc"])
        if started is None:
            warnings.append(
                f"{path}: started_utc {data['started_utc']!r} is not an ISO-8601 UTC timestamp; ignored"
            )
    device = data.get("device")
    if device is not None and not isinstance(device, str):
        warnings.append(f"{path}: device is not a string; ignored")
        device = None
    median = None
    if data.get("median_accuracy_m") is not None:
        median = _number(data["median_accuracy_m"])
        if median is None or median < 0:
            warnings.append(f"{path}: median_accuracy_m {data['median_accuracy_m']!r} is invalid; ignored")
            median = None
    duration = None
    if data.get("duration_sec") is not None:
        duration = _number(data["duration_sec"])
        if duration is None or duration <= 0:
            warnings.append(f"{path}: duration_sec {data['duration_sec']!r} is invalid; ignored")
            duration = None
    fixes = data.get("gps_fixes")
    if fixes is not None and (isinstance(fixes, bool) or not isinstance(fixes, int) or fixes < 0):
        warnings.append(f"{path}: gps_fixes {fixes!r} is invalid; ignored")
        fixes = None
    return _Sidecar(started, device, median, duration, fixes)


def _gps_coverage(gps: GpsLog, duration_s: float, max_gap_s: float | None) -> float:
    """Fraction of [0, duration] lying between consecutive usable fixes.

    With ``max_gap_s`` set, a pair of fixes further apart than that does not
    cover the time between them.
    """
    used = gps.used_fixes
    covered = 0.0
    for a, b in zip(used, used[1:]):
        if max_gap_s is not None and b.t_video_s - a.t_video_s > max_gap_s:
            continue
        covered += max(0.0, min(b.t_video_s, duration_s) - max(a.t_video_s, 0.0))
    return min(1.0, covered / duration_s)


def load_session(
    video_path: str | os.PathLike[str],
    gps_path: str | os.PathLike[str] | None = None,
    sidecar_path: str | os.PathLike[str] | None = None,
    *,
    max_accuracy_m: float,
    max_gap_s: float | None = None,
) -> Session:
    """Load a session; sibling GPS and sidecar files are found by naming convention.

    ``max_accuracy_m`` is the GPS accuracy filter. ``max_gap_s`` is optional: when
    given, gaps between usable fixes longer than it do not count towards
    ``gps_coverage`` and produce a warning; otherwise coverage is the span of
    usable fixes and the largest gap is only reported in ``gps.summary``.
    """
    video_path = Path(video_path)
    name = video_path.stem
    warnings: list[str] = []

    video = probe_video(video_path)  # errors name the file

    if gps_path is not None:
        gps_path = Path(gps_path)
        if not gps_path.is_file():
            raise SessionError(f"{gps_path}: GPS file not found")
    else:
        candidate = video_path.with_name(f"{name}_gps.csv")
        gps_path = candidate if candidate.is_file() else None

    if sidecar_path is not None:
        sidecar_path = Path(sidecar_path)
        if not sidecar_path.is_file():
            raise SessionError(f"{sidecar_path}: sidecar file not found")
    else:
        candidate = video_path.with_name(f"{name}.json")
        sidecar_path = candidate if candidate.is_file() else None

    sidecar = _read_sidecar(sidecar_path, warnings) if sidecar_path is not None else _Sidecar()

    gps = None
    coverage = None
    if gps_path is None:
        warnings.append(f"{video_path}: no GPS file (expected {name}_gps.csv); positions unavailable")
    else:
        gps = load_gps_csv(gps_path, max_accuracy_m)
        summary = gps.summary
        if gps.issues:
            n_dropped = sum(1 for i in gps.issues if i.row_dropped)
            warnings.append(
                f"{gps_path}: {len(gps.issues)} issue(s) loading GPS rows "
                f"({n_dropped} row(s) dropped); see session.gps.issues"
            )
        if summary.n_used == 0:
            warnings.append(
                f"{gps_path}: no usable GPS fixes ({summary.n_fixes} fix(es), "
                f"{summary.n_rows_read} row(s) read, accuracy limit {max_accuracy_m} m)"
            )
            coverage = 0.0 if video.duration_s is not None else None
        else:
            if video.duration_s is None:
                warnings.append(f"{video_path}: video duration unknown; GPS coverage not computed")
            else:
                coverage = _gps_coverage(gps, video.duration_s, max_gap_s)
                if coverage < MIN_GPS_COVERAGE:
                    warnings.append(
                        f"{gps_path}: usable GPS covers {coverage:.0%} of the "
                        f"{video.duration_s:.1f} s video (< {MIN_GPS_COVERAGE:.0%})"
                    )
            if max_gap_s is not None and summary.max_gap_s is not None and summary.max_gap_s > max_gap_s:
                warnings.append(
                    f"{gps_path}: largest gap between usable fixes is {summary.max_gap_s:.1f} s "
                    f"(> max_gap_s {max_gap_s} s); positions are unavailable inside it"
                )
            if summary.median_accuracy_m is not None and summary.median_accuracy_m > max_accuracy_m / 2:
                warnings.append(
                    f"{gps_path}: median accuracy of usable fixes is "
                    f"{summary.median_accuracy_m:.1f} m, worse than half the {max_accuracy_m} m limit"
                )
        if sidecar.gps_fixes is not None and sidecar.gps_fixes != summary.n_rows_read:
            warnings.append(
                f"{sidecar_path}: sidecar says {sidecar.gps_fixes} GPS fixes but "
                f"{gps_path} has {summary.n_rows_read} data row(s)"
            )

    if (
        sidecar.duration_sec is not None
        and video.duration_s is not None
        and abs(sidecar.duration_sec - video.duration_s) > MAX_DURATION_MISMATCH_S
    ):
        warnings.append(
            f"{sidecar_path}: sidecar duration {sidecar.duration_sec:.2f} s differs from the "
            f"video's {video.duration_s:.2f} s by more than {MAX_DURATION_MISMATCH_S} s"
        )

    return Session(
        name=name,
        video_path=video_path,
        gps_path=gps_path,
        sidecar_path=sidecar_path,
        video=video,
        gps=gps,
        gps_coverage=coverage,
        started_utc=sidecar.started_utc,
        device=sidecar.device,
        sidecar_median_accuracy_m=sidecar.median_accuracy_m,
        warnings=warnings,
    )
