"""GPS trace loading and camera-position queries.

The recorder writes one CSV per video. ``timestamp_sec`` is on the video clock
(0 is the first frame) and may be negative for fixes taken before it. Required
columns are ``timestamp_sec``, ``lat`` and ``lon``; ``accuracy_m``,
``speed_mps``, ``bearing_deg`` and ``utc_ms`` are optional and other columns
are ignored.

``load_gps_csv`` never drops anything silently. A row that cannot be a fix is
rejected and recorded as a ``GpsIssue``; a bad optional field is set to None and
recorded, and the fix is kept. Fixes whose reported accuracy is worse than the
threshold stay in the log with ``used = False`` so the database keeps the whole
trace; everything downstream (``position_at``, the summary) uses only ``used``
fixes.

``position_at`` answers "where was the camera at video time t" or returns None.
It interpolates only between two usable fixes that are close enough in time and
clamps to an end fix only within a short extrapolation window. It never guesses.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from roadsense.ingest import IngestError

REQUIRED_COLUMNS = ("timestamp_sec", "lat", "lon")
EARTH_RADIUS_M = 6_371_008.8
# Below this separation two fixes say nothing reliable about heading (GPS noise).
MIN_COURSE_DISTANCE_M = 1.0


class GpsError(IngestError):
    """The GPS file is missing, unreadable, or lacks required columns."""


@dataclass(frozen=True)
class GpsFix:
    """One GPS fix; the fields are the columns of the ``gps_fixes`` table."""

    seq: int
    t_video_s: float
    lat: float
    lon: float
    accuracy_m: float | None
    speed_mps: float | None
    bearing_deg: float | None
    utc_ms: int | None
    used: bool


@dataclass(frozen=True)
class GpsIssue:
    """A problem found while loading a CSV row.

    ``line`` is the physical line number in the file (the header is line 1).
    ``code`` is a stable short name, ``reason`` a readable explanation and
    ``row_dropped`` says whether the row was lost (True) or kept as a fix with
    the offending field set to None (False).
    """

    line: int
    code: str
    reason: str
    raw: str
    row_dropped: bool


@dataclass(frozen=True)
class GpsSummary:
    n_rows_read: int
    n_fixes: int
    n_used: int
    n_unknown_accuracy: int  # fixes with no accuracy; they count as usable
    rejected_by_reason: dict[str, int]  # row-dropping issues by code
    t_first_s: float | None
    t_last_s: float | None
    median_accuracy_m: float | None  # of used fixes with a known accuracy
    max_gap_s: float | None  # largest gap between consecutive used fixes
    path_length_m: float  # haversine length over used fixes


@dataclass(frozen=True)
class GpsLog:
    """Fixes sorted by time (strictly increasing), plus what was found loading them."""

    fixes: tuple[GpsFix, ...]
    issues: tuple[GpsIssue, ...]
    n_rows_read: int

    def __post_init__(self) -> None:
        for prev, cur in zip(self.fixes, self.fixes[1:]):
            if not cur.t_video_s > prev.t_video_s:
                raise ValueError(
                    "GpsLog fixes must have strictly increasing t_video_s "
                    f"(seq {prev.seq} at {prev.t_video_s}, seq {cur.seq} at {cur.t_video_s})"
                )

    @cached_property
    def used_fixes(self) -> tuple[GpsFix, ...]:
        return tuple(f for f in self.fixes if f.used)

    @cached_property
    def _used_times(self) -> list[float]:
        # Precomputed once so position_at can bisect instead of rescanning.
        return [f.t_video_s for f in self.used_fixes]

    @cached_property
    def summary(self) -> GpsSummary:
        used = self.used_fixes
        accuracies = [f.accuracy_m for f in used if f.accuracy_m is not None]
        rejected: dict[str, int] = {}
        for issue in self.issues:
            if issue.row_dropped:
                rejected[issue.code] = rejected.get(issue.code, 0) + 1
        gaps = [b.t_video_s - a.t_video_s for a, b in zip(used, used[1:])]
        length = sum(haversine_m(a.lat, a.lon, b.lat, b.lon) for a, b in zip(used, used[1:]))
        return GpsSummary(
            n_rows_read=self.n_rows_read,
            n_fixes=len(self.fixes),
            n_used=len(used),
            n_unknown_accuracy=sum(1 for f in self.fixes if f.accuracy_m is None),
            rejected_by_reason=rejected,
            t_first_s=self.fixes[0].t_video_s if self.fixes else None,
            t_last_s=self.fixes[-1].t_video_s if self.fixes else None,
            median_accuracy_m=statistics.median(accuracies) if accuracies else None,
            max_gap_s=max(gaps) if gaps else None,
            path_length_m=length,
        )


@dataclass(frozen=True)
class Position:
    """Interpolated camera position.

    ``gap_s`` is the time between the two bracketing fixes (0.0 at an exact fix
    or when clamped to an end fix, where ``extrapolated`` is True).
    ``accuracy_m`` is the worse of the two bracketing accuracies.
    """

    lat: float
    lon: float
    accuracy_m: float | None
    speed_mps: float | None
    bearing_deg: float | None
    gap_s: float
    extrapolated: bool


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres on a sphere of the mean Earth radius."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def _course_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, in [0, 360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    line: int
    raw: str
    t: float
    lat: float
    lon: float
    accuracy_m: float | None
    speed_mps: float | None
    bearing_deg: float | None
    utc_ms: int | None


def _float(text: str) -> float:
    """Parse a finite float; raises ValueError otherwise (also rejects ``1_0``)."""
    if "_" in text:
        raise ValueError("not a number")
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("not finite")
    return value


def _int(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        value = _float(text)  # tolerate "1.7926e12"
        if not value.is_integer():
            raise ValueError("not an integer") from None
        return int(value)


def _parse_row(
    line: int, raw: str, cells: dict[str, str], issues: list[GpsIssue]
) -> _Row | None:
    """Turn one CSV row into a ``_Row`` or record why it was rejected."""

    def reject(code: str, reason: str) -> None:
        issues.append(GpsIssue(line, code, reason, raw, row_dropped=True))

    required: dict[str, float] = {}
    for name, code in (("timestamp_sec", "timestamp"), ("lat", "lat"), ("lon", "lon")):
        text = cells.get(name, "")
        if text == "":
            reject(f"{code}_missing", f"{name} is blank")
            return None
        try:
            required[name] = _float(text)
        except ValueError as exc:
            reject(f"{code}_invalid", f"{name} {text!r}: {exc}")
            return None
    lat, lon = required["lat"], required["lon"]
    if not -90.0 <= lat <= 90.0:
        reject("lat_out_of_range", f"lat {lat} is outside [-90, 90]")
        return None
    if not -180.0 <= lon <= 180.0:
        reject("lon_out_of_range", f"lon {lon} is outside [-180, 180]")
        return None
    if lat == 0.0 and lon == 0.0:
        reject("null_island", "lat and lon are both 0 (no fix)")
        return None

    def optional(name: str, parse, valid, why: str):
        text = cells.get(name, "")
        if text == "":
            return None
        try:
            value = parse(text)
            if not valid(value):
                raise ValueError(why)
        except ValueError as exc:
            issues.append(
                GpsIssue(line, f"{name}_invalid", f"{name} {text!r}: {exc}; ignored", raw, False)
            )
            return None
        return value

    return _Row(
        line=line,
        raw=raw,
        t=required["timestamp_sec"],
        lat=lat,
        lon=lon,
        accuracy_m=optional("accuracy_m", _float, lambda v: v >= 0, "must be >= 0"),
        speed_mps=optional("speed_mps", _float, lambda v: v >= 0, "must be >= 0"),
        bearing_deg=optional("bearing_deg", _float, lambda v: 0 <= v <= 360, "must be in [0, 360]"),
        utc_ms=optional("utc_ms", _int, lambda v: v >= 0, "must be >= 0"),
    )


def _accuracy_key(row: _Row) -> tuple[bool, float]:
    # Unknown accuracy counts as worst.
    return (row.accuracy_m is None, row.accuracy_m or 0.0)


def load_gps_csv(path: str | os.PathLike[str], max_accuracy_m: float) -> GpsLog:
    """Load a recorder GPS CSV.

    Raises ``GpsError`` (naming the file) if it is unreadable, empty, or lacks a
    required column. Row-level problems are recorded in ``GpsLog.issues``.
    ``max_accuracy_m`` is the largest reported accuracy still marked ``used``
    (a fix exactly at the threshold is usable).
    """
    path = Path(path)
    if not (math.isfinite(max_accuracy_m) and max_accuracy_m > 0):
        raise ValueError(f"max_accuracy_m must be a positive number, got {max_accuracy_m!r}")

    rows: list[_Row] = []
    issues: list[GpsIssue] = []
    n_rows = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next((r for r in reader if r), None)
            if header is None:
                raise GpsError(f"{path}: file is empty (no header row)")
            columns = [c.strip() for c in header]
            missing = [c for c in REQUIRED_COLUMNS if c not in columns]
            if missing:
                raise GpsError(
                    f"{path}: missing required column(s) {missing}; header is {columns}"
                )
            index = {name: i for i, name in enumerate(columns)}
            for cells in reader:
                if not cells:  # blank line
                    continue
                n_rows += 1
                line = reader.line_num
                raw = ",".join(cells)
                if len(cells) != len(columns):
                    issues.append(
                        GpsIssue(
                            line,
                            "column_count",
                            f"row has {len(cells)} fields, header has {len(columns)}",
                            raw,
                            row_dropped=False,
                        )
                    )
                named = {
                    name: (cells[i].strip() if i < len(cells) else "")
                    for name, i in index.items()
                }
                row = _parse_row(line, raw, named, issues)
                if row is not None:
                    rows.append(row)
    except OSError as exc:
        raise GpsError(f"{path}: cannot read GPS file: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise GpsError(f"{path}: not a UTF-8 text file: {exc}") from exc
    except csv.Error as exc:
        raise GpsError(f"{path}: malformed CSV: {exc}") from exc

    for prev, cur in zip(rows, rows[1:]):
        if cur.t < prev.t:
            issues.append(
                GpsIssue(
                    cur.line,
                    "unsorted",
                    f"timestamp {cur.t} is earlier than the previous row ({prev.t}); "
                    "rows were sorted by time",
                    cur.raw,
                    row_dropped=False,
                )
            )
            break
    rows.sort(key=lambda r: r.t)  # stable: file order kept for equal times

    kept: list[_Row] = []
    for _, group in _group_by_time(rows):
        best = min(group, key=_accuracy_key)
        for row in group:
            if row is not best:
                issues.append(
                    GpsIssue(
                        row.line,
                        "duplicate_timestamp",
                        f"timestamp {row.t} repeated; kept the fix on line {best.line} "
                        "(smaller accuracy)",
                        row.raw,
                        row_dropped=True,
                    )
                )
        kept.append(best)

    fixes = tuple(
        GpsFix(
            seq=seq,
            t_video_s=r.t,
            lat=r.lat,
            lon=r.lon,
            accuracy_m=r.accuracy_m,
            speed_mps=r.speed_mps,
            bearing_deg=r.bearing_deg,
            utc_ms=r.utc_ms,
            used=not (r.accuracy_m is not None and r.accuracy_m > max_accuracy_m),
        )
        for seq, r in enumerate(kept)
    )
    issues.sort(key=lambda i: i.line)
    return GpsLog(fixes=fixes, issues=tuple(issues), n_rows_read=n_rows)


def _group_by_time(rows: list[_Row]):
    group: list[_Row] = []
    for row in rows:
        if group and row.t != group[0].t:
            yield group[0].t, group
            group = []
        group.append(row)
    if group:
        yield group[0].t, group


# ---------------------------------------------------------------------------
# Position queries
# ---------------------------------------------------------------------------


def _fix_position(fix: GpsFix, *, extrapolated: bool) -> Position:
    return Position(
        lat=fix.lat,
        lon=fix.lon,
        accuracy_m=fix.accuracy_m,
        speed_mps=fix.speed_mps,
        bearing_deg=fix.bearing_deg,
        gap_s=0.0,
        extrapolated=extrapolated,
    )


def _worse(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def position_at(
    log: GpsLog, t_video_s: float, max_gap_s: float, max_extrapolation_s: float
) -> Position | None:
    """Camera position at a video time, or None when the data does not support one.

    Only ``used`` fixes count. Between two fixes the position is linearly
    interpolated, unless they are more than ``max_gap_s`` apart (a dropout is not
    a straight line). Before the first or after the last fix, the end fix is
    returned (``extrapolated=True``) only within ``max_extrapolation_s``.
    """
    if not math.isfinite(t_video_s):
        raise ValueError(f"t_video_s must be finite, got {t_video_s!r}")
    if not (math.isfinite(max_gap_s) and max_gap_s > 0):
        raise ValueError(f"max_gap_s must be a positive number, got {max_gap_s!r}")
    if not (math.isfinite(max_extrapolation_s) and max_extrapolation_s >= 0):
        raise ValueError(f"max_extrapolation_s must be >= 0, got {max_extrapolation_s!r}")

    times = log._used_times
    if not times:
        return None
    used = log.used_fixes
    i = bisect_right(times, t_video_s)  # used[i-1] is the last fix at or before t

    if i > 0 and times[i - 1] == t_video_s:
        return _fix_position(used[i - 1], extrapolated=False)
    if i == 0:
        if times[0] - t_video_s > max_extrapolation_s:
            return None
        return _fix_position(used[0], extrapolated=True)
    if i == len(times):
        if t_video_s - times[-1] > max_extrapolation_s:
            return None
        return _fix_position(used[-1], extrapolated=True)

    a, b = used[i - 1], used[i]
    gap = b.t_video_s - a.t_video_s
    if gap > max_gap_s:
        return None
    frac = (t_video_s - a.t_video_s) / gap

    lat = a.lat + frac * (b.lat - a.lat)
    dlon = (b.lon - a.lon + 180.0) % 360.0 - 180.0  # shortest way round the antimeridian
    lon = (a.lon + frac * dlon + 180.0) % 360.0 - 180.0

    speed = None
    if a.speed_mps is not None and b.speed_mps is not None:
        speed = a.speed_mps + frac * (b.speed_mps - a.speed_mps)

    if a.bearing_deg is not None and b.bearing_deg is not None:
        delta = (b.bearing_deg - a.bearing_deg + 540.0) % 360.0 - 180.0  # shortest arc
        bearing = (a.bearing_deg + frac * delta) % 360.0
    elif haversine_m(a.lat, a.lon, b.lat, b.lon) > MIN_COURSE_DISTANCE_M:
        bearing = _course_deg(a.lat, a.lon, b.lat, b.lon)
    else:
        bearing = None

    return Position(
        lat=lat,
        lon=lon,
        accuracy_m=_worse(a.accuracy_m, b.accuracy_m),
        speed_mps=speed,
        bearing_deg=bearing,
        gap_s=gap,
        extrapolated=False,
    )
