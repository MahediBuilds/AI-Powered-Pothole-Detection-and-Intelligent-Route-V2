"""Interpolation along a road-edge polyline.

An edge has a nominal length (``length_m``, from the road data) and a geometry
(a list of ``(lat, lon)`` points). Offsets are measured in nominal metres; they
are mapped onto the polyline in proportion, so the polyline's own length may
differ slightly from ``length_m`` without offsets falling off the end.
"""

from __future__ import annotations

import math

import numpy as np

from roadsense.geo.projection import LocalProjection

LatLon = tuple[float, float]


def _positions(geometry: list[LatLon], length_m: float):
    """Cumulative metric distance at each vertex, and nominal-to-polyline scale."""
    proj = LocalProjection(geometry[0][0], geometry[0][1])
    lats = np.array([p[0] for p in geometry])
    lons = np.array([p[1] for p in geometry])
    x, y = proj.to_xy(lats, lons)
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    scale = total / length_m if length_m > 0 else 0.0
    return cum, total, scale


def _point_at(geometry: list[LatLon], cum, s: float) -> LatLon:
    n = len(geometry)
    if n == 1:
        return geometry[0]
    i = int(np.searchsorted(cum, s, side="right")) - 1
    i = min(max(i, 0), n - 2)
    seglen = float(cum[i + 1] - cum[i])
    t = 0.0 if seglen <= 0 else min(max((s - float(cum[i])) / seglen, 0.0), 1.0)
    (lat0, lon0), (lat1, lon1) = geometry[i], geometry[i + 1]
    return (lat0 + t * (lat1 - lat0), lon0 + t * (lon1 - lon0))


def point_at(geometry: list[LatLon], length_m: float, offset_m: float) -> LatLon:
    """The point ``offset_m`` nominal metres along the edge (clamped to the edge)."""
    if not geometry:
        raise ValueError("empty geometry")
    cum, total, scale = _positions(geometry, length_m)
    s = min(max(offset_m, 0.0), max(length_m, 0.0)) * scale
    return _point_at(geometry, cum, s)


def cut_polyline(
    geometry: list[LatLon], length_m: float, start_m: float, end_m: float
) -> list[LatLon]:
    """The part of the edge between two offsets as a list of ``(lat, lon)``.

    Offsets are clamped to ``[0, length_m]``. The result starts at the start
    point, includes every original vertex strictly between, and ends at the end
    point; cutting the whole edge returns the original vertices. A zero-length
    cut returns a single point.
    """
    if not geometry:
        raise ValueError("empty geometry")
    if end_m < start_m:
        raise ValueError(f"end {end_m} is before start {start_m}")
    cum, total, scale = _positions(geometry, length_m)
    hi = max(length_m, 0.0)
    s0 = min(max(start_m, 0.0), hi) * scale
    s1 = min(max(end_m, 0.0), hi) * scale
    if math.isclose(s0, 0.0, abs_tol=1e-9):
        s0 = 0.0
    if math.isclose(s1, total, abs_tol=1e-9):
        s1 = total
    first = geometry[0] if s0 == 0.0 else _point_at(geometry, cum, s0)
    last = geometry[-1] if s1 == total else _point_at(geometry, cum, s1)
    if s1 <= s0:
        return [first]
    inner = [geometry[i] for i in range(1, len(geometry) - 1) if s0 < cum[i] < s1]
    return [first, *inner, last]
