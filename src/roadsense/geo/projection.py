"""Local metric coordinates around a reference point.

``LocalProjection`` is an equirectangular projection: x grows east and y grows
north, both in metres from the reference point, using the same mean Earth radius
as ``roadsense.ingest.gps.haversine_m``.

Accuracy: the east-west scale is fixed at the reference latitude, so distances
are in error by about ``tan(lat) * dlat`` (relative, dlat in radians) away from
it. That is below 0.1% within 10 km and about 0.4% at 25 km north or south of the
reference at mid latitudes. It is intended for areas a few tens of kilometres
across and must not be used for continental distances or near the poles.
"""

from __future__ import annotations

import math

import numpy as np

from roadsense.ingest.gps import EARTH_RADIUS_M


class LocalProjection:
    """Equirectangular projection to metres around ``(lat0, lon0)``.

    ``to_xy`` and ``to_latlon`` accept floats or numpy arrays and return the same
    kind (floats in, floats out; arrays in, arrays out).
    """

    def __init__(self, lat0: float, lon0: float) -> None:
        if not (math.isfinite(lat0) and math.isfinite(lon0)):
            raise ValueError("reference point must be finite")
        if abs(lat0) >= 89.0:
            raise ValueError(f"reference latitude {lat0} is too close to a pole")
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self._cos0 = math.cos(math.radians(self.lat0))

    def to_xy(self, lat, lon):
        """Return ``(x, y)`` in metres east and north of the reference point."""
        x = EARTH_RADIUS_M * np.radians(np.asarray(lon) - self.lon0) * self._cos0
        y = EARTH_RADIUS_M * np.radians(np.asarray(lat) - self.lat0)
        if np.ndim(x) == 0:
            return float(x), float(y)
        return x, y

    def to_latlon(self, x, y):
        """Inverse of ``to_xy``; returns ``(lat, lon)`` in degrees."""
        lat = self.lat0 + np.degrees(np.asarray(y) / EARTH_RADIUS_M)
        lon = self.lon0 + np.degrees(np.asarray(x) / (EARTH_RADIUS_M * self._cos0))
        if np.ndim(lat) == 0:
            return float(lat), float(lon)
        return lat, lon
