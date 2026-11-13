import numpy as np
import pytest

from roadsense.geo.projection import LocalProjection
from roadsense.ingest.gps import haversine_m


def test_reference_point_is_origin():
    p = LocalProjection(51.5, -0.12)
    assert p.to_xy(51.5, -0.12) == (0.0, 0.0)


def test_axes_point_east_and_north():
    p = LocalProjection(51.5, -0.12)
    x, y = p.to_xy(51.501, -0.119)
    assert x > 0 and y > 0
    x, y = p.to_xy(51.499, -0.121)
    assert x < 0 and y < 0


def test_round_trip_scalar_and_array():
    p = LocalProjection(40.0, 10.0)
    lat, lon = p.to_latlon(*p.to_xy(40.123, 10.456))
    assert lat == pytest.approx(40.123, abs=1e-12)
    assert lon == pytest.approx(10.456, abs=1e-12)
    lats = np.array([39.9, 40.0, 40.2])
    lons = np.array([9.8, 10.0, 10.3])
    x, y = p.to_xy(lats, lons)
    assert isinstance(x, np.ndarray) and x.shape == (3,)
    la, lo = p.to_latlon(x, y)
    np.testing.assert_allclose(la, lats, atol=1e-12)
    np.testing.assert_allclose(lo, lons, atol=1e-12)


@pytest.mark.parametrize(
    "a,b",
    [
        ((51.5, -0.12), (51.51, -0.10)),  # about 2 km
        ((51.5, -0.12), (51.58, -0.02)),  # about 10 km
        ((51.5, -0.12), (51.42, -0.22)),
    ],
)
def test_distance_matches_haversine(a, b):
    p = LocalProjection(*a)
    xa, ya = p.to_xy(*a)
    xb, yb = p.to_xy(*b)
    planar = float(np.hypot(xb - xa, yb - ya))
    assert planar == pytest.approx(haversine_m(*a, *b), rel=1e-3)


def test_distance_between_two_points_far_from_reference():
    p = LocalProjection(51.5, -0.12)
    a, b = (51.52, -0.10), (51.521, -0.098)
    xa, ya = p.to_xy(*a)
    xb, yb = p.to_xy(*b)
    assert float(np.hypot(xb - xa, yb - ya)) == pytest.approx(haversine_m(*a, *b), rel=2e-3)


def test_invalid_reference_rejected():
    with pytest.raises(ValueError):
        LocalProjection(float("nan"), 0.0)
    with pytest.raises(ValueError):
        LocalProjection(90.0, 0.0)
