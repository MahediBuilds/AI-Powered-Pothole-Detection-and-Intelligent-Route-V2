import numpy as np
import pytest

from geo_fixtures import xy
from roadsense.geo.dedup import cluster_points
from roadsense.ingest.gps import haversine_m


def _pts(coords):
    ll = [xy(x, y) for x, y in coords]
    return [p[0] for p in ll], [p[1] for p in ll]


def test_empty_and_single():
    assert cluster_points([], [], 8.0) == []
    assert cluster_points([51.5], [-0.1], 8.0) == [0]


def test_two_close_points_merge_and_far_point_stays_apart():
    lats, lons = _pts([(0, 0), (5, 0), (100, 0)])
    assert cluster_points(lats, lons, 8.0) == [0, 0, 1]


def test_radius_boundary():
    lats, lons = _pts([(0, 0), (7.9, 0)])
    assert cluster_points(lats, lons, 8.0) == [0, 0]
    lats, lons = _pts([(0, 0), (8.1, 0)])
    assert cluster_points(lats, lons, 8.0) == [0, 1]


def test_chain_is_cut_into_bounded_clusters_not_merged():
    lats, lons = _pts([(0, 0), (6, 0), (12, 0), (18, 0), (60, 0)])
    # single linkage would give [0, 0, 0, 0, 1]; 0 and 18 are 18 m apart (> 8 m)
    labels = cluster_points(lats, lons, 8.0)
    assert labels[4] != labels[3]
    assert labels[0] != labels[3]
    assert len(set(labels)) == 3
    _assert_diameters(lats, lons, labels, 8.0)


def _assert_diameters(lats, lons, labels, radius):
    for c in set(labels):
        idx = [i for i, label in enumerate(labels) if label == c]
        for a in idx:
            for b in idx:
                assert haversine_m(lats[a], lons[a], lats[b], lons[b]) <= radius + 1e-6


def test_long_chain_gives_many_clusters_each_within_diameter():
    lats, lons = _pts([(7.0 * i, 0) for i in range(20)])
    labels = cluster_points(lats, lons, 8.0)
    assert len(set(labels)) >= 10  # each cluster holds at most 2 points 7 m apart
    _assert_diameters(lats, lons, labels, 8.0)


def test_three_in_a_line_closest_pair_is_joined():
    # A-B = 6 m, B-C = 5 m, A-C = 11 m: B-C is closest and is joined; A is left out
    lats, lons = _pts([(0, 0), (6, 0), (11, 0)])
    assert cluster_points(lats, lons, 8.0) == [0, 1, 1]
    # equal gaps: A-B = B-C = 6 m, A-C = 12 m -> two clusters; the tie goes to (i, j) = (0, 1)
    lats, lons = _pts([(0, 0), (6, 0), (12, 0)])
    assert cluster_points(lats, lons, 8.0) == [0, 0, 1]


def test_tight_group_is_one_cluster():
    lats, lons = _pts([(0, 0), (3, 0), (0, 3), (3, 3), (1.5, 1.5)])
    assert cluster_points(lats, lons, 8.0) == [0, 0, 0, 0, 0]


def test_duplicates_merge_and_exact_radius_pair_is_joined():
    assert cluster_points([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], 5.0) == [0, 0, 0]
    lats, lons = _pts([(0, 0), (8, 0)])
    exact = haversine_m(lats[0], lons[0], lats[1], lons[1])
    assert cluster_points(lats, lons, exact) == [0, 0]  # at the radius counts as within
    assert cluster_points(lats, lons, exact - 1e-3) == [0, 1]


def test_cluster_cannot_grow_past_radius_via_a_third_point():
    # A and C are 9 m apart; B sits between them (4.5 m from each). Joining all
    # three would break the 8 m bound, so only the first-sorted pair (A-B) merges.
    lats, lons = _pts([(0, 0), (4.5, 0), (9, 0)])
    labels = cluster_points(lats, lons, 8.0)
    assert labels == [0, 0, 1]
    _assert_diameters(lats, lons, labels, 8.0)


def test_random_points_respect_diameter_and_are_deterministic():
    rng = np.random.default_rng(42)
    xs, ys = rng.uniform(0, 200, 300), rng.uniform(0, 200, 300)
    lats, lons = zip(*[xy(a, b) for a, b in zip(xs, ys)])
    labels = cluster_points(lats, lons, 8.0)
    assert labels == cluster_points(lats, lons, 8.0)
    _assert_diameters(lats, lons, labels, 8.0)
    assert min(labels) == 0 and set(labels) == set(range(max(labels) + 1))
    first_seen = [labels.index(c) for c in range(max(labels) + 1)]
    assert first_seen == sorted(first_seen)


def test_labels_numbered_by_smallest_member_index():
    # input order interleaves two clusters: A B A B C
    lats, lons = _pts([(0, 0), (500, 0), (3, 0), (503, 0), (1000, 0)])
    assert cluster_points(lats, lons, 8.0) == [0, 1, 0, 1, 2]


def test_deterministic_under_repetition_and_numpy_input():
    rng = np.random.default_rng(0)
    xs, ys = rng.uniform(0, 300, 200), rng.uniform(0, 300, 200)
    lats, lons = zip(*[xy(a, b) for a, b in zip(xs, ys)])
    a = cluster_points(np.array(lats), np.array(lons), 10.0)
    assert a == cluster_points(list(lats), list(lons), 10.0)
    assert min(a) == 0 and set(a) == set(range(max(a) + 1))
    first_seen = [a.index(label) for label in range(max(a) + 1)]
    assert first_seen == sorted(first_seen)


def test_clusters_are_subsets_of_single_linkage_components():
    rng = np.random.default_rng(3)
    xs, ys = rng.uniform(0, 200, 80), rng.uniform(0, 200, 80)
    lats, lons = zip(*[xy(a, b) for a, b in zip(xs, ys)])
    labels = cluster_points(lats, lons, 15.0)
    n = len(xs)
    comp = list(range(n))
    for _ in range(n):
        for i in range(n):
            for j in range(n):
                if np.hypot(xs[i] - xs[j], ys[i] - ys[j]) <= 15.0 and comp[i] != comp[j]:
                    comp[i] = comp[j] = min(comp[i], comp[j])
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                assert comp[i] == comp[j]  # never joins what is not even chain-linked
    _assert_diameters(lats, lons, labels, 15.0)
    assert len(set(labels)) >= len(set(comp))


def test_zero_radius_merges_only_identical_points():
    assert cluster_points([1.0, 1.0, 1.0001], [2.0, 2.0, 2.0], 0.0) == [0, 0, 1]


def test_bad_input():
    with pytest.raises(ValueError):
        cluster_points([1.0], [1.0, 2.0], 5.0)
    with pytest.raises(ValueError):
        cluster_points([1.0], [1.0], -1.0)
    with pytest.raises(ValueError):
        cluster_points([float("nan")], [1.0], 5.0)


# --------------------------------------------------------------------------- refused-merge cache


def _reference_cluster(lats, lons, radius_m):
    """The plain greedy complete-linkage rule, with no caching and brute-force pairs."""
    from roadsense.ingest.gps import EARTH_RADIUS_M

    lat, lon = np.asarray(lats, float), np.asarray(lons, float)
    n = len(lat)
    phi, lam = np.radians(lat), np.radians(lon)
    pairs = []
    for i in range(n):
        j = np.arange(i + 1, n)
        if len(j) == 0:
            continue
        h = (
            np.sin((phi[j] - phi[i]) / 2) ** 2
            + np.cos(phi[i]) * np.cos(phi[j]) * np.sin((lam[j] - lam[i]) / 2) ** 2
        )
        dist = 2 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(h)))
        pairs.extend((float(d), i, int(k)) for d, k in zip(dist, j) if d <= radius_m)
    pairs.sort()
    within = {(i, j) for _, i, j in pairs}
    members = {i: [i] for i in range(n)}
    cluster = list(range(n))
    for _, i, j in pairs:
        ci, cj = cluster[i], cluster[j]
        if ci == cj:
            continue
        if all((min(a, b), max(a, b)) in within for a in members[ci] for b in members[cj]):
            for m in members[cj]:
                cluster[m] = ci
            members[ci].extend(members.pop(cj))
    labels, out = {}, []
    for c in cluster:
        labels.setdefault(c, len(labels))
        out.append(labels[c])
    return out


@pytest.mark.parametrize(
    "seed, n, spread_m, radius",
    [(1, 120, 400.0, 8.0), (2, 200, 40.0, 8.0), (3, 250, 12.0, 8.0), (4, 150, 25.0, 5.0)],
)
def test_cached_clustering_equals_the_plain_greedy_rule(seed, n, spread_m, radius):
    rng = np.random.default_rng(seed)
    pts = [(float(a), float(b)) for a, b in rng.uniform(0, spread_m, size=(n, 2))]
    lats, lons = _pts(pts)
    assert cluster_points(lats, lons, radius) == _reference_cluster(lats, lons, radius)


def test_cached_clustering_equals_the_plain_rule_with_many_duplicates():
    rng = np.random.default_rng(7)
    base = rng.uniform(0, 30, size=(12, 2))
    pts = [tuple(base[k]) for k in rng.integers(0, 12, size=200)]
    pts = [(x + float(rng.normal(0, 0.3)), y + float(rng.normal(0, 0.3))) for x, y in pts]
    pts += [pts[0]] * 20  # exact duplicates too
    lats, lons = _pts(pts)
    assert cluster_points(lats, lons, 8.0) == _reference_cluster(lats, lons, 8.0)


def test_many_sightings_at_one_spot_cluster_quickly():
    import time

    rng = np.random.default_rng(11)
    lats, lons = _pts([(float(a), float(b)) for a, b in rng.uniform(0, 4, size=(1000, 2))])
    start = time.monotonic()
    labels = cluster_points(lats, lons, 8.0)
    assert time.monotonic() - start < 10.0
    assert set(labels) == {0}  # every pair is within 8 m


def test_many_sightings_at_one_spot_with_refused_merges_cluster_quickly():
    import time

    rng = np.random.default_rng(12)
    lats, lons = _pts([(float(a), float(b)) for a, b in rng.uniform(0, 6, size=(1000, 2))])
    start = time.monotonic()
    labels = cluster_points(lats, lons, 4.0)  # many pairs are within 4 m, many whole clusters are not
    assert time.monotonic() - start < 10.0
    assert len(set(labels)) > 1
