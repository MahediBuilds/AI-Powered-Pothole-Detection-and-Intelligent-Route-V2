import numpy as np
import pytest

from geo_fixtures import edge_point, make_grid, xy
from roadsense.geo import mapmatch
from roadsense.geo.mapmatch import EdgeIndex
from roadsense.ingest.gps import haversine_m


@pytest.fixture(scope="module")
def graph():
    return make_grid()


@pytest.fixture(scope="module")
def index(graph):
    return EdgeIndex(graph)


def test_snap_on_straight_edge(index):
    # node 0 is (0, 0); node 1 is (100, 0). Point 40 m along, 5 m north of the street.
    s = index.snap(*xy(40, 5), max_dist_m=30)
    assert s is not None
    assert (s.edge_u, s.edge_v) == (0, 1) and s.edge_key == 0
    assert s.offset_m == pytest.approx(40.0, abs=0.2)
    assert s.distance_m == pytest.approx(5.0, abs=0.1)
    assert haversine_m(s.lat, s.lon, *xy(40, 0)) < 0.2


def test_snap_returns_canonical_edge_for_high_to_low_stored_edges(index):
    # edge between nodes 4 (0,100) and 8 (0,200): offset measured from node 4
    s = index.snap(*xy(2, 150), max_dist_m=30)
    assert (s.edge_u, s.edge_v) == (4, 8)
    assert s.offset_m == pytest.approx(50.0, abs=0.2)


def test_snap_at_node_and_ends_are_within_edge_length(index):
    s = index.snap(*xy(-50, 0), max_dist_m=100)  # beyond the west end of the grid
    assert (s.edge_u, s.edge_v) == (0, 4) or (s.edge_u, s.edge_v) == (0, 1)
    assert 0.0 <= s.offset_m <= 100.5
    s = index.snap(*xy(0, 0), max_dist_m=5)
    assert s.offset_m == pytest.approx(0.0, abs=0.1)


def test_snap_along_curved_edge_offset_follows_polyline(graph, index):
    # edge 9-10 bulges north through (125, 230) and (175, 230); nodes at y = 200
    for off in (10.0, 55.0, 100.0):
        lat, lon = edge_point(graph, 9, 10, 0, off)
        s = index.snap(lat, lon, max_dist_m=5)
        assert (s.edge_u, s.edge_v, s.edge_key) == (9, 10, 0)
        assert s.offset_m == pytest.approx(off, abs=0.3)
        assert s.distance_m < 0.2
    length = graph.g.edges[9, 10, 0]["length_m"]
    far = index.snap(*xy(150, 230.5), max_dist_m=5)  # on the bulge, midway
    assert far.offset_m == pytest.approx(length / 2, abs=1.0)
    assert length > 110.0  # longer than the straight 100 m


def test_snap_chooses_parallel_edge_by_geometry(graph, index):
    # edge 1-2 key 0 is straight at y = 0; key 1 bulges to y = -40 at x = 150
    on_alley = index.snap(*xy(150, -39), max_dist_m=5)
    assert on_alley.edge == (1, 2, 1)
    on_street = index.snap(*xy(150, 1), max_dist_m=5)
    assert on_street.edge == (1, 2, 0)


def test_snap_beyond_max_distance_is_none(index):
    assert index.snap(*xy(150, 500), max_dist_m=30) is None
    near = index.snap(*xy(40, 40), max_dist_m=45)  # 40 m from the nearest street
    assert near is not None and near.distance_m == pytest.approx(40.0, abs=0.2)
    assert index.snap(*xy(40, 40), max_dist_m=30) is None


def test_tie_is_deterministic_and_prefers_first_canonical_edge(index):
    # node 0 lies exactly on edges (0, 1), (0, 4) and the alley; first in sorted order wins
    first = index.snap(*xy(0, 0), max_dist_m=5)
    for _ in range(3):
        assert index.snap(*xy(0, 0), max_dist_m=5) == first
    assert first.edge == (0, 1, 0)
    assert first.offset_m == pytest.approx(0.0, abs=0.01)


def test_batch_equals_single_and_aligns_none(index):
    pts = [xy(40, 5), xy(500, 500), xy(150, -39), xy(2, 150), xy(150, 230.5)]
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    batch = index.snap_many(lats, lons, 30)
    assert batch[1] is None
    for i, (lat, lon) in enumerate(pts):
        assert batch[i] == index.snap(lat, lon, 30)


def test_chunking_gives_same_results(graph, monkeypatch):
    rng = np.random.default_rng(1)
    xs, ys = rng.uniform(-20, 320, 60), rng.uniform(-20, 320, 60)
    lats, lons = zip(*[xy(x, y) for x, y in zip(xs, ys)])
    idx = EdgeIndex(graph)
    whole = idx.snap_many(lats, lons, 40)
    monkeypatch.setattr(mapmatch, "_CHUNK_ELEMENTS", 1)  # one point per chunk
    assert idx.snap_many(lats, lons, 40) == whole


def test_empty_and_mismatched_input(index):
    assert index.snap_many([], [], 10) == []
    with pytest.raises(ValueError):
        index.snap_many([1.0], [1.0, 2.0], 10)


def test_nearest_node(index, graph):
    assert index.nearest_node(*xy(98, 3), max_dist_m=10) == 1
    assert index.nearest_node(*xy(98, 3), max_dist_m=2) is None
    assert index.nearest_node(*xy(299, 301), max_dist_m=10) == 15
    nodes = mapmatch.NodeIndex(graph)
    assert nodes.nearest_node(*xy(0, 0), max_dist_m=1) == 0


def test_node_tie_goes_to_lowest_id(graph):
    assert mapmatch.NodeIndex(graph).nearest_node(*xy(50, 0), max_dist_m=60) == 0
