import json

import pytest

from geo_fixtures import add_fixes, add_obs, add_session, edge_point, make_grid, path_points
from roadsense.config import IngestConfig, MappingConfig
from roadsense.geo.segments import rebuild_map
from roadsense.ingest.gps import haversine_m
from roadsense.storage import db, map_queries as mq


@pytest.fixture(scope="module")
def graph():
    return make_grid()


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "q.sqlite")
    db.init_schema(c)
    yield c
    c.close()


@pytest.fixture
def built(conn, graph):
    add_session(conn, "s1")
    add_session(conn, "s2", gps_source="synthetic")
    add_session(conn, "s3", job_status="failed")
    add_session(conn, "s4", job_status=None)
    add_fixes(conn, "s1", path_points(graph, [0, 1, 2]))
    add_obs(conn, "s1", *edge_point(graph, 0, 1, 0, 30), track=1)
    add_obs(conn, "s2", *edge_point(graph, 0, 1, 0, 32), track=1)
    add_obs(conn, "s1", *edge_point(graph, 1, 2, 0, 70), track=2)
    add_obs(conn, "s1", None, None, track=3)
    return rebuild_map(conn, graph, MappingConfig(), IngestConfig())


def test_empty_database(conn, graph):
    assert mq.list_potholes(conn) == [] and mq.list_coverage(conn, graph) == []
    assert mq.get_pothole(conn, 1) is None
    stats = mq.map_stats(conn)
    assert stats["latest_build"] is None and stats["n_potholes"] == 0
    assert stats["surveyed_km"] == 0.0 and stats["sessions_by_job_status"] == {}


def test_list_potholes_and_bbox(conn, built, graph):
    rows = mq.list_potholes(conn)
    # s1's two sightings are recorded; s2's sighting 2 m from the first is synthetic: a separate pothole
    assert [r["n_observations"] for r in rows] == [1, 1, 1]
    assert sorted(r["gps_source"] for r in rows) == ["recorded", "recorded", "synthetic"]
    json.dumps(rows)
    lat, lon = edge_point(graph, 0, 1, 0, 31)
    inside = (lat - 0.0001, lon - 0.0001, lat + 0.0001, lon + 0.0001)
    near = {r["id"] for r in rows if r["edge_u"] == 0}
    assert len(near) == 2 and {r["id"] for r in mq.list_potholes(conn, inside)} == near
    assert mq.list_potholes(conn, (0.0, 0.0, 1.0, 1.0)) == []


def test_get_pothole_includes_observations(conn, built):
    rows = mq.list_potholes(conn)
    recorded = next(r for r in rows if r["gps_source"] == "recorded" and r["edge_u"] == 0)
    synthetic = next(r for r in rows if r["gps_source"] == "synthetic")
    p = mq.get_pothole(conn, recorded["id"])
    assert p["n_observations"] == 1
    assert [(o["session_id"], o["session_gps_source"]) for o in p["observations"]] == [
        ("s1", "recorded"),
    ]  # fmt: skip
    assert {"track_id", "confidence", "t_video_s", "cam_lat", "image_path"} <= set(p["observations"][0])
    json.dumps(p)
    q = mq.get_pothole(conn, synthetic["id"])
    assert [(o["session_id"], o["session_gps_source"]) for o in q["observations"]] == [
        ("s2", "synthetic"),
    ]  # fmt: skip
    assert mq.get_pothole(conn, 99999) is None


def test_list_coverage_cuts_geometry(conn, built, graph):
    cov = mq.list_coverage(conn, graph)
    by_edge = {(c["edge_u"], c["edge_v"], c["edge_key"]): c for c in cov}
    assert set(by_edge) == {(0, 1, 0), (1, 2, 0)}
    c = by_edge[(0, 1, 0)]
    assert c["start_m"] == pytest.approx(0.0) and c["end_m"] == pytest.approx(100.0, abs=0.3)
    assert haversine_m(*c["coords"][0], graph.g.nodes[0]["lat"], graph.g.nodes[0]["lon"]) < 0.5
    assert haversine_m(*c["coords"][-1], graph.g.nodes[1]["lat"], graph.g.nodes[1]["lon"]) < 0.5
    json.dumps(cov)


def test_list_coverage_labels_the_source_and_never_counts_demo_over_recorded(conn, built, graph):
    # s2 (synthetic) has coverage only around its pothole, inside what s1 recorded
    assert {c["source"] for c in mq.list_coverage(conn, graph)} == {"recorded"}
    # a synthetic drive elsewhere shows up as its own kind
    add_session(conn, "s5", gps_source="synthetic")
    add_fixes(conn, "s5", [edge_point(graph, 12, 13, 0, 10), edge_point(graph, 12, 13, 0, 90)])
    rebuild_map(conn, graph, MappingConfig(), IngestConfig())
    cov = mq.list_coverage(conn, graph)
    assert {c["source"] for c in cov} == {"recorded", "synthetic"}
    demo = [c for c in cov if c["source"] == "synthetic"]
    assert {(c["edge_u"], c["edge_v"]) for c in demo} == {(12, 13)}
    totals = mq.coverage_totals(conn)
    assert totals["surveyed_m"] == pytest.approx(200.0, abs=0.6) and totals["edges"] == 2
    assert totals["demo_m"] == pytest.approx(90.0, abs=0.6) and totals["demo_edges"] == 1


def test_coverage_totals_of_an_empty_database(conn):
    assert mq.coverage_totals(conn) == {"surveyed_m": 0.0, "edges": 0, "demo_m": 0.0, "demo_edges": 0}


def test_subtract_intervals():
    sub = mq.subtract_intervals
    assert sub([(0, 10), (20, 30)], [(5, 25)]) == [(0, 5), (25, 30)]
    assert sub([(0, 10)], []) == [(0, 10)]
    assert sub([(0, 10)], [(0, 10)]) == []
    assert sub([(0, 10)], [(-5, 20)]) == []
    assert sub([(0, 10)], [(2, 3), (6, 7)]) == [(0, 2), (3, 6), (7, 10)]
    assert sub([(5, 10), (0, 6)], [(100, 200)]) == [(0, 10)]  # unsorted, overlapping input is merged
    assert sub([], [(0, 1)]) == []


def test_list_coverage_partial_stretch_and_bbox(conn, graph):
    add_session(conn, "s1")
    add_fixes(conn, "s1", [edge_point(graph, 0, 1, 0, 40), edge_point(graph, 0, 1, 0, 60)])
    rebuild_map(conn, graph, MappingConfig(), IngestConfig())
    (c,) = mq.list_coverage(conn, graph)
    assert c["start_m"] == pytest.approx(35.0, abs=0.3) and c["end_m"] == pytest.approx(65.0, abs=0.3)
    assert len(c["coords"]) == 2
    assert haversine_m(*c["coords"][0], *c["coords"][1]) == pytest.approx(30.0, abs=0.5)
    lat, lon = edge_point(graph, 14, 15, 0, 50)
    assert mq.list_coverage(conn, graph, (lat - 0.001, lon - 0.001, lat + 0.001, lon + 0.001)) == []
    lat, lon = edge_point(graph, 0, 1, 0, 50)
    assert len(mq.list_coverage(conn, graph, (lat - 0.001, lon - 0.001, lat + 0.001, lon + 0.001))) == 1


def test_list_coverage_needs_matching_graph(conn, built, graph):
    with pytest.raises(mq.MapQueryError, match="no road graph"):
        mq.list_coverage(conn, None)
    tiny = make_grid(rows=1, cols=2, oneway=(), curved=(), parallel=())
    with pytest.raises(mq.MapQueryError, match="different graph"):
        mq.list_coverage(conn, tiny)


def test_map_stats(conn, built):
    stats = mq.map_stats(conn)
    assert stats["sessions_by_job_status"] == {"done": 2, "failed": 1, "no_active_job": 1}
    assert stats["n_sessions"] == 4
    assert stats["n_observations"] == 4 and stats["n_observations_mapped"] == 3
    assert stats["n_potholes"] == 3
    assert stats["potholes_by_gps_source"] == {"recorded": 2, "synthetic": 1}
    assert stats["surveyed_km"] == pytest.approx(0.2, abs=0.002)
    assert stats["latest_build"]["id"] == built["build_id"]
    json.dumps(stats)


def test_works_with_default_row_factory(tmp_path, graph):
    c = db.connect(tmp_path / "plain.sqlite")
    db.init_schema(c)
    c.row_factory = None  # tuple rows
    add_session(c, "s1")
    add_obs(c, "s1", *edge_point(graph, 0, 1, 0, 30), track=1)
    rebuild_map(c, graph, MappingConfig(), IngestConfig())
    assert len(mq.list_potholes(c)) == 1 and mq.map_stats(c)["n_potholes"] == 1
    assert mq.get_pothole(c, mq.list_potholes(c)[0]["id"])["observations"]
    c.close()
