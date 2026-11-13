import json

import pytest

from geo_fixtures import add_fixes, add_obs, add_session, edge_point, make_grid, path_points, xy
from roadsense.config import IngestConfig, MappingConfig
from roadsense.geo import segments
from roadsense.geo.segments import merge_intervals, rebuild_map
from roadsense.storage import db


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "map.sqlite")
    db.init_schema(c)
    yield c
    c.close()


@pytest.fixture(scope="module")
def graph():
    return make_grid()


MAPPING = MappingConfig()  # dedup 8 m, snap 30 m
INGEST = IngestConfig()  # gap 3 s


def build(conn, graph, **kw):
    return rebuild_map(conn, graph, MAPPING, INGEST, **kw)


def pothole_rows(conn):
    return [dict(r) for r in conn.execute("SELECT * FROM potholes ORDER BY id")]


def coverage(conn, edge, session=None):
    sql = "SELECT start_m, end_m FROM edge_coverage WHERE edge_u=? AND edge_v=? AND edge_key=?"
    args = list(edge)
    if session:
        sql += " AND session_id=?"
        args.append(session)
    return [(r[0], r[1]) for r in conn.execute(sql + " ORDER BY start_m", args)]


def stats(conn, edge):
    r = conn.execute(
        "SELECT * FROM edge_stats WHERE edge_u=? AND edge_v=? AND edge_key=?", edge
    ).fetchone()
    return None if r is None else dict(r)


# --------------------------------------------------------------------------- potholes


def test_duplicate_sightings_across_sessions_become_one_pothole(conn, graph):
    add_session(conn, "s1")
    add_session(conn, "s2")
    p1, p2 = edge_point(graph, 0, 1, 0, 40), edge_point(graph, 0, 1, 0, 43)
    o1 = add_obs(conn, "s1", *p1, track=1, conf=0.6)
    o2 = add_obs(conn, "s2", *p2, track=1, conf=0.8)
    summary = build(conn, graph)

    rows = pothole_rows(conn)
    assert len(rows) == 1
    p = rows[0]
    assert p["n_observations"] == 2 and p["n_sessions"] == 2
    assert p["confidence"] == 0.8
    assert p["gps_source"] == "recorded"
    assert (p["edge_u"], p["edge_v"], p["edge_key"]) == (0, 1, 0)
    assert p["edge_offset_m"] == pytest.approx(41.5, abs=0.3)
    assert p["lat"] == pytest.approx((p1[0] + p2[0]) / 2)
    assert p["first_seen_utc"] == "2026-01-01T10:00:01+00:00"
    assert p["last_seen_utc"] == "2026-01-01T10:00:01+00:00"
    # both original observations are kept, linked, and carry the edge columns
    obs = conn.execute("SELECT id, pothole_id, edge_u, edge_v, edge_key, edge_offset_m, cam_lat"
                       " FROM observations ORDER BY id").fetchall()  # fmt: skip
    assert [o["id"] for o in obs] == [o1, o2]
    assert all(o["pothole_id"] == p["id"] for o in obs)
    assert all((o["edge_u"], o["edge_v"], o["edge_key"]) == (0, 1, 0) for o in obs)
    assert all(o["edge_offset_m"] == pytest.approx(p["edge_offset_m"]) for o in obs)
    assert [o["cam_lat"] for o in obs] == [p1[0], p2[0]]  # raw data untouched
    assert summary["n_potholes"] == 1 and summary["n_potholes_on_road"] == 1
    assert summary["n_observations"] == 2


def test_far_apart_sightings_stay_separate(conn, graph):
    add_session(conn, "s1")
    add_obs(conn, "s1", *edge_point(graph, 0, 1, 0, 20), track=1)
    add_obs(conn, "s1", *edge_point(graph, 0, 1, 0, 80), track=2)
    summary = build(conn, graph)
    rows = pothole_rows(conn)
    assert len(rows) == 2 and summary["n_potholes"] == 2
    assert [r["n_sessions"] for r in rows] == [1, 1]
    assert stats(conn, (0, 1, 0))["n_potholes"] == 2


def test_observations_without_position_are_kept_unlinked(conn, graph):
    add_session(conn, "s1")
    add_obs(conn, "s1", None, None, track=1)
    add_obs(conn, "s1", 51.5, None, track=2)
    add_obs(conn, "s1", *edge_point(graph, 0, 1, 0, 50), track=3)
    summary = build(conn, graph)
    assert summary["n_observations_without_position"] == 2
    assert summary["n_observations"] == 1 and summary["n_potholes"] == 1
    rows = conn.execute("SELECT track_id, pothole_id, edge_u, edge_offset_m FROM observations"
                        " ORDER BY track_id").fetchall()  # fmt: skip
    assert [(r["pothole_id"] is None, r["edge_u"] is None) for r in rows] == [
        (True, True), (True, True), (False, False),
    ]  # fmt: skip
    assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 3


def test_gps_source_labelling(conn, graph):
    add_session(conn, "rec1")
    add_session(conn, "rec2")
    add_session(conn, "syn1", gps_source="synthetic")
    add_session(conn, "syn2", gps_source="synthetic")
    spots = [10, 40, 70]
    pairs = [("rec1", "rec2"), ("syn1", "syn2"), ("rec1", "syn1")]
    for i, ((a, b), off) in enumerate(zip(pairs, spots)):
        add_obs(conn, a, *edge_point(graph, 0, 1, 0, off), track=10 + i)
        add_obs(conn, b, *edge_point(graph, 0, 1, 0, off + 1), track=20 + i)
    summary = build(conn, graph)
    by_offset = sorted(pothole_rows(conn), key=lambda r: r["edge_offset_m"])
    # a recorded and a synthetic sighting 1 m apart are two potholes, never one "mixed" pothole
    assert [r["gps_source"] for r in by_offset] == ["recorded", "synthetic", "recorded", "synthetic"]
    assert [r["n_sessions"] for r in by_offset] == [2, 2, 1, 1]
    assert [r["n_observations"] for r in by_offset] == [2, 2, 1, 1]
    assert summary["has_synthetic"] is True
    assert summary["n_potholes"] == 4
    assert summary["n_potholes_recorded"] == 2 and summary["n_potholes_synthetic"] == 2


def test_include_synthetic_false_ignores_synthetic_sessions(conn, graph):
    add_session(conn, "rec")
    add_session(conn, "syn", gps_source="synthetic")
    add_obs(conn, "rec", *edge_point(graph, 0, 1, 0, 20), track=1)
    add_obs(conn, "syn", *edge_point(graph, 0, 1, 0, 70), track=1)
    summary = build(conn, graph, include_synthetic=False)
    assert summary["n_sessions"] == 1 and summary["n_potholes"] == 1
    assert summary["has_synthetic"] is False
    assert pothole_rows(conn)[0]["gps_source"] == "recorded"
    syn_obs = conn.execute("SELECT pothole_id FROM observations WHERE session_id='syn'").fetchone()
    assert syn_obs["pothole_id"] is None
    params = json.loads(conn.execute("SELECT params_json FROM builds").fetchone()[0])
    assert params["include_synthetic"] is False and params["session_ids"] == ["rec"]
    assert build(conn, graph)["n_potholes"] == 2  # default includes synthetic


def test_ignored_sessions(conn, graph):
    here = edge_point(graph, 0, 1, 0, 50)
    add_session(conn, "excluded", include_in_map=0)
    add_session(conn, "failed", job_status="failed")
    add_session(conn, "running", job_status="running")
    add_session(conn, "nojob", job_status=None)
    add_session(conn, "inactive", active=False)  # done job but not the active one
    add_session(conn, "good")
    for i, sid in enumerate(["excluded", "failed", "running", "inactive", "good"]):
        add_obs(conn, sid, *here, track=i)
    # an observation from a superseded job of the good session must not count
    conn.execute("INSERT INTO jobs (id, session_id, status, created_at, updated_at)"
                 " VALUES ('old', 'good', 'done', 't', 't')")  # fmt: skip
    add_obs(conn, "good", *edge_point(graph, 2, 3, 0, 50), track=99, job_id="old")
    summary = build(conn, graph)
    assert summary["n_sessions"] == 1
    assert summary["n_observations"] == 1 and summary["n_potholes"] == 1
    assert pothole_rows(conn)[0]["n_sessions"] == 1
    linked = conn.execute("SELECT session_id FROM observations WHERE pothole_id IS NOT NULL").fetchall()
    assert [r[0] for r in linked] == ["good"]


def test_pothole_far_from_any_road_has_no_edge(conn, graph):
    add_session(conn, "s1")
    add_obs(conn, "s1", *xy(150, 400), track=1)
    summary = build(conn, graph)
    p = pothole_rows(conn)[0]
    assert p["edge_u"] is None and p["edge_offset_m"] is None
    assert summary["n_potholes_on_road"] == 0 and summary["n_potholes_off_road"] == 1
    o = conn.execute("SELECT pothole_id, edge_u FROM observations").fetchone()
    assert o["pothole_id"] == p["id"] and o["edge_u"] is None
    assert conn.execute("SELECT COUNT(*) FROM edge_stats").fetchone()[0] == 0


def test_build_without_graph(conn):
    add_session(conn, "s1")
    add_session(conn, "s2")
    add_fixes(conn, "s1", [xy(0, 0), xy(10, 0)])
    add_obs(conn, "s1", *xy(50, 0), track=1)
    add_obs(conn, "s2", *xy(52, 0), track=1)
    add_obs(conn, "s2", *xy(500, 0), track=2)
    summary = rebuild_map(conn, None, MAPPING, INGEST)
    assert summary["graph_sha256"] is None
    assert summary["n_potholes"] == 2 and summary["n_potholes_on_road"] == 0
    assert summary["n_potholes_off_road"] == 2 and summary["surveyed_m"] == 0
    rows = pothole_rows(conn)
    assert rows[0]["n_sessions"] == 2 and rows[0]["edge_u"] is None
    assert conn.execute("SELECT graph_sha256 FROM builds").fetchone()[0] is None
    assert conn.execute("SELECT COUNT(*) FROM edge_coverage").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM edge_stats").fetchone()[0] == 0


def test_session_started_utc_unknown_gives_null_times(conn, graph):
    add_session(conn, "s1", started_utc=None)
    add_obs(conn, "s1", *edge_point(graph, 0, 1, 0, 50), track=1)
    build(conn, graph)
    p = pothole_rows(conn)[0]
    assert p["first_seen_utc"] is None and p["last_seen_utc"] is None


def test_first_and_last_seen_span_sessions(conn, graph):
    add_session(conn, "s1", started_utc="2026-01-01T10:00:00Z")
    add_session(conn, "s2", started_utc="2026-02-01T10:00:00+00:00")
    add_obs(conn, "s1", *edge_point(graph, 0, 1, 0, 50), track=1, t=5.0)
    add_obs(conn, "s2", *edge_point(graph, 0, 1, 0, 51), track=1, t=7.0)
    build(conn, graph)
    p = pothole_rows(conn)[0]
    assert p["first_seen_utc"] == "2026-01-01T10:00:05+00:00"
    assert p["last_seen_utc"] == "2026-02-01T10:00:07+00:00"


# --------------------------------------------------------------------------- rebuild semantics


def test_rebuild_is_idempotent_and_replaces_previous_derived_rows(conn, graph):
    add_session(conn, "s1")
    add_fixes(conn, "s1", path_points(graph, [0, 1, 2]))
    add_obs(conn, "s1", *edge_point(graph, 0, 1, 0, 30), track=1)
    add_obs(conn, "s1", *edge_point(graph, 1, 2, 0, 60), track=2)
    # stale derived rows from an imaginary earlier build
    conn.execute("INSERT INTO builds (id, created_at, graph_sha256, params_json) VALUES (99,'t','x','{}')")
    conn.execute(
        "INSERT INTO potholes (id, build_id, lat, lon, n_observations, n_sessions, confidence,"
        " gps_source) VALUES (500, 99, 1, 1, 1, 1, 0.5, 'recorded')"
    )
    conn.execute("UPDATE observations SET pothole_id=500, edge_u=7, edge_v=8, edge_key=0 WHERE track_id=1")
    conn.execute("INSERT INTO edge_coverage (session_id, job_id, edge_u, edge_v, edge_key, start_m,"
                 " end_m) VALUES ('s1','job-s1',7,8,0,0,1)")  # fmt: skip
    conn.execute("INSERT INTO edge_stats VALUES (7,8,0,99,1,1,1,1)")
    conn.commit()

    first = build(conn, graph)

    def snapshot():
        tables = {
            "potholes": "SELECT id, edge_u, edge_v, edge_key, edge_offset_m, lat, lon,"
            " n_observations, n_sessions, confidence, gps_source FROM potholes ORDER BY id",
            "cov": "SELECT session_id, job_id, edge_u, edge_v, edge_key, start_m, end_m"
            " FROM edge_coverage ORDER BY edge_u, edge_v, start_m",
            "stats": "SELECT edge_u, edge_v, edge_key, length_m, surveyed_m, n_potholes,"
            " rate_per_100m FROM edge_stats ORDER BY edge_u, edge_v, edge_key",
            "obs": "SELECT id, pothole_id, edge_u, edge_v, edge_key, edge_offset_m FROM observations"
            " ORDER BY id",
        }
        return {k: [tuple(r) for r in conn.execute(q)] for k, q in tables.items()}

    snap1 = snapshot()
    assert conn.execute("SELECT COUNT(*) FROM builds").fetchone()[0] == 1
    assert conn.execute("SELECT id FROM builds").fetchone()[0] == first["build_id"]
    assert all(r[0] != 500 for r in snap1["potholes"]) and len(snap1["potholes"]) == 2
    assert all(r[2] != 7 for r in snap1["cov"]) and all(r[0] != 7 for r in snap1["stats"])
    second = build(conn, graph)
    assert snapshot() == snap1
    assert {k: v for k, v in second.items() if k != "build_id"} == {
        k: v for k, v in first.items() if k != "build_id"
    }
    assert conn.execute("SELECT COUNT(*) FROM builds").fetchone()[0] == 1


def test_rebuild_after_session_removed_from_map_drops_its_potholes(conn, graph):
    add_session(conn, "s1")
    add_obs(conn, "s1", *edge_point(graph, 0, 1, 0, 50), track=1)
    assert build(conn, graph)["n_potholes"] == 1
    conn.execute("UPDATE sessions SET include_in_map=0 WHERE id='s1'")
    summary = build(conn, graph)
    assert summary["n_potholes"] == 0 and summary["n_sessions"] == 0
    assert conn.execute("SELECT pothole_id FROM observations").fetchone()[0] is None


def test_failure_rolls_back_everything(conn, graph, monkeypatch):
    add_session(conn, "s1")
    add_obs(conn, "s1", *edge_point(graph, 0, 1, 0, 50), track=1)
    first = build(conn, graph)
    before = [tuple(r) for r in conn.execute("SELECT * FROM potholes")]

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    # fails after the old derived rows were already deleted inside the transaction
    monkeypatch.setattr(segments, "merge_intervals", boom)
    with pytest.raises(RuntimeError):
        build(conn, graph)
    assert conn.execute("SELECT id FROM builds").fetchone()[0] == first["build_id"]
    assert [tuple(r) for r in conn.execute("SELECT * FROM potholes")] == before
    assert conn.execute("SELECT pothole_id FROM observations").fetchone()[0] == before[0][0]
    assert not conn.in_transaction


def test_params_recorded(conn, graph):
    add_session(conn, "s1")
    build(conn, graph)
    row = conn.execute("SELECT graph_sha256, params_json FROM builds").fetchone()
    assert row["graph_sha256"] == graph.sha256
    params = json.loads(row["params_json"])
    assert params["mapping"] == {"dedup_radius_m": 8.0, "max_snap_m": 30.0}
    assert params["max_gps_gap_s"] == 3.0 and params["session_ids"] == ["s1"]
    assert params["jobs"] == {"s1": "job-s1"}  # which job of each session the build used


# --------------------------------------------------------------------------- coverage


def test_coverage_for_a_straight_drive(conn, graph):
    add_session(conn, "s1")
    add_fixes(conn, "s1", path_points(graph, [0, 1, 2]))
    summary = build(conn, graph)
    for edge in [(0, 1, 0), (1, 2, 0)]:
        assert coverage(conn, edge) == [(pytest.approx(0.0), pytest.approx(100.0, abs=0.3))]
        assert stats(conn, edge)["surveyed_m"] == pytest.approx(100.0, abs=0.3)
    assert summary["n_edges_with_coverage"] == 2
    assert summary["surveyed_m"] == pytest.approx(200.0, abs=0.6)
    job = conn.execute("SELECT DISTINCT job_id FROM edge_coverage").fetchall()
    assert [r[0] for r in job] == ["job-s1"]


@pytest.mark.parametrize("reverse", [False, True])
def test_coverage_across_a_junction(conn, graph, reverse):
    add_session(conn, "s1")
    a = edge_point(graph, 0, 1, 0, 80)
    b = edge_point(graph, 1, 2, 0, 20)
    add_fixes(conn, "s1", [b, a] if reverse else [a, b])
    build(conn, graph)
    # node 1 is the end of edge (0,1) and the start of edge (1,2); each covers up to the node
    assert coverage(conn, (0, 1, 0)) == [(pytest.approx(75.0, abs=0.3), pytest.approx(100.0, abs=0.3))]
    assert coverage(conn, (1, 2, 0)) == [(pytest.approx(0.0, abs=0.3), pytest.approx(25.0, abs=0.3))]


def test_fixes_on_unconnected_edges_cover_only_themselves(conn, graph):
    add_session(conn, "s1")
    add_fixes(conn, "s1", [edge_point(graph, 0, 1, 0, 50), edge_point(graph, 14, 15, 0, 50)])
    build(conn, graph)
    assert stats(conn, (0, 1, 0))["surveyed_m"] == pytest.approx(10.0, abs=0.1)
    assert stats(conn, (14, 15, 0))["surveyed_m"] == pytest.approx(10.0, abs=0.1)


def test_gps_gap_breaks_coverage_between_fixes(conn, graph):
    pts = [edge_point(graph, 0, 1, 0, 20), edge_point(graph, 0, 1, 0, 80)]
    add_session(conn, "close")
    add_fixes(conn, "close", pts, dt=1.0)
    add_session(conn, "gap")
    add_fixes(conn, "gap", pts, dt=10.0)
    build(conn, graph)
    assert coverage(conn, (0, 1, 0), "close") == [(pytest.approx(15, abs=0.3), pytest.approx(85, abs=0.3))]
    gap = coverage(conn, (0, 1, 0), "gap")
    assert len(gap) == 2  # only the 5 m either side of each fix
    assert sum(b - a for a, b in gap) == pytest.approx(20.0, abs=0.3)


def test_gap_exactly_at_limit_still_connects(conn, graph):
    add_session(conn, "s1")
    add_fixes(conn, "s1", [edge_point(graph, 0, 1, 0, 20), edge_point(graph, 0, 1, 0, 60)], dt=3.0)
    build(conn, graph)
    assert len(coverage(conn, (0, 1, 0))) == 1


def test_coverage_clipped_to_edge_length(conn, graph):
    add_session(conn, "s1")
    length = graph.g.edges[9, 10, 0]["length_m"]  # curved, longer than 100 m
    add_fixes(conn, "s1", [edge_point(graph, 9, 10, 0, 2), edge_point(graph, 9, 10, 0, length - 2)], dt=10)
    build(conn, graph)
    cov = coverage(conn, (9, 10, 0))
    assert cov[0][0] == 0.0 and cov[0][1] == pytest.approx(7.0, abs=0.3)
    assert cov[-1][1] == pytest.approx(length, abs=1e-9) and cov[-1][0] == pytest.approx(length - 7, abs=0.3)
    assert all(0.0 <= a <= b <= length for a, b in cov)


def test_unused_fixes_do_not_cover(conn, graph):
    add_session(conn, "s1")
    add_fixes(conn, "s1", path_points(graph, [0, 1]), used=0)
    summary = build(conn, graph)
    assert summary["n_edges_with_coverage"] == 0


def test_fixes_far_from_roads_are_ignored(conn, graph):
    add_session(conn, "s1")
    add_fixes(conn, "s1", [xy(150, 600), xy(160, 600)])
    assert build(conn, graph)["n_edges_with_coverage"] == 0


def test_pothole_edge_always_has_coverage_around_it(conn, graph):
    add_session(conn, "s1")  # no GPS fixes at all
    add_obs(conn, "s1", *edge_point(graph, 0, 1, 0, 40), track=1)
    build(conn, graph)
    cov = coverage(conn, (0, 1, 0), "s1")
    assert cov == [(pytest.approx(35.0, abs=0.3), pytest.approx(45.0, abs=0.3))]
    st = stats(conn, (0, 1, 0))
    assert st["n_potholes"] == 1 and st["surveyed_m"] == pytest.approx(10.0, abs=0.3)
    assert st["rate_per_100m"] == pytest.approx(10.0, rel=0.05)


def test_pothole_coverage_is_added_for_each_observing_session(conn, graph):
    add_session(conn, "s1")
    add_session(conn, "s2")
    add_obs(conn, "s1", *edge_point(graph, 0, 1, 0, 40), track=1)
    add_obs(conn, "s2", *edge_point(graph, 0, 1, 0, 42), track=1)
    build(conn, graph)
    assert len(coverage(conn, (0, 1, 0), "s1")) == 1 and len(coverage(conn, (0, 1, 0), "s2")) == 1
    assert stats(conn, (0, 1, 0))["surveyed_m"] < 15  # union, not the sum


# --------------------------------------------------------------------------- edge_stats


def test_edge_stats_union_across_sessions_and_rates(conn, graph):
    add_session(conn, "s1")
    add_session(conn, "s2")
    add_fixes(conn, "s1", path_points(graph, [0, 1]))  # full edge 0-1
    add_fixes(conn, "s2", path_points(graph, [1, 0]))  # same edge, other way
    add_obs(conn, "s1", *edge_point(graph, 0, 1, 0, 25), track=1)
    add_obs(conn, "s2", *edge_point(graph, 0, 1, 0, 75), track=1)
    build(conn, graph)
    st = stats(conn, (0, 1, 0))
    assert st["length_m"] == pytest.approx(100.0, abs=0.2)
    assert st["surveyed_m"] == pytest.approx(100.0, abs=0.5)  # union, not 200
    assert st["n_potholes"] == 2
    assert st["rate_per_100m"] == pytest.approx(2.0, rel=0.01)
    assert stats(conn, (4, 5, 0)) is None  # untouched edges have no row


def test_edge_with_coverage_but_no_potholes_has_zero_rate(conn, graph):
    add_session(conn, "s1")
    add_fixes(conn, "s1", path_points(graph, [2, 3]))
    build(conn, graph)
    st = stats(conn, (2, 3, 0))
    assert st["n_potholes"] == 0 and st["rate_per_100m"] == 0.0 and st["surveyed_m"] > 90


def test_edge_stats_use_canonical_ids_and_both_travel_directions(conn, graph):
    add_session(conn, "s1")
    add_fixes(conn, "s1", path_points(graph, [3, 2]))  # driven from the higher node id
    build(conn, graph)
    assert stats(conn, (2, 3, 0)) is not None
    assert conn.execute("SELECT COUNT(*) FROM edge_stats WHERE edge_u > edge_v").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM edge_coverage WHERE edge_u > edge_v").fetchone()[0] == 0


# --------------------------------------------------------------------------- helpers


def test_merge_intervals():
    assert merge_intervals([]) == []
    assert merge_intervals([(5, 8), (0, 2), (1, 3), (8, 9)]) == [(0, 3), (5, 9)]
    assert merge_intervals([(0, 10), (2, 3)]) == [(0, 10)]
