"""Regression tests: synthetic (demonstration) data never contaminates real-world numbers.

Covers the map build (``edge_stats`` from recorded sessions only, no mixed potholes),
the remembered demo-mode switch (off by default) and the router (demo mode off is
indistinguishable from a database without synthetic sessions; demo mode on flags
the whole plan, not only routes that pass a demo pothole).
"""

import pytest
from api_fixtures import make_api_cfg

from geo_fixtures import add_fixes, add_obs, add_session, edge_point, make_grid, path_points
from roadsense import mapstate
from roadsense.config import IngestConfig, MappingConfig, RoutingConfig
from roadsense.geo.segments import rebuild_map
from roadsense.routing import router as R
from roadsense.routing.router import plan_routes
from roadsense.storage import db

MAPPING, INGEST = MappingConfig(), IngestConfig()

STATS_SQL = (
    "SELECT edge_u, edge_v, edge_key, length_m, surveyed_m, n_potholes, rate_per_100m"
    " FROM edge_stats ORDER BY edge_u, edge_v, edge_key"
)


@pytest.fixture(scope="module")
def graph():
    return make_grid(parallel=())


@pytest.fixture
def new_db(tmp_path):
    made = []

    def make(name="d"):
        conn = db.connect(tmp_path / f"{name}.sqlite")
        db.init_schema(conn)
        made.append(conn)
        return conn

    yield make
    for c in made:
        c.close()


def add_recorded(conn, graph, sid="rec"):
    add_session(conn, sid)
    add_fixes(conn, sid, path_points(graph, [0, 1, 2]))
    add_obs(conn, sid, *edge_point(graph, 0, 1, 0, 40), track=1)


def add_synthetic(conn, graph, nodes, pothole, sid="syn"):
    """A synthetic session driving along ``nodes`` with one pothole at ``pothole`` = (edge, offset)."""
    add_session(conn, sid, gps_source="synthetic")
    add_fixes(conn, sid, path_points(graph, nodes))
    (u, v, k), off = pothole
    add_obs(conn, sid, *edge_point(graph, u, v, k, off), track=1)


def stats_rows(conn):
    return [tuple(r) for r in conn.execute(STATS_SQL)]


# --------------------------------------------------------------------------- map build


def test_edge_stats_come_from_recorded_sessions_only(new_db, graph):
    only = new_db("only")
    add_recorded(only, graph)
    s_only = rebuild_map(only, graph, MAPPING, INGEST)

    both = new_db("both")
    add_recorded(both, graph)
    # the synthetic drive overlaps the recorded one and goes on to a road nobody recorded
    add_synthetic(both, graph, [0, 1, 2, 3], ((0, 1, 0), 42))
    s_both = rebuild_map(both, graph, MAPPING, INGEST, include_synthetic=True)

    assert stats_rows(both) == stats_rows(only)
    assert not any(row[:3] == (2, 3, 0) for row in stats_rows(both))
    assert s_both["surveyed_m"] == pytest.approx(s_only["surveyed_m"])
    assert s_both["n_edges_with_coverage"] == s_only["n_edges_with_coverage"]
    assert s_only["demo_surveyed_m"] == 0 and s_only["has_synthetic"] is False
    # the synthetic session still has coverage rows (and its own length), kept apart
    assert both.execute("SELECT COUNT(*) FROM edge_coverage WHERE session_id = 'syn'").fetchone()[0] > 0
    assert both.execute(
        "SELECT COUNT(*) FROM edge_coverage WHERE session_id = 'syn' AND edge_u = 2 AND edge_v = 3"
    ).fetchone()[0] == 1
    assert s_both["demo_surveyed_m"] == pytest.approx(300.0, abs=2.0)
    assert s_both["n_potholes_recorded"] == s_only["n_potholes"] == 1
    assert s_both["n_potholes_synthetic"] == 1 and s_both["n_potholes"] == 2


@pytest.mark.parametrize("ids", [("a-rec", "b-syn"), ("a-syn", "b-rec")])
def test_a_recorded_and_a_synthetic_sighting_within_the_dedup_radius_are_two_potholes(
    new_db, graph, ids
):
    rec_id, syn_id = ids
    conn = new_db()
    add_session(conn, rec_id)
    add_session(conn, syn_id, gps_source="synthetic")
    add_obs(conn, rec_id, *edge_point(graph, 0, 1, 0, 40), track=1)
    add_obs(conn, syn_id, *edge_point(graph, 0, 1, 0, 41), track=1)  # 1 m away, radius is 8 m
    summary = rebuild_map(conn, graph, MAPPING, INGEST, include_synthetic=True)

    rows = [dict(r) for r in conn.execute("SELECT * FROM potholes ORDER BY id")]
    assert sorted(r["gps_source"] for r in rows) == ["recorded", "synthetic"]
    assert all(r["n_observations"] == 1 and r["n_sessions"] == 1 for r in rows)
    assert conn.execute("SELECT COUNT(*) FROM potholes WHERE gps_source = 'mixed'").fetchone()[0] == 0
    # each observation points at the pothole of its own kind
    for sid, source in ((rec_id, "recorded"), (syn_id, "synthetic")):
        (kind,) = conn.execute(
            "SELECT p.gps_source FROM observations o JOIN potholes p ON p.id = o.pothole_id"
            " WHERE o.session_id = ?",
            (sid,),
        ).fetchone()
        assert kind == source
    # only the recorded pothole counts towards the road statistics
    st = conn.execute("SELECT n_potholes FROM edge_stats WHERE edge_u = 0 AND edge_v = 1").fetchone()
    assert st["n_potholes"] == 1
    assert summary["n_potholes_recorded"] == 1 and summary["n_potholes_synthetic"] == 1


def test_synthetic_sessions_do_not_merge_into_recorded_potholes_across_sessions(new_db, graph):
    conn = new_db()
    add_session(conn, "r1")
    add_session(conn, "r2")
    add_session(conn, "s1", gps_source="synthetic")
    add_obs(conn, "r1", *edge_point(graph, 0, 1, 0, 40), track=1)
    add_obs(conn, "r2", *edge_point(graph, 0, 1, 0, 42), track=1)
    add_obs(conn, "s1", *edge_point(graph, 0, 1, 0, 41), track=1)
    rebuild_map(conn, graph, MAPPING, INGEST, include_synthetic=True)
    by_source = {
        r["gps_source"]: (r["n_observations"], r["n_sessions"])
        for r in conn.execute("SELECT * FROM potholes")
    }
    assert by_source == {"recorded": (2, 2), "synthetic": (1, 1)}


# --------------------------------------------------------------------------- demo-mode switch


def test_demo_mode_is_off_for_a_fresh_database_and_is_remembered(tmp_path, new_db, graph):
    cfg = make_api_cfg(tmp_path)
    conn = new_db()
    assert mapstate.get_include_synthetic(conn) is False
    add_recorded(conn, graph)
    add_synthetic(conn, graph, [12, 13, 14, 15], ((13, 14, 0), 50))

    summary = mapstate.rebuild_current(conn, cfg, graph)  # nothing remembered: demo mode off
    assert summary["n_sessions"] == 1 and summary["n_potholes_synthetic"] == 0
    assert summary["has_synthetic"] is False

    summary = mapstate.rebuild_current(conn, cfg, graph, include_synthetic=True)
    assert summary["n_sessions"] == 2 and summary["has_synthetic"] is True
    assert mapstate.get_include_synthetic(conn) is True
    assert mapstate.rebuild_current(conn, cfg, graph)["n_sessions"] == 2  # remembered

    mapstate.rebuild_current(conn, cfg, graph, include_synthetic=False)
    assert mapstate.rebuild_current(conn, cfg, graph)["n_sessions"] == 1
    conn.rollback()


# --------------------------------------------------------------------------- router


def _two_databases(tmp_path, new_db, graph):
    """Database A: one recorded session. Database B: the same plus a synthetic session far away."""
    cfg = make_api_cfg(tmp_path)
    a, b = new_db("a"), new_db("b")
    for conn in (a, b):
        add_recorded(conn, graph)
    add_synthetic(b, graph, [12, 13, 14, 15], ((13, 14, 0), 50))
    return cfg, a, b


def _plan(conn, graph, a, b, avoidance=0.5):
    from geo_fixtures import xy

    def node(n):
        return xy((n % 4) * 100.0, (n // 4) * 100.0)

    return plan_routes(conn, graph, node(a), node(b), avoidance, RoutingConfig(), MAPPING)


def test_with_demo_mode_off_the_router_ignores_the_synthetic_session_completely(
    tmp_path, new_db, graph
):
    cfg, a, b = _two_databases(tmp_path, new_db, graph)
    mapstate.rebuild_current(a, cfg, graph)
    mapstate.rebuild_current(b, cfg, graph)  # default: demo mode off
    a.commit()
    b.commit()
    for avoidance in (0.0, 0.5, 1.0):
        plan_a, plan_b = _plan(a, graph, 0, 3, avoidance), _plan(b, graph, 0, 3, avoidance)
        assert plan_b == plan_a
    plan_b = _plan(b, graph, 0, 3)
    plan_a = _plan(a, graph, 0, 3)
    assert plan_b["cost_model"]["prior_rate_per_100m"] == plan_a["cost_model"]["prior_rate_per_100m"]
    assert plan_b["cost_model"]["surveyed_total_m"] == plan_a["cost_model"]["surveyed_total_m"]
    assert plan_b["demo_mode"] is False and plan_b["uses_synthetic_data"] is False
    assert plan_b["caveats"] == R.CAVEATS
    # even a route over the synthetic road sees nothing of it
    over_a, over_b = _plan(a, graph, 12, 15), _plan(b, graph, 12, 15)
    assert over_b == over_a
    for route in over_b["routes"]:
        assert [s["status"] for s in route["sections"]] == ["unsurveyed"]
        assert route["demo_m"] == 0 and route["n_potholes_demo"] == 0


def test_demo_mode_on_flags_every_plan_even_a_route_that_passes_no_synthetic_pothole(
    tmp_path, new_db, graph
):
    cfg, a, b = _two_databases(tmp_path, new_db, graph)
    mapstate.rebuild_current(a, cfg, graph)
    mapstate.rebuild_current(b, cfg, graph, include_synthetic=True)
    a.commit()
    b.commit()

    plan = _plan(b, graph, 0, 3)  # the synthetic pothole is on the bottom row, far from this route
    assert plan["demo_mode"] is True and plan["uses_synthetic_data"] is True
    assert plan["caveats"][0] == R.SYNTHETIC_CAVEAT and plan["caveats"][1:] == R.CAVEATS
    assert all(r["n_potholes_demo"] == 0 and r["demo_m"] == 0 for r in plan["routes"])
    assert plan["cost_model"]["surveyed_total_m"] > _plan(a, graph, 0, 3)["cost_model"]["surveyed_total_m"]
    assert _plan(a, graph, 0, 3)["demo_mode"] is False  # the database without it is unaffected

    # a route along the synthetic drive: its sections are "demo", not "surveyed"
    over = _plan(b, graph, 12, 15, avoidance=0.0)
    shortest = next(r for r in over["routes"] if "Shortest" in r["labels"])
    assert over["demo_mode"] is True and over["caveats"][0] == R.SYNTHETIC_CAVEAT
    assert shortest["surveyed_m"] == 0 and shortest["surveyed_fraction"] == 0
    assert shortest["demo_m"] == pytest.approx(shortest["distance_m"], abs=1.0)
    assert shortest["unsurveyed_m"] == pytest.approx(0.0, abs=1.0)
    assert shortest["n_potholes_demo"] == 1 and shortest["n_potholes_observed"] == 1
    assert {s["status"] for s in shortest["sections"]} == {"demo"}
    assert all(s["surveyed"] is False for s in shortest["sections"])
    assert shortest["evidence"] == "insufficient"  # demonstration data is not a survey
    assert "demonstration" in shortest["explanation"]


def test_recorded_survey_wins_over_overlapping_demo_coverage_in_a_route(tmp_path, new_db, graph):
    cfg = make_api_cfg(tmp_path)
    conn = new_db()
    add_recorded(conn, graph)  # recorded over nodes 0-1-2
    add_synthetic(conn, graph, [0, 1, 2, 3], ((2, 3, 0), 50))
    mapstate.rebuild_current(conn, cfg, graph, include_synthetic=True)
    conn.commit()
    plan = _plan(conn, graph, 0, 3, avoidance=0.0)
    route = next(r for r in plan["routes"] if "Shortest" in r["labels"])
    # 0 -> 2 is recorded (about 200 m), 2 -> 3 is demonstration only; nothing is counted twice
    assert route["surveyed_m"] == pytest.approx(200.0, abs=3.0)
    assert route["demo_m"] == pytest.approx(100.0, abs=3.0)
    assert route["surveyed_m"] + route["demo_m"] + route["unsurveyed_m"] == pytest.approx(
        route["distance_m"], abs=0.2
    )
    assert route["surveyed_m"] + route["demo_m"] <= route["distance_m"] + 0.2
    statuses = [s["status"] for s in route["sections"]]
    assert statuses[0] == "surveyed" and statuses[-1] == "demo"


def test_an_off_road_synthetic_pothole_still_puts_the_plan_in_demo_mode(tmp_path, new_db, graph):
    from geo_fixtures import xy

    cfg = make_api_cfg(tmp_path)
    conn = new_db()
    add_recorded(conn, graph)
    add_session(conn, "syn", gps_source="synthetic")
    add_obs(conn, "syn", *xy(150, 600), track=1)  # nowhere near a road
    summary = mapstate.rebuild_current(conn, cfg, graph, include_synthetic=True)
    conn.commit()
    assert summary["n_potholes_synthetic"] == 1 and summary["n_potholes_on_road"] == 1
    plan = _plan(conn, graph, 0, 3)
    assert plan["demo_mode"] is True and plan["caveats"][0] == R.SYNTHETIC_CAVEAT
