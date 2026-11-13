import json

import pytest
from api_fixtures import grid_fixes, post_session, wait_job, write_gps_csv
from geo_fixtures import make_grid, xy

from roadsense.routing.graph import make_road_graph
from roadsense.storage import db

ORIGIN = dict(zip(("lat", "lon"), xy(0, 0)))
DEST = dict(zip(("lat", "lon"), xy(300, 0)))


def code(response):
    return response.json()["error"]["code"]


def isolated_node_graph():
    """The grid with node 15 cut off from every road."""
    grid = make_grid()
    g = grid.g.copy()
    g.remove_edges_from([(u, v, k) for u, v, k in list(g.edges(keys=True)) if 15 in (u, v)])
    return make_road_graph(g, {"source": "test"})


@pytest.fixture
def mapped(make_client, one_pothole_video, tmp_path):
    client = make_client(graph=make_grid())
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    wait_job(client, post_session(client, one_pothole_video, gps).json()["session"]["id"])
    return client


def test_routes_happy_path(mapped):
    r = mapped.post("/api/routes", json={"origin": ORIGIN, "destination": DEST, "avoidance": 0.5})
    assert r.status_code == 200
    plan = r.json()
    assert {"routes", "cost_model", "uses_synthetic_data", "caveats"} <= set(plan)
    assert 1 <= len(plan["routes"]) <= 3
    route = plan["routes"][0]
    for key in (
        "labels", "recommended", "distance_m", "duration_s", "n_potholes_observed", "potholes",
        "surveyed_m", "surveyed_fraction", "expected_potholes_unsurveyed",
        "expected_potholes_total", "geometry", "sections", "explanation",
    ):  # fmt: skip
        assert key in route, key
    assert route["distance_m"] == pytest.approx(300, abs=5)
    assert sum(1 for x in plan["routes"] if x["recommended"]) == 1
    assert plan["uses_synthetic_data"] is False


def test_routes_default_avoidance(mapped):
    r = mapped.post("/api/routes", json={"origin": ORIGIN, "destination": DEST})
    assert r.status_code == 200 and r.json()["avoidance"] == 0.5


def test_routes_no_graph_is_409(make_client):
    client = make_client(graph=None)
    r = client.post("/api/routes", json={"origin": ORIGIN, "destination": DEST, "avoidance": 0.5})
    assert r.status_code == 409 and code(r) == "no_graph"


def test_routes_outside_graph_is_422(mapped):
    far = {"lat": 0.0, "lon": 0.0}
    for body in (
        {"origin": far, "destination": DEST, "avoidance": 0.5},
        {"origin": ORIGIN, "destination": far, "avoidance": 0.5},
    ):
        r = mapped.post("/api/routes", json=body)
        assert r.status_code == 422 and code(r) == "outside_graph"


def test_routes_no_route_is_404(make_client):
    client = make_client(graph=isolated_node_graph())
    isolated = dict(zip(("lat", "lon"), xy(300, 300)))  # node 15
    r = client.post("/api/routes", json={"origin": ORIGIN, "destination": isolated, "avoidance": 0.5})
    assert r.status_code == 404 and code(r) == "no_route"


@pytest.mark.parametrize(
    "body",
    [
        {"origin": {"lat": 91, "lon": 0}, "destination": DEST, "avoidance": 0.5},
        {"origin": {"lat": 0, "lon": 181}, "destination": DEST, "avoidance": 0.5},
        {"origin": ORIGIN, "destination": DEST, "avoidance": 1.5},
        {"origin": ORIGIN, "destination": DEST, "avoidance": -0.1},
        {"origin": ORIGIN, "avoidance": 0.5},
        {"origin": ORIGIN, "destination": {"lat": "north", "lon": 1}, "avoidance": 0.5},
        {"origin": ORIGIN, "destination": DEST, "avoidance": "high"},
        {"origin": ORIGIN, "destination": DEST, "avoidance": 0.5, "surprise": True},
        [],
    ],
)
def test_routes_validation_is_422_invalid_input(mapped, body):
    r = mapped.post("/api/routes", json=body)
    assert r.status_code == 422 and code(r) == "invalid_input"
    assert r.json()["error"]["message"]


def test_routes_info_with_graph(mapped):
    info = mapped.get("/api/routes/info").json()
    assert info["available"] is True and info["graph"]["loaded"] is True
    assert info["graph"]["n_nodes"] == 16 and info["graph"]["attribution"]
    assert info["coverage"]["edges"] >= 1 and info["coverage"]["surveyed_m"] > 0
    assert info["potholes_on_road"] == 1
    assert info["cost_model"] == {
        "max_detour_per_pothole_m": 300.0,
        "assumed_rate_per_100m": 1.0,
        "prior_strength_m": 500.0,
        "unsurveyed_rate_per_100m": None,
        "min_surveyed_fraction": 0.2,
        "adequate_surveyed_fraction": 0.8,
    }
    assert info["demo"] == {"active": False, "demo_m": 0.0, "potholes_on_road": 0}


def test_routes_info_without_graph(make_client):
    info = make_client(graph=None).get("/api/routes/info").json()
    assert info["available"] is False and info["graph"]["loaded"] is False
    assert info["coverage"] == {"surveyed_m": 0.0, "edges": 0} and info["potholes_on_road"] == 0


def test_routes_recovers_when_the_map_was_built_with_another_graph(
    make_client, one_pothole_video, tmp_path
):
    client = make_client(graph=make_grid())
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    wait_job(client, post_session(client, one_pothole_video, gps).json()["session"]["id"])
    conn = db.connect(client.cfg.db_path)
    conn.execute("UPDATE builds SET graph_sha256 = 'stale'")
    conn.commit()
    conn.close()
    r = client.post("/api/routes", json={"origin": ORIGIN, "destination": DEST, "avoidance": 0.5})
    assert r.status_code == 200


def _write_example(client, session_id="x"):
    example = {
        "origin": {"lat": 1.0, "lon": 2.0},
        "destination": {"lat": 3.0, "lon": 4.0},
        "description": "demo",
        "synthetic": True,
        "session_id": session_id,
    }
    demo = client.cfg.data_dir / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "demo_route.json").write_text(json.dumps(example), encoding="utf-8")
    return example, demo / "demo_route.json"


@pytest.fixture
def demo_client(make_client, one_pothole_video, tmp_path):
    """A client with a processed synthetic session; demo mode still off."""
    client = make_client(graph=make_grid())
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    sid = post_session(client, one_pothole_video, gps, gps_source="synthetic", role="demo").json()[
        "session"
    ]["id"]
    wait_job(client, sid)
    client.sid = sid
    return client


def test_demo_route_example_absent_without_a_file(make_client):
    r = make_client().get("/api/demo/route-example")
    assert r.status_code == 404 and code(r) == "not_found"


def test_demo_route_example_unreadable_file_is_404(make_client):
    client = make_client()
    _, path = _write_example(client)
    path.write_text("{not json", encoding="utf-8")
    assert client.get("/api/demo/route-example").status_code == 404


def test_demo_route_example_is_served_while_its_session_is_on_the_map(demo_client):
    example, _ = _write_example(demo_client, demo_client.sid)
    demo_client.post("/api/map/rebuild", json={"include_synthetic": True})
    r = demo_client.get("/api/demo/route-example")
    assert r.status_code == 200 and r.json() == example


def test_demo_route_example_is_404_when_demo_mode_is_off(demo_client):
    _write_example(demo_client, demo_client.sid)
    r = demo_client.get("/api/demo/route-example")  # the session exists, demo mode is off
    assert r.status_code == 404 and code(r) == "not_found"
    demo_client.post("/api/map/rebuild", json={"include_synthetic": True})
    assert demo_client.get("/api/demo/route-example").status_code == 200
    demo_client.post("/api/map/rebuild", json={"include_synthetic": False})
    assert demo_client.get("/api/demo/route-example").status_code == 404


def test_demo_route_example_is_404_when_the_session_is_gone(demo_client):
    _write_example(demo_client, demo_client.sid)
    demo_client.post("/api/map/rebuild", json={"include_synthetic": True})
    assert demo_client.get("/api/demo/route-example").status_code == 200
    assert demo_client.delete(f"/api/sessions/{demo_client.sid}").status_code == 200
    assert demo_client.get("/api/demo/route-example").status_code == 404


def test_demo_route_example_is_404_when_the_session_is_not_synthetic_or_not_named(
    make_client, one_pothole_video, tmp_path
):
    client = make_client(graph=make_grid())
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    real = post_session(client, one_pothole_video, gps).json()["session"]["id"]
    wait_job(client, real)
    client.post("/api/map/rebuild", json={"include_synthetic": True})
    _write_example(client, real)  # a recorded session is never a demonstration
    assert client.get("/api/demo/route-example").status_code == 404
    _write_example(client, "no-such-session")
    assert client.get("/api/demo/route-example").status_code == 404
    _, path = _write_example(client)
    path.write_text(json.dumps({"origin": {}, "destination": {}}), encoding="utf-8")  # no session_id
    assert client.get("/api/demo/route-example").status_code == 404
    path.write_text("[1, 2]", encoding="utf-8")
    assert client.get("/api/demo/route-example").status_code == 404


def test_demo_route_example_is_404_when_the_session_is_excluded_from_the_map(demo_client):
    _write_example(demo_client, demo_client.sid)
    demo_client.post("/api/map/rebuild", json={"include_synthetic": True})
    demo_client.patch(f"/api/sessions/{demo_client.sid}", json={"include_in_map": False})
    assert demo_client.get("/api/demo/route-example").status_code == 404
