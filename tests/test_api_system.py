import hashlib
import logging

import pytest
from api_fixtures import grid_fixes, make_api_cfg, post_session, wait_job, write_gps_csv
from fastapi.testclient import TestClient
from geo_fixtures import make_grid
from pipeline_fixtures import FakeDetector

from roadsense import __version__
from roadsense.api.app import create_app
from roadsense.routing.graph import save_graph
from roadsense.storage import db


def code(response):
    return response.json()["error"]["code"]


def never_called():
    raise AssertionError("the detector must not be built by this request")


def test_health(make_client):
    assert make_client().get("/api/health").json() == {"status": "ok", "version": __version__}


def test_system_shape_and_does_not_load_the_model(make_client, tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not really a model")
    client = make_client(graph=make_grid(), detector_factory=never_called)
    s = client.get("/api/system").json()
    assert s["version"] == __version__ and s["include_synthetic"] is False  # demo mode starts off
    assert s["demo"] == {"mode": False, "sessions": 0, "active": False}
    assert s["model"] == {
        "weights": str(weights),
        "exists": True,
        "sha256": hashlib.sha256(b"not really a model").hexdigest(),
        "loaded": False,
        "device": None,
        "architecture": "YOLOv8s detection, single class 'pothole'",
    }
    assert s["graph"]["loaded"] is True and s["graph"]["n_nodes"] == 16
    assert s["graph"]["bbox"] and s["graph"]["sha256"] and s["graph"]["error"] is None
    assert set(s["settings"]) == {"detection", "tracking", "mapping", "ingest", "routing"}
    assert s["settings"]["detection"] == {
        "conf": 0.25,
        "iou": 0.5,
        "imgsz": 640,
        "partial_frame_error_fraction": 0.02,
        "max_frame_error_fraction": 0.25,
    }
    assert s["settings"]["tracking"]["frame_stride"] == 3
    assert s["settings"]["tracking"]["vehicle_frame_stride"] == 1
    assert s["settings"]["tracking"]["vehicle_min_hits"] == 6
    assert s["settings"]["routing"]["unsurveyed_rate_per_100m"] is None
    assert client.get("/api/stats").status_code == 200
    assert client.get("/api/routes/info").status_code == 200


def test_system_reports_missing_weights(make_client):
    s = make_client(detector_factory=never_called).get("/api/system").json()
    assert s["model"]["exists"] is False and s["model"]["sha256"] is None
    assert s["graph"] == {"loaded": False, "path": s["graph"]["path"], "error": None}


def test_system_reports_loaded_detector_after_a_job(make_client, one_pothole_video):
    client = make_client()
    assert client.get("/api/system").json()["model"]["loaded"] is False
    wait_job(client, post_session(client, one_pothole_video).json()["session"]["id"])
    model = client.get("/api/system").json()["model"]
    assert model["loaded"] is True and model["device"] == "cpu"  # FakeDetector.info()


def test_weights_hash_is_cached(make_client, tmp_path, monkeypatch):
    (tmp_path / "weights.pt").write_bytes(b"abc")
    client = make_client()
    first = client.get("/api/system").json()["model"]["sha256"]
    assert client.app.state.rs._weights_cache[1] == first

    def fail(*args, **kwargs):
        pytest.fail("hashed again")

    monkeypatch.setattr(hashlib, "sha256", fail)
    assert client.get("/api/system").json()["model"]["sha256"] == first


def test_stats_shape_and_counts(make_client, one_pothole_video, tmp_path):
    client = make_client(graph=make_grid())
    empty = client.get("/api/stats").json()
    assert empty["sessions"] == {
        "total": 0,
        "by_status": {"done": 0, "running": 0, "queued": 0, "failed": 0},
        "by_gps_source": {"recorded": 0, "synthetic": 0, "none": 0},
    }
    assert empty["potholes"]["total"] == 0 and empty["build"]["has_graph"] is True

    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    a = post_session(client, one_pothole_video, gps).json()["session"]["id"]
    gps_demo = write_gps_csv(tmp_path / "gd.csv", grid_fixes(y=100.0))  # another street
    b = post_session(client, one_pothole_video, gps_demo, gps_source="synthetic").json()["session"]["id"]
    c = post_session(client, one_pothole_video).json()["session"]["id"]
    for sid in (a, b, c):
        wait_job(client, sid)
    s = client.get("/api/stats").json()
    assert s["sessions"]["total"] == 3
    assert s["sessions"]["by_status"] == {"done": 3, "running": 0, "queued": 0, "failed": 0}
    assert s["sessions"]["by_gps_source"] == {"recorded": 1, "synthetic": 1, "none": 1}
    assert s["observations"] == {"total": 3, "with_position": 2, "without_position": 1}
    # demo mode is off: the synthetic session is stored but not on the map
    assert s["potholes"]["total"] == 1
    assert s["potholes"]["by_gps_source"] == {"recorded": 1, "synthetic": 0, "mixed": 0}
    assert s["potholes"]["on_road"] == 1 and s["potholes"]["off_road"] == 0
    assert s["coverage"]["edges"] >= 1 and s["coverage"]["surveyed_m"] > 0
    assert s["coverage"]["demo_m"] == 0 and s["coverage"]["demo_edges"] == 0
    assert s["demo"] == {"mode": False, "sessions": 1, "active": False, "potholes": 0}
    assert set(s["build"]) == {"id", "created_at", "has_graph", "include_synthetic"}

    # demo mode on: demonstration data is counted on its own, recorded numbers do not move
    client.post("/api/map/rebuild", json={"include_synthetic": True})
    d = client.get("/api/stats").json()
    assert d["potholes"]["by_gps_source"] == {"recorded": 1, "synthetic": 1, "mixed": 0}
    assert d["potholes"]["total"] == 2
    assert d["coverage"]["surveyed_m"] == s["coverage"]["surveyed_m"]
    assert d["coverage"]["edges"] == s["coverage"]["edges"]
    assert d["coverage"]["demo_m"] > 0 and d["coverage"]["demo_edges"] >= 1
    assert d["demo"] == {"mode": True, "sessions": 1, "active": True, "potholes": 1}


def test_unknown_api_paths_use_the_error_shape(make_client):
    client = make_client()
    for method in ("get", "post", "delete", "patch"):
        r = getattr(client, method)("/api/nothing/here")
        assert r.status_code == 404 and code(r) == "not_found"
        assert set(r.json()["error"]) == {"code", "message"}


def test_unexpected_exception_is_a_generic_500_and_is_logged(make_client, monkeypatch, caplog):
    from roadsense.api import system

    def boom(conn):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(system, "get_include_synthetic", boom)
    client = make_client(raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR):
        r = client.get("/api/system")
    assert r.status_code == 500 and code(r) == "internal"
    assert "secret" not in r.text
    assert any(
        rec.exc_info and "secret internal detail" in str(rec.exc_info[1]) for rec in caplog.records
    )


def test_static_index_is_served_at_root(make_client):
    client = make_client()
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()


def test_malformed_graph_file_is_reported_not_fatal(tmp_path):
    cfg = make_api_cfg(tmp_path)
    cfg.routing.graph_path.parent.mkdir(parents=True)
    cfg.routing.graph_path.write_text("{ definitely not a graph", encoding="utf-8")
    with TestClient(create_app(cfg, detector_factory=FakeDetector)) as client:
        g = client.get("/api/system").json()["graph"]
        assert g["loaded"] is False and "not valid JSON" in g["error"]
        assert client.get("/api/routes/info").json()["available"] is False
        r = client.post(
            "/api/routes",
            json={"origin": {"lat": 1, "lon": 1}, "destination": {"lat": 2, "lon": 2}},
        )
        assert r.status_code == 409 and code(r) == "no_graph"


def test_graph_is_loaded_from_the_configured_file(tmp_path):
    cfg = make_api_cfg(tmp_path)
    graph = make_grid()
    save_graph(graph, cfg.routing.graph_path)
    with TestClient(create_app(cfg, detector_factory=FakeDetector)) as client:
        g = client.get("/api/system").json()["graph"]
        assert g["loaded"] is True and g["n_nodes"] == 16 and g["error"] is None


def test_startup_rebuilds_the_map_when_the_graph_changed(tmp_path, one_pothole_video):
    cfg = make_api_cfg(tmp_path)
    grid = make_grid()
    other = make_grid(rows=3, cols=3)
    assert grid.sha256 != other.sha256
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    with TestClient(create_app(cfg, detector_factory=FakeDetector, graph=grid)) as client:
        wait_job(client, post_session(client, one_pothole_video, gps).json()["session"]["id"])

    def latest_sha():
        conn = db.connect(cfg.db_path)
        try:
            return conn.execute("SELECT graph_sha256 FROM builds ORDER BY id DESC").fetchone()[0]
        finally:
            conn.close()

    assert latest_sha() == grid.sha256
    with TestClient(create_app(cfg, detector_factory=FakeDetector, graph=other)) as client:
        assert latest_sha() == other.sha256
        assert client.get("/api/coverage").status_code == 200  # no graph mismatch error
    with TestClient(create_app(cfg, detector_factory=FakeDetector, graph=None)) as client:
        assert latest_sha() is None
        assert len(client.get("/api/potholes").json()["potholes"]) == 1


def test_startup_fails_jobs_left_over_from_a_previous_run(tmp_path, one_pothole_video):
    cfg = make_api_cfg(tmp_path)
    with TestClient(create_app(cfg, detector_factory=FakeDetector, graph=None)) as client:
        sid = post_session(client, one_pothole_video).json()["session"]["id"]
        wait_job(client, sid)
    conn = db.connect(cfg.db_path)
    conn.execute("UPDATE jobs SET status = 'running'")
    conn.commit()
    conn.close()
    with TestClient(create_app(cfg, detector_factory=FakeDetector, graph=None)) as client:
        job = client.get(f"/api/sessions/{sid}/status").json()["job"]
        assert job["status"] == "failed" and "interrupted" in job["error"]


def test_web_files_are_revalidated_but_api_responses_are_not_marked(make_client):
    client = make_client()
    assert client.get("/").headers.get("cache-control") == "no-cache"
    assert "cache-control" not in client.get("/api/health").headers
