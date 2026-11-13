import pytest
from api_fixtures import grid_fixes, post_session, wait_job, write_gps_csv
from geo_fixtures import make_grid, xy


@pytest.fixture
def mapped(make_client, one_pothole_video, tmp_path):
    """A client with the grid graph and one finished, mapped, recorded session."""
    client = make_client(graph=make_grid())
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    sid = post_session(client, one_pothole_video, gps).json()["session"]["id"]
    assert wait_job(client, sid)["status"] == "done"
    return client, sid


def test_potholes_list_shape(mapped):
    client, sid = mapped
    body = client.get("/api/potholes").json()
    assert set(body) == {"potholes", "note"} and "camera position" in body["note"]
    (p,) = body["potholes"]
    assert set(p) == {
        "id", "lat", "lon", "confidence", "n_observations", "n_sessions", "gps_source",
        "on_road", "first_seen_utc", "last_seen_utc",
    }  # fmt: skip
    assert p["gps_source"] == "recorded" and p["on_road"] is True
    assert p["n_observations"] == 1 and p["n_sessions"] == 1


def test_potholes_bbox_filter(mapped):
    client, _ = mapped
    (p,) = client.get("/api/potholes").json()["potholes"]
    inside = f"{p['lat'] - 0.001},{p['lon'] - 0.001},{p['lat'] + 0.001},{p['lon'] + 0.001}"
    assert len(client.get("/api/potholes", params={"bbox": inside}).json()["potholes"]) == 1
    outside = f"{p['lat'] + 1},{p['lon']},{p['lat'] + 2},{p['lon'] + 1}"
    assert client.get("/api/potholes", params={"bbox": outside}).json()["potholes"] == []


@pytest.mark.parametrize(
    "bbox", ["1,2,3", "a,b,c,d", "1,2,3,4,5", "5,5,1,1", "nan,0,1,1", "inf,0,1,1"]
)
def test_bad_bbox_is_422_invalid_input(mapped, bbox):
    client, _ = mapped
    for url in ("/api/potholes", "/api/coverage"):
        r = client.get(url, params={"bbox": bbox})
        assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"


def test_pothole_detail_has_observations_with_session(mapped):
    client, sid = mapped
    (p,) = client.get("/api/potholes").json()["potholes"]
    d = client.get(f"/api/potholes/{p['id']}").json()
    assert d["id"] == p["id"] and d["on_road"] is True
    (o,) = d["observations"]
    assert o["session_id"] == sid and o["session_name"]
    assert len(o["box"]) == 4 and o["position"] is not None and o["pothole_id"] == p["id"]
    assert o["image_url"].startswith("/api/artifacts/")
    assert client.get(o["image_url"]).status_code == 200


def test_unknown_pothole_is_404_and_non_integer_id_is_422(mapped):
    client, _ = mapped
    r = client.get("/api/potholes/99999")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = client.get("/api/potholes/abc")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"


def test_coverage_shape(mapped):
    client, _ = mapped
    body = client.get("/api/coverage").json()
    assert set(body) == {"stretches", "surveyed_m", "demo_stretches", "demo_m"}
    assert body["demo_stretches"] == [] and body["demo_m"] == 0
    # the video is 3 s long and the car drives 20 m/s: 60 m filmed plus 5 m either side of the
    # first and last fix. The GPS fixes logged before the first frame and after the last one
    # (the trace runs from -1 s to 4 s) survey nothing.
    assert body["stretches"] and body["surveyed_m"] == pytest.approx(70.0, abs=2.0)
    for stretch in body["stretches"]:
        assert len(stretch) >= 2 and all(len(pt) == 2 for pt in stretch)
    lat, lon = xy(100, 0)  # a stretch has a vertex at the graph node
    near = f"{lat - 0.0003},{lon - 0.0003},{lat + 0.0003},{lon + 0.0003}"
    assert client.get("/api/coverage", params={"bbox": near}).json()["stretches"]
    far = "10,10,11,11"
    assert client.get("/api/coverage", params={"bbox": far}).json() == {
        "stretches": [], "surveyed_m": 0, "demo_stretches": [], "demo_m": 0,
    }  # fmt: skip


def test_coverage_keeps_recorded_and_demo_lengths_apart(make_client, one_pothole_video, tmp_path):
    client = make_client(graph=make_grid())
    real_gps = write_gps_csv(tmp_path / "r.csv", grid_fixes())
    demo_gps = write_gps_csv(tmp_path / "d.csv", grid_fixes(y=100.0))  # another street
    wait_job(client, post_session(client, one_pothole_video, real_gps).json()["session"]["id"])
    wait_job(
        client,
        post_session(client, one_pothole_video, demo_gps, gps_source="synthetic", role="demo").json()[
            "session"
        ]["id"],
    )
    off = client.get("/api/coverage").json()  # demo mode is off: nothing synthetic on the map
    assert off["demo_stretches"] == [] and off["demo_m"] == 0
    assert off["surveyed_m"] == pytest.approx(70.0, abs=2.0)

    client.post("/api/map/rebuild", json={"include_synthetic": True})
    on = client.get("/api/coverage").json()
    assert on["surveyed_m"] == pytest.approx(off["surveyed_m"])  # recorded numbers do not move
    assert on["stretches"] == off["stretches"]
    assert on["demo_m"] == pytest.approx(70.0, abs=2.0) and on["demo_stretches"]
    far = "10,10,11,11"
    assert client.get("/api/coverage", params={"bbox": far}).json()["demo_stretches"] == []


def test_coverage_is_empty_without_a_graph(make_client, one_pothole_video, tmp_path):
    client = make_client(graph=None)
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    wait_job(client, post_session(client, one_pothole_video, gps).json()["session"]["id"])
    assert client.get("/api/coverage").json() == {
        "stretches": [], "surveyed_m": 0.0, "demo_stretches": [], "demo_m": 0.0,
    }  # fmt: skip
    (p,) = client.get("/api/potholes").json()["potholes"]
    assert p["on_road"] is False


def test_rebuild_with_include_synthetic_false_then_true_is_remembered(
    make_client, one_pothole_video, tmp_path
):
    client = make_client(graph=make_grid())
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    real = post_session(client, one_pothole_video, gps).json()["session"]["id"]
    wait_job(client, real)
    synth = post_session(
        client, one_pothole_video, gps, gps_source="synthetic", role="demo"
    ).json()["session"]["id"]
    wait_job(client, synth)
    # demo mode is off until someone turns it on: the synthetic session is processed but left out
    assert client.get("/api/system").json()["include_synthetic"] is False
    (p,) = client.get("/api/potholes").json()["potholes"]
    assert p["gps_source"] == "recorded" and p["n_sessions"] == 1

    # turning demo mode on: the same place gives two potholes, one per kind, never a mixed one
    r = client.post("/api/map/rebuild", json={"include_synthetic": True})
    assert r.status_code == 200 and isinstance(r.json(), dict)
    ps = client.get("/api/potholes").json()["potholes"]
    assert sorted(p["gps_source"] for p in ps) == ["recorded", "synthetic"]
    assert all(p["n_sessions"] == 1 for p in ps)
    assert client.get("/api/system").json()["include_synthetic"] is True
    assert client.get("/api/stats").json()["build"]["include_synthetic"] is True

    # the choice is remembered: an automatic rebuild (PATCH) and an empty-body rebuild keep it
    client.patch(f"/api/sessions/{real}", json={"include_in_map": True})
    assert len(client.get("/api/potholes").json()["potholes"]) == 2
    client.post("/api/map/rebuild")
    assert len(client.get("/api/potholes").json()["potholes"]) == 2
    assert client.get("/api/system").json()["include_synthetic"] is True

    client.post("/api/map/rebuild", json={"include_synthetic": False})
    (p,) = client.get("/api/potholes").json()["potholes"]
    assert p["gps_source"] == "recorded" and p["n_sessions"] == 1
    assert client.get("/api/system").json()["include_synthetic"] is False
    assert client.get("/api/stats").json()["build"]["include_synthetic"] is False
    client.post("/api/map/rebuild")
    assert len(client.get("/api/potholes").json()["potholes"]) == 1


def test_a_finished_job_rebuilds_with_the_remembered_choice(make_client, one_pothole_video, tmp_path):
    client = make_client(graph=make_grid())
    client.post("/api/map/rebuild", json={"include_synthetic": False})
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    sid = post_session(client, one_pothole_video, gps, gps_source="synthetic").json()["session"]["id"]
    wait_job(client, sid)
    assert client.get("/api/potholes").json()["potholes"] == []  # excluded, though processed
    client.post("/api/map/rebuild", json={"include_synthetic": True})
    assert len(client.get("/api/potholes").json()["potholes"]) == 1


def test_rebuild_rejects_a_bad_body(make_client):
    client = make_client()
    r = client.post("/api/map/rebuild", json={"include_synthetic": "maybe"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"
