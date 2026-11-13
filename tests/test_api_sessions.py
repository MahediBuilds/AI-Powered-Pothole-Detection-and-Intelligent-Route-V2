import threading

import pytest
from api_fixtures import grid_fixes, post_session, wait_job, write_gps_csv
from pipeline_fixtures import FakeDetector

from roadsense.api import sessions as sessions_api
from roadsense.api.errors import ApiError


def error_of(response):
    body = response.json()
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message"}
    return body["error"]["code"], body["error"]["message"]


@pytest.fixture
def uploaded(make_client, one_pothole_video, tmp_path):
    """A client and a finished session with GPS."""
    client = make_client()
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    r = post_session(client, one_pothole_video, gps)
    sid = r.json()["session"]["id"]
    assert wait_job(client, sid)["status"] == "done"
    return client, sid


def test_upload_happy_path_to_done_with_report_and_detections(uploaded):
    client, sid = uploaded
    s = client.get(f"/api/sessions/{sid}").json()
    assert s["gps_source"] == "recorded" and s["role"] == "survey" and s["include_in_map"] is True
    assert s["gps"] == {"n_fixes": 6, "n_used": 6, "median_accuracy_m": 5.0}
    assert s["job"]["status"] == "done" and s["job"]["progress"] == 1.0
    assert (s["n_observations"], s["n_with_position"]) == (1, 1)
    report = s["report"]
    assert report["frames_total"] == 30 and report["n_tracks_confirmed"] == 1
    assert "warnings" in report and "detector" in report and report["gps"]["source"] == "recorded"

    det = client.get(f"/api/sessions/{sid}/detections").json()
    assert det["session_id"] == sid and det["gps_source"] == "recorded"
    assert det["job_id"] == s["job"]["id"]
    (o,) = det["observations"]
    assert set(o) == {
        "id", "track_id", "n_frames", "confidence", "first_t_video_s", "t_video_s", "frame_index",
        "box", "frame_w", "frame_h", "position", "pothole_id", "image_url",
    }  # fmt: skip
    assert len(o["box"]) == 4 and (o["frame_w"], o["frame_h"]) == (160, 120)
    assert set(o["position"]) == {"lat", "lon", "accuracy_m"}
    assert o["pothole_id"] is not None  # the build ran after the job

    status = client.get(f"/api/sessions/{sid}/status").json()
    assert status["session_id"] == sid and status["job"]["status"] == "done"
    listed = client.get("/api/sessions").json()["sessions"]
    assert [x["id"] for x in listed] == [sid]


def test_detection_image_is_served(uploaded):
    client, sid = uploaded
    url = client.get(f"/api/sessions/{sid}/detections").json()["observations"][0]["image_url"]
    assert url.startswith(f"/api/artifacts/{sid}/")
    r = client.get(url)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"


def test_track_endpoint_lists_gps_points(uploaded):
    client, sid = uploaded
    track = client.get(f"/api/sessions/{sid}/track").json()
    assert track["gps_source"] == "recorded" and len(track["points"]) == 6
    lat, lon, t, used = track["points"][0]
    assert t == pytest.approx(-1.0) and used is True


def test_upload_files_are_stored_under_the_session_folder(uploaded, one_pothole_video):
    client, sid = uploaded
    folder = client.cfg.uploads.dir / sid
    stem = one_pothole_video.stem
    assert {p.name for p in folder.iterdir()} == {f"{stem}.mp4", f"{stem}_gps.csv"}
    assert not [p for p in client.cfg.uploads.dir.iterdir() if p.name.startswith(".incoming")]


def test_gps_less_upload_has_source_none_and_no_positions(make_client, one_pothole_video):
    client = make_client()
    r = post_session(client, one_pothole_video, gps_source="synthetic")  # ignored without a GPS file
    sid = r.json()["session"]["id"]
    assert r.json()["session"]["gps_source"] == "none"
    assert r.json()["session"]["gps"]["n_fixes"] == 0
    assert wait_job(client, sid)["status"] == "done"
    s = client.get(f"/api/sessions/{sid}").json()
    assert (s["n_observations"], s["n_with_position"]) == (1, 0)
    (o,) = client.get(f"/api/sessions/{sid}/detections").json()["observations"]
    assert o["position"] is None
    assert any("no GPS" in w for w in s["report"]["warnings"])
    assert client.get("/api/potholes").json()["potholes"] == []
    assert client.get(f"/api/sessions/{sid}/track").json()["points"] == []


def test_synthetic_flag_is_stored_and_reported(make_client, one_pothole_video, tmp_path):
    client = make_client()
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    r = post_session(client, one_pothole_video, gps, gps_source="synthetic", role="demo")
    sid = r.json()["session"]["id"]
    assert r.json()["session"]["gps_source"] == "synthetic" and r.json()["session"]["role"] == "demo"
    wait_job(client, sid)
    s = client.get(f"/api/sessions/{sid}").json()
    assert s["gps_source"] == "synthetic"
    assert any("synthetic" in w for w in s["report"]["warnings"])
    assert client.get(f"/api/sessions/{sid}/detections").json()["gps_source"] == "synthetic"
    assert client.get("/api/potholes").json()["potholes"] == []  # demo mode is off by default
    client.post("/api/map/rebuild", json={"include_synthetic": True})
    (p,) = client.get("/api/potholes").json()["potholes"]
    assert p["gps_source"] == "synthetic"


def test_include_in_map_false_at_upload(make_client, one_pothole_video, tmp_path):
    client = make_client()
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    sid = post_session(client, one_pothole_video, gps, include_in_map=False).json()["session"]["id"]
    wait_job(client, sid)
    assert client.get(f"/api/sessions/{sid}").json()["include_in_map"] is False
    assert client.get("/api/potholes").json()["potholes"] == []


def test_unsupported_extension_is_400(make_client, tmp_path):
    client = make_client()
    path = tmp_path / "notes.txt"
    path.write_bytes(b"hello")
    r = post_session(client, path)
    assert r.status_code == 400 and error_of(r)[0] == "unsupported_file"
    assert client.get("/api/sessions").json() == {"sessions": []}


def test_oversized_upload_is_413(make_client, tmp_path):
    client = make_client(uploads={"max_upload_mb": 1})
    big = tmp_path / "big.mp4"
    big.write_bytes(b"\0" * (3 * (1 << 20)))
    r = post_session(client, big)
    assert r.status_code == 413 and error_of(r)[0] == "too_large"
    assert client.get("/api/sessions").json() == {"sessions": []}
    assert not client.cfg.uploads.dir.exists() or list(client.cfg.uploads.dir.iterdir()) == []


def test_copy_limit_is_enforced_while_streaming(tmp_path):
    class Upload:
        def __init__(self, data):
            import io

            self.file = io.BytesIO(data)

    dest = tmp_path / "out.bin"
    assert sessions_api._copy_limited(Upload(b"x" * 100), dest, 0, 100) == 100
    with pytest.raises(ApiError) as exc:
        sessions_api._copy_limited(Upload(b"x" * 101), dest, 0, 100)
    assert exc.value.status == 413 and exc.value.code == "too_large"
    with pytest.raises(ApiError):  # the limit covers all files together
        sessions_api._copy_limited(Upload(b"x" * 60), dest, 50, 100)


def test_corrupt_video_is_422_invalid_video_and_leaves_nothing(make_client, tmp_path):
    client = make_client()
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    r = post_session(client, bad)
    code, message = error_of(r)
    assert r.status_code == 422 and code == "invalid_video" and message
    assert client.get("/api/sessions").json() == {"sessions": []}
    assert list(client.cfg.uploads.dir.iterdir()) == []


def test_malformed_gps_csv_is_422_invalid_gps(make_client, one_pothole_video, tmp_path):
    client = make_client()
    bad = tmp_path / "bad.csv"
    bad.write_text("when,where\n1,2\n", encoding="utf-8")
    r = post_session(client, one_pothole_video, bad)
    assert r.status_code == 422 and error_of(r)[0] == "invalid_gps"
    assert "timestamp_sec" in r.json()["error"]["message"]
    assert client.get("/api/sessions").json() == {"sessions": []}
    assert list(client.cfg.uploads.dir.iterdir()) == []


def test_unknown_session_is_404_everywhere(make_client):
    client = make_client()
    for method, url in [
        ("get", "/api/sessions/nope"),
        ("get", "/api/sessions/nope/status"),
        ("get", "/api/sessions/nope/detections"),
        ("get", "/api/sessions/nope/track"),
        ("post", "/api/sessions/nope/reprocess"),
        ("delete", "/api/sessions/nope"),
    ]:
        r = getattr(client, method)(url)
        assert r.status_code == 404 and error_of(r)[0] == "not_found", url
    r = client.patch("/api/sessions/nope", json={"include_in_map": False})
    assert r.status_code == 404 and error_of(r)[0] == "not_found"


def test_reprocess_creates_a_new_job(uploaded):
    client, sid = uploaded
    first = client.get(f"/api/sessions/{sid}/status").json()["job"]["id"]
    r = client.post(f"/api/sessions/{sid}/reprocess")
    assert r.status_code == 202
    assert r.json()["job"]["id"] != first and r.json()["session"]["id"] == sid
    job = wait_job(client, sid)
    assert job["status"] == "done" and job["id"] == r.json()["job"]["id"]
    # the old job's observations stay in the database but only the latest counts
    assert client.get(f"/api/sessions/{sid}").json()["n_observations"] == 1
    assert len(client.get("/api/potholes").json()["potholes"]) == 1


class BlockingFactory:
    """A detector factory that waits until released, keeping the first job ``running``."""

    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()

    def __call__(self):
        self.started.set()
        assert self.release.wait(30)
        return FakeDetector()


def test_reprocess_and_delete_are_409_while_running_or_queued(
    make_client, one_pothole_video, tmp_path
):
    factory = BlockingFactory()
    client = make_client(detector_factory=factory)
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    running = post_session(client, one_pothole_video, gps).json()["session"]["id"]
    assert factory.started.wait(10)
    queued = post_session(client, one_pothole_video, gps).json()["session"]["id"]
    try:
        # the first job is still "queued" while the detector loads, then "running"
        assert client.get(f"/api/sessions/{running}/status").json()["job"]["status"] in ("queued", "running")
        assert client.get(f"/api/sessions/{queued}/status").json()["job"]["status"] == "queued"
        for sid in (running, queued):
            r = client.post(f"/api/sessions/{sid}/reprocess")
            assert r.status_code == 409 and error_of(r)[0] == "job_running"
            r = client.delete(f"/api/sessions/{sid}")
            assert r.status_code == 409 and error_of(r)[0] == "job_running"
        assert (client.cfg.uploads.dir / running).is_dir()
    finally:
        factory.release.set()
    assert wait_job(client, running)["status"] == "done"
    assert wait_job(client, queued)["status"] == "done"
    assert client.delete(f"/api/sessions/{queued}").status_code == 200


def test_patch_include_in_map_changes_the_pothole_list(uploaded):
    client, sid = uploaded
    assert len(client.get("/api/potholes").json()["potholes"]) == 1
    r = client.patch(f"/api/sessions/{sid}", json={"include_in_map": False})
    assert r.status_code == 200 and r.json()["include_in_map"] is False
    assert client.get("/api/potholes").json()["potholes"] == []
    r = client.patch(f"/api/sessions/{sid}", json={"include_in_map": True})
    assert r.json()["include_in_map"] is True
    assert len(client.get("/api/potholes").json()["potholes"]) == 1


def test_patch_rejects_a_bad_body(uploaded):
    client, sid = uploaded
    for body in ({}, {"include_in_map": "maybe"}, {"include_in_map": True, "extra": 1}):
        r = client.patch(f"/api/sessions/{sid}", json=body)
        assert r.status_code == 422 and error_of(r)[0] == "invalid_input"


def test_delete_removes_rows_and_files(uploaded):
    client, sid = uploaded
    url = client.get(f"/api/sessions/{sid}/detections").json()["observations"][0]["image_url"]
    assert client.get(url).status_code == 200
    assert (client.cfg.uploads.dir / sid).is_dir() and (client.cfg.uploads.artifacts_dir / sid).is_dir()
    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.get("/api/sessions").json() == {"sessions": []}
    assert not (client.cfg.uploads.dir / sid).exists()
    assert not (client.cfg.uploads.artifacts_dir / sid).exists()
    assert client.get(url).status_code == 404
    assert client.get("/api/potholes").json()["potholes"] == []
    stats = client.get("/api/stats").json()
    assert stats["sessions"]["total"] == 0 and stats["observations"]["total"] == 0


def test_artifacts_reject_traversal_and_missing_files(uploaded, tmp_path):
    client, sid = uploaded
    (client.cfg.uploads.dir / "secret.txt").write_text("secret", encoding="utf-8")
    attempts = [
        "..%2Fuploads%2Fsecret.txt",
        "%2e%2e/uploads/secret.txt",
        f"{sid}/..%2F..%2Fuploads%2Fsecret.txt",
        "..%2Fapp.sqlite",
        "%2e%2e%5Cuploads%5Csecret.txt",
        "C:%5CWindows%5Cwin.ini",
        "%2FWindows%2Fwin.ini",
        "does/not/exist.jpg",
        f"{sid}",  # a directory
    ]
    for attempt in attempts:
        r = client.get(f"/api/artifacts/{attempt}")
        assert r.status_code == 404, attempt
        assert error_of(r)[0] == "not_found"
        assert b"secret" not in r.content


def test_failing_detector_factory_gives_a_failed_job_with_the_message(
    make_client, one_pothole_video, tmp_path
):
    def broken():
        raise RuntimeError("weights are missing")

    client = make_client(detector_factory=broken)
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    sid = post_session(client, one_pothole_video, gps).json()["session"]["id"]
    job = wait_job(client, sid)
    assert job["status"] == "failed" and "weights are missing" in job["error"]
    s = client.get(f"/api/sessions/{sid}").json()
    assert s["job"]["status"] == "failed" and s["report"] is None and s["n_observations"] == 0
    assert client.get(f"/api/sessions/{sid}/detections").json()["observations"] == []
    assert client.get("/api/stats").json()["sessions"]["by_status"]["failed"] == 1
    # a failed session can be reprocessed
    assert client.post(f"/api/sessions/{sid}/reprocess").status_code == 202
    assert wait_job(client, sid)["status"] == "failed"


def test_missing_video_field_is_a_contract_shaped_422(make_client):
    client = make_client()
    r = client.post("/api/sessions", data={"role": "survey"})
    assert r.status_code == 422 and error_of(r)[0] == "invalid_input"
    assert "video" in r.json()["error"]["message"]


def test_bad_form_values_are_422(make_client, one_pothole_video):
    client = make_client()
    for field in ({"role": "boss"}, {"gps_source": "none"}, {"include_in_map": "perhaps"}):
        r = post_session(client, one_pothole_video, **field)
        assert r.status_code == 422 and error_of(r)[0] == "invalid_input", field
    assert client.get("/api/sessions").json() == {"sessions": []}


def test_sessions_are_listed_newest_first(make_client, one_pothole_video, tmp_path):
    client = make_client()
    ids = [post_session(client, one_pothole_video).json()["session"]["id"] for _ in range(3)]
    for sid in ids:
        wait_job(client, sid)
    assert [s["id"] for s in client.get("/api/sessions").json()["sessions"]] == ids[::-1]


def test_demo_sidecar_with_default_form_fields_is_stored_synthetic(
    make_client, one_pothole_video, tmp_path
):
    import json

    client = make_client()
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    side = tmp_path / "meta.json"
    side.write_text(json.dumps({"format": "road-recorder/1", "synthetic": True}), encoding="utf-8")
    r = post_session(client, one_pothole_video, gps, side)  # gps_source left at its default
    assert r.json()["session"]["gps_source"] == "synthetic"


def test_successful_upload_leaves_no_incoming_folder(make_client, one_pothole_video, tmp_path):
    client = make_client()
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    sid = post_session(client, one_pothole_video, gps).json()["session"]["id"]
    assert {p.name for p in client.cfg.uploads.dir.iterdir()} == {sid}  # no .incoming-* leftovers
    assert {p.suffix for p in (client.cfg.uploads.dir / sid).iterdir()} >= {".mp4", ".csv"}
    wait_job(client, sid)
