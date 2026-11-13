import json

import pytest
from pipeline_fixtures import (
    FakeDetector,
    add_session,
    make_cfg,
    moving_object,
    write_object_video,
)

from roadsense.pipeline import PIPELINE_VERSION, PipelineError, process_session
from roadsense.storage import db
from roadsense.storage import jobs as jobs_db
from roadsense.storage import observations as obs_db

# Walking pace (about 0.9 m/s), so the session uses the walking stride of 3.
LINEAR_FIXES = [(float(t), 50.0 + 0.000005 * t, 8.0 + 0.00001 * t, 5.0) for t in range(0, 8)]


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "p.sqlite")
    db.init_schema(c)
    yield c
    c.close()


@pytest.fixture
def cfg(tmp_path):
    return make_cfg(tmp_path)


def one_object_video(tmp_path, n_frames=60):
    return write_object_video(tmp_path, n_frames, [moving_object(0, 11)])


def test_one_moving_object_is_one_observation(conn, cfg, tmp_path):
    sid = add_session(conn, one_object_video(tmp_path), LINEAR_FIXES)
    report = process_session(conn, sid, FakeDetector(), cfg)
    rows = obs_db.list_observations(conn, sid)
    assert len(rows) == 1 and report["n_tracks_confirmed"] == 1
    row = rows[0]
    assert row["track_id"] == 1 and row["n_frames"] == 4
    assert row["confidence"] == pytest.approx(0.9)
    assert row["frame_index"] == 9 and row["first_t_video_s"] == pytest.approx(0.0, abs=1e-6)
    assert row["t_video_s"] == pytest.approx(0.9, abs=1e-3)
    assert (row["frame_w"], row["frame_h"]) == (160, 120)
    # the box is the LAST detection: object at frame 9 starts at x = 10 + 2*9
    assert row["x1"] == pytest.approx(28, abs=3) and row["x2"] == pytest.approx(52, abs=3)
    assert row["edge_u"] is None and row["pothole_id"] is None
    assert report["frames_total"] == 60 and report["frames_processed"] == 20
    assert report["n_tracks"] == 1 and report["n_tracks_rejected_short"] == 0


def test_two_objects_at_different_times_are_two_observations(conn, cfg, tmp_path):
    video = write_object_video(
        tmp_path, 60, [moving_object(0, 11), moving_object(30, 41, x0=100, y=70)]
    )
    sid = add_session(conn, video, LINEAR_FIXES)
    report = process_session(conn, sid, FakeDetector(), cfg)
    rows = obs_db.list_observations(conn, sid)
    assert [r["track_id"] for r in rows] == [1, 2]
    assert rows[0]["t_video_s"] < 1.5 < 3.0 < rows[1]["t_video_s"]
    assert report["n_tracks_confirmed"] == 2


def test_short_blip_is_rejected_and_counted(conn, cfg, tmp_path):
    video = write_object_video(tmp_path, 40, [moving_object(0, 11), moving_object(30, 32, x0=100, y=70)])
    sid = add_session(conn, video, LINEAR_FIXES)
    report = process_session(conn, sid, FakeDetector(), cfg)
    assert obs_db.count_observations(conn, sid) == 1
    assert report["n_tracks"] == 2 and report["n_tracks_rejected_short"] == 1


def test_no_gps_gives_null_position_and_warning(conn, cfg, tmp_path):
    sid = add_session(conn, one_object_video(tmp_path))
    report = process_session(conn, sid, FakeDetector(), cfg)
    (row,) = obs_db.list_observations(conn, sid)
    assert row["cam_lat"] is None and row["cam_lon"] is None and row["cam_accuracy_m"] is None
    assert any("no GPS data" in w for w in report["warnings"])
    assert report["n_observations_without_position"] == 1
    assert report["n_observations_with_position"] == 0
    assert report["gps"] == {"source": "none", "n_fixes": 0, "n_used": 0}


def test_gps_positions_are_interpolated(conn, cfg, tmp_path):
    sid = add_session(conn, one_object_video(tmp_path), LINEAR_FIXES)
    report = process_session(conn, sid, FakeDetector(), cfg)
    (row,) = obs_db.list_observations(conn, sid)
    t = row["t_video_s"]
    assert row["cam_lat"] == pytest.approx(50.0 + 0.000005 * t, abs=1e-9)
    assert row["cam_lon"] == pytest.approx(8.0 + 0.00001 * t, abs=1e-9)
    assert row["cam_accuracy_m"] == 5.0
    assert report["n_observations_with_position"] == 1
    assert report["gps"]["n_fixes"] == len(LINEAR_FIXES)
    assert not any("no GPS" in w for w in report["warnings"])


def test_gps_dropout_gives_null_position(conn, cfg, tmp_path):
    # fixes at 0, 1 then 5, 6: the 4 s hole is longer than max_gps_gap_s (3 s)
    fixes = [(0.0, 50.0, 8.0), (1.0, 50.00001, 8.0), (5.0, 50.00005, 8.0), (6.0, 50.00006, 8.0)]
    video = write_object_video(
        tmp_path, 60, [moving_object(0, 8), moving_object(30, 41, x0=100, y=70)]
    )
    sid = add_session(conn, video, fixes)
    report = process_session(conn, sid, FakeDetector(), cfg)
    first, second = obs_db.list_observations(conn, sid)
    assert first["t_video_s"] < 1.0 and first["cam_lat"] is not None
    assert 3.0 <= second["t_video_s"] <= 5.0
    assert second["cam_lat"] is None and second["cam_lon"] is None
    assert report["n_observations_without_position"] == 1
    assert any("no camera position" in w for w in report["warnings"])


def test_frame_errors_are_counted_and_skipped(conn, cfg, tmp_path):
    sid = add_session(conn, one_object_video(tmp_path), LINEAR_FIXES)
    detector = FakeDetector(fail_on={1})
    report = process_session(conn, sid, detector, cfg)
    assert report["n_frame_errors"] == 1
    assert report["frames_processed"] == 20 and detector.calls == 20
    assert report["outcome"] == "partial"  # 1 of 20 is 5%, above the 2% of isolated failures
    assert obs_db.count_observations(conn, sid) == 1  # the track survives the skipped frames
    assert any("could not be analysed" in w for w in report["warnings"])


def failed_job_checks(conn, cfg, sid):
    job = jobs_db.latest_job_for_session(conn, sid)
    assert job["status"] == "failed" and job["error"]
    assert obs_db.count_observations_for_job(conn, job["id"]) == 0
    assert conn.execute("SELECT active_job_id FROM sessions WHERE id=?", (sid,)).fetchone()[0] is None
    assert not (cfg.uploads.artifacts_dir / sid / job["id"]).exists()
    return job


def test_detector_that_always_fails_fails_the_job(conn, cfg, tmp_path):
    sid = add_session(conn, one_object_video(tmp_path), LINEAR_FIXES)
    with pytest.raises(PipelineError):
        process_session(conn, sid, FakeDetector(always_fail=True), cfg)
    failed_job_checks(conn, cfg, sid)


def test_first_thirty_frames_all_failing_aborts(conn, cfg, tmp_path):
    video = write_object_video(tmp_path, 120, [moving_object(0, 11)])  # 40 processed frames
    sid = add_session(conn, video, LINEAR_FIXES)
    with pytest.raises(PipelineError, match="first 30"):
        process_session(conn, sid, FakeDetector(fail_on=set(range(30))), cfg)
    job = failed_job_checks(conn, cfg, sid)
    assert "first 30" in job["error"]


def test_twenty_nine_failures_then_success_is_not_aborted(conn, cfg, tmp_path):
    video = write_object_video(tmp_path, 120, [moving_object(40, 99)])
    sid = add_session(conn, video, LINEAR_FIXES)
    # the share of failed frames is not what is tested here: allow any
    cfg = make_cfg(
        tmp_path,
        detection={"partial_frame_error_fraction": 1.0, "max_frame_error_fraction": 1.0},
    )
    report = process_session(conn, sid, FakeDetector(fail_on=set(range(29))), cfg)
    assert report["n_frame_errors"] == 29


def test_unexpected_exception_fails_the_job(conn, cfg, tmp_path):
    sid = add_session(conn, one_object_video(tmp_path), LINEAR_FIXES)
    with pytest.raises(PipelineError, match="fake crash"):
        process_session(conn, sid, FakeDetector(crash_on=5), cfg)
    job = failed_job_checks(conn, cfg, sid)
    assert "RuntimeError" in job["error"]


def test_missing_video_fails_the_job(conn, cfg, tmp_path):
    sid = add_session(conn, tmp_path / "nope.mp4", LINEAR_FIXES)
    with pytest.raises(PipelineError, match="not found"):
        process_session(conn, sid, FakeDetector(), cfg)
    failed_job_checks(conn, cfg, sid)


def test_unknown_session_raises_without_creating_a_job(conn, cfg):
    with pytest.raises(PipelineError, match="no such session"):
        process_session(conn, "missing", FakeDetector(), cfg)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_annotated_image_exists_and_path_is_relative(conn, cfg, tmp_path):
    import cv2

    video = write_object_video(tmp_path, 60, [moving_object(0, 11)])
    sid = add_session(conn, video, LINEAR_FIXES)
    report = process_session(conn, sid, FakeDetector(), cfg)
    (row,) = obs_db.list_observations(conn, sid)
    rel = row["image_path"]
    assert rel == f"{sid}/{report['job_id']}/track_1.jpg"
    assert "\\" not in rel and ":" not in rel
    path = cfg.uploads.artifacts_dir / rel
    assert path.is_file()
    image = cv2.imread(str(path))
    assert image is not None and image.shape[:2] == (120, 160)
    # the red box outline was drawn on the (otherwise grey/white) frame
    b, g, r = (image[..., i].astype(int) for i in range(3))
    assert ((r - g > 100) & (r - b > 100)).sum() > 20


def test_large_frames_are_downscaled_in_the_annotation(tmp_path):
    import numpy as np

    from roadsense.detection.detector import Detection
    from roadsense.pipeline.process import annotated_jpeg

    import cv2

    big = np.full((1080, 1920, 3), 40, np.uint8)
    data = annotated_jpeg(big, Detection(100, 100, 400, 300, 0.77, 0, "pothole"))
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert max(image.shape[:2]) == 1280


def test_job_rows_are_filled(conn, cfg, tmp_path):
    sid = add_session(conn, one_object_video(tmp_path), LINEAR_FIXES, name="route")
    stages = []
    report = process_session(conn, sid, FakeDetector(), cfg, progress=lambda s, f: stages.append((s, f)))
    job = jobs_db.get_job(conn, report["job_id"])
    assert job["status"] == "done" and job["progress"] == 1.0
    assert job["pipeline_version"] == PIPELINE_VERSION == report["pipeline_version"]
    assert job["model_sha256"] == "f" * 64
    assert json.loads(job["summary_json"]) == report
    params = json.loads(job["params_json"])
    assert params["tracking"]["frame_stride"] == 3 and params["detection"]["conf"] == 0.25
    assert params["ingest"]["max_gps_gap_s"] == 3.0
    assert conn.execute("SELECT active_job_id FROM sessions WHERE id=?", (sid,)).fetchone()[0] == job["id"]
    assert report["session_id"] == sid and report["session_name"] == "route"
    assert report["video"]["width"] == 160 and report["video"]["timeline_source"] in ("pts", "index_fps")
    assert report["frame_stride"] == 3 and report["detector"]["weights"] == "fake.pt"
    assert report["elapsed_s"] >= 0
    json.dumps(report)  # serialisable
    assert stages[0] == ("detecting", 0.0) and stages[-1] == ("done", 1.0)
    assert report["outcome"] == "complete"
    assert report["tracking"] == {
        "mode": "walking",
        "median_speed_mps": pytest.approx(0.9, abs=0.1),
        "upper_quartile_speed_mps": pytest.approx(0.9, abs=0.1),
        "frame_stride": 3,
        "min_hits": 3,
        "max_age_source_frames": 15,
    }
    assert ("saving", 1.0) in stages


def test_progress_is_visible_to_another_connection(conn, cfg, tmp_path):
    seen = []

    class Spy(FakeDetector):
        def detect(self, image):
            other = db.connect(tmp_path / "p.sqlite")
            try:
                seen.append(jobs_db.latest_job_for_session(other, sid)["status"])
            finally:
                other.close()
            return super().detect(image)

    sid = add_session(conn, one_object_video(tmp_path, 12), LINEAR_FIXES)
    process_session(conn, sid, Spy(), cfg)
    assert set(seen) == {"running"}


def test_reprocessing_creates_new_job_and_keeps_old_observations(conn, cfg, tmp_path):
    sid = add_session(conn, one_object_video(tmp_path), LINEAR_FIXES)
    first = process_session(conn, sid, FakeDetector(), cfg)
    second = process_session(conn, sid, FakeDetector(), cfg)
    assert first["job_id"] != second["job_id"]
    active = conn.execute("SELECT active_job_id FROM sessions WHERE id=?", (sid,)).fetchone()[0]
    assert active == second["job_id"]
    assert obs_db.count_observations_for_job(conn, first["job_id"]) == 1
    assert obs_db.count_observations_for_job(conn, second["job_id"]) == 1
    assert obs_db.count_observations(conn, sid) == 1
    assert all(r["job_id"] == second["job_id"] for r in obs_db.list_observations(conn, sid))
    assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2


def test_failed_reprocess_keeps_previous_active_job(conn, cfg, tmp_path):
    sid = add_session(conn, one_object_video(tmp_path), LINEAR_FIXES)
    first = process_session(conn, sid, FakeDetector(), cfg)
    with pytest.raises(PipelineError):
        process_session(conn, sid, FakeDetector(always_fail=True), cfg)
    active = conn.execute("SELECT active_job_id FROM sessions WHERE id=?", (sid,)).fetchone()[0]
    assert active == first["job_id"]
    assert obs_db.count_observations(conn, sid) == 1


def test_given_job_id_is_used(conn, cfg, tmp_path):
    sid = add_session(conn, one_object_video(tmp_path, 12), LINEAR_FIXES)
    jid = jobs_db.create_job(conn, sid)
    conn.commit()
    report = process_session(conn, sid, FakeDetector(), cfg, job_id=jid)
    assert report["job_id"] == jid
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_synthetic_gps_is_flagged(conn, cfg, tmp_path):
    sid = add_session(conn, one_object_video(tmp_path, 12), LINEAR_FIXES, gps_source="synthetic")
    report = process_session(conn, sid, FakeDetector(), cfg)
    assert report["gps"]["source"] == "synthetic"
    assert any("synthetic" in w for w in report["warnings"])


# --------------------------------------------------------------------------- real model


@pytest.mark.slow
def test_end_to_end_with_real_model(conn, tmp_path):
    """3 RDD2022 validation images, 12 frames each, through the real detector.

    Only checks that the run completes and reports; the number of potholes is not asserted.
    """
    import cv2
    import numpy as np

    from roadsense.config import project_root
    from roadsense.detection.detector import Detector

    pytest.importorskip("ultralytics", reason="ultralytics is not installed")
    root = project_root()
    weights = root / "runs" / "train" / "baseline" / "weights" / "best.pt"
    images_dir = root / "data" / "processed" / "rdd2022_pothole" / "images" / "val"
    labels_dir = root / "data" / "processed" / "rdd2022_pothole" / "labels" / "val"
    if not weights.is_file():
        pytest.skip(f"{weights} not found")
    if not (images_dir.is_dir() and labels_dir.is_dir()):
        pytest.skip("RDD2022 validation data not found")
    images = []
    for label in sorted(labels_dir.glob("*.txt")):
        if label.stat().st_size > 0 and (images_dir / f"{label.stem}.jpg").is_file():
            images.append(images_dir / f"{label.stem}.jpg")
        if len(images) == 3:
            break
    if len(images) < 3:
        pytest.skip("fewer than 3 positive validation images found")

    video = tmp_path / "rdd.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (640, 480))
    if not writer.isOpened():
        pytest.skip("MJPG VideoWriter not available")
    for path in images:
        frame = cv2.resize(cv2.imread(str(path)), (640, 480))
        for _ in range(12):
            writer.write(frame)
    writer.release()

    cfg = make_cfg(tmp_path)
    detector = Detector(cfg.detection)
    fixes = [(float(t), 50.0 + 0.000005 * t, 8.0, 5.0) for t in range(6)]
    sid = add_session(conn, video, fixes)
    report = process_session(conn, sid, detector, cfg)
    assert report["frames_total"] == 36 and report["frames_processed"] == 12
    assert report["n_frame_errors"] == 0
    assert report["detector"]["weights_sha256"] == detector.weights_sha256
    assert report["n_tracks_confirmed"] == len(obs_db.list_observations(conn, sid))
    job = jobs_db.get_job(conn, report["job_id"])
    assert job["status"] == "done" and job["model_sha256"] == detector.weights_sha256
    for row in obs_db.list_observations(conn, sid):
        assert (cfg.uploads.artifacts_dir / row["image_path"]).is_file()
