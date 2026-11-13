"""Pipeline behaviour added in the tracking/outcome review: tracking modes, outcomes,
retained frames and the progress shown while the video is read."""

import json

import numpy as np
import pytest
from pipeline_fixtures import (
    FakeDetector,
    add_session,
    make_cfg,
    moving_object,
    write_object_video,
)

from roadsense.api.views import job_view
from roadsense.config import TrackingConfig
from roadsense.detection.detector import Detection
from roadsense.ingest.gps import GpsFix
from roadsense.pipeline import PipelineError, process_session
from roadsense.pipeline import process as process_mod
from roadsense.pipeline.process import annotated_jpeg, plan_tracking, session_median_speed
from roadsense.storage import db
from roadsense.storage import jobs as jobs_db
from roadsense.storage import observations as obs_db

# about 0.9 m/s and 11-18 m/s: positions only, no recorded speed
WALKING = [(float(t), 50.0 + 0.000005 * t, 8.0 + 0.00001 * t, 5.0) for t in range(0, 8)]
VEHICLE = [(float(t), 50.0 + 0.0001 * t, 8.0 + 0.0002 * t, 5.0) for t in range(0, 8)]


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "p.sqlite")
    db.init_schema(c)
    yield c
    c.close()


@pytest.fixture
def cfg(tmp_path):
    return make_cfg(tmp_path)


def fix(seq, t, lat=50.0, lon=8.0, speed=None, used=True):
    return GpsFix(seq, t, lat, lon, 5.0, speed, None, None, used)


# --------------------------------------------------------------------------- mode selection


def test_median_speed_uses_recorded_speeds_inside_the_video_only():
    fixes = [fix(0, -2.0, speed=30.0), fix(1, 0.0, speed=1.0), fix(2, 1.0, speed=2.0),
             fix(3, 2.0, speed=3.0), fix(4, 99.0, speed=30.0), fix(5, 1.5, speed=50.0, used=False)]  # fmt: skip
    assert session_median_speed(fixes, 10.0) == 2.0  # pre-roll, post-roll and unused ignored


def test_median_speed_is_derived_from_positions_when_speeds_are_missing():
    # 0.0001 degrees of latitude per second is about 11.1 m/s
    fixes = [fix(i, float(i), lat=50.0 + 0.0001 * i) for i in range(6)]
    assert session_median_speed(fixes, 10.0) == pytest.approx(11.12, abs=0.05)
    # fewer than three recorded speeds: positions are used instead of those one or two values
    fixes[2] = fix(2, 2.0, lat=50.0002, speed=0.5)
    assert session_median_speed(fixes, 10.0) == pytest.approx(11.12, abs=0.05)


def test_median_speed_ignores_pre_roll_when_deriving():
    pre = [fix(i, -5.0 + i, lat=50.0 + 0.001 * i) for i in range(4)]  # fast, before the video
    inside = [fix(4 + i, float(i), lat=50.5 + 0.000005 * i) for i in range(5)]  # walking
    assert session_median_speed(pre + inside, 4.0) == pytest.approx(0.56, abs=0.05)


def test_median_speed_is_none_without_usable_fixes():
    assert session_median_speed([], 10.0) is None
    assert session_median_speed([fix(0, 1.0)], 10.0) is None  # one fix: no speed
    assert session_median_speed([fix(0, 1.0, used=False), fix(1, 2.0, used=False)], 10.0) is None
    assert session_median_speed([fix(0, 1.0), fix(1, 1.0, lat=50.1)], 10.0) is None  # same time


def test_plan_tracking_modes():
    cfg = TrackingConfig()
    walking = [fix(i, float(i), speed=0.8) for i in range(5)]
    vehicle = [fix(i, float(i), speed=5.0) for i in range(5)]
    edge = [fix(i, float(i), speed=2.5) for i in range(5)]
    w, v, e = (plan_tracking(cfg, f, 10.0) for f in (walking, vehicle, edge))
    assert (w.mode, w.frame_stride, w.min_hits, w.max_age_source_frames) == ("walking", 3, 3, 15)
    assert (v.mode, v.frame_stride, v.min_hits, v.max_age_source_frames) == ("vehicle", 1, 6, 15)
    assert e.mode == "vehicle"  # "at or above" the threshold
    assert (v.median_speed_mps, v.upper_quartile_speed_mps) == (5.0, 5.0)
    d = plan_tracking(cfg, [], 10.0)
    assert (d.mode, d.median_speed_mps, d.frame_stride, d.min_hits) == ("default", None, 3, 3)
    assert d.upper_quartile_speed_mps is None
    custom = TrackingConfig(frame_stride=2, max_age_frames=4, vehicle_frame_stride=1, vehicle_min_hits=8)
    assert plan_tracking(custom, vehicle, 10.0).max_age_source_frames == 8  # 4 analysed at stride 2


def test_session_stationary_for_most_of_its_fixes_is_still_planned_as_vehicle():
    cfg = TrackingConfig()
    # 60% stopped, then 6 m/s: the median is 0 but the upper quartile is 6
    fixes = [fix(i, float(i), speed=0.0 if i < 12 else 6.0) for i in range(20)]
    plan = plan_tracking(cfg, fixes, 100.0)
    assert plan.median_speed_mps == 0.0 and plan.upper_quartile_speed_mps == 6.0
    assert plan.mode == "vehicle" and plan.frame_stride == 1


def test_walking_session_with_a_few_noisy_fast_fixes_stays_walking():
    cfg = TrackingConfig()
    # 4 of 20 fixes (under a quarter) read 8 m/s
    fixes = [fix(i, float(i), speed=8.0 if i in (3, 7, 11, 15) else 0.8) for i in range(20)]
    plan = plan_tracking(cfg, fixes, 100.0)
    assert plan.mode == "walking" and plan.upper_quartile_speed_mps < 2.5


def test_upper_quartile_of_one_sample_is_that_sample():
    from roadsense.pipeline.process import session_upper_quartile_speed

    one = [fix(0, 1.0, speed=4.0)]  # a single recorded speed is the last resort
    assert session_upper_quartile_speed(one, 10.0) == 4.0
    assert plan_tracking(TrackingConfig(), one, 10.0).mode == "vehicle"
    assert session_upper_quartile_speed([], 10.0) is None


def test_walking_session_keeps_stride_three(conn, cfg, tmp_path):
    video = write_object_video(tmp_path, 60, [moving_object(0, 11)])
    sid = add_session(conn, video, WALKING)
    report = process_session(conn, sid, FakeDetector(), cfg)
    assert report["frame_stride"] == 3 and report["frames_processed"] == 20
    assert report["tracking"]["mode"] == "walking"
    assert report["tracking"]["frame_stride"] == 3 and report["tracking"]["min_hits"] == 3
    assert report["tracking"]["max_age_source_frames"] == 15
    assert report["tracking"]["median_speed_mps"] == pytest.approx(0.9, abs=0.1)
    assert report["tracking"]["upper_quartile_speed_mps"] == pytest.approx(0.9, abs=0.1)


def test_vehicle_session_analyses_every_frame_and_needs_six_hits(conn, cfg, tmp_path):
    # object A is visible for 9 frames, B for 5: at stride 1 both have as many hits
    video = write_object_video(
        tmp_path, 60, [moving_object(0, 8, speed=1), moving_object(30, 34, x0=100, y=70, speed=1)]
    )
    sid = add_session(conn, video, VEHICLE)  # no recorded speed: derived from the positions
    report = process_session(conn, sid, FakeDetector(), cfg)
    assert report["frame_stride"] == 1 and report["frames_processed"] == report["frames_total"] == 60
    t = report["tracking"]
    assert t["mode"] == "vehicle" and t["frame_stride"] == 1 and t["min_hits"] == 6
    assert t["max_age_source_frames"] == 15 and t["median_speed_mps"] > 10
    assert report["n_tracks"] == 2 and report["n_tracks_confirmed"] == 1  # B has 5 < 6 hits
    assert report["n_tracks_rejected_short"] == 1
    (row,) = obs_db.list_observations(conn, sid)
    assert row["n_frames"] == 9


def test_recorded_speed_decides_over_positions(conn, cfg, tmp_path):
    video = write_object_video(tmp_path, 30, [moving_object(0, 11)])
    sid = add_session(conn, video, WALKING)  # positions say 0.9 m/s
    conn.execute("UPDATE gps_fixes SET speed_mps = 8.0 WHERE session_id = ?", (sid,))
    conn.commit()
    report = process_session(conn, sid, FakeDetector(), cfg)
    assert report["tracking"]["mode"] == "vehicle"
    assert report["tracking"]["median_speed_mps"] == 8.0 and report["frame_stride"] == 1


def test_no_gps_uses_the_default_settings(conn, cfg, tmp_path):
    video = write_object_video(tmp_path, 30, [moving_object(0, 11)])
    sid = add_session(conn, video)
    report = process_session(conn, sid, FakeDetector(), cfg)
    assert report["tracking"] == {
        "mode": "default",
        "median_speed_mps": None,
        "upper_quartile_speed_mps": None,
        "frame_stride": 3,
        "min_hits": 3,
        "max_age_source_frames": 15,
    }
    assert report["frame_stride"] == 3


def test_pre_roll_fixes_do_not_make_a_walking_video_a_vehicle_one(conn, cfg, tmp_path):
    fast_pre_roll = [(-3.0 + i, 50.0 + 0.001 * i, 8.0, 5.0) for i in range(3)]
    video = write_object_video(tmp_path, 30, [moving_object(0, 11)])
    sid = add_session(conn, video, fast_pre_roll + [(t, 51.0 + 5e-6 * t, 8.0, 5.0) for t, *_ in WALKING])
    report = process_session(conn, sid, FakeDetector(), cfg)
    assert report["tracking"]["mode"] == "walking" and report["frame_stride"] == 3


def test_params_record_the_configured_values(conn, cfg, tmp_path):
    video = write_object_video(tmp_path, 30, [moving_object(0, 11)])
    sid = add_session(conn, video, VEHICLE)
    report = process_session(conn, sid, FakeDetector(), cfg)
    params = json.loads(jobs_db.get_job(conn, report["job_id"])["params_json"])
    assert params["tracking"]["frame_stride"] == 3 and params["tracking"]["vehicle_min_hits"] == 6
    assert params["detection"]["max_frame_error_fraction"] == 0.25


# --------------------------------------------------------------------------- outcomes


def static_video(tmp_path, n_processed=100):
    """A static object in every frame; walking GPS means every third frame is analysed."""
    n = 3 * n_processed
    return write_object_video(tmp_path, n, [moving_object(0, n - 1, speed=0)])


def spread(n):
    """``n`` failed calls spread over calls 30..99, so the first-30 check is not hit."""
    return {30 + (70 * k) // n for k in range(n)} if n else set()


def failed_job(conn, cfg, sid):
    job = jobs_db.latest_job_for_session(conn, sid)
    assert job["status"] == "failed" and job["error"]
    assert obs_db.count_observations_for_job(conn, job["id"]) == 0
    assert conn.execute("SELECT active_job_id FROM sessions WHERE id=?", (sid,)).fetchone()[0] is None
    assert not (cfg.uploads.artifacts_dir / sid / job["id"]).exists()
    return job


def test_one_failed_frame_in_a_hundred_is_complete_with_a_count_warning(conn, cfg, tmp_path):
    sid = add_session(conn, static_video(tmp_path), WALKING)
    report = process_session(conn, sid, FakeDetector(fail_on=spread(1)), cfg)
    assert report["frames_processed"] == 100 and report["n_frame_errors"] == 1
    assert report["outcome"] == "complete"
    assert any("1 of 100 processed frame(s) could not be analysed" in w for w in report["warnings"])
    assert obs_db.count_observations(conn, sid) == 1
    job = jobs_db.get_job(conn, report["job_id"])
    assert job["status"] == "done" and job_view(job)["outcome"] == "complete"


def test_ten_percent_failed_frames_is_partial_and_keeps_observations(conn, cfg, tmp_path):
    sid = add_session(conn, static_video(tmp_path), WALKING)
    report = process_session(conn, sid, FakeDetector(fail_on=spread(10)), cfg)
    assert report["outcome"] == "partial" and report["n_frame_errors"] == 10
    assert report["warnings"][0] == (
        "10 of 100 analysed frames (10.0%) could not be analysed; "
        "potholes in those frames are missing from the results."
    )
    assert obs_db.count_observations(conn, sid) == 1  # kept
    job = jobs_db.get_job(conn, report["job_id"])
    assert job["status"] == "done"
    view = job_view(job)
    assert view["status"] == "done" and view["outcome"] == "partial"
    assert conn.execute("SELECT active_job_id FROM sessions WHERE id=?", (sid,)).fetchone()[0] == job["id"]


def test_forty_percent_failed_frames_fails_the_job(conn, cfg, tmp_path):
    sid = add_session(conn, static_video(tmp_path), WALKING)
    with pytest.raises(PipelineError) as err:
        process_session(conn, sid, FakeDetector(fail_on=spread(40)), cfg)
    job = failed_job(conn, cfg, sid)
    for text in ("40 of 100", "40.0%", "25%"):
        assert text in job["error"] and text in str(err.value)
    assert job_view(job)["outcome"] is None


@pytest.mark.parametrize(
    ("n_failed", "outcome"),
    [(0, "complete"), (2, "complete"), (3, "partial"), (25, "partial"), (26, None)],
)
def test_outcome_boundaries_are_at_the_configured_fractions(conn, cfg, tmp_path, n_failed, outcome):
    # exactly 2 of 100 (2%) is still isolated; exactly 25 of 100 is still partial
    sid = add_session(conn, static_video(tmp_path), WALKING)
    detector = FakeDetector(fail_on=spread(n_failed))
    if outcome is None:
        with pytest.raises(PipelineError):
            process_session(conn, sid, detector, cfg)
        failed_job(conn, cfg, sid)
    else:
        assert process_session(conn, sid, detector, cfg)["outcome"] == outcome


def test_thresholds_come_from_the_config(conn, tmp_path):
    cfg = make_cfg(tmp_path, detection={"partial_frame_error_fraction": 0.0, "max_frame_error_fraction": 0.05})
    sid = add_session(conn, static_video(tmp_path), WALKING)
    assert process_session(conn, sid, FakeDetector(fail_on=spread(1)), cfg)["outcome"] == "partial"
    with pytest.raises(PipelineError, match="6 of 100.*5%"):
        process_session(conn, sid, FakeDetector(fail_on=spread(6)), cfg)


def test_early_checks_still_apply(conn, cfg, tmp_path):
    sid = add_session(conn, static_video(tmp_path, 50), WALKING)
    with pytest.raises(PipelineError, match="first 30"):
        process_session(conn, sid, FakeDetector(fail_on=set(range(30))), cfg)
    short = add_session(conn, static_video(tmp_path, 20), WALKING)
    with pytest.raises(PipelineError, match="every one of the 20"):
        process_session(conn, short, FakeDetector(always_fail=True), cfg)


def test_job_view_outcome_for_old_summaries_and_other_states(conn, cfg, tmp_path):
    sid = add_session(conn, static_video(tmp_path, 20), WALKING)
    report = process_session(conn, sid, FakeDetector(), cfg)
    old = {k: v for k, v in report.items() if k not in ("outcome", "tracking")}
    jobs_db.update_summary(conn, report["job_id"], old)
    conn.commit()
    assert job_view(jobs_db.get_job(conn, report["job_id"]))["outcome"] == "complete"

    queued = jobs_db.create_job(conn, sid)
    running = jobs_db.create_job(conn, sid)
    jobs_db.mark_running(conn, running, stage="detecting")
    for jid in (queued, running):
        assert job_view(jobs_db.get_job(conn, jid))["outcome"] is None
    done_no_summary = jobs_db.create_job(conn, sid)
    jobs_db.mark_done(conn, done_no_summary, {})
    assert job_view(jobs_db.get_job(conn, done_no_summary))["outcome"] == "complete"
    assert job_view(None) is None


# --------------------------------------------------------------------------- retained frames


class Capture:
    """Records the per-track entries retained after every frame and the run object."""

    def __init__(self, monkeypatch):
        self.snapshots = []
        self.run = None
        original = process_mod._Run.note_best_frames
        capture = self

        def spy(run, frame_index, image):
            original(run, frame_index, image)
            capture.run = run
            capture.snapshots.append(dict(run.best_frames))

        monkeypatch.setattr(process_mod._Run, "note_best_frames", spy)


def test_only_encoded_bytes_are_retained_per_track(conn, cfg, tmp_path, monkeypatch):
    cap = Capture(monkeypatch)
    sid = add_session(conn, write_object_video(tmp_path, 60, [moving_object(0, 11)]), WALKING)
    process_session(conn, sid, FakeDetector(), cfg)
    kept = [entry for snap in cap.snapshots for entry in snap.values()]
    assert kept, "the run should have retained a best frame for its live track"
    for entry in kept:
        assert isinstance(entry.jpeg, bytes) and entry.jpeg[:2] == b"\xff\xd8"
        assert (entry.width, entry.height) == (160, 120)
        assert not any(isinstance(v, np.ndarray) for v in entry)
    assert cap.run.best_frames == {}  # everything was written out or dropped


def test_rejected_track_leaves_no_file_and_no_entry(conn, cfg, tmp_path, monkeypatch):
    cap = Capture(monkeypatch)
    video = write_object_video(tmp_path, 40, [moving_object(0, 11), moving_object(30, 32, x0=100, y=70)])
    sid = add_session(conn, video, WALKING)
    report = process_session(conn, sid, FakeDetector(), cfg)
    assert report["n_tracks"] == 2 and report["n_tracks_rejected_short"] == 1
    folder = cfg.uploads.artifacts_dir / sid / report["job_id"]
    assert sorted(p.name for p in folder.iterdir()) == ["track_1.jpg"]
    assert cap.run.best_frames == {}
    assert any(2 in snap for snap in cap.snapshots)  # it was retained while the track was live


def test_stored_image_is_what_annotating_the_best_frame_gives(conn, cfg, tmp_path):
    from roadsense.ingest.video import iter_frames

    video = write_object_video(tmp_path, 60, [moving_object(0, 11)])
    sid = add_session(conn, video, WALKING)
    report = process_session(conn, sid, FakeDetector(), cfg)
    (row,) = obs_db.list_observations(conn, sid)
    # all detections tie on confidence, so the best frame is the track's last one
    frame = next(f for f in iter_frames(video, 1) if f.index == row["frame_index"])
    det = Detection(row["x1"], row["y1"], row["x2"], row["y2"], 0.9, 0, "pothole")
    stored = (cfg.uploads.artifacts_dir / row["image_path"]).read_bytes()
    assert stored == annotated_jpeg(frame.image, det)
    assert (row["frame_w"], row["frame_h"]) == (160, 120) and report["n_tracks_confirmed"] == 1


def test_original_frame_size_is_stored_when_the_image_is_downscaled(conn, cfg, tmp_path, monkeypatch):
    import cv2

    monkeypatch.setattr(process_mod, "MAX_IMAGE_SIDE", 100)  # the 160 px frames get shrunk
    sid = add_session(conn, write_object_video(tmp_path, 60, [moving_object(0, 11)]), WALKING)
    process_session(conn, sid, FakeDetector(), cfg)
    (row,) = obs_db.list_observations(conn, sid)
    assert (row["frame_w"], row["frame_h"]) == (160, 120)  # the original size, not the JPEG's
    image = cv2.imread(str(cfg.uploads.artifacts_dir / row["image_path"]))
    assert max(image.shape[:2]) == 100


# --------------------------------------------------------------------------- reading stage


def test_job_shows_reading_video_without_progress_until_detection_starts(conn, cfg, tmp_path, monkeypatch):
    seen = {}

    def peek(label):
        other = db.connect(tmp_path / "p.sqlite")
        try:
            job = jobs_db.latest_job_for_session(other, sid)
            seen[label] = (job["status"], job["stage"], job["progress"])
        finally:
            other.close()

    probe, timeline = process_mod.probe_video, process_mod.read_timeline

    def probe_spy(path):
        peek("probe")
        return probe(path)

    def timeline_spy(path):
        peek("timeline")
        return timeline(path)

    monkeypatch.setattr(process_mod, "probe_video", probe_spy)
    monkeypatch.setattr(process_mod, "read_timeline", timeline_spy)

    class Spy(FakeDetector):
        def detect(self, image):
            if self.calls == 0:
                peek("first frame")
            return super().detect(image)

    sid = add_session(conn, write_object_video(tmp_path, 30, [moving_object(0, 11)]), WALKING)
    stages = []
    process_session(conn, sid, Spy(), cfg, progress=lambda s, f: stages.append((s, f)))
    assert seen["probe"] == ("running", "reading video", None)
    assert seen["timeline"] == ("running", "reading video", None)
    assert seen["first frame"] == ("running", "detecting", 0.0)
    assert stages[0] == ("detecting", 0.0)  # no progress callback with a missing fraction
    assert all(isinstance(f, float) for _, f in stages)
