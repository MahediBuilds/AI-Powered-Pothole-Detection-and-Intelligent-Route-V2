import json
import sqlite3
import time

import pytest

from roadsense.storage import db, jobs, observations


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "jobs.sqlite")
    db.init_schema(c)
    for sid in ("s1", "s2"):
        c.execute(
            "INSERT INTO sessions (id, name, video_path, created_at) VALUES (?, 'n', 'v.mp4', 't')",
            (sid,),
        )
    c.commit()
    yield c
    c.close()


def obs_row(track, t=1.0, **extra):
    return {
        "track_id": track, "n_frames": 3, "confidence": 0.8, "t_video_s": t,
        "frame_index": 30, "x1": 1.0, "y1": 2.0, "x2": 30.0, "y2": 40.0, **extra,
    }  # fmt: skip


def test_create_job_is_queued(conn):
    jid = jobs.create_job(conn, "s1")
    job = jobs.get_job(conn, jid)
    assert job["status"] == "queued" and job["session_id"] == "s1"
    assert job["created_at"] == job["updated_at"]
    assert jobs.get_job(conn, "nope") is None


def test_create_job_unknown_session_violates_fk(conn):
    with pytest.raises(sqlite3.IntegrityError):
        jobs.create_job(conn, "missing")


def test_running_progress_and_updated_at(conn):
    jid = jobs.create_job(conn, "s1")
    before = jobs.get_job(conn, jid)["updated_at"]
    time.sleep(0.01)
    jobs.mark_running(conn, jid, stage="detecting", pipeline_version="v", model_sha256="abc",
                      params={"b": 1, "a": 2})  # fmt: skip
    job = jobs.get_job(conn, jid)
    assert job["status"] == "running" and job["stage"] == "detecting" and job["progress"] == 0.0
    assert job["updated_at"] > before
    assert job["pipeline_version"] == "v" and job["model_sha256"] == "abc"
    assert json.loads(job["params_json"]) == {"a": 2, "b": 1}
    jobs.set_progress(conn, jid, "saving", 7.0)  # clamped
    job = jobs.get_job(conn, jid)
    assert (job["stage"], job["progress"]) == ("saving", 1.0)


def test_mark_done_sets_summary_and_active_job(conn):
    jid = jobs.create_job(conn, "s1")
    jobs.mark_running(conn, jid)
    jobs.mark_done(conn, jid, {"n": 2})
    job = jobs.get_job(conn, jid)
    assert job["status"] == "done" and job["progress"] == 1.0
    assert json.loads(job["summary_json"]) == {"n": 2}
    active = conn.execute("SELECT active_job_id FROM sessions WHERE id='s1'").fetchone()[0]
    assert active == jid
    assert conn.execute("SELECT active_job_id FROM sessions WHERE id='s2'").fetchone()[0] is None


def test_mark_failed_keeps_active_job_unchanged(conn):
    j1 = jobs.create_job(conn, "s1")
    jobs.mark_done(conn, j1, {})
    j2 = jobs.create_job(conn, "s1")
    jobs.mark_failed(conn, j2, "boom")
    assert jobs.get_job(conn, j2)["status"] == "failed"
    assert jobs.get_job(conn, j2)["error"] == "boom"
    assert conn.execute("SELECT active_job_id FROM sessions WHERE id='s1'").fetchone()[0] == j1


def test_updates_on_unknown_job_raise(conn):
    with pytest.raises(KeyError):
        jobs.mark_failed(conn, "nope", "x")
    with pytest.raises(KeyError):
        jobs.set_progress(conn, "nope", "x", 0.5)


def test_fail_stale_running_jobs(conn):
    q = jobs.create_job(conn, "s1")
    r = jobs.create_job(conn, "s1")
    jobs.mark_running(conn, r)
    d = jobs.create_job(conn, "s2")
    jobs.mark_done(conn, d, {})
    assert jobs.fail_stale_running_jobs(conn) == 2
    assert jobs.get_job(conn, q)["status"] == "failed"
    assert jobs.get_job(conn, r)["error"] == jobs.STALE_JOB_ERROR
    assert jobs.get_job(conn, d)["status"] == "done"


def test_latest_job_for_session(conn):
    assert jobs.latest_job_for_session(conn, "s1") is None
    j1 = jobs.create_job(conn, "s1")
    j2 = jobs.create_job(conn, "s1")
    jobs.create_job(conn, "s2")
    assert jobs.latest_job_for_session(conn, "s1")["id"] == j2
    assert j1 != j2


def test_no_function_commits(tmp_path):
    path = tmp_path / "c.sqlite"
    a = db.connect(path)
    db.init_schema(a)
    a.execute("INSERT INTO sessions (id, name, video_path, created_at) VALUES ('s','n','v','t')")
    a.commit()
    jobs.create_job(a, "s")
    b = db.connect(path)
    try:
        assert b.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        a.commit()
        assert b.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    finally:
        a.close()
        b.close()


def test_insert_and_list_observations_active_job_only(conn):
    j1 = jobs.create_job(conn, "s1")
    assert observations.insert_observations(conn, "s1", j1, [obs_row(1, 2.0), obs_row(2, 1.0, cam_lat=5.0,
                                            cam_lon=6.0, image_path="s1/j/track_2.jpg")]) == 2  # fmt: skip
    assert observations.list_observations(conn, "s1") == []  # no active job yet
    jobs.mark_done(conn, j1, {})
    rows = observations.list_observations(conn, "s1")
    assert [r["track_id"] for r in rows] == [2, 1]  # ordered by time
    assert rows[0]["image_path"] == "s1/j/track_2.jpg" and rows[0]["cam_lat"] == 5.0
    assert rows[1]["cam_lat"] is None and rows[1]["edge_u"] is None and rows[1]["pothole_id"] is None
    assert observations.count_observations(conn, "s1") == 2
    assert observations.count_positioned_observations(conn, "s1") == 1

    j2 = jobs.create_job(conn, "s1")
    observations.insert_observations(conn, "s1", j2, [obs_row(1)])
    jobs.mark_done(conn, j2, {})
    assert observations.count_observations(conn, "s1") == 1  # only the new active job
    assert observations.count_observations_for_job(conn, j1) == 2  # old rows remain
    assert len(observations.list_observations_for_job(conn, j1)) == 2


def test_insert_observations_validates_and_enforces_unique(conn):
    j1 = jobs.create_job(conn, "s1")
    bad = obs_row(1)
    del bad["x1"]
    with pytest.raises(KeyError, match="x1"):
        observations.insert_observations(conn, "s1", j1, [obs_row(2), bad])
    assert observations.count_observations_for_job(conn, j1) == 0
    observations.insert_observations(conn, "s1", j1, [obs_row(1)])
    with pytest.raises(sqlite3.IntegrityError):
        observations.insert_observations(conn, "s1", j1, [obs_row(1)])
