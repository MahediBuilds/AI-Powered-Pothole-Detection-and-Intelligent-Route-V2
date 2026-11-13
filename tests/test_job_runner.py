import json
import sqlite3
import threading
import time

import pytest
from pipeline_fixtures import FakeDetector, add_session, make_cfg, moving_object, write_object_video

from roadsense.pipeline.runner import JobRunner
from roadsense.storage import db
from roadsense.storage import jobs as jobs_db
from roadsense.storage import observations as obs_db


@pytest.fixture
def env(tmp_path):
    path = tmp_path / "r.sqlite"
    conn = db.connect(path)
    db.init_schema(conn)
    video = write_object_video(tmp_path, 30, [moving_object(0, 11)])
    fixes = [(float(t), 50.0 + 0.0001 * t, 8.0, 5.0) for t in range(4)]
    sessions = [add_session(conn, video, fixes, name=f"s{i}") for i in range(2)]
    yield path, make_cfg(tmp_path), conn, sessions
    conn.close()


@pytest.fixture
def runners():
    made = []

    def make(*args, **kwargs):
        r = JobRunner(*args, **kwargs)
        made.append(r)
        r.start()
        return r

    yield make
    for r in made:
        r.stop(5)


def job(path, jid):
    c = db.connect(path)
    try:
        return jobs_db.get_job(c, jid)
    finally:
        c.close()


def test_two_sessions_finish_done_in_order_with_one_detector(env, runners):
    path, cfg, conn, (s1, s2) = env
    built = []
    order = []

    def factory():
        built.append(threading.current_thread().name)
        return FakeDetector()

    runner = runners(path, cfg, factory, on_done=lambda c, s, j: order.append(s))
    j1, j2 = runner.submit(s1), runner.submit(s2)
    assert job(path, j1)["status"] in ("queued", "running", "done")
    assert runner.wait_idle(30)
    assert job(path, j1)["status"] == "done" and job(path, j2)["status"] == "done"
    assert order == [s1, s2]
    assert built == ["job-runner"]  # built once, on the worker thread
    assert job(path, j1)["created_at"] <= job(path, j2)["created_at"]
    check = db.connect(path)
    try:
        assert obs_db.count_observations(check, s1) == 1 and obs_db.count_observations(check, s2) == 1
    finally:
        check.close()


def test_submit_creates_queued_row_immediately(env):
    path, cfg, conn, (s1, _) = env
    runner = JobRunner(path, cfg, FakeDetector)  # not started: the job just waits
    jid = runner.submit(s1)
    assert job(path, jid)["status"] == "queued"
    runner.start()
    try:
        assert runner.wait_idle(30)
        assert job(path, jid)["status"] == "done"
    finally:
        runner.stop(5)


def test_submit_unknown_session_raises(env):
    import sqlite3

    path, cfg, *_ = env
    with pytest.raises(sqlite3.IntegrityError):
        JobRunner(path, cfg, FakeDetector).submit("missing")


def test_failing_factory_fails_job_and_next_job_still_runs(env, runners):
    path, cfg, conn, (s1, s2) = env
    calls = []

    def factory():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("no weights here")
        return FakeDetector()

    runner = runners(path, cfg, factory)
    j1 = runner.submit(s1)
    j2 = runner.submit(s2)
    assert runner.wait_idle(30)
    failed = job(path, j1)
    assert failed["status"] == "failed" and "no weights here" in failed["error"]
    assert job(path, j2)["status"] == "done"
    assert len(calls) == 2  # retried on the next job, then reused


def test_failing_detector_fails_job_and_worker_survives(env, runners):
    path, cfg, conn, (s1, s2) = env
    detectors = [FakeDetector(always_fail=True), FakeDetector()]
    runner = runners(path, cfg, lambda: detectors[0])
    j1 = runner.submit(s1)
    assert runner.wait_idle(30)
    assert job(path, j1)["status"] == "failed"
    # the detector is cached, so the same (failing) one is reused; the worker is still alive
    j2 = runner.submit(s2)
    assert runner.wait_idle(30)
    assert job(path, j2)["status"] == "failed"


def test_on_done_called_with_connection_session_and_job(env, runners):
    path, cfg, conn, (s1, _) = env
    got = []

    def on_done(c, sid, jid):
        got.append((sid, jid, c.execute("SELECT status FROM jobs WHERE id=?", (jid,)).fetchone()[0]))

    runner = runners(path, cfg, FakeDetector, on_done=on_done)
    jid = runner.submit(s1)
    assert runner.wait_idle(30)
    assert got == [(s1, jid, "done")]


def test_on_done_not_called_for_failed_job(env, runners):
    path, cfg, conn, (s1, _) = env
    got = []
    runner = runners(path, cfg, lambda: FakeDetector(always_fail=True), on_done=lambda *a: got.append(a))
    runner.submit(s1)
    assert runner.wait_idle(30)
    assert got == []


def test_on_done_exception_is_a_warning_not_a_failure(env, runners):
    path, cfg, conn, (s1, s2) = env

    def on_done(c, sid, jid):
        raise ValueError("map build exploded")

    runner = runners(path, cfg, FakeDetector, on_done=on_done)
    runner.max_post_retries = 0  # no retries: the warning is written at once
    j1 = runner.submit(s1)
    j2 = runner.submit(s2)
    assert runner.wait_idle(30)
    for jid, sid in ((j1, s1), (j2, s2)):
        row = job(path, jid)
        assert row["status"] == "done"
        summary = json.loads(row["summary_json"])
        assert any("map build exploded" in w for w in summary["warnings"])
        assert summary["n_tracks_confirmed"] == 1  # the rest of the report is intact
        check = db.connect(path)
        try:
            assert check.execute("SELECT active_job_id FROM sessions WHERE id=?", (sid,)).fetchone()[0] == jid
        finally:
            check.close()


def wait_for(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.02)


def warnings_of(path, jid):
    raw = job(path, jid)["summary_json"]
    return json.loads(raw).get("warnings", []) if raw else []


def test_failed_on_done_is_retried_while_idle_and_the_debt_is_cleared(env, runners):
    path, cfg, conn, (s1, _) = env
    calls = []

    def on_done(c, sid, jid):
        calls.append(jid)
        if len(calls) < 3:
            raise sqlite3.OperationalError("database is locked")

    runner = runners(path, cfg, FakeDetector, on_done=on_done)
    runner.retry_interval_s = 0.02
    j1 = runner.submit(s1)
    assert runner.wait_idle(30)
    wait_for(lambda: len(calls) == 3 and runner._owed is None)
    assert calls == [j1, j1, j1]  # the same job's step, retried until it worked
    row = job(path, j1)
    assert row["status"] == "done" and not warnings_of(path, j1)  # no warning: it did succeed
    time.sleep(0.15)
    assert len(calls) == 3  # and it is not repeated once settled


def test_on_done_retries_run_out_then_a_warning_is_recorded(env, runners):
    path, cfg, conn, (s1, _) = env
    calls = []

    def on_done(c, sid, jid):
        calls.append(jid)
        raise ValueError("map build exploded")

    runner = runners(path, cfg, FakeDetector, on_done=on_done)
    runner.retry_interval_s = 0.02
    runner.max_post_retries = 2
    j1 = runner.submit(s1)
    wait_for(lambda: any("map build exploded" in w for w in warnings_of(path, j1)))
    assert len(calls) == 3  # first attempt plus two retries
    assert job(path, j1)["status"] == "done" and runner._owed is None
    time.sleep(0.1)
    assert len(calls) == 3  # no further attempts after the warning


def test_a_later_successful_on_done_settles_an_earlier_failure(env, runners):
    path, cfg, conn, (s1, s2) = env
    calls = []

    def on_done(c, sid, jid):
        calls.append(sid)
        if sid == s1:
            raise sqlite3.OperationalError("database is locked")

    runner = runners(path, cfg, FakeDetector, on_done=on_done)
    runner.retry_interval_s = 60.0  # a retry would never get its turn during the test
    j1 = runner.submit(s1)
    assert runner.wait_idle(30)
    assert runner._owed is not None
    runner.submit(s2)
    assert runner.wait_idle(30)
    assert runner._owed is None  # a rebuild is a full rebuild, so it covered the first job too
    assert calls == [s1, s2] and not warnings_of(path, j1)


def test_wait_idle_times_out_when_not_started(env):
    path, cfg, conn, (s1, _) = env
    runner = JobRunner(path, cfg, FakeDetector)
    runner.submit(s1)
    assert runner.wait_idle(0.1) is False
    assert JobRunner(path, cfg, FakeDetector).wait_idle(0.1) is True  # nothing submitted


def test_stop_leaves_waiting_jobs_queued(env):
    path, cfg, conn, (s1, s2) = env
    started = threading.Event()
    release = threading.Event()

    class Slow(FakeDetector):
        def detect(self, image):
            started.set()
            release.wait(10)
            return super().detect(image)

    runner = JobRunner(path, cfg, Slow)
    runner.start()
    j1, j2 = runner.submit(s1), runner.submit(s2)
    assert started.wait(10)
    stopper = threading.Thread(target=runner.stop, args=(20,))
    stopper.start()
    release.set()
    stopper.join(25)
    assert job(path, j1)["status"] == "done"
    assert job(path, j2)["status"] == "queued"
    c = db.connect(path)
    try:
        assert jobs_db.fail_stale_running_jobs(c) == 1
        c.commit()
    finally:
        c.close()
    assert job(path, j2)["status"] == "failed"


def test_stop_returns_false_while_a_job_blocks_and_true_after(env, runners):
    path, cfg, conn, sessions = env
    entered, release = threading.Event(), threading.Event()

    def factory():
        entered.set()
        release.wait(10)
        return FakeDetector()

    runner = runners(path, cfg, factory)
    runner.submit(sessions[0])
    assert entered.wait(10)
    assert runner.stop(timeout=0.1) is False  # the worker is still inside the job
    release.set()
    assert runner.stop(timeout=10) is True


def test_stop_on_a_runner_that_never_started_is_true(env):
    path, cfg, _, _ = env
    assert JobRunner(path, cfg, FakeDetector).stop() is True
