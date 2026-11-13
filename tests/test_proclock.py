"""Regression tests: a second server must not fail another live server's jobs."""

import json
import subprocess
import sys
import time

import pytest
import yaml
from api_fixtures import make_api_cfg
from fastapi.testclient import TestClient
from pipeline_fixtures import FakeDetector

from geo_fixtures import add_session
from roadsense.api.app import create_app
from roadsense.storage import db
from roadsense.storage.jobs import STALE_JOB_ERROR
from roadsense.storage.proclock import AlreadyRunningError, ProcessLock, lock_path_for


@pytest.fixture
def lock_file(tmp_path):
    return tmp_path / "sub" / "x.sqlite.lock"


# --------------------------------------------------------------------------- ProcessLock


def test_lock_path_is_next_to_the_database(tmp_path):
    assert lock_path_for(tmp_path / "a.sqlite").name == "a.sqlite.lock"


def test_a_second_lock_on_the_same_path_fails_in_the_same_process(lock_file):
    first = ProcessLock(lock_file, role="server").acquire()
    try:
        assert first.held
        second = ProcessLock(lock_file, role="command line")
        with pytest.raises(AlreadyRunningError, match="already working on this database") as exc:
            second.acquire()
        assert "server" in str(exc.value) and "process id" in str(exc.value)  # names the holder
        assert not second.held
        assert first.held  # the failed attempt did not disturb the holder
    finally:
        first.release()


def test_release_then_acquire_again_works(lock_file):
    lock = ProcessLock(lock_file).acquire()
    lock.release()
    assert not lock.held
    again = ProcessLock(lock_file).acquire()  # a new instance
    again.release()
    lock.acquire()  # and the old instance can be reused
    assert lock.held
    lock.release()


def test_acquire_and_release_are_idempotent(lock_file):
    lock = ProcessLock(lock_file)
    assert lock.acquire() is lock.acquire()
    lock.release()
    lock.release()  # releasing twice, or never having acquired, is harmless
    ProcessLock(lock_file).release()


def test_context_manager_holds_only_inside_the_block(lock_file):
    with ProcessLock(lock_file) as lock:
        assert lock.held
        with pytest.raises(AlreadyRunningError):
            ProcessLock(lock_file).acquire()
    ProcessLock(lock_file).acquire().release()


def test_locks_on_different_paths_do_not_interfere(tmp_path):
    a = ProcessLock(tmp_path / "a.lock").acquire()
    b = ProcessLock(tmp_path / "b.lock").acquire()
    a.release()
    b.release()


def test_the_info_file_names_the_holder_while_held_and_is_removed_after(lock_file):
    lock = ProcessLock(lock_file, role="demo script").acquire()
    info = json.loads(lock.info_path.read_text(encoding="utf-8"))
    assert info["role"] == "demo script" and isinstance(info["pid"], int) and info["started_utc"]
    lock.release()
    assert not lock.info_path.exists()
    assert lock_file.exists()  # the lock file itself is never deleted


def test_a_lock_held_by_another_process_blocks_us_until_that_process_dies(lock_file, tmp_path):
    code = (
        "import sys, time\n"
        "from roadsense.storage.proclock import ProcessLock\n"
        "lock = ProcessLock(sys.argv[1], role='other').acquire()  # keep a reference alive\n"
        "print('ready', flush=True)\n"
        "time.sleep(120)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(lock_file)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(AlreadyRunningError) as exc:
            ProcessLock(lock_file).acquire()
        assert "other" in str(exc.value)
    finally:
        child.kill()  # no clean-up code runs: the operating system must release the lock
        child.wait(timeout=30)
        child.stdout.close()
    # the operating system may take a moment to drop a dead process's locks (seen on Windows)
    deadline = time.monotonic() + 10
    while True:
        try:
            ProcessLock(lock_file).acquire().release()
            break
        except AlreadyRunningError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


# --------------------------------------------------------------------------- the application


def _app(cfg):
    return TestClient(create_app(cfg, detector_factory=FakeDetector, graph=None))


def _job_status(cfg, job_id):
    conn = db.connect(cfg.db_path)
    try:
        return tuple(conn.execute("SELECT status, error FROM jobs WHERE id = ?", (job_id,)).fetchone())
    finally:
        conn.close()


def test_a_second_app_on_the_same_database_does_not_fail_the_live_apps_jobs(tmp_path):
    cfg = make_api_cfg(tmp_path)
    with _app(cfg) as client:
        conn = db.connect(cfg.db_path)
        try:
            job_id = add_session(conn, "live", job_status="running")
        finally:
            conn.close()
        assert client.get("/api/health").status_code == 200

        with pytest.raises(AlreadyRunningError):
            with _app(cfg):
                pytest.fail("the second app must not start")  # pragma: no cover
        assert _job_status(cfg, job_id) == ("running", None)  # untouched
        assert client.get("/api/health").status_code == 200  # and the live app is unaffected

        with pytest.raises(AlreadyRunningError):  # a failed start did not take over the lock either
            with _app(cfg):
                pytest.fail("the second app must not start")  # pragma: no cover

    # the first app has shut down: a new one can start, and only now is the job known to be orphaned
    with _app(cfg):
        status, error = _job_status(cfg, job_id)
        assert status == "failed" and error == STALE_JOB_ERROR


def test_queued_jobs_are_also_left_alone_by_a_refused_start(tmp_path):
    cfg = make_api_cfg(tmp_path)
    with _app(cfg):
        conn = db.connect(cfg.db_path)
        try:
            job_id = add_session(conn, "q", job_status="queued")
        finally:
            conn.close()
        with pytest.raises(AlreadyRunningError):
            with _app(cfg):
                pass  # pragma: no cover
        assert _job_status(cfg, job_id)[0] == "queued"


def test_the_lock_is_released_when_the_app_fails_to_start_for_another_reason(tmp_path):
    cfg = make_api_cfg(tmp_path)
    cfg.db_path.parent.mkdir(parents=True)
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(db.SchemaVersionError):
        with _app(cfg):
            pass  # pragma: no cover
    ProcessLock(lock_path_for(cfg.db_path)).acquire().release()  # not left locked


def test_lock_is_kept_after_shutdown_when_the_worker_outlives_the_stop_timeout(tmp_path):
    import gc
    import threading

    from pipeline_fixtures import add_session as add_pipeline_session

    from roadsense.api import app as app_mod

    cfg = make_api_cfg(tmp_path)
    started, release = threading.Event(), threading.Event()

    def blocking_factory():
        started.set()
        release.wait(60)
        return FakeDetector()

    before = list(app_mod._HELD_LOCKS)
    thread = None
    try:
        app = create_app(cfg, detector_factory=blocking_factory, graph=None, stop_timeout_s=0.05)
        with TestClient(app):
            conn = db.connect(cfg.db_path)
            try:
                sid = add_pipeline_session(conn, tmp_path / "missing.mp4")
            finally:
                conn.close()
            runner = app.state.rs.runner
            runner.submit(sid)
            assert started.wait(30)  # the worker is now stuck inside the job
            thread = runner._thread
        # lifespan over, stop() timed out: the lock object must still be referenced
        assert len(app_mod._HELD_LOCKS) == len(before) + 1
        gc.collect()
        with pytest.raises(AlreadyRunningError):
            ProcessLock(lock_path_for(cfg.db_path)).acquire()
    finally:
        release.set()
        if thread is not None:
            thread.join(30)
        for lock in app_mod._HELD_LOCKS[len(before) :]:
            lock.release()
        del app_mod._HELD_LOCKS[len(before) :]
    ProcessLock(lock_path_for(cfg.db_path)).acquire().release()  # free again


def test_python_m_roadsense_api_refuses_while_the_database_is_in_use(tmp_path, capsys):
    from roadsense.api.__main__ import main

    cfg = make_api_cfg(tmp_path)
    path = tmp_path / "app.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(cfg.data_dir),
                "db_path": str(cfg.db_path),
                "uploads": {
                    "dir": str(cfg.uploads.dir),
                    "artifacts_dir": str(cfg.uploads.artifacts_dir),
                },
            }
        ),
        encoding="utf-8",
    )
    with ProcessLock(lock_path_for(cfg.db_path), role="server"):
        assert main(["--config", str(path)]) == 1
    assert "already working on this database" in capsys.readouterr().err
