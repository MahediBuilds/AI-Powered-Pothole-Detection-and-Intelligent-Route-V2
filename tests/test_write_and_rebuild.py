"""Regression tests: the map rebuild survives a locked database and never leaves a half-done state."""

import json
import sqlite3

import pytest
from api_fixtures import make_api_cfg
from pipeline_fixtures import FakeDetector

from geo_fixtures import add_fixes, add_obs, add_session, edge_point, make_grid, path_points
from roadsense import mapstate
from roadsense.mapstate import build_is_current, write_and_rebuild
from roadsense.storage import db


@pytest.fixture(scope="module")
def graph():
    return make_grid()


@pytest.fixture
def cfg(tmp_path):
    return make_api_cfg(tmp_path)


@pytest.fixture
def conn(cfg):
    c = db.connect(cfg.db_path)
    db.init_schema(c)
    c.execute("PRAGMA busy_timeout=50")  # a lock is noticed after 50 ms, not 5 s
    yield c
    c.close()


@pytest.fixture
def other(cfg, conn):
    """A second connection, to hold the write lock."""
    c = db.connect(cfg.db_path)
    yield c
    c.close()


class Sleeps(list):
    """An injected ``sleep`` that records the delays (and optionally acts on each call)."""

    def __init__(self, on_call=None):
        super().__init__()
        self.on_call = on_call

    def __call__(self, delay):
        self.append(delay)
        if self.on_call is not None:
            self.on_call(len(self))


def seed(conn, graph, sid="s1"):
    add_session(conn, sid)
    add_fixes(conn, sid, path_points(graph, [0, 1, 2]))
    add_obs(conn, sid, *edge_point(graph, 0, 1, 0, 40), track=1)


def snapshot(conn):
    return {
        "builds": [tuple(r) for r in conn.execute("SELECT id, graph_sha256, params_json FROM builds")],
        "potholes": [tuple(r) for r in conn.execute("SELECT * FROM potholes ORDER BY id")],
        "cov": [tuple(r) for r in conn.execute("SELECT * FROM edge_coverage ORDER BY rowid")],
        "obs": [tuple(r) for r in conn.execute("SELECT id, pothole_id FROM observations ORDER BY id")],
    }


# --------------------------------------------------------------------------- locking


def test_succeeds_after_waiting_for_a_short_lock(conn, other, cfg, graph):
    seed(conn, graph)
    other.execute("BEGIN IMMEDIATE")  # another writer holds the lock

    def release_after_first_wait(n):
        if n == 1:
            other.rollback()

    sleep = Sleeps(release_after_first_wait)
    summary = write_and_rebuild(conn, cfg, graph, base_delay_s=0.01, sleep=sleep)
    assert sleep == [0.01]  # one failed attempt, one pause, then it went through
    assert summary["n_potholes"] == 1
    assert conn.execute("SELECT COUNT(*) FROM potholes").fetchone()[0] == 1
    assert not conn.in_transaction  # committed


def test_the_pause_doubles_between_attempts(conn, other, cfg, graph):
    seed(conn, graph)
    other.execute("BEGIN IMMEDIATE")
    sleep = Sleeps(lambda n: other.rollback() if n == 3 else None)
    write_and_rebuild(conn, cfg, graph, base_delay_s=0.01, sleep=sleep)
    assert sleep == [0.01, 0.02, 0.04]


def test_a_lock_that_never_clears_raises_after_exactly_attempts_tries_and_changes_nothing(
    conn, other, cfg, graph
):
    seed(conn, graph)
    write_and_rebuild(conn, cfg, graph)  # a good build is in place
    # change the data so that a successful rebuild would look different
    add_obs(conn, "s1", *edge_point(graph, 1, 2, 0, 50), track=2)
    before = snapshot(conn)
    assert before["potholes"] and before["obs"][-1][1] is None  # the new sighting is not mapped yet
    other.execute("BEGIN IMMEDIATE")
    sleep = Sleeps()
    attempts = []
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        write_and_rebuild(
            conn, cfg, graph, attempts=4, base_delay_s=0.01, sleep=sleep,
            before=lambda c: attempts.append(1),
        )  # fmt: skip
    assert sleep == [0.01, 0.02, 0.04]  # pauses between 4 attempts, none after the last
    assert attempts == []  # the lock was never obtained, so no attempt got as far as `before`
    assert not conn.in_transaction
    other.rollback()
    assert snapshot(conn) == before  # the previous build is intact


def test_each_attempt_that_gets_the_lock_runs_before_once_and_only_one_commit_lands(conn, cfg, graph):
    seed(conn, graph)
    calls = []

    def before(c):
        calls.append(1)
        c.execute("UPDATE sessions SET name = name || '!' WHERE id = 's1'")
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")

    sleep = Sleeps()
    write_and_rebuild(conn, cfg, graph, before=before, base_delay_s=0.01, sleep=sleep)
    assert len(calls) == 2 and sleep == [0.01]
    # the failed first attempt was rolled back: the change is applied once, not twice
    assert conn.execute("SELECT name FROM sessions WHERE id = 's1'").fetchone()[0] == "s1!"


def test_the_last_lock_error_is_raised_when_every_attempt_fails_inside_the_transaction(conn, cfg, graph):
    seed(conn, graph)

    def before(c):
        raise sqlite3.OperationalError("database table is locked")

    sleep = Sleeps()
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        write_and_rebuild(conn, cfg, graph, before=before, attempts=3, base_delay_s=0.01, sleep=sleep)
    assert sleep == [0.01, 0.02] and not conn.in_transaction


def test_a_busy_error_is_retried_too(conn, cfg, graph):
    seed(conn, graph)
    calls = []

    def before(c):
        calls.append(1)
        if len(calls) < 2:
            raise sqlite3.OperationalError("database is busy")

    write_and_rebuild(conn, cfg, graph, before=before, base_delay_s=0.0, sleep=Sleeps())
    assert len(calls) == 2


# --------------------------------------------------------------------------- other errors


def test_an_error_that_is_not_a_lock_is_not_retried(conn, cfg, graph):
    seed(conn, graph)
    calls = []

    def before(c):
        calls.append(1)
        raise ValueError("not a lock problem")

    sleep = Sleeps()
    with pytest.raises(ValueError):
        write_and_rebuild(conn, cfg, graph, before=before, sleep=sleep)
    assert calls == [1] and sleep == [] and not conn.in_transaction


def test_an_operational_error_that_is_not_a_lock_is_not_retried(conn, cfg, graph):
    seed(conn, graph)
    sleep = Sleeps()
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        write_and_rebuild(
            conn, cfg, graph, before=lambda c: c.execute("SELECT * FROM nothing_here"), sleep=sleep
        )  # fmt: skip
    assert sleep == [] and not conn.in_transaction


def test_before_and_the_rebuild_are_atomic(conn, cfg, graph, monkeypatch):
    seed(conn, graph)
    write_and_rebuild(conn, cfg, graph)
    before = snapshot(conn)

    def boom(*args, **kwargs):
        raise RuntimeError("rebuild failed")

    monkeypatch.setattr(mapstate, "rebuild_map", boom)
    with pytest.raises(RuntimeError, match="rebuild failed"):
        write_and_rebuild(
            conn, cfg, graph, include_synthetic=True,
            before=lambda c: c.execute("UPDATE sessions SET include_in_map = 0"),
        )  # fmt: skip
    assert conn.execute("SELECT include_in_map FROM sessions").fetchone()[0] == 1  # rolled back
    assert not conn.in_transaction
    assert snapshot(conn) == before
    # the remembered demo-mode choice made inside the failed transaction is gone too
    assert conn.execute("SELECT 1 FROM meta WHERE key = 'include_synthetic'").fetchone() is None


def test_a_connection_with_an_open_transaction_is_refused_and_left_alone(conn, cfg, graph):
    seed(conn, graph)
    conn.execute("UPDATE sessions SET name = 'pending'")  # opens a transaction
    assert conn.in_transaction
    with pytest.raises(RuntimeError, match="no open transaction"):
        write_and_rebuild(conn, cfg, graph)
    assert conn.in_transaction  # not rolled back behind the caller's back
    assert conn.execute("SELECT name FROM sessions").fetchone()[0] == "pending"
    conn.rollback()


def test_attempts_must_be_positive(conn, cfg, graph):
    with pytest.raises(ValueError):
        write_and_rebuild(conn, cfg, graph, attempts=0)


def test_works_after_plain_selects_and_after_a_commit(conn, cfg, graph):
    seed(conn, graph)
    conn.execute("SELECT * FROM sessions").fetchall()
    conn.execute("SELECT * FROM potholes").fetchall()
    assert not conn.in_transaction  # a read opens no transaction in this isolation mode
    write_and_rebuild(conn, cfg, graph)
    conn.execute("UPDATE sessions SET name = 'x'")
    conn.commit()
    write_and_rebuild(conn, cfg, graph)
    assert conn.execute("SELECT COUNT(*) FROM builds").fetchone()[0] == 1


# --------------------------------------------------------------------------- build_is_current


def test_build_is_current_with_no_build_is_false(conn, cfg, graph):
    assert build_is_current(conn, graph, cfg) is False
    assert build_is_current(conn, None, cfg) is False


def test_build_is_current_without_sessions_and_without_a_graph(conn, cfg, graph):
    write_and_rebuild(conn, cfg, None)
    assert build_is_current(conn, None, cfg) is True
    assert build_is_current(conn, graph, cfg) is False  # the build was made with no graph


def test_build_is_current_with_sessions_and_a_graph(conn, cfg, graph):
    seed(conn, graph)
    write_and_rebuild(conn, cfg, graph)
    assert build_is_current(conn, graph, cfg) is True
    assert build_is_current(conn, make_grid(rows=3, cols=3), cfg) is False  # another graph
    assert build_is_current(conn, None, cfg) is False


def test_a_job_that_finished_without_a_rebuild_makes_the_build_stale(conn, cfg, graph):
    seed(conn, graph)
    write_and_rebuild(conn, cfg, graph)
    seed(conn, graph, "s2")  # a new session whose job is done, but no rebuild happened
    assert build_is_current(conn, graph, cfg) is False
    write_and_rebuild(conn, cfg, graph)
    assert build_is_current(conn, graph, cfg) is True


def test_a_reprocessed_session_makes_the_build_stale(conn, cfg, graph):
    seed(conn, graph)
    write_and_rebuild(conn, cfg, graph)
    conn.execute(
        "INSERT INTO jobs (id, session_id, status, created_at, updated_at)"
        " VALUES ('job2', 's1', 'done', 't', 't')"
    )
    conn.execute("UPDATE sessions SET active_job_id = 'job2' WHERE id = 's1'")
    conn.commit()
    assert build_is_current(conn, graph, cfg) is False


def test_a_session_changing_include_in_map_or_failing_makes_the_build_stale(conn, cfg, graph):
    seed(conn, graph)
    write_and_rebuild(conn, cfg, graph)
    conn.execute("UPDATE sessions SET include_in_map = 0")
    conn.commit()
    assert build_is_current(conn, graph, cfg) is False
    write_and_rebuild(conn, cfg, graph)
    assert build_is_current(conn, graph, cfg) is True
    # a session that is not counted at all (no done job) changes nothing
    add_session(conn, "queued", job_status="queued")
    assert build_is_current(conn, graph, cfg) is True


def test_an_old_build_without_a_jobs_record_is_rebuilt(conn, cfg, graph):
    seed(conn, graph)
    write_and_rebuild(conn, cfg, graph)
    params = json.loads(conn.execute("SELECT params_json FROM builds").fetchone()[0])
    del params["jobs"]
    conn.execute("UPDATE builds SET params_json = ?", (json.dumps(params),))
    conn.commit()
    assert build_is_current(conn, graph, cfg) is False
    # the same with no sessions at all
    conn.execute("DELETE FROM sessions")
    conn.commit()
    assert build_is_current(conn, graph, cfg) is False
    write_and_rebuild(conn, cfg, graph)
    assert build_is_current(conn, graph, cfg) is True


def test_a_build_with_unreadable_params_is_rebuilt(conn, cfg, graph):
    write_and_rebuild(conn, cfg, graph)
    conn.execute("UPDATE builds SET params_json = 'not json'")
    conn.commit()
    assert build_is_current(conn, graph, cfg) is False


def test_changing_demo_mode_makes_the_build_stale(conn, cfg, graph):
    seed(conn, graph)
    add_session(conn, "syn", gps_source="synthetic")
    write_and_rebuild(conn, cfg, graph)  # demo mode off: the synthetic session is left out
    assert build_is_current(conn, graph, cfg) is True
    mapstate.set_include_synthetic(conn, True)
    conn.commit()
    assert build_is_current(conn, graph, cfg) is False  # the build must now include it
    write_and_rebuild(conn, cfg, graph)
    assert build_is_current(conn, graph, cfg) is True
    write_and_rebuild(conn, cfg, graph, include_synthetic=False)
    assert build_is_current(conn, graph, cfg) is True


def test_ensure_map_current_rebuilds_only_when_needed(conn, cfg, graph):
    seed(conn, graph)
    assert mapstate.ensure_map_current(conn, cfg, graph) is True
    conn.commit()
    assert mapstate.ensure_map_current(conn, cfg, graph) is False


# --------------------------------------------------------------------------- restart rebuilds


def test_a_restart_rebuilds_a_map_that_missed_a_finished_job(tmp_path, graph):
    """A job finished but the rebuild never happened (locked database, crash): the next start repairs it."""
    from fastapi.testclient import TestClient

    from roadsense.api.app import create_app

    cfg = make_api_cfg(tmp_path)
    with TestClient(create_app(cfg, detector_factory=FakeDetector, graph=graph)):
        pass  # first start: schema and an empty build
    conn = db.connect(cfg.db_path)
    try:
        seed(conn, graph)  # a done job with a pothole, but no rebuild
        assert build_is_current(conn, graph, cfg) is False
        assert conn.execute("SELECT COUNT(*) FROM potholes").fetchone()[0] == 0
    finally:
        conn.close()
    with TestClient(create_app(cfg, detector_factory=FakeDetector, graph=graph)) as client:
        assert len(client.get("/api/potholes").json()["potholes"]) == 1
    conn = db.connect(cfg.db_path)
    try:
        assert build_is_current(conn, graph, cfg) is True
    finally:
        conn.close()
    # a start with nothing to repair does not rebuild again
    conn = db.connect(cfg.db_path)
    build_before = conn.execute("SELECT MAX(id), created_at FROM builds").fetchone()
    conn.close()
    with TestClient(create_app(cfg, detector_factory=FakeDetector, graph=graph)):
        pass
    conn = db.connect(cfg.db_path)
    try:
        assert tuple(conn.execute("SELECT MAX(id), created_at FROM builds").fetchone()) == tuple(
            build_before
        )
    finally:
        conn.close()


# --------------------------------------------------------------------------- settings and version


def test_changing_a_mapping_setting_makes_the_build_stale(conn, cfg, graph):
    seed(conn, graph)
    write_and_rebuild(conn, cfg, graph)
    assert build_is_current(conn, graph, cfg) is True
    changed = cfg.model_copy(
        update={
            "mapping": cfg.mapping.model_copy(
                update={"dedup_radius_m": cfg.mapping.dedup_radius_m + 1.0}
            )
        }
    )
    assert build_is_current(conn, graph, changed) is False
    write_and_rebuild(conn, changed, graph)
    assert build_is_current(conn, graph, changed) is True
    assert build_is_current(conn, graph, cfg) is False


def test_changing_the_gps_gap_setting_makes_the_build_stale(conn, cfg, graph):
    seed(conn, graph)
    write_and_rebuild(conn, cfg, graph)
    changed = cfg.model_copy(
        update={
            "ingest": cfg.ingest.model_copy(update={"max_gps_gap_s": cfg.ingest.max_gps_gap_s + 1})
        }
    )
    assert build_is_current(conn, graph, changed) is False


def test_a_different_pipeline_version_makes_the_build_stale(conn, cfg, graph):
    seed(conn, graph)
    write_and_rebuild(conn, cfg, graph)
    assert build_is_current(conn, graph, cfg) is True
    conn.execute("UPDATE builds SET pipeline_version = '0.0.0-old'")
    conn.commit()
    assert build_is_current(conn, graph, cfg) is False
    write_and_rebuild(conn, cfg, graph)
    assert build_is_current(conn, graph, cfg) is True


def test_unchanged_settings_do_not_loop_rebuilds(conn, cfg, graph):
    seed(conn, graph)
    assert mapstate.ensure_map_current(conn, cfg, graph) is True
    conn.commit()
    for _ in range(3):
        assert mapstate.ensure_map_current(conn, cfg, graph) is False
    assert conn.execute("SELECT COUNT(*) FROM builds").fetchone()[0] == 1
