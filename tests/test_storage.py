import sqlite3

import pytest

from roadsense.storage import db


def test_connect_creates_parent_and_sets_pragmas(tmp_path):
    path = tmp_path / "nested" / "dir" / "app.sqlite"
    conn = db.connect(path)
    try:
        assert path.parent.is_dir()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert conn.row_factory is not None
    finally:
        conn.close()


def test_foreign_keys_are_enforced(tmp_path):
    conn = db.connect(tmp_path / "fk.sqlite")
    try:
        conn.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE child (pid INTEGER REFERENCES parent(id))")
        with pytest.raises(Exception, match="FOREIGN KEY"):
            conn.execute("INSERT INTO child (pid) VALUES (99)")
    finally:
        conn.close()


def test_init_schema_creates_meta_with_version(tmp_path):
    conn = db.connect(tmp_path / "a.sqlite")
    try:
        db.init_schema(conn)
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        assert row["value"] == str(db.SCHEMA_VERSION)
    finally:
        conn.close()


def test_init_schema_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "a.sqlite")
    try:
        db.init_schema(conn)
        db.init_schema(conn)
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
        assert [(r["key"], r["value"]) for r in rows] == [("schema_version", str(db.SCHEMA_VERSION))]
    finally:
        conn.close()


def test_reopening_existing_database_works(tmp_path):
    path = tmp_path / "a.sqlite"
    for _ in range(2):
        conn = db.connect(path)
        try:
            db.init_schema(conn)
        finally:
            conn.close()


def test_schema_version_mismatch_raises(tmp_path):
    path = tmp_path / "a.sqlite"
    conn = db.connect(path)
    try:
        db.init_schema(conn)
        conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
        conn.commit()
        with pytest.raises(db.SchemaVersionError, match="999"):
            db.init_schema(conn)
    finally:
        conn.close()


# --------------------------------------------------------------------------- real schema


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "schema.sqlite")
    db.init_schema(c)
    yield c
    c.close()


def _session(conn, sid="s1", **extra):
    cols = {"id": sid, "name": "n", "video_path": "v.mp4", "created_at": "t", **extra}
    conn.execute(
        f"INSERT INTO sessions ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        list(cols.values()),
    )


def _job(conn, jid="j1", sid="s1", status="done"):
    conn.execute(
        "INSERT INTO jobs (id, session_id, status, created_at, updated_at) VALUES (?,?,?,?,?)",
        (jid, sid, status, "t", "t"),
    )


def _observation(conn, sid="s1", jid="j1", track=1, pothole_id=None):
    conn.execute(
        "INSERT INTO observations (session_id, job_id, track_id, n_frames, confidence, t_video_s,"
        " frame_index, x1, y1, x2, y2, pothole_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (sid, jid, track, 3, 0.9, 1.0, 30, 0, 0, 10, 10, pothole_id),
    )


def _count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_all_tables_exist(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "meta", "sessions", "gps_fixes", "jobs", "builds", "potholes",
        "observations", "edge_coverage", "edge_stats",
    } <= names  # fmt: skip
    indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {
        "idx_observations_session", "idx_observations_edge", "idx_observations_pothole",
        "idx_edge_coverage_edge", "idx_edge_coverage_session", "idx_jobs_session",
        "idx_potholes_edge",
    } <= indexes  # fmt: skip


def test_deleting_session_cascades(conn):
    _session(conn)
    _job(conn)
    conn.execute("INSERT INTO gps_fixes (session_id, seq, t_video_s, lat, lon) VALUES ('s1',0,0,1,2)")
    _observation(conn)
    conn.execute(
        "INSERT INTO edge_coverage (session_id, job_id, edge_u, edge_v, edge_key, start_m, end_m)"
        " VALUES ('s1','j1',1,2,0,0,5)"
    )
    conn.execute("DELETE FROM sessions WHERE id='s1'")
    for table in ("gps_fixes", "jobs", "observations", "edge_coverage"):
        assert _count(conn, table) == 0, table


def test_deleting_build_cascades_and_nulls_observation_pothole(conn):
    _session(conn)
    _job(conn)
    conn.execute("INSERT INTO builds (id, created_at, graph_sha256, params_json) VALUES (1,'t','g','{}')")
    conn.execute(
        "INSERT INTO potholes (id, build_id, edge_u, edge_v, edge_key, edge_offset_m, lat, lon,"
        " n_observations, n_sessions, confidence, gps_source) VALUES (7,1,1,2,0,3.0,0,0,1,1,0.5,'recorded')"
    )
    conn.execute(
        "INSERT INTO edge_stats (edge_u, edge_v, edge_key, build_id, length_m, surveyed_m,"
        " n_potholes, rate_per_100m) VALUES (1,2,0,1,100,50,1,2)"
    )
    _observation(conn, pothole_id=7)
    assert conn.execute("SELECT pothole_id FROM observations").fetchone()[0] == 7
    conn.execute("DELETE FROM builds WHERE id=1")
    assert _count(conn, "potholes") == 0 and _count(conn, "edge_stats") == 0
    assert _count(conn, "observations") == 1
    assert conn.execute("SELECT pothole_id FROM observations").fetchone()[0] is None


def test_check_constraints(conn):
    with pytest.raises(sqlite3.IntegrityError):
        _session(conn, role="admin")
    with pytest.raises(sqlite3.IntegrityError):
        _session(conn, include_in_map=2)
    _session(conn)
    _job(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _job(conn, jid="j2", status="paused")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO edge_coverage (session_id, job_id, edge_u, edge_v, edge_key, start_m, end_m)"
            " VALUES ('s1','j1',1,2,0,9,1)"
        )


def test_session_defaults(conn):
    _session(conn)
    row = conn.execute("SELECT role, include_in_map, active_job_id FROM sessions").fetchone()
    assert (row["role"], row["include_in_map"], row["active_job_id"]) == ("survey", 1, None)


def test_duplicate_job_track_rejected(conn):
    _session(conn)
    _job(conn)
    _observation(conn, track=1)
    with pytest.raises(sqlite3.IntegrityError):
        _observation(conn, track=1)
    _observation(conn, track=2)


def test_bind_graph_set_once_and_mismatch_raises(conn):
    db.bind_graph(conn, "abc")
    db.bind_graph(conn, "abc")  # same hash is fine
    with pytest.raises(db.GraphMismatchError, match="abc"):
        db.bind_graph(conn, "def")
    assert conn.execute("SELECT value FROM meta WHERE key='graph_sha256'").fetchone()[0] == "abc"
