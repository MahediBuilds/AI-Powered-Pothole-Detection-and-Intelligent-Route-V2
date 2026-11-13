import dataclasses
import json
import shutil
from datetime import datetime, timedelta, timezone

import pytest
from ingest_fixtures import write_synthetic_video

from roadsense.ingest.gps import GpsError, GpsFix, GpsLog
from roadsense.ingest.session import SessionError, load_session
from roadsense.ingest.video import VideoError
from roadsense.storage import db
from roadsense.storage.sessions import load_gps_fixes, save_session

MAX_ACC = 25.0
GPS_HEADER = "timestamp_sec,lat,lon,accuracy_m,speed_mps,bearing_deg,altitude_m,utc_ms"
VIDEO_FPS = 2.0  # 20 synthetic frames -> a 10 s video
VIDEO_S = 10.0


def gps_rows(times, accuracy=5.0, lat0=10.0, lon0=50.0):
    """Obviously fake fixes drifting north-east from (10, 50)."""
    return [
        f"{t:.3f},{lat0 + 0.00001 * i:.7f},{lon0 + 0.00001 * i:.7f},{accuracy},1.0,45.0,0.0,{1000 + i}"
        for i, t in enumerate(times)
    ]


def sidecar(**overrides):
    data = {
        "format": "road-recorder/1",
        "started_utc": "2026-01-02T03:04:05.678Z",
        "duration_sec": VIDEO_S,
        "gps_fixes": 11,
        "device": "Test Phone",
        "median_accuracy_m": 4.5,
    }
    data.update(overrides)
    return data


class Rec:
    """Builds a session directory: <stem>.mp4, <stem>_gps.csv, <stem>.json."""

    def __init__(self, tmp_path):
        self.dir = tmp_path / "rec"
        self.dir.mkdir()
        self.video = write_synthetic_video(self.dir, 20, VIDEO_FPS)
        stem = "road_test"
        renamed = self.video.with_name(stem + self.video.suffix)
        self.video.rename(renamed)
        self.video = renamed
        self.stem = stem

    def gps(self, times=tuple(range(-2, 11)), header=GPS_HEADER, rows=None, name=None):
        path = self.dir / (name or f"{self.stem}_gps.csv")
        lines = rows if rows is not None else gps_rows(times)
        path.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")
        return path

    def sidecar(self, data=None, text=None, name=None):
        path = self.dir / (name or f"{self.stem}.json")
        path.write_text(text if text is not None else json.dumps(data or sidecar()), encoding="utf-8")
        return path

    def load(self, **kw):
        return load_session(self.video, max_accuracy_m=MAX_ACC, **kw)


@pytest.fixture
def rec(tmp_path):
    return Rec(tmp_path)


# ---------------------------------------------------------------------------
# Discovery and the happy path
# ---------------------------------------------------------------------------


def test_sibling_files_are_discovered_and_clean_session_has_no_warnings(rec):
    times = [-1, 0, 2, 4, 6, 8, 10, 11]  # pre-roll and overrun, 8 fixes
    gps = rec.gps(times=times)
    sc = rec.sidecar(data=sidecar(gps_fixes=len(times)))
    s = rec.load()
    assert s.name == "road_test"
    assert s.video_path == rec.video and s.gps_path == gps and s.sidecar_path == sc
    assert s.video.duration_s == pytest.approx(VIDEO_S)
    assert s.gps is not None and s.gps.summary.n_used == 8
    assert s.device == "Test Phone" and s.sidecar_median_accuracy_m == 4.5
    assert s.started_utc == datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=timezone.utc)
    assert s.gps_coverage == pytest.approx(1.0)
    assert s.warnings == []


def test_explicit_paths_override_naming_convention(rec, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    gps = other / "trace.csv"
    shutil.copy(rec.gps(times=range(0, 11)), gps)
    sc = other / "meta.json"
    sc.write_text(json.dumps(sidecar(gps_fixes=11)), encoding="utf-8")
    s = load_session(rec.video, gps, sc, max_accuracy_m=MAX_ACC)
    assert s.gps_path == gps and s.sidecar_path == sc
    assert s.gps.summary.n_fixes == 11 and s.device == "Test Phone"


def test_explicit_missing_files_are_errors(rec, tmp_path):
    with pytest.raises(SessionError, match="missing_gps.csv"):
        load_session(rec.video, tmp_path / "missing_gps.csv", max_accuracy_m=MAX_ACC)
    with pytest.raises(SessionError, match="missing.json"):
        load_session(rec.video, sidecar_path=tmp_path / "missing.json", max_accuracy_m=MAX_ACC)


def test_missing_video_is_error_naming_file(tmp_path):
    with pytest.raises(VideoError) as exc:
        load_session(tmp_path / "nope.mp4", max_accuracy_m=MAX_ACC)
    assert "nope.mp4" in str(exc.value)


def test_video_without_gps_loads_with_warning(rec):
    s = rec.load()
    assert s.gps is None and s.gps_path is None and s.sidecar_path is None
    assert s.gps_coverage is None
    assert len(s.warnings) == 1 and "no GPS file" in s.warnings[0]
    assert s.started_utc is None and s.device is None and s.sidecar_median_accuracy_m is None


def test_unreadable_gps_file_is_error_not_warning(rec):
    rec.gps(header="timestamp_sec,lat", rows=["0,10"])
    with pytest.raises(GpsError, match="missing required column"):
        rec.load()


def test_gps_with_no_usable_fixes_warns(rec):
    rec.gps(rows=gps_rows(range(0, 11), accuracy=99.0))
    s = rec.load()
    assert s.gps.summary.n_used == 0 and s.gps_coverage == 0.0
    assert any("no usable GPS fixes" in w for w in s.warnings)


def test_gps_row_issues_are_surfaced(rec):
    rec.gps(rows=[*gps_rows(range(0, 11)), "bad,10,50,5,1,1,0,1"])
    s = rec.load()
    assert len(s.gps.issues) == 1
    assert any("1 issue(s)" in w and "1 row(s) dropped" in w for w in s.warnings)


# ---------------------------------------------------------------------------
# Sidecar handling
# ---------------------------------------------------------------------------


def test_malformed_sidecar_json_is_warning(rec):
    rec.gps(times=range(0, 11))
    rec.sidecar(text="{not json")
    s = rec.load()
    assert s.sidecar_path is not None and s.device is None and s.started_utc is None
    assert any("cannot read as JSON" in w for w in s.warnings)
    assert s.gps is not None


def test_non_object_sidecar_is_warning(rec):
    rec.sidecar(text="[1, 2]")
    s = rec.load()
    assert any("not a JSON object" in w for w in s.warnings)


def test_wrong_format_sidecar_is_warning_and_data_ignored(rec):
    rec.sidecar(data=sidecar(format="other-recorder/2"))
    s = rec.load()
    assert any("other-recorder/2" in w for w in s.warnings)
    assert s.device is None and s.started_utc is None and s.sidecar_median_accuracy_m is None


def test_missing_format_key_is_warning(rec):
    data = sidecar()
    del data["format"]
    rec.sidecar(data=data)
    assert any("format" in w for w in rec.load().warnings)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-10-22T11:18:59.017Z", datetime(2026, 10, 22, 11, 18, 59, 17000, tzinfo=timezone.utc)),
        ("2026-10-22T11:18:59Z", datetime(2026, 10, 22, 11, 18, 59, tzinfo=timezone.utc)),
        ("2026-10-22T11:18:59.1234567Z", datetime(2026, 10, 22, 11, 18, 59, 123456, tzinfo=timezone.utc)),
        ("2026-10-22T11:18:59+00:00", datetime(2026, 10, 22, 11, 18, 59, tzinfo=timezone.utc)),
        ("2026-10-22T16:48:59+05:30", datetime(2026, 10, 22, 11, 18, 59, tzinfo=timezone.utc)),
    ],
)
def test_started_utc_formats_are_parsed_to_utc(rec, text, expected):
    rec.sidecar(data=sidecar(started_utc=text))
    s = rec.load()
    assert s.started_utc == expected
    assert s.started_utc.utcoffset() == timedelta(0)
    assert not any("not an ISO-8601" in w for w in s.warnings)


@pytest.mark.parametrize(
    "bad", ["yesterday", "2026-10-22 11:18:59", "2026-10-22T11:18:59", "2026-13-40T11:18:59Z", 12345]
)
def test_bad_started_utc_becomes_none_with_warning(rec, bad):
    rec.sidecar(data=sidecar(started_utc=bad))
    s = rec.load()
    assert s.started_utc is None
    assert any("not an ISO-8601" in w for w in s.warnings)
    assert s.device == "Test Phone"  # the rest of the sidecar is still used


def test_invalid_sidecar_fields_are_ignored_with_warnings(rec):
    rec.sidecar(data=sidecar(device=5, median_accuracy_m="n/a", duration_sec=-1, gps_fixes="many"))
    s = rec.load()
    assert s.device is None and s.sidecar_median_accuracy_m is None
    joined = " ".join(s.warnings)
    for field in ("device", "median_accuracy_m", "duration_sec", "gps_fixes"):
        assert field in joined


def test_duration_mismatch_warns_only_beyond_half_a_second(rec):
    rec.sidecar(data=sidecar(duration_sec=VIDEO_S + 0.4))
    assert not any("differs" in w for w in rec.load().warnings)
    rec.sidecar(data=sidecar(duration_sec=VIDEO_S + 0.6))
    assert any("differs" in w and "duration" in w for w in rec.load().warnings)


def test_sidecar_gps_fix_count_mismatch_warns(rec):
    rec.gps(times=range(0, 11))  # 11 rows
    rec.sidecar(data=sidecar(gps_fixes=37))
    assert any("37 GPS fixes" in w and "11 data row" in w for w in rec.load().warnings)
    rec.sidecar(data=sidecar(gps_fixes=11))
    assert not any("GPS fixes" in w for w in rec.load().warnings)


# ---------------------------------------------------------------------------
# GPS cross-checks
# ---------------------------------------------------------------------------


def test_coverage_for_gps_that_stops_half_way(rec):
    rec.gps(times=[0, 1, 2, 3, 4, 5])
    s = rec.load()
    assert s.gps_coverage == pytest.approx(0.5)
    assert any("covers 50%" in w for w in s.warnings)


def test_coverage_clips_preroll_and_overrun_to_video(rec):
    rec.gps(times=[-4, -3, 0, 5, 10, 12])
    assert rec.load().gps_coverage == pytest.approx(1.0)


def test_coverage_ignores_rejected_fixes(rec):
    rows = gps_rows([0, 1, 2, 3, 4, 5], accuracy=5.0) + gps_rows([6, 7, 8, 9, 10], accuracy=99.0)
    rec.gps(rows=rows)
    assert rec.load().gps_coverage == pytest.approx(0.5)


def test_coverage_with_max_gap_excludes_dropouts_and_warns(rec):
    rec.gps(times=[0, 1, 2, 8, 9, 10])
    without = rec.load()
    assert without.gps_coverage == pytest.approx(1.0)
    assert without.gps.summary.max_gap_s == pytest.approx(6.0)
    gap_aware = rec.load(max_gap_s=3.0)
    assert gap_aware.gps_coverage == pytest.approx(0.4)
    assert any("largest gap" in w and "6.0 s" in w for w in gap_aware.warnings)


def test_poor_median_accuracy_warns(rec):
    rec.gps(rows=gps_rows(range(0, 11), accuracy=15.0))
    assert any("median accuracy" in w for w in rec.load().warnings)
    rec.gps(rows=gps_rows(range(0, 11), accuracy=12.5))  # exactly half the limit: fine
    assert not any("median accuracy" in w for w in rec.load().warnings)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.sqlite")
    db.init_schema(c)
    yield c
    c.close()


def count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.fixture
def saved_session(rec):
    rows = [
        "-2.000,10.0000000,50.0000000,5.5,0.5,90.0,0,1000",
        "-1.000,10.0000100,50.0000100,6.5,,,0,",  # blank optional fields
        "0.500,10.0000200,50.0000200,40.0,0.5,91.0,0,1002",  # rejected by the filter
        "2.000,10.0000300,50.0000300,,0.5,92.0,0,1003",  # unknown accuracy
        "4.000,10.0000400,50.0000400,7.5,0.5,93.0,0,1004",
    ]
    rec.gps(rows=rows)
    rec.sidecar(data=sidecar(gps_fixes=5, median_accuracy_m=99.0))
    return rec.load()


def test_save_session_writes_session_row_and_all_fixes(conn, saved_session):
    s = saved_session
    sid = save_session(conn, s)
    assert len(sid) == 32 and int(sid, 16) >= 0  # uuid4 hex
    assert count(conn, "sessions") == 1 and count(conn, "gps_fixes") == 5
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
    assert row["name"] == "road_test"
    assert row["video_path"] == str(s.video_path) and row["gps_path"] == str(s.gps_path)
    assert row["started_utc"] == "2026-01-02T03:04:05.678000+00:00"
    assert row["duration_s"] == pytest.approx(VIDEO_S)
    assert row["device"] == "Test Phone"
    assert row["median_accuracy_m"] == 6.5  # used fixes with known accuracy: 5.5, 6.5, 7.5
    assert (row["role"], row["include_in_map"]) == ("survey", 1)
    assert row["active_job_id"] is None
    created = datetime.fromisoformat(row["created_at"])
    assert created.utcoffset() == timedelta(0)
    assert abs(datetime.now(timezone.utc) - created) < timedelta(minutes=1)
    assert conn.execute("SELECT COUNT(*) FROM gps_fixes WHERE used = 0").fetchone()[0] == 1
    for table in ("jobs", "observations", "edge_coverage", "builds", "potholes"):
        assert count(conn, table) == 0


def test_role_and_include_in_map_are_stored(conn, saved_session):
    sid = save_session(conn, saved_session, role="test", include_in_map=False)
    row = conn.execute("SELECT role, include_in_map FROM sessions WHERE id = ?", (sid,)).fetchone()
    assert (row["role"], row["include_in_map"]) == ("test", 0)


def test_gps_log_round_trips_through_database(conn, saved_session):
    sid = save_session(conn, saved_session)
    loaded = load_gps_fixes(conn, sid)
    assert loaded == list(saved_session.gps.fixes)
    assert [f.seq for f in loaded] == [0, 1, 2, 3, 4]
    assert [f.used for f in loaded] == [True, True, False, True, True]
    assert loaded[1].speed_mps is None and loaded[1].utc_ms is None
    rebuilt = GpsLog(tuple(loaded), (), len(loaded))
    assert rebuilt.summary.n_used == saved_session.gps.summary.n_used
    assert rebuilt.summary.path_length_m == pytest.approx(saved_session.gps.summary.path_length_m)


def test_fixes_are_returned_in_seq_order_regardless_of_insert_order(conn, saved_session):
    s = saved_session
    sid = save_session(conn, s)
    conn.execute("UPDATE gps_fixes SET seq = seq + 100 WHERE seq IN (0, 1)")
    assert [f.seq for f in load_gps_fixes(conn, sid)] == [2, 3, 4, 100, 101]


def test_median_accuracy_falls_back_to_sidecar_then_none(conn, rec):
    rec.gps(rows=gps_rows(range(0, 11), accuracy=99.0))  # nothing usable
    rec.sidecar(data=sidecar(median_accuracy_m=8.25))
    sid = save_session(conn, rec.load())
    assert conn.execute("SELECT median_accuracy_m FROM sessions WHERE id=?", (sid,)).fetchone()[0] == 8.25
    rec.sidecar(data=sidecar(median_accuracy_m=None))
    sid2 = save_session(conn, rec.load())
    assert conn.execute("SELECT median_accuracy_m FROM sessions WHERE id=?", (sid2,)).fetchone()[0] is None


def test_session_without_gps_saves_with_null_gps_path_and_no_fixes(conn, rec):
    s = rec.load()
    sid = save_session(conn, s)
    row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    assert row["gps_path"] is None and row["started_utc"] is None and row["device"] is None
    assert load_gps_fixes(conn, sid) == []


def test_two_saves_get_distinct_ids(conn, saved_session):
    assert save_session(conn, saved_session) != save_session(conn, saved_session)
    assert count(conn, "sessions") == 2 and count(conn, "gps_fixes") == 10


def test_save_is_atomic_when_a_fix_insert_fails_midway(conn, saved_session):
    f = saved_session.gps.fixes
    clash = dataclasses.replace(f[2], seq=f[1].seq)  # duplicate primary key on the 3rd fix
    broken_gps = GpsLog.__new__(GpsLog)  # bypass nothing important: same ordering, bad seq
    object.__setattr__(broken_gps, "fixes", (f[0], f[1], clash, f[3], f[4]))
    object.__setattr__(broken_gps, "issues", ())
    object.__setattr__(broken_gps, "n_rows_read", 5)
    broken = dataclasses.replace(saved_session, gps=broken_gps)
    with pytest.raises(Exception, match="UNIQUE"):
        save_session(conn, broken)
    assert count(conn, "sessions") == 0 and count(conn, "gps_fixes") == 0
    assert not conn.in_transaction
    save_session(conn, saved_session)  # the connection is still usable
    assert count(conn, "sessions") == 1 and count(conn, "gps_fixes") == 5


def test_save_is_atomic_when_session_insert_fails(conn, saved_session):
    with pytest.raises(Exception, match="CHECK"):
        save_session(conn, saved_session, role="nonsense")
    assert count(conn, "sessions") == 0 and count(conn, "gps_fixes") == 0


def test_failed_save_inside_callers_transaction_only_undoes_itself(conn, saved_session):
    conn.execute("BEGIN")
    conn.execute("INSERT INTO meta (key, value) VALUES ('mine', '1')")
    with pytest.raises(Exception, match="CHECK"):
        save_session(conn, saved_session, role="nonsense")
    assert conn.in_transaction  # the caller's transaction is still open
    sid = save_session(conn, saved_session)  # and a good save joins it
    conn.commit()
    assert conn.execute("SELECT value FROM meta WHERE key='mine'").fetchone()[0] == "1"
    assert count(conn, "sessions") == 1
    assert len(load_gps_fixes(conn, sid)) == 5


def test_save_is_durable_after_return(tmp_path, saved_session):
    path = tmp_path / "durable.sqlite"
    c = db.connect(path)
    db.init_schema(c)
    sid = save_session(c, saved_session)
    other = db.connect(path)  # a second connection sees it without any explicit commit
    try:
        assert len(load_gps_fixes(other, sid)) == 5
    finally:
        other.close()
        c.close()


def test_deleting_session_cascades_to_fixes(conn, saved_session):
    sid = save_session(conn, saved_session)
    other = save_session(conn, saved_session)
    conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    conn.commit()
    assert load_gps_fixes(conn, sid) == []
    assert len(load_gps_fixes(conn, other)) == 5
    assert count(conn, "gps_fixes") == 5


def test_save_and_load_work_on_a_plain_connection_without_row_factory(tmp_path, saved_session):
    import sqlite3

    path = tmp_path / "plain.sqlite"
    setup = db.connect(path)
    db.init_schema(setup)
    setup.close()
    c = sqlite3.connect(path)  # default row_factory, foreign keys off
    sid = save_session(c, saved_session)
    assert load_gps_fixes(c, sid) == list(saved_session.gps.fixes)
    c.close()
