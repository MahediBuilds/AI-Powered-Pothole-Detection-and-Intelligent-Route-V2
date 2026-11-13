import json

import pytest
from api_fixtures import grid_fixes, make_api_cfg, write_gps_csv

from roadsense.ingest.gps import GpsError
from roadsense.ingest.video import VideoError
from roadsense.pipeline.intake import (
    UnsupportedFileError,
    create_session_from_files,
    delete_session_files,
    safe_stem,
)
from roadsense.storage import db


@pytest.fixture
def env(tmp_path):
    cfg = make_api_cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    yield cfg, conn, tmp_path
    conn.close()


def sidecar(path):
    path.write_text(
        json.dumps({"format": "road-recorder/1", "device": "Test Phone", "duration_sec": 3.0}),
        encoding="utf-8",
    )
    return path


def row(conn, sid):
    return conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()


def test_copies_files_under_recorder_names_and_saves(env, one_pothole_video):
    cfg, conn, tmp_path = env
    gps = write_gps_csv(tmp_path / "anything.csv", grid_fixes())
    side = sidecar(tmp_path / "meta.json")
    sid = create_session_from_files(conn, cfg, video=one_pothole_video, gps=gps, sidecar=side)
    conn.commit()
    folder = cfg.uploads.dir / sid
    stem = one_pothole_video.stem
    assert {p.name for p in folder.iterdir()} == {f"{stem}.mp4", f"{stem}_gps.csv", f"{stem}.json"}
    s = row(conn, sid)
    assert s["name"] == stem and s["gps_source"] == "recorded" and s["role"] == "survey"
    assert s["device"] == "Test Phone" and s["include_in_map"] == 1
    assert s["video_path"] == str(folder / f"{stem}.mp4") and s["gps_path"].endswith("_gps.csv")
    assert conn.execute("SELECT COUNT(*) FROM gps_fixes WHERE session_id = ?", (sid,)).fetchone()[0] == 6
    assert one_pothole_video.is_file() and gps.is_file()  # the sources are only copied


def test_name_overrides_the_stem_and_is_sanitised(env, one_pothole_video):
    cfg, conn, _ = env
    sid = create_session_from_files(conn, cfg, video=one_pothole_video, name="My Road (take 2).MOV")
    assert row(conn, sid)["name"] == "My_Road_take_2"
    assert (cfg.uploads.dir / sid / "My_Road_take_2.mp4").is_file()  # the real extension is kept


@pytest.mark.parametrize(
    "name, expected",
    [
        ("road_20261022.mp4", "road_20261022"),
        ("a b/c d.mp4", "c_d"),
        ("..\\..\\evil.mp4", "evil"),
        ("???.mp4", "session"),
        ("", "session"),
        ("--ok--.mp4", "--ok--"),
        ("x" * 300 + ".mp4", "x" * 80),
        ("naïve café.mp4", "na_ve_caf"),
    ],
)
def test_safe_stem(name, expected):
    assert safe_stem(name) == expected


def test_gps_source_rules(env, one_pothole_video):
    cfg, conn, tmp_path = env
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    none = create_session_from_files(conn, cfg, video=one_pothole_video)
    none_synth = create_session_from_files(
        conn, cfg, video=one_pothole_video, gps_source="synthetic"
    )
    recorded = create_session_from_files(conn, cfg, video=one_pothole_video, gps=gps)
    synthetic = create_session_from_files(
        conn, cfg, video=one_pothole_video, gps=gps, gps_source="synthetic", role="demo",
        include_in_map=False,
    )  # fmt: skip
    assert [row(conn, s)["gps_source"] for s in (none, none_synth, recorded, synthetic)] == [
        "none", "none", "recorded", "synthetic",
    ]  # fmt: skip
    assert row(conn, synthetic)["role"] == "demo" and row(conn, synthetic)["include_in_map"] == 0


def test_invalid_video_removes_copies_and_raises_unchanged(env):
    cfg, conn, tmp_path = env
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"garbage" * 50)
    with pytest.raises(VideoError):
        create_session_from_files(conn, cfg, video=bad)
    assert list(cfg.uploads.dir.iterdir()) == []
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_invalid_gps_removes_copies_and_raises_unchanged(env, one_pothole_video):
    cfg, conn, tmp_path = env
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(GpsError):
        create_session_from_files(conn, cfg, video=one_pothole_video, gps=bad)
    assert list(cfg.uploads.dir.iterdir()) == []
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_unsupported_extension_and_bad_arguments(env, tmp_path, one_pothole_video):
    cfg, conn, _ = env
    text = tmp_path / "clip.txt"
    text.write_text("x")
    with pytest.raises(UnsupportedFileError):
        create_session_from_files(conn, cfg, video=text)
    with pytest.raises(ValueError):
        create_session_from_files(conn, cfg, video=one_pothole_video, role="admin")
    with pytest.raises(ValueError):
        create_session_from_files(conn, cfg, video=one_pothole_video, gps_source="guessed")
    assert not cfg.uploads.dir.exists() or list(cfg.uploads.dir.iterdir()) == []


def test_missing_source_file_leaves_nothing(env, tmp_path):
    cfg, conn, _ = env
    with pytest.raises(OSError):
        create_session_from_files(conn, cfg, video=tmp_path / "missing.mp4")
    assert list(cfg.uploads.dir.iterdir()) == []


def test_delete_session_files_removes_both_folders(env, one_pothole_video):
    cfg, conn, _ = env
    sid = create_session_from_files(conn, cfg, video=one_pothole_video)
    art = cfg.uploads.artifacts_dir / sid / "job"
    art.mkdir(parents=True)
    (art / "track_1.jpg").write_bytes(b"x")
    other = cfg.uploads.dir / "other-session"
    other.mkdir()
    delete_session_files(cfg, sid)
    assert not (cfg.uploads.dir / sid).exists() and not (cfg.uploads.artifacts_dir / sid).exists()
    assert other.is_dir() and cfg.uploads.dir.is_dir() and cfg.uploads.artifacts_dir.is_dir()
    delete_session_files(cfg, sid)  # already gone: no error


@pytest.mark.parametrize("bad", ["", ".", "..", "../x", "a/b", "a\\b", "..\\..\\x"])
def test_delete_session_files_refuses_unsafe_ids(env, bad):
    cfg, _, tmp_path = env
    cfg.uploads.dir.mkdir(parents=True)
    victim = tmp_path / "data" / "keep.txt"
    victim.write_text("keep")
    with pytest.raises(ValueError):
        delete_session_files(cfg, bad)
    assert victim.is_file() and cfg.uploads.dir.is_dir()


# ---------------------------------------------------------------- synthetic sidecar flag


def _gps_and_sidecar(tmp_path, payload):
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    side = tmp_path / "meta.json"
    side.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return gps, side


def test_sidecar_synthetic_true_overrides_recorded(env, one_pothole_video):
    cfg, conn, tmp_path = env
    gps, side = _gps_and_sidecar(tmp_path, {"format": "road-recorder/1", "synthetic": True})
    sid = create_session_from_files(
        conn, cfg, video=one_pothole_video, gps=gps, sidecar=side, gps_source="recorded"
    )
    assert row(conn, sid)["gps_source"] == "synthetic"


def test_sidecar_without_the_flag_leaves_gps_source_unchanged(env, one_pothole_video):
    cfg, conn, tmp_path = env
    gps, side = _gps_and_sidecar(tmp_path, {"format": "road-recorder/1", "synthetic": False})
    sid = create_session_from_files(conn, cfg, video=one_pothole_video, gps=gps, sidecar=side)
    assert row(conn, sid)["gps_source"] == "recorded"


@pytest.mark.parametrize("payload", ["{not json", "[1, 2]", '"synthetic"', ""])
def test_malformed_sidecar_is_ignored_for_the_synthetic_flag(env, one_pothole_video, payload):
    cfg, conn, tmp_path = env
    gps, side = _gps_and_sidecar(tmp_path, payload)
    sid = create_session_from_files(conn, cfg, video=one_pothole_video, gps=gps, sidecar=side)
    assert row(conn, sid)["gps_source"] == "recorded"


def test_synthetic_sidecar_without_gps_stays_none(env, one_pothole_video):
    cfg, conn, tmp_path = env
    _, side = _gps_and_sidecar(tmp_path, {"synthetic": True})
    sid = create_session_from_files(conn, cfg, video=one_pothole_video, sidecar=side)
    assert row(conn, sid)["gps_source"] == "none"


# ---------------------------------------------------------------- move instead of copy


def _staged(tmp_path, video):
    stage = tmp_path / "stage"
    stage.mkdir()
    v = stage / "video.mp4"
    v.write_bytes(video.read_bytes())
    gps = write_gps_csv(stage / "gps.csv", grid_fixes())
    return stage, v, gps


def test_move_true_moves_the_files(env, one_pothole_video):
    cfg, conn, tmp_path = env
    _, v, gps = _staged(tmp_path, one_pothole_video)
    video_bytes, gps_bytes = v.read_bytes(), gps.read_bytes()
    sid = create_session_from_files(conn, cfg, video=v, gps=gps, move=True)
    folder = cfg.uploads.dir / sid
    assert not v.exists() and not gps.exists()
    assert (folder / "video.mp4").read_bytes() == video_bytes
    assert (folder / "video_gps.csv").read_bytes() == gps_bytes


def test_move_false_leaves_the_sources(env, one_pothole_video):
    cfg, conn, tmp_path = env
    _, v, gps = _staged(tmp_path, one_pothole_video)
    create_session_from_files(conn, cfg, video=v, gps=gps)
    assert v.is_file() and gps.is_file()


def test_move_falls_back_to_shutil_move_when_replace_fails(env, one_pothole_video, monkeypatch):
    import os

    cfg, conn, tmp_path = env
    _, v, _ = _staged(tmp_path, one_pothole_video)

    def broken(*a, **k):
        raise OSError("cross-device")

    monkeypatch.setattr(os, "replace", broken)
    sid = create_session_from_files(conn, cfg, video=v, move=True)
    assert not v.exists() and (cfg.uploads.dir / sid / "video.mp4").is_file()


def test_failed_validation_with_move_leaves_nothing_behind(env, one_pothole_video):
    cfg, conn, tmp_path = env
    stage, v, _ = _staged(tmp_path, one_pothole_video)
    bad = stage / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(GpsError):
        create_session_from_files(conn, cfg, video=v, gps=bad, move=True)
    assert list(cfg.uploads.dir.iterdir()) == []
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
