import json
import shutil

import pytest
import yaml
from api_fixtures import grid_fixes, make_api_cfg, write_gps_csv
from geo_fixtures import make_grid
from pipeline_fixtures import FakeDetector

from roadsense import cli
from roadsense.routing.graph import save_graph
from roadsense.storage import db


@pytest.fixture
def cfg_file(tmp_path):
    """A config file on temporary paths, with the grid road graph in place."""
    cfg = make_api_cfg(tmp_path)
    save_graph(make_grid(), cfg.routing.graph_path)
    path = tmp_path / "app.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(cfg.data_dir),
                "db_path": str(cfg.db_path),
                "uploads": {"dir": str(cfg.uploads.dir), "artifacts_dir": str(cfg.uploads.artifacts_dir)},
                "routing": {"graph_path": str(cfg.routing.graph_path)},
            }
        ),
        encoding="utf-8",
    )
    return path, cfg


@pytest.fixture
def recording(tmp_path, one_pothole_video):
    """A recorder-style folder: <stem>.mp4, <stem>_gps.csv and <stem>.json side by side."""
    folder = tmp_path / "recs" / "road_a"
    folder.mkdir(parents=True)
    video = folder / "road_a.mp4"
    shutil.copyfile(one_pothole_video, video)
    write_gps_csv(folder / "road_a_gps.csv", grid_fixes())
    (folder / "road_a.json").write_text(
        json.dumps({"format": "road-recorder/1", "device": "Phone", "duration_sec": 3.0}),
        encoding="utf-8",
    )
    return video


def run(argv, cfg_path, **kwargs):
    return cli.main([*argv, "--config", str(cfg_path)], detector_factory=kwargs.pop("factory", FakeDetector))


def db_rows(cfg, sql):
    conn = db.connect(cfg.db_path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def test_add_picks_up_sibling_files_processes_and_rebuilds(cfg_file, recording, capsys):
    path, cfg = cfg_file
    assert run(["add", str(recording)], path) == 0
    out = capsys.readouterr().out
    assert "added road_a.mp4" in out and "1 pothole(s) tracked" in out and "map rebuilt" in out
    (s,) = db_rows(cfg, "SELECT * FROM sessions")
    assert s["name"] == "road_a" and s["gps_source"] == "recorded" and s["device"] == "Phone"
    assert s["active_job_id"] is not None
    assert db_rows(cfg, "SELECT COUNT(*) FROM observations")[0][0] == 1
    assert db_rows(cfg, "SELECT COUNT(*) FROM potholes")[0][0] == 1
    assert (cfg.uploads.dir / s["id"] / "road_a_gps.csv").is_file()
    assert (cfg.uploads.dir / s["id"] / "road_a.json").is_file()


def test_add_with_explicit_files_and_options(cfg_file, one_pothole_video, tmp_path, capsys):
    path, cfg = cfg_file
    gps = write_gps_csv(tmp_path / "trace.csv", grid_fixes())
    code = run(
        ["add", str(one_pothole_video), "--gps", str(gps), "--synthetic-gps", "--role", "demo", "--no-map"],
        path,
    )
    assert code == 0
    (s,) = db_rows(cfg, "SELECT * FROM sessions")
    assert (s["gps_source"], s["role"], s["include_in_map"]) == ("synthetic", "demo", 0)
    assert db_rows(cfg, "SELECT COUNT(*) FROM potholes")[0][0] == 0  # not in the map


def test_add_without_gps_stores_none(cfg_file, one_pothole_video):
    path, cfg = cfg_file
    assert run(["add", str(one_pothole_video)], path) == 0
    (s,) = db_rows(cfg, "SELECT * FROM sessions")
    assert s["gps_source"] == "none"
    assert db_rows(cfg, "SELECT cam_lat FROM observations")[0][0] is None


def test_add_synthetic_gps_without_a_gps_file_is_an_error(cfg_file, one_pothole_video, capsys):
    path, cfg = cfg_file
    assert run(["add", str(one_pothole_video), "--synthetic-gps"], path) == cli.EXIT_ERROR
    assert "needs a GPS file" in capsys.readouterr().err
    assert db_rows(cfg, "SELECT COUNT(*) FROM sessions")[0][0] == 0


def test_add_no_process_only_registers(cfg_file, recording, capsys):
    path, cfg = cfg_file
    never = pytest.fail
    assert run(["add", str(recording), "--no-process"], path, factory=lambda: never("detector built")) == 0
    assert db_rows(cfg, "SELECT COUNT(*) FROM sessions")[0][0] == 1
    assert db_rows(cfg, "SELECT COUNT(*) FROM jobs")[0][0] == 0
    assert "map rebuilt" not in capsys.readouterr().out


def test_add_reports_a_bad_video_and_a_missing_file(cfg_file, tmp_path, capsys):
    path, cfg = cfg_file
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nope" * 100)
    assert run(["add", str(bad)], path) == cli.EXIT_ERROR
    assert run(["add", str(tmp_path / "missing.mp4")], path) == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert "bad.mp4" in err and "not found" in err
    assert db_rows(cfg, "SELECT COUNT(*) FROM sessions")[0][0] == 0
    assert list(cfg.uploads.dir.iterdir()) == []


def test_add_reports_a_failing_detector(cfg_file, recording, capsys):
    path, cfg = cfg_file

    def broken():
        raise RuntimeError("no weights")

    assert run(["add", str(recording)], path, factory=broken) == cli.EXIT_ERROR
    assert "no weights" in capsys.readouterr().err


def test_add_dir_finds_videos_recursively(cfg_file, recording, tmp_path, capsys):
    path, cfg = cfg_file
    second = tmp_path / "recs" / "deeper" / "road_b"
    second.mkdir(parents=True)
    shutil.copyfile(recording, second / "road_b.mp4")
    write_gps_csv(second / "road_b_gps.csv", grid_fixes())
    (tmp_path / "recs" / "notes.txt").write_text("not a video")
    assert run(["add-dir", str(tmp_path / "recs")], path) == 0
    out = capsys.readouterr().out
    assert "2 added, 0 failed" in out
    names = sorted(r["name"] for r in db_rows(cfg, "SELECT name FROM sessions"))
    assert names == ["road_a", "road_b"]
    sources = {r["name"]: r["gps_source"] for r in db_rows(cfg, "SELECT name, gps_source FROM sessions")}
    assert sources == {"road_a": "recorded", "road_b": "recorded"}
    assert db_rows(cfg, "SELECT COUNT(*) FROM jobs WHERE status = 'done'")[0][0] == 2


def test_add_dir_errors(cfg_file, tmp_path, capsys):
    path, _ = cfg_file
    assert run(["add-dir", str(tmp_path / "nowhere")], path) == cli.EXIT_ERROR
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run(["add-dir", str(empty)], path) == cli.EXIT_ERROR
    assert "no video files" in capsys.readouterr().err


def test_add_dir_continues_after_a_bad_video(cfg_file, recording, tmp_path, capsys):
    path, cfg = cfg_file
    (recording.parent / "a_corrupt.mp4").write_bytes(b"junk" * 100)
    assert run(["add-dir", str(recording.parent)], path) == cli.EXIT_ERROR
    assert "1 added, 1 failed" in capsys.readouterr().out
    assert [r["name"] for r in db_rows(cfg, "SELECT name FROM sessions")] == ["road_a"]


def test_list_shows_sessions_with_status_and_counts(cfg_file, recording, capsys):
    path, _ = cfg_file
    assert run(["list"], path) == 0
    assert "no sessions" in capsys.readouterr().out
    run(["add", str(recording)], path)
    capsys.readouterr()
    assert run(["list"], path) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "road_a" in ln)
    assert "recorded" in line and "done" in line and line.split()[-3:] == ["1", "1", "yes"]


def test_rebuild_map_demo_mode_flags_and_remembered(cfg_file, one_pothole_video, tmp_path, capsys):
    path, cfg = cfg_file
    gps = write_gps_csv(tmp_path / "g.csv", grid_fixes())
    run(["add", str(one_pothole_video), "--gps", str(gps), "--synthetic-gps"], path)
    # demo mode is off by default: the synthetic session is processed but left out of the map
    assert db_rows(cfg, "SELECT COUNT(*) FROM potholes")[0][0] == 0
    capsys.readouterr()
    assert run(["rebuild-map"], path) == 0  # no flag keeps the remembered choice (off)
    assert "demo mode off" in capsys.readouterr().out
    assert db_rows(cfg, "SELECT COUNT(*) FROM potholes")[0][0] == 0

    assert run(["rebuild-map", "--include-synthetic"], path) == 0
    assert "demo mode ON" in capsys.readouterr().out
    assert [r[0] for r in db_rows(cfg, "SELECT gps_source FROM potholes")] == ["synthetic"]
    assert db_rows(cfg, "SELECT value FROM meta WHERE key = 'include_synthetic'")[0][0] == "true"
    assert run(["rebuild-map"], path) == 0  # remembered: still on
    assert db_rows(cfg, "SELECT COUNT(*) FROM potholes")[0][0] == 1

    assert run(["rebuild-map", "--exclude-synthetic"], path) == 0
    assert "demo mode off" in capsys.readouterr().out
    assert db_rows(cfg, "SELECT COUNT(*) FROM potholes")[0][0] == 0
    assert db_rows(cfg, "SELECT value FROM meta WHERE key = 'include_synthetic'")[0][0] == "false"


def test_rebuild_map_flags_are_mutually_exclusive(cfg_file):
    path, _ = cfg_file
    with pytest.raises(SystemExit) as exc:
        run(["rebuild-map", "--include-synthetic", "--exclude-synthetic"], path)
    assert exc.value.code == 2


def test_add_and_reset_refuse_while_another_process_holds_the_database(
    cfg_file, recording, capsys
):
    from roadsense.storage.proclock import ProcessLock, lock_path_for

    path, cfg = cfg_file
    with ProcessLock(lock_path_for(cfg.db_path), role="server"):
        capsys.readouterr()
        assert run(["add", str(recording)], path) == cli.EXIT_ERROR
        assert "already working on this database" in capsys.readouterr().err
        assert run(["add-dir", str(recording.parent)], path) == cli.EXIT_ERROR
        assert run(["reset", "--yes"], path) == cli.EXIT_ERROR
        assert run(["list"], path) == 0 and run(["rebuild-map"], path) == 0  # read/rebuild are safe
    assert not db_rows(cfg, "SELECT * FROM sessions")  # nothing was added behind the lock
    assert run(["add", str(recording)], path) == 0  # the lock was released again


def test_reset_refuses_without_yes(cfg_file, recording, capsys):
    path, cfg = cfg_file
    run(["add", str(recording)], path)
    capsys.readouterr()
    assert run(["reset"], path) == cli.EXIT_USAGE
    assert "--yes" in capsys.readouterr().err
    assert cfg.db_path.is_file() and any(cfg.uploads.dir.iterdir())
    assert any(cfg.uploads.artifacts_dir.iterdir())


def test_reset_yes_deletes_database_uploads_and_artifacts_only(cfg_file, recording):
    path, cfg = cfg_file
    run(["add", str(recording)], path)
    demo = cfg.data_dir / "demo"
    demo.mkdir()
    (demo / "demo_route.json").write_text("{}")
    assert run(["reset", "--yes"], path) == 0
    assert not cfg.db_path.exists() and not cfg.uploads.dir.exists() and not cfg.uploads.artifacts_dir.exists()
    # the (empty) process-lock file is never deleted; everything else of the database is gone
    assert not any(
        p.name.startswith("app.sqlite") and p.name != "app.sqlite.lock" for p in cfg.data_dir.iterdir()
    )
    assert (demo / "demo_route.json").is_file() and cfg.routing.graph_path.is_file()
    assert run(["list"], path) == 0  # a fresh database is created on demand
    assert run(["reset", "--yes"], path) == 0  # and resetting nothing is fine


def test_reset_refuses_to_delete_project_directories(tmp_path, capsys):
    cfg = make_api_cfg(tmp_path)
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(cfg.data_dir),
                "db_path": str(cfg.db_path),
                "uploads": {"dir": str(cfg.data_dir), "artifacts_dir": str(cfg.uploads.artifacts_dir)},
            }
        ),
        encoding="utf-8",
    )
    cfg.data_dir.mkdir(parents=True)
    (cfg.data_dir / "keep.txt").write_text("keep")
    assert run(["reset", "--yes"], path) == cli.EXIT_ERROR
    assert (cfg.data_dir / "keep.txt").is_file()


def test_bad_config_is_a_usage_error(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("nonsense_key: 1\n")
    assert cli.main(["list", "--config", str(bad)]) == cli.EXIT_USAGE
    assert "nonsense_key" in capsys.readouterr().err
