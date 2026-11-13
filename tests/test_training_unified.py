"""Tests for the unified-run extensions of the training entrypoint (no ultralytics import)."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from roadsense.config import ConfigError, load_train_config, project_root
from roadsense.training import train
from roadsense.training.train import (
    DatasetConfigError,
    RunDirError,
    build_ultralytics_args,
    guard_run_dir,
    load_dataset_yaml,
    pin_epochs_callback,
    read_list_file,
    write_resolved_dataset_yaml,
)

ROOT = project_root()
BASELINE = ROOT / "configs" / "train_baseline.yaml"
UNIFIED = ROOT / "configs" / "train_unified_v1.yaml"
OPTIONAL_KEYS = {"optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
                 "warmup_bias_lr", "cos_lr", "time", "save_period", "fraction"}  # fmt: skip


def make_list_dataset(tmp_path: Path, extra_lines=()):
    """A unified-style dataset root: images/<source>/<split>/..., lists/*.txt, dataset.yaml."""
    root = tmp_path / "unified"
    lines = {"train": [], "val": []}
    for source, split, n in (("rdd2022", "train", 2), ("hrp4k", "train", 2), ("rdd2022", "val", 1)):
        folder = root / "images" / source / split
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            name = f"{source}_{i}.jpg"
            (folder / name).write_bytes(b"x")
            lines["val" if split == "val" else "train"].append(f"images/{source}/{split}/{name}")
    (root / "lists").mkdir()
    (root / "lists" / "train.txt").write_text("\n".join(lines["train"] + list(extra_lines)) + "\n")
    (root / "lists" / "val_pooled.txt").write_text("\n".join(lines["val"]) + "\n")
    ds = root / "dataset.yaml"
    ds.write_text(yaml.safe_dump({"train": "lists/train.txt", "val": "lists/val_pooled.txt",
                                  "names": {0: "pothole"}}))  # fmt: skip
    return root, ds


def test_baseline_arguments_carry_no_optional_keys():
    args = build_ultralytics_args(load_train_config(BASELINE))
    assert not (OPTIONAL_KEYS & set(args))


def test_unified_config_arguments():
    cfg = load_train_config(UNIFIED)
    args = build_ultralytics_args(cfg)
    assert args["imgsz"] == 1280 and args["seed"] == 42 and args["deterministic"] is True
    assert args["optimizer"] == "AdamW" and args["lr0"] == 0.001 and args["cos_lr"] is True
    assert args["warmup_bias_lr"] == 0.0
    assert args["save_period"] == 5 and args["time"] == 2.5 and args["epochs"] == 50
    assert args["name"] == "unified_v1_y8s_1280" and args["exist_ok"] is False
    assert cfg.model == str(ROOT / "pretrained" / "baseline_yolov8s_rdd2022_best.pt")
    assert cfg.data == ROOT / "data/processed/unified_v1/dataset.yaml"
    base = build_ultralytics_args(load_train_config(BASELINE))
    for key in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
                "perspective", "flipud", "fliplr", "mosaic", "mixup", "close_mosaic"):  # fmt: skip
        assert args[key] == base[key], key  # same augmentation as the baseline


def test_unified_args_never_reference_test_split():
    args = build_ultralytics_args(load_train_config(UNIFIED))
    assert "split" not in args
    for key, value in args.items():
        assert "test" not in key.lower() and "test" not in str(value).lower(), (key, value)


def test_pretrained_weights_are_a_copy_of_the_archive_best():
    copy = ROOT / "pretrained" / "baseline_yolov8s_rdd2022_best.pt"
    archive = ROOT / "model_archive" / "baseline_yolov8s_rdd2022" / "weights" / "best.pt"
    if not (copy.is_file() and archive.is_file()):
        pytest.skip("weights not present")
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == hashlib.sha256(archive.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "bad", [{"optim": {"optimizer": "Lion"}}, {"optim": {"lr0": 0}}, {"optim": {"lr0": 5}},
            {"time_hours": 0}, {"save_period": 0}, {"fraction": 0}, {"fraction": 1.5},
            {"optim": {"unknown": 1}}]  # fmt: skip
)
def test_invalid_optional_fields_rejected(write_yaml, bad):
    with pytest.raises(ConfigError):
        load_train_config(write_yaml("t.yaml", bad))


def test_list_dataset_validates_and_resolves_to_absolute_lists(tmp_path):
    root, ds_yaml = make_list_dataset(tmp_path)
    ds = load_dataset_yaml(ds_yaml)
    run_dir = tmp_path / "run"
    out = write_resolved_dataset_yaml(ds, ds_yaml, run_dir)
    resolved = yaml.safe_load(out.read_text())
    assert resolved["path"] == root.resolve().as_posix()
    for split, name, count in (("train", "train.txt", 4), ("val", "val_pooled.txt", 1)):
        assert resolved[split] == (run_dir / "lists" / name).resolve().as_posix()
        entries = (run_dir / "lists" / name).read_text().splitlines()
        assert len(entries) == count
        assert all(Path(e).is_absolute() and Path(e).is_file() for e in entries)
    assert "\r" not in (run_dir / "lists" / "train.txt").read_text()


@pytest.mark.parametrize(
    "line, message",
    [("images/rdd2022/test/a.jpg", "test directory"),
     ("images/hrp4k/Test/a.jpg", "test directory"),
     ("/abs/a.jpg", "not found"),
     ("../x/a.jpg", "not found"),
     ("images/rdd2022/train/missing.jpg", "not found")]  # fmt: skip
)
def test_list_entries_are_checked(tmp_path, line, message):
    _, ds_yaml = make_list_dataset(tmp_path, extra_lines=[line])
    with pytest.raises(DatasetConfigError, match=message):
        load_dataset_yaml(ds_yaml)


def test_list_entry_formats_are_not_assumed(tmp_path):
    root, ds_yaml = make_list_dataset(tmp_path)
    first = "images/rdd2022/train/rdd2022_0.jpg"
    absolute = (root / first).resolve()
    entries = [
        first,                                      # relative to the dataset root
        first.replace("/", "\\"),                   # backslash form
        "./" + first,                               # dot-slash form
        "../images/rdd2022/train/rdd2022_1.jpg",    # relative to the list file's directory
        absolute.as_posix(),                        # absolute
    ]
    (root / "lists" / "mixed.txt").write_text(chr(10).join(entries) + chr(10))
    resolved = read_list_file(root / "lists" / "mixed.txt", root)
    assert [p.name for p in resolved] == ["rdd2022_0.jpg"] * 3 + ["rdd2022_1.jpg", "rdd2022_0.jpg"]
    assert all(p.is_absolute() and p.is_file() for p in resolved)


def test_empty_list_rejected(tmp_path):
    root, ds_yaml = make_list_dataset(tmp_path)
    (root / "lists" / "val_pooled.txt").write_text("\n\n")
    with pytest.raises(DatasetConfigError, match="empty"):
        read_list_file(root / "lists" / "val_pooled.txt", root)


def test_dry_run_with_list_dataset_records_list_hashes(write_yaml, tmp_path, capsys):
    root, ds_yaml = make_list_dataset(tmp_path)
    weights = tmp_path / "w" / "init.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")
    cfg_path = write_yaml(
        "t.yaml",
        {"data": str(ds_yaml), "model": str(weights), "project": str(tmp_path / "runs"),
         "name": "u", "device": "cpu", "imgsz": 320, "batch": 2, "epochs": 2, "patience": 1,
         "time_hours": 0.5, "save_period": 1, "optim": {"optimizer": "AdamW", "lr0": 0.001}},
    )  # fmt: skip
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["optimizer"] == "AdamW" and printed["time"] == 0.5 and printed["save_period"] == 1
    record = json.loads((tmp_path / "runs" / "u" / "run_config.json").read_text())
    assert record["dataset_lists_sha256"] == {
        "lists/train.txt": hashlib.sha256((root / "lists" / "train.txt").read_bytes()).hexdigest(),
        "lists/val_pooled.txt": hashlib.sha256((root / "lists" / "val_pooled.txt").read_bytes()).hexdigest(),
    }
    assert record["init_weights_sha256"] == hashlib.sha256(b"weights").hexdigest()
    # a second dry run is allowed (only our own files are present), a populated one is refused
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 0
    (tmp_path / "runs" / "u" / "weights").mkdir()
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 2


def test_missing_initial_weights_fail_before_writing(write_yaml, tmp_path, capsys):
    _, ds_yaml = make_list_dataset(tmp_path)
    cfg_path = write_yaml(
        "t.yaml",
        {"data": str(ds_yaml), "model": str(tmp_path / "nope" / "w.pt"),
         "project": str(tmp_path / "runs"), "device": "cpu", "imgsz": 320, "epochs": 2,
         "patience": 1},
    )  # fmt: skip
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 2
    assert "initial weights not found" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_guard_run_dir_blocks_baseline_and_archive():
    for protected in ("runs/train/baseline", "runs/train/baseline/weights", "model_archive",
                      "model_archive/baseline_yolov8s_rdd2022"):  # fmt: skip
        with pytest.raises(RunDirError):
            guard_run_dir(ROOT / protected)
    guard_run_dir(ROOT / "runs" / "train" / "unified_v1_y8s_1280")
    guard_run_dir(ROOT / "runs" / "train" / "baseline_other")


def test_run_refuses_protected_run_dir(write_yaml, tmp_path, capsys):
    _, ds_yaml = make_list_dataset(tmp_path)
    baseline_dir = ROOT / "runs" / "train" / "baseline"
    cfg_path = write_yaml(
        "t.yaml",
        {"data": str(ds_yaml), "project": str(ROOT / "runs" / "train"), "name": "baseline",
         "exist_ok": True, "device": "cpu", "imgsz": 320, "epochs": 2, "patience": 1},
    )  # fmt: skip
    before = sorted(p.name for p in baseline_dir.iterdir()) if baseline_dir.exists() else None
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 2
    assert "protected" in capsys.readouterr().err
    after = sorted(p.name for p in baseline_dir.iterdir()) if baseline_dir.exists() else None
    assert before == after


def test_resume_requires_existing_checkpoint(tmp_path, capsys):
    assert train.main(["--resume", str(tmp_path / "last.pt")]) == 2
    assert "checkpoint not found" in capsys.readouterr().err


def test_resume_refuses_protected_location(capsys):
    last = ROOT / "model_archive" / "baseline_yolov8s_rdd2022" / "weights" / "last.pt"
    if not last.is_file():
        pytest.skip("archive not present")
    assert train.main(["--resume", str(last)]) == 2
    assert "protected" in capsys.readouterr().err


def _fake_trainer(time, epochs, epoch=3):
    calls = []
    trainer = SimpleNamespace(
        args=SimpleNamespace(time=time, epochs=epochs), epochs=epochs, epoch=epoch, stop=False,
        _setup_scheduler=lambda: calls.append("setup"),
        scheduler=SimpleNamespace(last_epoch=0),
    )  # fmt: skip
    return trainer, calls


def test_pin_epochs_callback_only_shrinks_an_overshoot():
    trainer, calls = _fake_trainer(2.5, 61)  # time mode wants 61 epochs, 50 configured
    pin_epochs_callback(50)(trainer)
    assert trainer.epochs == trainer.args.epochs == 50
    assert calls == ["setup"] and trainer.scheduler.last_epoch == 3
    pin_epochs_callback(50)(trainer)  # already at the configured count: nothing happens
    assert calls == ["setup"]


def test_pin_epochs_callback_leaves_a_shorter_time_fit_alone():
    # a late start: the time budget only fits 41 epochs. Ultralytics has already rebuilt the
    # schedule for 41; the callback must not stretch it back to 50.
    trainer, calls = _fake_trainer(1.2, 41)
    pin_epochs_callback(50)(trainer)
    assert trainer.epochs == trainer.args.epochs == 41 and calls == []


def test_pin_epochs_callback_ignores_runs_without_time_mode():
    trainer, calls = _fake_trainer(None, 90)
    pin_epochs_callback(50)(trainer)
    assert trainer.epochs == 90 and calls == []


def test_wall_clock_cap_stops_only_after_the_cap():
    now = [1000.0]
    on_start, on_batch_end = train.wall_clock_cap_callbacks(0.5, clock=lambda: now[0])
    trainer, _ = _fake_trainer(None, 50)
    on_batch_end(trainer)  # before the start event: no effect
    assert trainer.stop is False
    on_start(trainer)
    now[0] += 1799
    on_batch_end(trainer)
    assert trainer.stop is False
    now[0] += 2
    on_batch_end(trainer)
    assert trainer.stop is True
    assert trainer.epochs == 50  # the schedule is untouched


def test_time_hours_flag_overrides_config_and_is_recorded(write_yaml, tmp_path, pothole_dataset_yaml, capsys):
    project_dir = tmp_path / "runs"
    cfg_path = _flag_config(write_yaml, pothole_dataset_yaml, project_dir, time_hours=2.5)
    assert train.main(["--config", str(cfg_path), "--time-hours", "1.75", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["time"] == 1.75
    record = json.loads((project_dir / "baseline" / "run_config.json").read_text(encoding="utf-8"))
    assert record["cli_overrides"] == {"time_hours": 1.75}
    assert record["train_config"]["time_hours"] == 1.75


def test_time_hours_flag_enables_the_cap_when_config_has_none(write_yaml, tmp_path, pothole_dataset_yaml, capsys):
    cfg_path = _flag_config(write_yaml, pothole_dataset_yaml, tmp_path / "runs")
    assert train.main(["--config", str(cfg_path), "--time-hours", "0.9", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["time"] == 0.9


def test_invalid_time_hours_flag_rejected(write_yaml, tmp_path, pothole_dataset_yaml, capsys):
    cfg_path = _flag_config(write_yaml, pothole_dataset_yaml, tmp_path / "runs")
    assert train.main(["--config", str(cfg_path), "--time-hours", "0", "--dry-run"]) == 2
    assert "time_hours" in capsys.readouterr().err


def test_unified_config_patience_does_not_end_the_run_early():
    cfg = load_train_config(UNIFIED)
    assert cfg.patience == cfg.epochs == 50  # early stopping effectively off; selection is local


def _flag_config(write_yaml, ds_yaml, project_dir, **extra):
    return write_yaml(
        "t.yaml",
        {"data": str(ds_yaml), "project": str(project_dir), "device": "cpu", "imgsz": 320,
         "batch": 4, "epochs": 2, "patience": 1, **extra},
    )  # fmt: skip
