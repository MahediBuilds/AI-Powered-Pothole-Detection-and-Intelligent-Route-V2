import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from roadsense.config import load_train_config, project_root
from roadsense.training import train
from roadsense.training.train import (
    DatasetConfigError,
    RunDirError,
    build_ultralytics_args,
    load_dataset_yaml,
    prepare_run_dir,
    validate_dataset_config,
    write_resolved_dataset_yaml,
)

ROOT = project_root()
BASELINE = ROOT / "configs" / "train_baseline.yaml"

REQUIRED_KEYS = {
    "data", "imgsz", "batch", "epochs", "patience", "seed", "deterministic", "device",
    "workers", "project", "name", "exist_ok", "val", "plots", "save",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "close_mosaic",
}  # fmt: skip

GOOD_DS = {"path": "x", "train": "images/train", "val": "images/val", "names": {0: "pothole"}}


def test_args_from_baseline_config():
    cfg = load_train_config(BASELINE)
    args = build_ultralytics_args(cfg)
    assert REQUIRED_KEYS <= set(args)
    assert args["imgsz"] == cfg.imgsz == 640
    assert args["batch"] == 16
    assert args["epochs"] == 100
    assert args["patience"] == 20
    assert args["seed"] == 42
    assert args["deterministic"] is True
    assert args["device"] == "0"
    assert args["workers"] == 4
    assert args["name"] == "baseline"
    assert args["exist_ok"] is False
    assert args["data"] == str(ROOT / "data/processed/rdd2022_pothole/dataset.yaml")
    assert args["val"] is True and args["plots"] is True and args["save"] is True
    assert args["flipud"] == 0.0
    assert "single_cls" not in args


def test_args_do_not_reference_test_split():
    args = build_ultralytics_args(load_train_config(BASELINE))
    assert "split" not in args
    for key, value in args.items():
        assert "test" not in key.lower()
        assert "test" not in str(value).lower(), (key, value)


def test_train_module_never_calls_test_split():
    source = Path(train.__file__).read_text(encoding="utf-8")
    assert 'split="test"' not in source and "split='test'" not in source


@pytest.mark.parametrize("names", [{0: "pothole"}, ["pothole"]])
def test_validator_accepts_single_class_pothole(names):
    validate_dataset_config({**GOOD_DS, "names": names})
    validate_dataset_config({**GOOD_DS, "names": names, "nc": 1})


def test_validator_rejects_bad_datasets():
    bad_cases = {
        "multi-class": {**GOOD_DS, "names": {0: "pothole", 1: "crack"}},
        "wrong name": {**GOOD_DS, "names": {0: "crack"}},
        "wrong id": {**GOOD_DS, "names": {1: "pothole"}},
        "wrong nc": {**GOOD_DS, "nc": 2},
        "no names": {k: v for k, v in GOOD_DS.items() if k != "names"},
        "missing val": {k: v for k, v in GOOD_DS.items() if k != "val"},
        "missing train": {k: v for k, v in GOOD_DS.items() if k != "train"},
        "empty val": {**GOOD_DS, "val": ""},
        "not a mapping": ["pothole"],
    }
    for label, ds in bad_cases.items():
        with pytest.raises(DatasetConfigError):
            validate_dataset_config(copy.deepcopy(ds))
        assert label  # keeps the failing case visible in tracebacks


def _project_config(write_yaml, ds_path, project_dir, **extra):
    return write_yaml(
        "train.yaml",
        {"data": str(ds_path), "project": str(project_dir), "device": "cpu", "seed": 123,
         "imgsz": 320, "batch": 4, "epochs": 2, "patience": 1, **extra},
    )  # fmt: skip


def test_dry_run_writes_run_config_and_does_not_import_ultralytics(
    write_yaml, tmp_path, pothole_dataset_yaml, capsys
):
    assert "ultralytics" not in sys.modules
    manifest = pothole_dataset_yaml.parent / "manifest.csv"
    manifest.write_text("source_id\n", encoding="utf-8")
    project_dir = tmp_path / "proj" / "runs"
    cfg_path = _project_config(write_yaml, pothole_dataset_yaml, project_dir)

    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 0

    record = json.loads((project_dir / "baseline" / "run_config.json").read_text(encoding="utf-8"))
    assert record["train_config"]["seed"] == 123
    assert record["train_config"]["model"] == "yolov8s.pt"
    assert record["train_config"]["imgsz"] == 320
    assert record["train_config"]["batch"] == 4
    assert record["ultralytics_args"]["seed"] == 123
    assert record["dataset_yaml_sha256"] == hashlib.sha256(pothole_dataset_yaml.read_bytes()).hexdigest()
    assert record["dataset_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert "python" in record["environment"] and "platform" in record["environment"]

    resolved = project_dir / "baseline" / "dataset_resolved.yaml"
    assert record["dataset_yaml"] == str(pothole_dataset_yaml)
    assert record["dataset_resolved_yaml"] == str(resolved)
    assert record["dataset_resolved_yaml_sha256"] == hashlib.sha256(resolved.read_bytes()).hexdigest()
    assert record["ultralytics_args"]["data"] == str(resolved)  # Ultralytics gets the resolved copy
    resolved_ds = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    assert Path(resolved_ds["path"]).is_absolute()
    assert resolved_ds["path"] == pothole_dataset_yaml.parent.as_posix()
    assert resolved_ds["names"] == {0: "pothole"} and resolved_ds["val"] == "images/val"

    printed = json.loads(capsys.readouterr().out)
    assert printed == record["ultralytics_args"]
    assert "ultralytics" not in sys.modules


def test_dry_run_without_manifest_records_none(write_yaml, tmp_path, pothole_dataset_yaml):
    project_dir = tmp_path / "runs"
    cfg_path = _project_config(write_yaml, pothole_dataset_yaml, project_dir)
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 0
    record = json.loads((project_dir / "baseline" / "run_config.json").read_text(encoding="utf-8"))
    assert record["dataset_manifest_sha256"] is None


def test_device_override(write_yaml, tmp_path, pothole_dataset_yaml, capsys):
    cfg_path = _project_config(write_yaml, pothole_dataset_yaml, tmp_path / "runs", device="0")
    assert train.main(["--config", str(cfg_path), "--device", "cpu", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["device"] == "cpu"


def test_batch_and_workers_overrides_are_applied_and_recorded(
    write_yaml, tmp_path, pothole_dataset_yaml, capsys
):
    project_dir = tmp_path / "runs"
    cfg_path = _project_config(write_yaml, pothole_dataset_yaml, project_dir)
    argv = ["--config", str(cfg_path), "--batch", "64", "--workers", "12", "--dry-run"]
    assert train.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["batch"] == 64 and printed["workers"] == 12
    record = json.loads((project_dir / "baseline" / "run_config.json").read_text(encoding="utf-8"))
    assert record["cli_overrides"] == {"batch": 64, "workers": 12}
    assert record["train_config"]["batch"] == 64
    assert record["train_config"]["seed"] == 123  # everything else still comes from the file


def test_no_overrides_records_empty_mapping(write_yaml, tmp_path, pothole_dataset_yaml):
    project_dir = tmp_path / "runs"
    cfg_path = _project_config(write_yaml, pothole_dataset_yaml, project_dir)
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 0
    record = json.loads((project_dir / "baseline" / "run_config.json").read_text(encoding="utf-8"))
    assert record["cli_overrides"] == {}


def test_invalid_batch_override_is_rejected(write_yaml, tmp_path, pothole_dataset_yaml, capsys):
    cfg_path = _project_config(write_yaml, pothole_dataset_yaml, tmp_path / "runs")
    assert train.main(["--config", str(cfg_path), "--batch", "0", "--dry-run"]) == 2
    assert "batch" in capsys.readouterr().err


def test_missing_dataset_yaml_fails_with_converter_hint(write_yaml, tmp_path, capsys):
    cfg_path = _project_config(write_yaml, tmp_path / "nope" / "dataset.yaml", tmp_path / "runs")
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "converter" in err
    assert not (tmp_path / "runs").exists()  # nothing written before validation passes


def test_multiclass_dataset_fails_before_writing(write_yaml, tmp_path, capsys):
    ds = tmp_path / "ds" / "dataset.yaml"
    ds.parent.mkdir()
    ds.write_text(yaml.safe_dump({**GOOD_DS, "names": {0: "pothole", 1: "crack"}}), encoding="utf-8")
    cfg_path = _project_config(write_yaml, ds, tmp_path / "runs")
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 2
    assert "exactly one class" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_invalid_config_returns_error_code(write_yaml, tmp_path, capsys):
    cfg_path = write_yaml("train.yaml", {"imgsz": 100})
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 2
    assert "imgsz" in capsys.readouterr().err


def test_repeated_dry_run_allowed_but_populated_run_dir_refused(
    write_yaml, tmp_path, pothole_dataset_yaml
):
    project_dir = tmp_path / "runs"
    cfg_path = _project_config(write_yaml, pothole_dataset_yaml, project_dir)
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 0
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 0  # only run_config.json inside
    (project_dir / "baseline" / "weights").mkdir()
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 2


def test_fresh_run_dir_tells_ultralytics_exist_ok(write_yaml, tmp_path, pothole_dataset_yaml, capsys):
    # Found by a real run: this module creates the run directory before Ultralytics
    # starts, so Ultralytics must always be told to use it, or it renames the run.
    project_dir = tmp_path / "runs"
    cfg_path = _project_config(write_yaml, pothole_dataset_yaml, project_dir)
    assert not (project_dir / "baseline").exists()
    assert train.main(["--config", str(cfg_path), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["exist_ok"] is True
    record = json.loads((project_dir / "baseline" / "run_config.json").read_text(encoding="utf-8"))
    assert record["ultralytics_args"]["exist_ok"] is True
    assert record["train_config"]["exist_ok"] is False  # the configured intent is kept


def test_prepare_run_dir(tmp_path):
    run_dir = tmp_path / "r"
    assert prepare_run_dir(run_dir, exist_ok=False) is False  # absent
    run_dir.mkdir()
    assert prepare_run_dir(run_dir, exist_ok=False) is True  # empty: must tell Ultralytics
    (run_dir / "results.csv").write_text("x", encoding="utf-8")
    with pytest.raises(RunDirError):
        prepare_run_dir(run_dir, exist_ok=False)
    assert prepare_run_dir(run_dir, exist_ok=True) is False


def test_dataset_dirs_must_exist_and_be_non_empty(tmp_path, pothole_dataset_yaml):
    load_dataset_yaml(pothole_dataset_yaml)  # sanity: valid fixture passes
    (pothole_dataset_yaml.parent / "images" / "val" / "img.jpg").unlink()
    with pytest.raises(DatasetConfigError, match="'val' image directory is empty"):
        load_dataset_yaml(pothole_dataset_yaml)
    (pothole_dataset_yaml.parent / "images" / "val").rmdir()
    with pytest.raises(DatasetConfigError, match="'val' images not found"):
        load_dataset_yaml(pothole_dataset_yaml)


def test_relative_path_key_and_missing_path_key_resolve_against_yaml_dir(tmp_path):
    ds_dir = tmp_path / "ds"
    for split in ("train", "val"):
        (ds_dir / "data" / split).mkdir(parents=True)
        (ds_dir / "data" / split / "a.jpg").write_bytes(b"x")
    yaml_path = ds_dir / "dataset.yaml"
    body = {"train": "train", "val": "val", "names": {0: "pothole"}}
    for path_key, expected in ((None, ds_dir), ("data", ds_dir / "data")):
        content = {**body, **({"path": path_key} if path_key else {})}
        if path_key is None:
            content = {"train": "data/train", "val": "data/val", "names": {0: "pothole"}}
        yaml_path.write_text(yaml.safe_dump(content), encoding="utf-8")
        ds = load_dataset_yaml(yaml_path)
        out = write_resolved_dataset_yaml(ds, yaml_path, tmp_path / f"run_{path_key}")
        assert yaml.safe_load(out.read_text(encoding="utf-8"))["path"] == expected.resolve().as_posix()
