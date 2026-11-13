from pathlib import Path

import pytest

from roadsense.config import (
    AppConfig,
    ConfigError,
    TrainConfig,
    load_app_config,
    load_train_config,
    project_root,
)

ROOT = project_root()


def test_project_root_contains_pyproject():
    assert (ROOT / "pyproject.toml").is_file()


def test_shipped_configs_load():
    app = load_app_config(ROOT / "configs" / "app.yaml")
    train = load_train_config(ROOT / "configs" / "train_baseline.yaml")
    assert isinstance(app, AppConfig)
    assert isinstance(train, TrainConfig)
    assert train.augment.flipud == 0.0
    assert train.model == "yolov8s.pt"
    assert TrainConfig.model_fields["model"].default == "yolov8s.pt"
    assert train.data == ROOT / "data/processed/rdd2022_pothole/dataset.yaml"


def test_valid_app_config(write_yaml):
    path = write_yaml("app.yaml", {"data_dir": "d", "db_path": "d/x.sqlite"})
    cfg = load_app_config(path)
    assert cfg.data_dir == ROOT / "d"
    assert cfg.db_path == ROOT / "d" / "x.sqlite"
    assert cfg.runs_dir == ROOT / "runs"  # default, also resolved


def test_empty_file_gives_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_app_config(path).data_dir == ROOT / "data"


def test_valid_train_config(write_yaml):
    path = write_yaml("t.yaml", {"imgsz": 320, "batch": 8, "epochs": 10, "patience": 10, "device": "cpu"})
    cfg = load_train_config(path)
    assert (cfg.imgsz, cfg.batch, cfg.epochs, cfg.patience, cfg.device) == (320, 8, 10, 10, "cpu")
    assert cfg.seed == 42 and cfg.deterministic is True and cfg.exist_ok is False


def test_relative_paths_resolve_against_project_root_not_cwd(write_yaml, tmp_path, monkeypatch):
    cfg_path = write_yaml("t.yaml", {"data": "some/dataset.yaml", "project": "runs/x"})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = load_train_config(cfg_path)
    assert cfg.data == ROOT / "some" / "dataset.yaml"
    assert cfg.project == ROOT / "runs" / "x"
    assert cfg.data.is_absolute()


def test_explicit_root_is_used(write_yaml, tmp_path):
    root = tmp_path / "proj"
    cfg = load_app_config(write_yaml("a.yaml", {"db_path": "db/x.sqlite"}), root=root)
    assert cfg.db_path == root / "db" / "x.sqlite"


def test_absolute_path_is_kept(write_yaml, tmp_path):
    target = tmp_path / "abs"
    cfg = load_app_config(write_yaml("a.yaml", {"data_dir": str(target)}))
    assert cfg.data_dir == target


def test_relative_path_escaping_root_rejected(write_yaml):
    with pytest.raises(ConfigError, match="data_dir.*escapes the project root"):
        load_app_config(write_yaml("a.yaml", {"data_dir": "../outside"}))


def test_model_path_resolves_but_bare_name_does_not(write_yaml):
    bare = load_train_config(write_yaml("a.yaml", {"model": "yolov8n.pt"}))
    assert bare.model == "yolov8n.pt"
    path = load_train_config(write_yaml("b.yaml", {"model": "weights/best.pt"}))
    assert Path(path.model) == ROOT / "weights" / "best.pt"


@pytest.mark.parametrize(
    "name",
    ["yolov8s-seg.pt", "YOLOv8s-SEG.pt", "yolov8s-cls.pt", "yolov8s-pose.pt", "yolov8s-obb.pt",
     "weights/yolov8s-seg.pt"],
)  # fmt: skip
def test_non_detection_weights_rejected(write_yaml, name):
    with pytest.raises(ConfigError, match="model.*box-detection"):
        load_train_config(write_yaml("a.yaml", {"model": name}))


def test_unknown_key_rejected_and_error_names_file_and_field(write_yaml):
    path = write_yaml("app.yaml", {"data_dir": "data", "bogus": 1})
    with pytest.raises(ConfigError) as exc:
        load_app_config(path)
    assert str(path) in str(exc.value)
    assert "bogus" in str(exc.value)


def test_unknown_train_key_rejected(write_yaml):
    with pytest.raises(ConfigError, match="single_cls"):
        load_train_config(write_yaml("t.yaml", {"single_cls": True}))


def test_wrong_type_rejected(write_yaml):
    path = write_yaml("t.yaml", {"batch": "sixteen"})
    with pytest.raises(ConfigError) as exc:
        load_train_config(path)
    assert str(path) in str(exc.value) and "batch" in str(exc.value)


@pytest.mark.parametrize("bad", ["16", 16.0, True])
def test_strict_int_does_not_coerce(write_yaml, bad):
    with pytest.raises(ConfigError, match="batch"):
        load_train_config(write_yaml("t.yaml", {"batch": bad}))


def test_device_must_be_string(write_yaml):
    with pytest.raises(ConfigError, match="device"):
        load_train_config(write_yaml("t.yaml", {"device": 0}))
    with pytest.raises(ConfigError, match="device"):
        load_train_config(write_yaml("t2.yaml", {"device": "gpu"}))


def test_int_accepted_for_float_augment(write_yaml):
    cfg = load_train_config(write_yaml("t.yaml", {"augment": {"mosaic": 1, "flipud": 0}}))
    assert cfg.augment.mosaic == 1


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"imgsz": 650}, "imgsz"),
        ({"imgsz": 0}, "imgsz"),
        ({"epochs": 10, "patience": 11}, "patience"),
        ({"patience": -1}, "patience"),
        ({"seed": -1}, "seed"),
        ({"batch": 0}, "batch"),
        ({"epochs": 0, "patience": 0}, "epochs"),
        ({"name": "a/b"}, "name"),
    ],
)
def test_numeric_validation_rules(write_yaml, overrides, field):
    with pytest.raises(ConfigError, match=field):
        load_train_config(write_yaml("t.yaml", overrides))


def test_unknown_augmentation_key_rejected(write_yaml):
    path = write_yaml("t.yaml", {"augment": {"cutmix": 0.5}})
    with pytest.raises(ConfigError, match="augment.cutmix"):
        load_train_config(path)


def test_augmentation_range_rejected(write_yaml):
    with pytest.raises(ConfigError, match="augment.fliplr"):
        load_train_config(write_yaml("t.yaml", {"augment": {"fliplr": 1.5}}))


def test_missing_file_and_bad_yaml_and_non_mapping(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_app_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_app_config(bad)
    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_app_config(lst)


# --------------------------------------------------------------------------- tracking / detection


def test_tracking_and_detection_defaults_match_the_shipped_config():
    shipped = load_app_config(ROOT / "configs" / "app.yaml")
    default = AppConfig()
    assert shipped.tracking == default.tracking
    assert shipped.detection == default.detection
    t = default.tracking
    assert (t.frame_stride, t.iou_match, t.max_age_frames, t.min_hits) == (3, 0.2, 5, 3)
    assert (t.vehicle_speed_mps, t.vehicle_frame_stride, t.vehicle_min_hits) == (2.5, 1, 6)
    d = default.detection
    assert (d.partial_frame_error_fraction, d.max_frame_error_fraction) == (0.02, 0.25)


@pytest.mark.parametrize(
    ("section", "overrides", "field"),
    [
        ("tracking", {"vehicle_speed_mps": 0}, "vehicle_speed_mps"),
        ("tracking", {"vehicle_speed_mps": -1.0}, "vehicle_speed_mps"),
        ("tracking", {"vehicle_speed_mps": float("inf")}, "vehicle_speed_mps"),
        ("tracking", {"vehicle_frame_stride": 0}, "vehicle_frame_stride"),
        ("tracking", {"vehicle_frame_stride": 1.5}, "vehicle_frame_stride"),
        ("tracking", {"vehicle_min_hits": 0}, "vehicle_min_hits"),
        ("tracking", {"vehicle_min_hits": "6"}, "vehicle_min_hits"),
        ("detection", {"partial_frame_error_fraction": -0.1}, "partial_frame_error_fraction"),
        ("detection", {"max_frame_error_fraction": 1.5}, "max_frame_error_fraction"),
        ("detection", {"partial_frame_error_fraction": 0.5, "max_frame_error_fraction": 0.1},
         "must not exceed"),
    ],
)  # fmt: skip
def test_new_tracking_and_detection_fields_are_validated(write_yaml, section, overrides, field):
    with pytest.raises(ConfigError, match=field):
        load_app_config(write_yaml("a.yaml", {section: overrides}))


def test_error_fraction_boundaries_are_accepted(write_yaml):
    cfg = load_app_config(
        write_yaml(
            "a.yaml",
            {"detection": {"partial_frame_error_fraction": 0, "max_frame_error_fraction": 1}},
        )
    )
    assert cfg.detection.partial_frame_error_fraction == 0
    equal = {"partial_frame_error_fraction": 0.3, "max_frame_error_fraction": 0.3}
    assert load_app_config(write_yaml("b.yaml", {"detection": equal})).detection.max_frame_error_fraction == 0.3
