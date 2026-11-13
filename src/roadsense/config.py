"""Configuration loading and validation.

Configs are YAML files validated by strict pydantic models. Unknown keys and
wrong types are errors, and every error message names the file and the field.

Relative paths in a config resolve against the project root (the directory
containing ``pyproject.toml``), never the current working directory. A relative
path that would leave the project root (for example ``../elsewhere``) is
rejected; an absolute path is an explicit choice and is accepted as written.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)


class ConfigError(Exception):
    """A config file is missing, unreadable, or invalid."""


def project_root() -> Path:
    """Return the directory containing ``pyproject.toml``."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ConfigError(
        "cannot locate the project root (no pyproject.toml above "
        f"{Path(__file__).resolve()}); run from a source checkout"
    )


def _resolve_project_path(value: Path, info: ValidationInfo) -> Path:
    if value.is_absolute():
        return value
    root = Path((info.context or {}).get("root") or project_root())
    root = Path(os.path.normpath(root))
    resolved = Path(os.path.normpath(root / value))
    if not resolved.is_relative_to(root):
        raise ValueError(f"relative path {str(value)!r} escapes the project root")
    return resolved


ProjectPath = Annotated[Path, AfterValidator(_resolve_project_path)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


Number = StrictFloat | StrictInt


class IngestConfig(_Strict):
    """Provisional ingestion thresholds; see configs/app.yaml."""

    max_gps_accuracy_m: Number = 25.0
    max_gps_gap_s: Number = 3.0
    max_gps_extrapolation_s: Number = 1.0

    @field_validator("max_gps_accuracy_m", "max_gps_gap_s")
    @classmethod
    def _positive(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"must be a finite number > 0, got {value}")
        return value

    @field_validator("max_gps_extrapolation_s")
    @classmethod
    def _non_negative_finite(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"must be a finite number >= 0, got {value}")
        return value


def _finite_positive(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"must be a finite number > 0, got {value}")
    return value


def _unit_interval(value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"must be between 0 and 1, got {value}")
    return value


_INFER_DEVICE_RE = re.compile(r"^(auto|cpu|\d+)$")


class DetectionConfig(_Strict):
    """Inference with the trained detector. Thresholds are PROVISIONAL."""

    weights: ProjectPath = Path("runs/train/baseline/weights/best.pt")
    conf: Number = 0.25  # minimum detection confidence
    iou: Number = 0.5  # NMS IoU threshold
    imgsz: StrictInt = 640
    device: StrictStr = "auto"  # "auto" = GPU 0 when CUDA is available, else CPU
    # Share of analysed frames the detector may fail on before a job is no longer
    # a plain success: up to ``partial_frame_error_fraction`` the job is "complete"
    # (isolated failures, only counted), above it "partial" (kept, with a warning),
    # above ``max_frame_error_fraction`` the job fails.
    partial_frame_error_fraction: Number = 0.02
    max_frame_error_fraction: Number = 0.25

    @field_validator("conf", "iou", "partial_frame_error_fraction", "max_frame_error_fraction")
    @classmethod
    def _unit(cls, value: float) -> float:
        return _unit_interval(value)

    @model_validator(mode="after")
    def _error_fractions_ordered(self) -> "DetectionConfig":
        if self.partial_frame_error_fraction > self.max_frame_error_fraction:
            raise ValueError(
                "partial_frame_error_fraction must not exceed max_frame_error_fraction"
            )
        return self

    @field_validator("imgsz")
    @classmethod
    def _imgsz(cls, value: int) -> int:
        if value < 32 or value % 32 != 0:
            raise ValueError(f"must be a positive multiple of 32, got {value}")
        return value

    @field_validator("device")
    @classmethod
    def _device(cls, value: str) -> str:
        if not _INFER_DEVICE_RE.match(value):
            raise ValueError(f"must be 'auto', 'cpu' or a GPU index like '0', got {value!r}")
        return value


class TrackingConfig(_Strict):
    """Frame sampling and track association. PROVISIONAL.

    ``frame_stride``, ``max_age_frames`` and ``min_hits`` are the settings for slow
    (walking) sessions. A session whose upper-quartile GPS speed is at least
    ``vehicle_speed_mps`` is a vehicle recording: a pothole then moves further
    between analysed frames than its own height, so every ``vehicle_frame_stride``-th
    frame is analysed and ``vehicle_min_hits`` detections are needed. The track age
    limit is the same time in both modes: ``max_age_frames`` analysed frames at
    ``frame_stride``, i.e. ``max_age_frames * frame_stride`` source frames.
    """

    frame_stride: StrictInt = 3  # run the detector on every Nth frame (slow sessions)
    iou_match: Number = 0.2  # minimum IoU to continue a track
    max_age_frames: StrictInt = 5  # analysed frames (at frame_stride) a track survives unmatched
    min_hits: StrictInt = 3  # detections needed before a track counts as a pothole
    vehicle_speed_mps: Number = 2.5  # upper-quartile GPS speed at or above which a session is a vehicle one
    vehicle_frame_stride: StrictInt = 1  # frame stride for vehicle sessions
    vehicle_min_hits: StrictInt = 6  # detections needed for a track in vehicle sessions

    @field_validator("iou_match")
    @classmethod
    def _unit(cls, value: float) -> float:
        return _unit_interval(value)

    @field_validator("frame_stride", "min_hits", "vehicle_frame_stride", "vehicle_min_hits")
    @classmethod
    def _at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"must be >= 1, got {value}")
        return value

    @field_validator("vehicle_speed_mps")
    @classmethod
    def _positive(cls, value: float) -> float:
        return _finite_positive(value)

    @field_validator("max_age_frames")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"must be >= 0, got {value}")
        return value


class MappingConfig(_Strict):
    """Turning observations into mapped potholes. PROVISIONAL."""

    dedup_radius_m: Number = 8.0  # largest distance allowed between any two sightings of one mapped pothole
    max_snap_m: Number = 30.0  # farthest a point may be from a road to be put on it

    @field_validator("dedup_radius_m", "max_snap_m")
    @classmethod
    def _positive(cls, value: float) -> float:
        return _finite_positive(value)


class RoutingConfig(_Strict):
    graph_path: ProjectPath = Path("data/graphs/road_graph.json")
    # At full avoidance, one expected pothole costs as much as this many metres.
    max_detour_per_pothole_m: Number = 300.0
    # Potholes per 100 m assumed for road with no survey evidence. It is an
    # ASSUMPTION, not a measurement; it keeps unsurveyed road from being costed
    # as if it had no potholes.
    assumed_rate_per_100m: Number = 1.0
    # How many metres of survey the assumption above weighs as much as. Below
    # this much surveyed road the observed rate barely moves the assumed one.
    prior_strength_m: Number = 500.0
    # Overrides the two settings above with one fixed rate (None = not set).
    unsurveyed_rate_per_100m: Number | None = None
    # Share of a route that must be surveyed for its pothole estimate to count
    # as partly (min) or adequately evidence-based; below min it is "insufficient".
    min_surveyed_fraction: Number = 0.2
    adequate_surveyed_fraction: Number = 0.8
    default_speed_kph: Number = 30.0  # used when a road has no usable speed limit

    @field_validator("max_detour_per_pothole_m", "default_speed_kph", "prior_strength_m")
    @classmethod
    def _positive(cls, value: float) -> float:
        return _finite_positive(value)

    @field_validator("unsurveyed_rate_per_100m", "assumed_rate_per_100m")
    @classmethod
    def _rate(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"must be a finite number >= 0 or null, got {value}")
        return value

    @field_validator("min_surveyed_fraction", "adequate_surveyed_fraction")
    @classmethod
    def _fraction(cls, value: float) -> float:
        return _unit_interval(value)

    @model_validator(mode="after")
    def _fractions_ordered(self) -> "RoutingConfig":
        if self.min_surveyed_fraction > self.adequate_surveyed_fraction:
            raise ValueError("min_surveyed_fraction must not exceed adequate_surveyed_fraction")
        return self


class UploadConfig(_Strict):
    dir: ProjectPath = Path("data/uploads")
    artifacts_dir: ProjectPath = Path("data/artifacts")  # annotated detection images
    max_upload_mb: StrictInt = 2048

    @field_validator("max_upload_mb")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"must be >= 1, got {value}")
        return value


class AppConfig(_Strict):
    data_dir: ProjectPath = Path("data")
    db_path: ProjectPath = Path("data/roadsense.sqlite")
    runs_dir: ProjectPath = Path("runs")
    web_dir: ProjectPath = Path("src/roadsense/web/static")
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    mapping: MappingConfig = Field(default_factory=MappingConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    uploads: UploadConfig = Field(default_factory=UploadConfig)


# Defaults match Ultralytics' defaults except flipud, which is stated explicitly.
UnitFloat = Annotated[StrictFloat | StrictInt, Field(ge=0, le=1)]


class AugmentConfig(_Strict):
    """Allow-list of Ultralytics augmentation overrides."""

    hsv_h: UnitFloat = 0.015
    hsv_s: UnitFloat = 0.7
    hsv_v: UnitFloat = 0.4
    degrees: Annotated[StrictFloat | StrictInt, Field(ge=0, le=180)] = 0.0
    translate: UnitFloat = 0.1
    scale: Annotated[StrictFloat | StrictInt, Field(ge=0)] = 0.5
    shear: Annotated[StrictFloat | StrictInt, Field(ge=0, le=180)] = 0.0
    perspective: Annotated[StrictFloat | StrictInt, Field(ge=0, le=0.001)] = 0.0
    flipud: UnitFloat = 0.0
    fliplr: UnitFloat = 0.5
    mosaic: UnitFloat = 1.0
    mixup: UnitFloat = 0.0
    close_mosaic: Annotated[StrictInt, Field(ge=0)] = 10


class OptimConfig(_Strict):
    """Optional optimiser overrides, passed to Ultralytics only when set.

    Left at None they are omitted, so Ultralytics' own choice applies (the baseline
    run used ``optimizer=auto``). An explicit ``optimizer`` makes Ultralytics honour
    ``lr0``; with ``auto`` it ignores ``lr0`` and picks both from the iteration count.
    """

    optimizer: Literal["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSprop", "MuSGD"] | None = None
    lr0: Annotated[StrictFloat | StrictInt, Field(gt=0, le=1)] | None = None
    lrf: Annotated[StrictFloat | StrictInt, Field(gt=0, le=1)] | None = None
    momentum: Annotated[StrictFloat | StrictInt, Field(ge=0, lt=1)] | None = None
    weight_decay: Annotated[StrictFloat | StrictInt, Field(ge=0, le=0.1)] | None = None
    warmup_epochs: Annotated[StrictFloat | StrictInt, Field(ge=0)] | None = None
    warmup_bias_lr: Annotated[StrictFloat | StrictInt, Field(ge=0, le=1)] | None = None
    cos_lr: StrictBool | None = None


_DEVICE_RE = re.compile(r"^(cpu|mps|\d+(,\d+)*)$")


# Ultralytics names weights for other tasks with these suffixes (yolov8s-seg.pt, ...).
_NON_DETECTION_SUFFIXES = ("-seg", "-cls", "-pose", "-obb")


class TrainConfig(_Strict):
    model: StrictStr = "yolov8s.pt"
    data: ProjectPath = Path("data/processed/rdd2022_pothole/dataset.yaml")
    imgsz: StrictInt = 640
    batch: StrictInt = 16
    epochs: StrictInt = 100
    patience: StrictInt = 20
    seed: StrictInt = 42
    deterministic: StrictBool = True
    device: StrictStr = "0"
    workers: StrictInt = 4
    project: ProjectPath = Path("runs/train")
    name: StrictStr = "baseline"
    exist_ok: StrictBool = False
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    # Optional run controls, omitted from the Ultralytics arguments when None.
    # time_hours is a hard wall-clock cap on training (Ultralytics ``time``); the
    # epoch count and learning-rate schedule stay as configured (see train.py).
    time_hours: Annotated[StrictFloat | StrictInt, Field(gt=0, le=72)] | None = None
    save_period: Annotated[StrictInt, Field(ge=1)] | None = None  # keep epochN.pt every N epochs
    fraction: Annotated[StrictFloat | StrictInt, Field(gt=0, le=1)] | None = None  # smoke tests

    @field_validator("model")
    @classmethod
    def _model_path(cls, value: str, info: ValidationInfo) -> str:
        # A bare name such as "yolov8s.pt" is a weights name Ultralytics resolves
        # itself. Anything containing a path separator is a file path and is
        # resolved against the project root like the other paths.
        if not value:
            raise ValueError("must not be empty")
        stem = Path(value.replace("\\", "/")).stem.lower()
        if stem.endswith(_NON_DETECTION_SUFFIXES):
            raise ValueError(
                f"must be box-detection weights, got {value!r} "
                "(segmentation, classification, pose and OBB models are not supported)"
            )
        if "/" in value or "\\" in value:
            return str(_resolve_project_path(Path(value), info))
        return value

    @field_validator("imgsz")
    @classmethod
    def _imgsz(cls, value: int) -> int:
        if value < 32 or value % 32 != 0:
            raise ValueError(f"must be a positive multiple of 32, got {value}")
        return value

    @field_validator("batch", "epochs")
    @classmethod
    def _at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"must be >= 1, got {value}")
        return value

    @field_validator("patience")
    @classmethod
    def _patience(cls, value: int, info: ValidationInfo) -> int:
        epochs = info.data.get("epochs")
        if value < 0:
            raise ValueError(f"must be >= 0, got {value}")
        if epochs is not None and value > epochs:
            raise ValueError(f"must be <= epochs ({epochs}), got {value}")
        return value

    @field_validator("seed", "workers")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"must be >= 0, got {value}")
        return value

    @field_validator("device")
    @classmethod
    def _device(cls, value: str) -> str:
        if not _DEVICE_RE.match(value):
            raise ValueError(
                f"must be 'cpu', 'mps', or GPU indices like '0' or '0,1', got {value!r}"
            )
        return value

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not value or any(c in value for c in "/\\") or value in {".", ".."}:
            raise ValueError("must be a plain directory name")
        return value


def _format_errors(path: Path, exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"]) or "<file>"
        lines.append(f"{path}: {field}: {err['msg']}")
    return "\n".join(lines)


def _load(
    path: str | os.PathLike[str],
    model_cls: type[_Strict],
    root: Path | None,
    overrides: dict[str, Any] | None,
) -> Any:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read config file: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping of settings")
    if overrides:
        raw = {**raw, **overrides}
    try:
        return model_cls.model_validate(raw, context={"root": root or project_root()})
    except ValidationError as exc:
        raise ConfigError(_format_errors(path, exc)) from exc


def load_app_config(
    path: str | os.PathLike[str], root: Path | None = None
) -> AppConfig:
    """Load and validate an application config; ``root`` defaults to the project root."""
    return _load(path, AppConfig, root, None)


def load_train_config(
    path: str | os.PathLike[str],
    root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """Load and validate a training config.

    ``overrides`` (for example ``{"device": "cpu"}``) are applied to the raw
    mapping before validation, so they are validated like file values.
    """
    return _load(path, TrainConfig, root, overrides)


# ---------------------------------------------------------------------------
# Dataset conversion config
# ---------------------------------------------------------------------------

# Country names become folder names, so keep them to a safe alphabet.
CountryName = Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9_]+$")]
Fraction = Annotated[StrictFloat | StrictInt, Field(ge=0, le=1)]


class SplitFractions(_Strict):
    train: Fraction = 0.8
    val: Fraction = 0.1
    test: Fraction = 0.1

    @model_validator(mode="after")
    def _check(self) -> "SplitFractions":
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1, got {total}")
        if self.train <= 0:
            raise ValueError("train fraction must be > 0")
        return self


class SplitConfig(_Strict):
    strategy: Literal["blocks", "country", "similarity"] = "blocks"
    seed: Annotated[StrictInt, Field(ge=0)] = 42
    block_size: Annotated[StrictInt, Field(ge=1)] = 200  # used only by 'blocks'
    # Used only by 'similarity': max Hamming distance (of 64-bit dHash) that links two images.
    max_hash_distance_bits: Annotated[StrictInt, Field(ge=0, le=16)] = 8
    fractions: SplitFractions = Field(default_factory=SplitFractions)
    val_countries: list[CountryName] = Field(default_factory=list)
    test_countries: list[CountryName] = Field(default_factory=list)


def _unique(values: list[str], what: str) -> list[str]:
    if len(set(values)) != len(values):
        raise ValueError(f"{what} must not contain duplicates")
    return values


class DatasetConfig(_Strict):
    raw_dir: ProjectPath = Path("data/raw/rdd2022")
    out_dir: ProjectPath = Path("data/processed/rdd2022_pothole")
    countries: list[CountryName]
    pothole_labels: list[StrictStr] = Field(default_factory=lambda: ["D40"])
    negative_countries: list[CountryName]
    split: SplitConfig = Field(default_factory=SplitConfig)
    train_negative_ratio: Annotated[StrictFloat | StrictInt, Field(ge=0)] | None = 2.0
    min_box_side_px: Annotated[StrictFloat | StrictInt, Field(ge=0)] = 2
    link_mode: Literal["hardlink", "copy"] = "hardlink"

    @field_validator("countries")
    @classmethod
    def _countries(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must not be empty")
        return _unique(value, "countries")

    @field_validator("pothole_labels")
    @classmethod
    def _labels(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must not be empty")
        if any(label != label.strip() or not label for label in value):
            raise ValueError("labels must be non-empty without surrounding whitespace")
        return _unique(value, "pothole_labels")

    @field_validator("negative_countries")
    @classmethod
    def _negative_countries(cls, value: list[str], info: ValidationInfo) -> list[str]:
        _unique(value, "negative_countries")
        countries = info.data.get("countries")
        if countries is not None and not set(value) <= set(countries):
            extra = sorted(set(value) - set(countries))
            raise ValueError(f"must be a subset of countries, not in countries: {extra}")
        return value

    @model_validator(mode="after")
    def _split_countries(self) -> "DatasetConfig":
        if self.split.strategy != "country":
            return self
        val, test = self.split.val_countries, self.split.test_countries
        if not val:
            raise ValueError("split.val_countries must not be empty when split.strategy is 'country'")
        _unique(val, "split.val_countries")
        _unique(test, "split.test_countries")
        for name, group in (("val_countries", val), ("test_countries", test)):
            if not set(group) <= set(self.countries):
                raise ValueError(f"split.{name} must be a subset of countries")
        if set(val) & set(test):
            raise ValueError("split.val_countries and split.test_countries must be disjoint")
        if not set(self.countries) - set(val) - set(test):
            raise ValueError("at least one country must be left for the train split")
        return self


def load_dataset_config(
    path: str | os.PathLike[str], root: Path | None = None
) -> DatasetConfig:
    """Load and validate a dataset conversion config."""
    return _load(path, DatasetConfig, root, None)
