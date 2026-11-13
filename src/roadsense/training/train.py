"""Training entrypoint for the single-class pothole detector.

Usage::

    python -m roadsense.training.train --config configs/train_baseline.yaml \
        [--device cpu] [--batch 64] [--workers 8] [--dry-run]

Steps: load and validate the config, verify the dataset yaml (exactly one class
named ``pothole``, ``train`` and ``val`` entries whose image directories exist
and are not empty), write ``<project>/<name>/dataset_resolved.yaml`` (the same
dataset with an absolute ``path:``, because Ultralytics' handling of relative
dataset roots has varied between versions) and the resolved configuration to
``<project>/<name>/run_config.json``, then train with Ultralytics on the
resolved yaml. Checkpoints and ``results.csv`` are Ultralytics' own outputs under
``<project>/<name>/``.

This module never evaluates on the test split. Only the ``val`` split is used,
for Ultralytics' per-epoch validation and best-checkpoint selection. Held-out
test evaluation belongs to ``roadsense.training.evaluate`` behind an explicit
flag.

``--dry-run`` does everything except import ultralytics and train.

Dataset ``train``/``val`` entries may be image directories (the baseline) or text
files listing images. The entry format is not assumed (relative to the dataset
root, relative to the list file, or absolute; either slash style). Ultralytics
resolves relative entries against the current directory, so the resolved copy
written into the run directory (``lists/<name>.txt``) holds absolute paths, built
on the machine that trains. Every listed image must exist and none may sit under a
``test`` directory.

Optional config fields (``optim``, ``time_hours``, ``save_period``, ``fraction``)
are passed to Ultralytics only when set; the baseline config sets none of them.
``time_hours`` is a wall-clock cap that uses Ultralytics' ``time`` mode: after every
epoch the trainer sets ``epochs = ceil(time / mean epoch time)`` and rebuilds the
learning-rate schedule. A callback makes this shrink-only (never above the configured
``epochs``), so a short budget (for example a late start) gives a coherent, shorter
schedule that still anneals and reaches close_mosaic, while a long budget does not
stretch the run.

``--resume runs/<project>/<name>/weights/last.pt [--time-hours H]`` continues an
interrupted run with its own wall-clock cap (Ultralytics' time mode is off for the
resumed process; the cap restarts from zero there, so pass the remaining hours).
The saved epoch count and schedule are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from roadsense.config import ConfigError, TrainConfig, load_train_config, project_root

# Written next to dataset.yaml by roadsense.data.convert_rdd2022.
MANIFEST_NAME = "manifest.csv"
RUN_CONFIG_NAME = "run_config.json"
RESOLVED_YAML_NAME = "dataset_resolved.yaml"
RESOLVED_LISTS_DIR = "lists"
# Run directories this module must never write into, relative to the project root.
PROTECTED_RUN_ROOTS = ("model_archive", "runs/train/baseline")
_PACKAGES = (
    "roadsense",
    "ultralytics",
    "torch",
    "torchvision",
    "opencv-python",
    "numpy",
    "pyyaml",
    "pydantic",
)


class DatasetConfigError(Exception):
    """The dataset yaml is missing or does not describe a single-class pothole dataset."""


class RunDirError(Exception):
    """The run output directory cannot be used as configured."""


class ModelTaskError(Exception):
    """The configured weights are not a box-detection model."""


class ModelFileError(Exception):
    """The configured weights file (a path, not a bare Ultralytics name) does not exist."""


def validate_dataset_config(ds: Any) -> None:
    """Check a parsed dataset yaml. Pure function; raises DatasetConfigError."""
    if not isinstance(ds, dict):
        raise DatasetConfigError("dataset yaml must be a mapping")
    names = ds.get("names")
    if isinstance(names, dict):
        keys = list(names.keys())
        class_names = list(names.values())
        if keys != [0]:
            raise DatasetConfigError(
                f"dataset must have exactly one class with id 0, got ids {keys}"
            )
    elif isinstance(names, list):
        class_names = names
    else:
        raise DatasetConfigError("dataset yaml has no usable 'names' (expected {0: pothole})")
    if class_names != ["pothole"]:
        raise DatasetConfigError(
            f"dataset must declare exactly one class named 'pothole', got {class_names}"
        )
    if "nc" in ds and ds["nc"] != 1:
        raise DatasetConfigError(f"dataset 'nc' must be 1, got {ds['nc']!r}")
    for split in ("train", "val"):
        if not ds.get(split):
            raise DatasetConfigError(f"dataset yaml has no '{split}' entry")


def dataset_root(ds: dict[str, Any], yaml_path: Path) -> Path:
    """Directory the dataset's ``train``/``val`` entries are relative to.

    The ``path`` key if present (a relative one is taken relative to the yaml's
    directory), otherwise the yaml's own directory.
    """
    base = yaml_path.resolve().parent
    if ds.get("path"):
        candidate = Path(str(ds["path"]))
        return candidate if candidate.is_absolute() else (base / candidate).resolve()
    return base


def _entries(value: Any) -> list[str]:
    return [str(e) for e in (value if isinstance(value, list) else [value])]


def is_list_entry(entry: str) -> bool:
    return entry.lower().endswith(".txt")


def resolve_list_entry(entry: str, list_path: Path, root: Path) -> Path | None:
    """Absolute path of one list entry, or None if no candidate exists.

    The entry format is not assumed: it may use either slash style, be absolute, relative to
    the dataset root, or relative to the list file's directory (including the
    ``./`` form Ultralytics itself resolves against the list file).
    """
    norm = Path(entry.replace("\\", "/"))
    candidates = [norm] if norm.is_absolute() else [root / norm, list_path.parent / norm]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def read_list_file(list_path: Path, root: Path) -> list[Path]:
    """Resolved absolute image paths named by a list file; blank lines are ignored.

    Raises DatasetConfigError for an empty list, an entry under a ``test``
    directory, or an image that does not exist.
    """
    lines = [ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise DatasetConfigError(f"image list is empty: {list_path}")
    root_resolved = root.resolve()
    out: list[Path] = []
    for ln in lines:
        if "test" in (part.lower() for part in Path(ln.replace(chr(92), "/")).parts[:-1]):
            raise DatasetConfigError(f"{list_path}: entry is under a test directory: {ln}")
        path = resolve_list_entry(ln, list_path, root)
        if path is None:
            raise DatasetConfigError(f"{list_path}: listed image not found: {ln}")
        try:
            parts = path.relative_to(root_resolved).parts[:-1]
        except ValueError:  # outside the dataset root: judge every directory component
            parts = path.parts[:-1]
        if "test" in (part.lower() for part in parts):
            raise DatasetConfigError(f"{list_path}: entry is under a test directory: {ln}")
        out.append(path)
    return out


def check_dataset_dirs(ds: dict[str, Any], yaml_path: Path) -> None:
    """Check that the train and val entries exist and hold at least one image.

    A directory must hold at least one file; a list file must list only existing
    images (see ``read_list_file``).
    """
    root = dataset_root(ds, yaml_path)
    for split in ("train", "val"):
        for entry in _entries(ds[split]):
            target = (root / entry).resolve()
            if target.is_file():
                if is_list_entry(entry):
                    read_list_file(target, root)
                continue
            if not target.is_dir():
                raise DatasetConfigError(f"'{split}' images not found: {target}")
            if not any(p.is_file() for p in target.iterdir()):
                raise DatasetConfigError(f"'{split}' image directory is empty: {target}")


def write_resolved_dataset_yaml(ds: dict[str, Any], yaml_path: Path, run_dir: Path) -> Path:
    """Write ``run_dir/dataset_resolved.yaml``: the dataset with an absolute ``path:``.

    List-file entries are rewritten as absolute-path lists under ``run_dir/lists``.
    """
    root = dataset_root(ds, yaml_path)
    resolved = {"path": root.as_posix()}
    resolved.update({k: v for k, v in ds.items() if k != "path"})
    run_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        entries = _entries(ds[split])
        if not any(is_list_entry(e) for e in entries):
            continue
        out_entries = []
        for entry in entries:
            if not is_list_entry(entry):
                out_entries.append(entry)
                continue
            images = read_list_file((root / entry).resolve(), root)
            out_list = run_dir / RESOLVED_LISTS_DIR / Path(entry).name
            out_list.parent.mkdir(parents=True, exist_ok=True)
            out_list.write_text(
                "\n".join(p.as_posix() for p in images) + "\n",
                encoding="utf-8",
            )
            out_entries.append(out_list.resolve().as_posix())
        resolved[split] = out_entries if isinstance(ds[split], list) else out_entries[0]
    out = run_dir / RESOLVED_YAML_NAME
    out.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return out


def load_dataset_yaml(path: Path) -> dict[str, Any]:
    """Read and validate the dataset yaml at ``path``."""
    if not path.is_file():
        raise DatasetConfigError(
            f"dataset yaml not found: {path}\n"
            "Run the dataset converter first to create it."
        )
    try:
        ds = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DatasetConfigError(f"{path}: invalid YAML: {exc}") from exc
    try:
        validate_dataset_config(ds)
        check_dataset_dirs(ds, path)
    except DatasetConfigError as exc:
        raise DatasetConfigError(f"{path}: {exc}") from exc
    return ds


def build_ultralytics_args(cfg: TrainConfig, data_yaml: Path | None = None) -> dict[str, Any]:
    """Build the keyword arguments for ``YOLO(model).train(**args)``. Pure function.

    ``data_yaml`` is the dataset yaml handed to Ultralytics (the resolved copy in
    a real run); it defaults to the configured one.
    """
    args: dict[str, Any] = {
        "data": str(data_yaml if data_yaml is not None else cfg.data),
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "epochs": cfg.epochs,
        "patience": cfg.patience,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "device": cfg.device,
        "workers": cfg.workers,
        "project": str(cfg.project),
        "name": cfg.name,
        "exist_ok": cfg.exist_ok,
        "val": True,
        "plots": True,
        "save": True,
        **cfg.augment.model_dump(),
    }
    # Optional settings: absent from the arguments unless configured, so the
    # baseline config keeps producing exactly the arguments it always did.
    args.update({k: v for k, v in cfg.optim.model_dump().items() if v is not None})
    if cfg.time_hours is not None:
        args["time"] = cfg.time_hours
    if cfg.save_period is not None:
        args["save_period"] = cfg.save_period
    if cfg.fraction is not None:
        args["fraction"] = cfg.fraction
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    """Installed versions, read from metadata without importing the packages."""
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def is_weights_path(model: str) -> bool:
    """True for a weights file path, False for a bare name Ultralytics resolves itself."""
    return "/" in model or "\\" in model


def list_hashes(yaml_path: Path) -> dict[str, str]:
    """SHA-256 of every image list file named by the dataset yaml ({} for directory datasets)."""
    ds = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = dataset_root(ds, yaml_path)
    out: dict[str, str] = {}
    for split in ("train", "val"):
        for entry in _entries(ds[split]):
            if is_list_entry(entry):
                out[entry] = sha256_file((root / entry).resolve())
    return out


def pin_epochs_callback(epochs: int):
    """Ultralytics callback: in time mode, the configured epoch count is an upper bound only.

    In time mode the trainer overwrites ``epochs`` after every epoch with
    ``ceil(time / mean epoch time)`` and rebuilds the learning-rate schedule to
    fill the time budget. That is wanted when the budget is short (a late start
    then yields a coherent, shorter cosine schedule that still reaches the
    close_mosaic phase) but not when time is plentiful (50 configured epochs
    must not become 90). So the value is shrink-only: ``min(epochs, trainer
    epochs)``. Only an overshoot is undone, with the two steps the trainer
    itself performs (``_setup_scheduler`` then ``last_epoch``).
    """

    def on_fit_epoch_end(trainer) -> None:
        if trainer.args.time and trainer.epochs > epochs:
            trainer.epochs = trainer.args.epochs = epochs
            trainer._setup_scheduler()
            trainer.scheduler.last_epoch = trainer.epoch

    return on_fit_epoch_end


def wall_clock_cap_callbacks(hours: float, clock=None):
    """(on_train_start, on_train_batch_end) callbacks implementing a hard wall-clock cap.

    Used for ``--resume`` instead of Ultralytics' time mode: in time mode a resumed
    trainer assigns ``ceil(remaining time / mean epoch time)`` to the TOTAL epoch
    count, ignoring the epochs already done, which can end the run at once. Here
    the saved epoch count and schedule are kept and the cap only stops training
    (the trainer validates, saves and finishes normally after a stop).
    """
    import time as _time

    now = clock or _time.time
    state: dict[str, float] = {}

    def on_train_start(trainer) -> None:
        state["t0"] = now()

    def on_train_batch_end(trainer) -> None:
        if "t0" in state and now() - state["t0"] > hours * 3600:
            trainer.stop = True

    return on_train_start, on_train_batch_end


def guard_run_dir(run_dir: Path, root: Path | None = None) -> None:
    """Refuse run directories inside the archived baseline run or the model archive."""
    base = (root or project_root()).resolve()
    target = run_dir.resolve()
    for protected in PROTECTED_RUN_ROOTS:
        guarded = (base / protected).resolve()
        if target == guarded or guarded in target.parents:
            raise RunDirError(f"refusing to write into protected location: {guarded}")


def prepare_run_dir(run_dir: Path, exist_ok: bool) -> bool:
    """Check the run directory is usable; return True if Ultralytics must be told exist_ok.

    Ultralytics silently renames the run (``baseline`` -> ``baseline2``) when the
    directory exists and ``exist_ok`` is false, which would separate the outputs
    from ``run_config.json``. So an existing directory is an error unless it only
    holds only files written by this module (``run_config.json``,
    ``dataset_resolved.yaml``) left by an earlier dry run or aborted launch.
    """
    if exist_ok or not run_dir.exists():
        return False
    ours = {RUN_CONFIG_NAME, RESOLVED_YAML_NAME, RESOLVED_LISTS_DIR}
    leftovers = [p.name for p in run_dir.iterdir() if p.name not in ours]
    if leftovers:
        raise RunDirError(
            f"run directory already exists and is not empty: {run_dir}\n"
            "Choose a different 'name' or set exist_ok: true."
        )
    return True


def write_run_config(
    cfg: TrainConfig,
    args: dict[str, Any],
    config_path: Path,
    run_dir: Path,
    resolved_yaml: Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Write the fully resolved configuration to ``run_dir/run_config.json``.

    ``overrides`` are the command-line values that replaced config-file values;
    they are recorded so the run can be told apart from the file alone.
    """
    manifest = cfg.data.parent / MANIFEST_NAME
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path.resolve()),
        "config_file_sha256": sha256_file(config_path),
        "cli_overrides": dict(overrides or {}),
        "train_config": cfg.model_dump(mode="json"),
        "ultralytics_args": args,
        "dataset_yaml": str(cfg.data),
        "dataset_yaml_sha256": sha256_file(cfg.data),
        "dataset_resolved_yaml": str(resolved_yaml),
        "dataset_resolved_yaml_sha256": sha256_file(resolved_yaml),
        "dataset_manifest": str(manifest) if manifest.is_file() else None,
        "dataset_manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
        "dataset_lists_sha256": list_hashes(cfg.data),
        "init_weights_sha256": sha256_file(Path(cfg.model)) if is_weights_path(cfg.model) else None,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / RUN_CONFIG_NAME
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def run(
    config_path: Path,
    device: str | None = None,
    dry_run: bool = False,
    batch: int | None = None,
    workers: int | None = None,
    time_hours: float | None = None,
) -> dict[str, Any]:
    """Validate, record, and (unless ``dry_run``) train. Returns the Ultralytics args.

    ``device``, ``batch``, ``workers`` and ``time_hours`` replace the config-file
    values when given. They depend on the machine and the clock, not on the
    experiment design (``time_hours`` is the wall-clock cap, see the module docstring).
    """
    given = {"device": device, "batch": batch, "workers": workers, "time_hours": time_hours}
    overrides = {key: value for key, value in given.items() if value is not None}
    cfg = load_train_config(config_path, overrides=overrides or None)
    ds = load_dataset_yaml(cfg.data)
    run_dir = cfg.project / cfg.name
    guard_run_dir(run_dir)
    if is_weights_path(cfg.model) and not Path(cfg.model).is_file():
        raise ModelFileError(f"initial weights not found: {cfg.model}")
    prepare_run_dir(run_dir, cfg.exist_ok)  # raises if the directory holds another run
    resolved_yaml = write_resolved_dataset_yaml(ds, cfg.data, run_dir)
    args = build_ultralytics_args(cfg, resolved_yaml)
    # The run directory now exists, because this module has just written into it.
    # Without exist_ok Ultralytics would rename the run ("baseline" -> "baseline-2")
    # and separate its outputs from run_config.json. The config's own exist_ok has
    # already been enforced by prepare_run_dir above.
    args["exist_ok"] = True
    write_run_config(cfg, args, config_path, run_dir, resolved_yaml, overrides)
    if dry_run:
        print(json.dumps(args, indent=2))
        return args

    from ultralytics import YOLO  # lazy: heavy import, absent from the core install

    model = YOLO(cfg.model)
    if model.task != "detect":
        raise ModelTaskError(
            f"{cfg.model} is a '{model.task}' model; this pipeline trains box detection only"
        )
    if cfg.time_hours is not None:
        model.add_callback("on_fit_epoch_end", pin_epochs_callback(cfg.epochs))
    model.train(**args)
    return args


def resume(
    last_pt: Path,
    time_hours: float | None = None,
    device: str | None = None,
    workers: int | None = None,
    batch: int | None = None,
) -> None:
    """Continue an interrupted run from ``last_pt`` (optimiser, EMA and epoch are restored).

    Ultralytics' resume honours only a few overrides (batch, workers, device,
    time, ...); the rest come from the checkpoint. The wall-clock cap restarts
    from zero in this process, so pass the hours that remain.
    """
    last_pt = Path(last_pt)
    if not last_pt.is_file():
        raise ModelFileError(f"checkpoint not found: {last_pt}")
    guard_run_dir(last_pt.resolve().parent.parent)

    from ultralytics import YOLO

    model = YOLO(str(last_pt))
    train_args = dict(model.ckpt.get("train_args", {}))
    overrides: dict[str, Any] = {"resume": True}
    for key, value in (("device", device), ("workers", workers), ("batch", batch)):
        if value is not None:
            overrides[key] = value
    if train_args.get("time") or time_hours is not None:
        # Own wall-clock cap; Ultralytics' time mode is switched off for the resumed process.
        overrides["time"] = None
        cap = time_hours if time_hours is not None else float(train_args["time"])
        on_start, on_batch_end = wall_clock_cap_callbacks(cap)
        model.add_callback("on_train_start", on_start)
        model.add_callback("on_train_batch_end", on_batch_end)
    model.train(**overrides)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the single-class pothole detector.")
    parser.add_argument("--config", type=Path, help="training config yaml")
    parser.add_argument("--device", help="override the config device, e.g. cpu or 0")
    parser.add_argument("--batch", type=int, help="override the config batch size")
    parser.add_argument("--workers", type=int, help="override the config dataloader workers")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate, write run_config.json and print the arguments; do not train",
    )
    parser.add_argument("--resume", type=Path, metavar="LAST_PT", help="resume from this last.pt")
    parser.add_argument(
        "--time-hours",
        type=float,
        help="wall-clock cap in hours: overrides the config's time_hours for a fresh run; "
        "with --resume, the cap for the resumed process (the hours that remain)",
    )
    ns = parser.parse_args(argv)
    try:
        if ns.resume is not None:
            resume(ns.resume, ns.time_hours, ns.device, ns.workers, ns.batch)
            return 0
        if ns.config is None:
            parser.error("--config is required unless --resume is given")
        run(
            ns.config,
            device=ns.device,
            dry_run=ns.dry_run,
            batch=ns.batch,
            workers=ns.workers,
            time_hours=ns.time_hours,
        )
    except (ConfigError, DatasetConfigError, RunDirError, ModelTaskError, ModelFileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
