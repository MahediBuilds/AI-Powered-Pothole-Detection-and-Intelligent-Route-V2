"""scripts/export_checkpoints.py in a subprocess (it imports torch/ultralytics, which other tests must not see)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "pretrained" / "baseline_yolov8s_rdd2022_best.pt"
SCRIPT = ROOT / "scripts" / "export_checkpoints.py"

pytestmark = pytest.mark.skipif(not WEIGHTS.is_file(), reason="baseline weights not present")


def run(run_dir, out_dir, *flags):
    return subprocess.run([sys.executable, "-I", str(SCRIPT), str(run_dir), str(out_dir), "--quiet", *flags],
                          capture_output=True, text=True, timeout=300)


def make_run(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    for name in ("epoch0.pt", "epoch5.pt", "last.pt", "best.pt"):
        shutil.copyfile(WEIGHTS, run_dir / "weights" / name)
    return run_dir


def test_only_epochs_skips_best_and_last(tmp_path):
    run_dir = make_run(tmp_path)
    out = tmp_path / "pack"
    assert run(run_dir, out, "--only-epochs").returncode == 0
    assert sorted(p.name for p in (out / "weights").iterdir()) == ["epoch0.pt", "epoch5.pt"]
    index = json.loads((out / "weights_index.json").read_text())
    assert sorted(index) == ["epoch0.pt", "epoch5.pt"] and index["epoch0.pt"]["sha256"]


def test_full_export_adds_best_and_last_and_is_incremental(tmp_path):
    run_dir = make_run(tmp_path)
    out = tmp_path / "pack"
    assert run(run_dir, out, "--only-epochs").returncode == 0
    first = (out / "weights" / "epoch0.pt").stat().st_mtime_ns
    assert run(run_dir, out).returncode == 0
    assert sorted(p.name for p in (out / "weights").iterdir()) == ["best.pt", "epoch0.pt", "epoch5.pt", "last.pt"]
    assert (out / "weights" / "epoch0.pt").stat().st_mtime_ns == first  # unchanged sources are not re-exported
