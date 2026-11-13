#!/usr/bin/env bash
# Runs ON the rented GPU instance, from the unpacked project root.
#
#   bash scripts/vast_remote.sh setup            install dependencies, check the GPU
#   bash scripts/vast_remote.sh data             fetch RDD2022, convert, verify against the local digest
#   bash scripts/vast_remote.sh train <batch> <workers>   start training detached, return at once
#   bash scripts/vast_remote.sh resume           continue an interrupted run from last.pt
#   bash scripts/vast_remote.sh status           one-screen progress report
#   bash scripts/vast_remote.sh pack             bundle the run directory and logs into results.tgz
#
# Every stage appends to logs/<stage>.log. Nothing here evaluates the test split.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

RUN_DIR="runs/train/baseline"
DATASET="data/processed/rdd2022_pothole"
RAW="data/raw/rdd2022"

pick_python() {
    local candidate
    for candidate in /venv/main/bin/python python python3 /opt/conda/bin/python; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import torch" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    echo "no python with torch found" >&2
    return 1
}

PY="$(pick_python)"

gpu_check() {
    "$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is not available"
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 2**30
x = torch.randn(2048, 2048, device="cuda")
y = (x @ x).sum().item()
print("gpu", name, "%.1f GiB" % total, "matmul ok", y == y)
try:
    import torchvision
except ImportError:
    print("torchvision not installed yet")
else:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]], device="cuda")
    keep = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
    print("torchvision", torchvision.__version__, "cuda nms ok", keep.tolist() == [0])
EOF
}

stage_setup() {
    echo "== python: $PY"; "$PY" --version
    nvidia-smi
    nproc; free -g | head -2; df -h . | tail -1
    gpu_check
    if ! "$PY" -c "import ctypes; ctypes.CDLL('libGL.so.1')" >/dev/null 2>&1; then
        (apt-get update -qq && (apt-get install -y -qq libgl1 libglib2.0-0 || apt-get install -y -qq libgl1 libglib2.0-0t64)) || echo "apt install failed; continuing"
    fi
    # Keep the image's own torch build: it is the one matched to this GPU and driver.
    "$PY" -m pip freeze | grep -iE '^(torch|torchvision)==' > logs/torch_constraints.txt || true
    echo "== torch constraints:"; cat logs/torch_constraints.txt
    # Pillow and NumPy are pinned to the versions used locally: the split depends on the image hash.
    "$PY" -m pip install --no-input -q -c logs/torch_constraints.txt "pillow==12.3.0" "numpy==2.2.6" pyyaml "pydantic>=2" "ultralytics==8.4.210"
    "$PY" -m pip install --no-input -q --no-deps -e .
    "$PY" -c "import cv2, PIL, numpy, ultralytics, roadsense; print('cv2', cv2.__version__, 'pillow', PIL.__version__, 'numpy', numpy.__version__, 'ultralytics', ultralytics.__version__)"
    gpu_check
    "$PY" -m pip freeze > logs/pip_freeze.txt
    echo "== setup ok"
}

stage_data() {
    [ -f logs/dataset_digest_local.json ] || { echo "missing logs/dataset_digest_local.json (upload it first)"; exit 1; }
    "$PY" -m roadsense.data.fetch_rdd2022 --countries India Japan Czech United_States China_MotorBike
    # --overwrite so the stage can be re-run after a failure; the fetch skips countries already present.
    "$PY" -m roadsense.data.convert_rdd2022 --config configs/dataset_rdd2022.yaml --overwrite
    "$PY" scripts/dataset_digest.py "$DATASET" > logs/dataset_digest_remote.json
    "$PY" - <<'EOF'
import json, sys
local = json.load(open("logs/dataset_digest_local.json", encoding="utf-8-sig"))
remote = json.load(open("logs/dataset_digest_remote.json", encoding="utf-8"))
if local != remote:
    print("DATASET DIGEST MISMATCH"); print(json.dumps({"local": local, "remote": remote}, indent=1)); sys.exit(1)
print("dataset digest identical to the locally verified dataset:", remote["manifest_sha256"])
EOF
    "$PY" scripts/verify_converted_dataset.py "$DATASET" "$RAW" logs/verify_converted_remote.json
    echo "== data ok"
}

launch_detached() {
    # $1 = log file, $2 = command. Survives the SSH session closing.
    local log="$1" cmd="$2"
    if [ -f logs/train.pid ] && kill -0 "$(cat logs/train.pid)" 2>/dev/null; then
        echo "training already running (pid $(cat logs/train.pid))"; return 1
    fi
    rm -f logs/train.exit logs/train.finished
    date -u +%FT%TZ >> logs/train.started
    PYTHONUNBUFFERED=1 nohup setsid bash -c "$cmd; echo \$? > logs/train.exit; date -u +%FT%TZ > logs/train.finished" >> "$log" 2>&1 < /dev/null &
    echo $! > logs/train.pid
}

stage_train() {
    local batch="$1" workers="$2"
    launch_detached logs/train.log "'$PY' -m roadsense.training.train --config configs/train_baseline.yaml --batch $batch --workers $workers"
    echo "== training started, pid $(cat logs/train.pid), batch $batch, workers $workers"
}

stage_resume() {
    # Continue an interrupted run from its last checkpoint with the arguments stored in it.
    [ -f "$RUN_DIR/weights/last.pt" ] || { echo "no $RUN_DIR/weights/last.pt to resume from"; exit 1; }
    launch_detached logs/train.log "'$PY' -c \"from ultralytics import YOLO; YOLO('$RUN_DIR/weights/last.pt').train(resume=True)\""
    echo "== resume started, pid $(cat logs/train.pid)"
}

stage_status() {
    echo "started: $(tr '\n' ' ' < logs/train.started 2>/dev/null || echo -)  finished: $(cat logs/train.finished 2>/dev/null || echo -)  exit: $(cat logs/train.exit 2>/dev/null || echo running)  now: $(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    if [ -f "$RUN_DIR/results.csv" ]; then
        echo "epochs logged: $(($(wc -l < "$RUN_DIR/results.csv") - 1))"
        head -1 "$RUN_DIR/results.csv"; tail -2 "$RUN_DIR/results.csv"
    fi
    grep -a "Reducing to batch" logs/train.log || true
    free -g | sed -n 2p
    tail -c 1500 logs/train.log | tr '\r' '\n' | tail -6
}

stage_pack() {
    # tar exits 1 when a file changes while being read (training still running); that is tolerated.
    [ -f logs/train.exit ] || echo "warning: training has not finished; packing a snapshot"
    tar czf results.tgz "$RUN_DIR" logs "$DATASET/conversion_report.json" "$DATASET/manifest.csv" || [ $? -eq 1 ]
    sha256sum results.tgz "$RUN_DIR"/weights/*.pt | tee logs/results_sha256.txt
    ls -l results.tgz
}

stage="${1:-}"
case "$stage" in
    setup) stage_setup 2>&1 | tee -a logs/setup.log ;;
    data) stage_data 2>&1 | tee -a logs/data.log ;;
    train) stage_train "${2:?batch}" "${3:?workers}" ;;
    resume) stage_resume ;;
    status) stage_status ;;
    pack) stage_pack ;;
    *) echo "usage: $0 setup|data|train <batch> <workers>|resume|status|pack" >&2; exit 2 ;;
esac
