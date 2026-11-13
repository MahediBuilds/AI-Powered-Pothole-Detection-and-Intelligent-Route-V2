#!/usr/bin/env bash
# Runs ON the rented GPU instance, from the unpacked project root (e.g. /workspace/roadsense).
# One-run helper for the unified_v1 fine-tune (configs/train_unified_v1.yaml). The baseline
# helper scripts/vast_remote.sh is separate and untouched.
#
#   bash scripts/vast_remote_unified.sh setup                    install pinned dependencies, check the GPU
#   bash scripts/vast_remote_unified.sh data <digest_sha256>     verify + unpack the uploaded bundle against the pinned digest
#   bash scripts/vast_remote_unified.sh train [batch] [workers]  start training + watchdog, detached; returns at once
#                                                                (time cap = min(2.5, (200 - elapsed_min - 12) / 60) h)
#   bash scripts/vast_remote_unified.sh status                   one-screen progress report
#   bash scripts/vast_remote_unified.sh resume [hours]           continue from last.pt after a crash (restarts the watchdog)
#   bash scripts/vast_remote_unified.sh pack                     stripped checkpoints + logs + SHA256SUMS into pack_out/
#   bash scripts/vast_remote_unified.sh t0 [unix_epoch]          (re)set the instance clock start used by the watchdog
#   bash scripts/vast_remote_unified.sh selfdestruct --check     verify CONTAINER_ID / CONTAINER_API_KEY / vastai CLI; destroys nothing
#   bash scripts/vast_remote_unified.sh selfdestruct --now       destroy THIS instance (Vast.ai) right now; --cancel disarms the watchdog's
#   Optional billing backstop: run `SELFDESTRUCT=1 bash scripts/vast_remote_unified.sh train ...` and the watchdog
#   destroys the instance at T0 + SELFDESTRUCT_AT_MIN (205) minutes at the earliest, after packing, never before.
#   Anything not yet downloaded by then is lost: download first, or disarm with `selfdestruct --cancel`.
#
# Every stage appends to logs/<stage>.log. Nothing here reads or evaluates a test split: the
# bundle contains none (cloud_bundle.py refuses otherwise) and the data stage re-checks.
#
# Environment overrides (used by the local smoke test; the defaults are the real run):
#   CONFIG RUN_DIR DATASET UPLOAD_DIR PY TOTAL_LIMIT_MIN WATCHDOG_POLL_S INIT_WEIGHTS_SHA256
#   TIME_CAP_H (2.5)  TIME_MARGIN_MIN (12)  SELFDESTRUCT (1 to arm)  SELFDESTRUCT_AT_MIN (205)
#   T0_EPOCH   unix time the instance billing started, if known (else the first stage call is T0)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

CONFIG="${CONFIG:-configs/train_unified_v1.yaml}"
RUN_DIR="${RUN_DIR:-runs/train/unified_v1_y8s_1280}"
DATASET="${DATASET:-data/processed/unified_v1}"
UPLOAD_DIR="${UPLOAD_DIR:-upload}"
PACK_DIR="pack_out"
WEIGHTS_NAME="baseline_yolov8s_rdd2022_best.pt"
# sha256 of model_archive/baseline_yolov8s_rdd2022/weights/best.pt
INIT_WEIGHTS_SHA256="${INIT_WEIGHTS_SHA256:-2aabdd131d1664089e06e3219fabcaed4cc96610fa4603ff165429f92696d798}"
# Independent watchdog: stop training and pack when this many minutes have passed since T0.
# Worst case billing: 200 min x $0.70/h = $2.33 (+ pack and download), below the $3 budget even at the price ceiling.
TOTAL_LIMIT_MIN="${TOTAL_LIMIT_MIN:-200}"
WATCHDOG_POLL_S="${WATCHDOG_POLL_S:-30}"
# Training time cap passed to Ultralytics: the ceiling, shortened when the start is late, so the cosine
# schedule is shortened coherently instead of the cap cutting it: min(TIME_CAP_H, (limit - elapsed - margin) / 60).
TIME_CAP_H="${TIME_CAP_H:-2.5}"
TIME_MARGIN_MIN="${TIME_MARGIN_MIN:-12}"
MIN_TRAIN_H="${MIN_TRAIN_H:-0.5}"
# Optional self-destroy backstop (Vast.ai). Never earlier than the watchdog's own hard limit.
SELFDESTRUCT_AT_MIN="${SELFDESTRUCT_AT_MIN:-205}"
MAX_WORKERS=12

pick_python() {
    if [ -n "${PY:-}" ]; then echo "$PY"; return 0; fi
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

now() { date +%s; }
log() { echo "[$(date -u +%FT%TZ)] $*"; }

ensure_t0() {
    if [ -n "${T0_EPOCH:-}" ]; then echo "$T0_EPOCH" > logs/instance_t0; fi
    [ -f logs/instance_t0 ] || now > logs/instance_t0
}
elapsed_min() { echo $(( ( $(now) - $(cat logs/instance_t0) ) / 60 )); }

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
    ensure_t0
    echo "== python: $PY"; "$PY" --version
    nvidia-smi
    nproc; free -g | head -2; df -h . | tail -1
    gpu_check
    if ! "$PY" -c "import ctypes; ctypes.CDLL('libGL.so.1')" >/dev/null 2>&1; then
        (apt-get update -qq && (apt-get install -y -qq libgl1 libglib2.0-0 || apt-get install -y -qq libgl1 libglib2.0-0t64)) || echo "apt install failed; continuing"
    fi
    # Same pins as the baseline rental (logs of 2026-11-11): keep the image's torch build,
    # Pillow 12.3.0, NumPy 2.2.6, Ultralytics 8.4.210.
    "$PY" -m pip freeze | grep -iE '^(torch|torchvision)==' > logs/torch_constraints.txt || true
    echo "== torch constraints:"; cat logs/torch_constraints.txt
    "$PY" -m pip install --no-input -q -c logs/torch_constraints.txt "pillow==12.3.0" "numpy==2.2.6" pyyaml "pydantic>=2" "ultralytics==8.4.210"
    "$PY" -m pip install --no-input -q --no-deps -e .
    "$PY" -c "import cv2, PIL, numpy, ultralytics, roadsense; print('cv2', cv2.__version__, 'pillow', PIL.__version__, 'numpy', numpy.__version__, 'ultralytics', ultralytics.__version__)"
    gpu_check
    "$PY" -m pip freeze > logs/pip_freeze.txt
    echo "== elapsed since T0: $(elapsed_min) min"
    echo "== setup ok"
}

stage_data() {
    ensure_t0
    local digest="${1:?usage: data <digest_sha256 printed by cloud_bundle.py build>}"
    [ -f "$UPLOAD_DIR/bundle_digest.json" ] || { echo "missing $UPLOAD_DIR/bundle_digest.json (upload not finished?)"; exit 1; }
    [ -f "$UPLOAD_DIR/$WEIGHTS_NAME" ] || { echo "missing $UPLOAD_DIR/$WEIGHTS_NAME"; exit 1; }
    local got
    got="$(sha256sum "$UPLOAD_DIR/$WEIGHTS_NAME" | cut -d' ' -f1)"
    [ "$got" = "$INIT_WEIGHTS_SHA256" ] || { echo "INITIAL WEIGHTS SHA256 MISMATCH: $got"; exit 1; }
    mkdir -p pretrained
    cp "$UPLOAD_DIR/$WEIGHTS_NAME" "pretrained/$WEIGHTS_NAME"
    echo "initial weights ok: $got"
    # Fails (exit 2) on a damaged part, a digest that differs from the pinned one, a file-set or content
    # mismatch, or any path under a 'test' directory.
    "$PY" scripts/cloud_bundle.py verify --upload-dir "$UPLOAD_DIR" --dest . --expect-digest-sha256 "$digest"
    [ ! -e "$DATASET/lists/rdd2022_test.txt" ] && [ ! -e "$DATASET/lists/hrp4k_test.txt" ] || { echo "test lists present"; exit 1; }
    if find "$DATASET" -type d -iname test | grep -q .; then echo "a test directory exists under $DATASET"; exit 1; fi
    echo "train images: $(wc -l < "$DATASET/lists/train.txt")  val images: $(wc -l < "$DATASET/lists/val_pooled.txt")"
    echo "$digest" > logs/data.ok
    df -h . | tail -1
    echo "== elapsed since T0: $(elapsed_min) min"
    echo "== data ok"
}

auto_workers() {
    # Baseline evidence: 31 GB host, batch 64 at 640: 6 workers ran (about 23 GB used), 12 were OOM-killed,
    # i.e. roughly 3.5 GB per worker plus the main process. Same rule here (batch 16 at 1280 buffers
    # about the same bytes per worker). The container's cgroup limit counts, not only MemTotal.
    "$PY" - "$MAX_WORKERS" <<'EOF'
import os, sys
cap = int(sys.argv[1])
mem = 0
for line in open("/proc/meminfo"):
    if line.startswith("MemAvailable:"):
        mem = int(line.split()[1]) * 1024
for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    try:
        value = open(path).read().strip()
        if value.isdigit() and int(value) < (1 << 60):
            mem = min(mem, int(value)) if mem else int(value)
    except OSError:
        pass
gb = mem / 2**30
by_ram = int((gb - 8) // 3.5)
by_cpu = (os.cpu_count() or 4) - 2
print(max(2, min(cap, by_ram, by_cpu)))
EOF
}

guard_run_dir() {
    case "$RUN_DIR" in
        */baseline|*/baseline/*|runs/train/baseline*|model_archive*|*/model_archive/*)
            echo "refusing protected run dir: $RUN_DIR"; exit 1 ;;
    esac
    if [ -e "$RUN_DIR/weights" ] || [ -e "$RUN_DIR/results.csv" ]; then
        echo "refusing: $RUN_DIR already holds training output (use 'resume' to continue it)"; exit 1
    fi
}

launch_detached() {
    # $1 = log file, $2 = command. Survives the SSH session closing; its own process group.
    local log="$1" cmd="$2"
    if [ -f logs/train.pid ] && kill -0 "$(cat logs/train.pid)" 2>/dev/null; then
        echo "training already running (pid $(cat logs/train.pid))"; return 1
    fi
    rm -f logs/train.exit logs/train.finished logs/watchdog.fired
    date -u +%FT%TZ >> logs/train.started
    PYTHONUNBUFFERED=1 nohup setsid bash -c "$cmd; echo \$? > logs/train.exit; date -u +%FT%TZ > logs/train.finished" >> "$log" 2>&1 < /dev/null &
    echo $! > logs/train.pid
}

launch_watchdog() {
    if [ -f logs/watchdog.pid ] && kill -0 "$(cat logs/watchdog.pid)" 2>/dev/null; then
        echo "watchdog already running (pid $(cat logs/watchdog.pid))"; return 0
    fi
    nohup setsid bash "$0" watchdog >> logs/watchdog.log 2>&1 < /dev/null &
    echo $! > logs/watchdog.pid
    echo "== watchdog started, pid $(cat logs/watchdog.pid), limit ${TOTAL_LIMIT_MIN} min after T0 ($(date -u -d "@$(( $(cat logs/instance_t0) + TOTAL_LIMIT_MIN * 60 ))" +%FT%TZ 2>/dev/null || echo "T0+${TOTAL_LIMIT_MIN}min"))"
}

stage_train() {
    ensure_t0
    local batch="${1:-16}" workers="${2:-auto}"
    [ -f logs/data.ok ] || { echo "run the data stage first (logs/data.ok missing)"; exit 1; }
    if [ -f logs/train.pid ] && kill -0 "$(cat logs/train.pid)" 2>/dev/null; then
        echo "training already running (pid $(cat logs/train.pid))"; exit 1
    fi
    guard_run_dir
    if [ "$workers" = "auto" ]; then workers="$(auto_workers)"; fi
    local hours
    hours="$(time_cap_hours "$(elapsed_min)")"
    if "$PY" -c "import sys; sys.exit(0 if float('$hours') >= $MIN_TRAIN_H else 1)"; then :; else
        echo "only $hours h of the budget are left before the ${TOTAL_LIMIT_MIN} min limit (need at least $MIN_TRAIN_H): do not start; destroy the instance"; exit 1
    fi
    arm_selfdestruct
    echo "== batch $batch, workers $workers, time cap $hours h, free RAM: $(free -g | awk '/Mem:/ {print $7}') GB available, $(nproc) vCPUs"
    # Validates the config, the lists and the initial weights, and writes run_config.json; trains nothing.
    "$PY" -m roadsense.training.train --config "$CONFIG" --batch "$batch" --workers "$workers" --time-hours "$hours" --dry-run > logs/dry_run.json
    launch_detached logs/train.log "'$PY' -m roadsense.training.train --config '$CONFIG' --batch $batch --workers $workers --time-hours $hours"
    echo "== training started, pid $(cat logs/train.pid), elapsed since T0: $(elapsed_min) min"
    launch_watchdog
}

time_cap_hours() {
    # $1 = elapsed minutes. min(TIME_CAP_H, (TOTAL_LIMIT_MIN - elapsed - TIME_MARGIN_MIN) / 60), 2 decimals.
    "$PY" -c "print(round(min($TIME_CAP_H, ($TOTAL_LIMIT_MIN - $1 - $TIME_MARGIN_MIN) / 60), 2))"
}

stage_resume() {
    ensure_t0
    [ -f "$RUN_DIR/weights/last.pt" ] || { echo "no $RUN_DIR/weights/last.pt to resume from"; exit 1; }
    local hours="${1:-}"
    if [ -z "$hours" ]; then
        # remaining watchdog time minus the margin for packing, never more than the cap
        hours="$("$PY" -c "print(round(max(0.05, min($TIME_CAP_H, ($TOTAL_LIMIT_MIN - $(elapsed_min) - $TIME_MARGIN_MIN) / 60)), 2))")"
    fi
    echo "== resuming with a $hours h cap"
    launch_detached logs/train.log "'$PY' -m roadsense.training.train --resume '$RUN_DIR/weights/last.pt' --time-hours $hours"
    echo "== resume started, pid $(cat logs/train.pid)"
    launch_watchdog
}

export_checkpoints() {
    "$PY" scripts/export_checkpoints.py "$RUN_DIR" "$PACK_DIR" --quiet >> logs/export.log 2>&1 || true
}

export_new_epochs() {
    # Cheap check first (no Python): only when an epochN.pt has no stripped copy yet is the exporter
    # started (it imports torch), and then at low priority and for epoch files only. Every 30 s the
    # watchdog would otherwise import torch and re-strip last/best on every epoch.
    local f need=0
    for f in "$RUN_DIR"/weights/epoch*.pt; do
        [ -e "$f" ] || continue
        [ -f "$PACK_DIR/weights/$(basename "$f")" ] || need=1
    done
    [ "$need" = 1 ] || return 0
    nice -n 10 "$PY" scripts/export_checkpoints.py "$RUN_DIR" "$PACK_DIR" --only-epochs --quiet >> logs/export.log 2>&1 || true
}

stage_pack() {
    # One packer at a time (the watchdog and an operator may both call it). A stale lock after a
    # killed pack is removed with: rmdir logs/pack.lock
    if ! mkdir logs/pack.lock 2>/dev/null; then echo "another pack is running (stale? rmdir logs/pack.lock)"; return 0; fi
    local rc=0
    pack_work || rc=$?
    rmdir logs/pack.lock 2>/dev/null || true
    return $rc
}

pack_work() {
    [ -f logs/train.exit ] || echo "warning: training has not finished; packing a snapshot"
    mkdir -p "$PACK_DIR"
    rm -f "$PACK_DIR/DONE"
    export_checkpoints
    # run directory without the (large, unstripped) weights, plus logs and the dataset manifest
    local extras=()
    [ -f "$DATASET/manifest.csv" ] && extras+=("$DATASET/manifest.csv")
    # tar exits 1 when a file changes while being read; tolerated.
    tar czf "$PACK_DIR/run_files.tgz" --exclude="$RUN_DIR/weights" --exclude=logs/pack.lock "$RUN_DIR" logs "${extras[@]}" || [ $? -eq 1 ]
    ( cd "$PACK_DIR" && find . -type f ! -name SHA256SUMS ! -name DONE -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
    cat "$PACK_DIR/SHA256SUMS"
    cp "$PACK_DIR/SHA256SUMS" logs/results_sha256.txt
    du -sh "$PACK_DIR"
    date -u +%FT%TZ > "$PACK_DIR/DONE"
    echo "== pack ok ($(elapsed_min) min since T0)"
}

stage_status() {
    ensure_t0
    local exitc="running"
    [ -f logs/train.exit ] && exitc="$(cat logs/train.exit)"
    echo "elapsed since T0: $(elapsed_min) min of ${TOTAL_LIMIT_MIN} (watchdog)  now: $(date -u +%FT%TZ)"
    echo "started: $(tr '\n' ' ' < logs/train.started 2>/dev/null || echo -)  finished: $(cat logs/train.finished 2>/dev/null || echo -)  exit: $exitc"
    if [ -f logs/watchdog.pid ] && kill -0 "$(cat logs/watchdog.pid)" 2>/dev/null; then echo "watchdog: running"; else echo "watchdog: NOT running"; fi
    [ -f logs/watchdog.fired ] && echo "WATCHDOG FIRED: $(cat logs/watchdog.fired)"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    if [ -f "$RUN_DIR/results.csv" ]; then
        echo "epochs logged: $(($(wc -l < "$RUN_DIR/results.csv") - 1))"
        head -1 "$RUN_DIR/results.csv" | cut -d, -f1,2,6-9
        tail -3 "$RUN_DIR/results.csv" | cut -d, -f1,2,6-9
    fi
    echo "checkpoints: $(ls "$RUN_DIR/weights" 2>/dev/null | tr '\n' ' ')"
    echo "exported:    $(ls "$PACK_DIR/weights" 2>/dev/null | tr '\n' ' ')  pack DONE: $(cat "$PACK_DIR/DONE" 2>/dev/null || echo no)"
    grep -a "Reducing to batch\|OutOfMemory\|Killed" logs/train.log || true
    free -g | sed -n 2p
    tail -c 1500 logs/train.log 2>/dev/null | tr '\r' '\n' | tail -4
}

sd_enabled() { [ -f logs/selfdestruct.enabled ]; }

arm_selfdestruct() {
    # SELFDESTRUCT=1 at the train stage arms the backstop; it is recorded in a file so a later
    # resume (a new shell) keeps it. Refuses a deadline earlier than the watchdog's hard limit.
    [ "${SELFDESTRUCT:-0}" = 1 ] || return 0
    if [ "$SELFDESTRUCT_AT_MIN" -lt "$TOTAL_LIMIT_MIN" ]; then
        echo "SELFDESTRUCT_AT_MIN ($SELFDESTRUCT_AT_MIN) must not be below TOTAL_LIMIT_MIN ($TOTAL_LIMIT_MIN)"; exit 1
    fi
    if ! selfdestruct_check > /dev/null 2>&1; then
        echo "SELFDESTRUCT=1 but the check fails (run: selfdestruct --check): not armed"; return 0
    fi
    echo "$SELFDESTRUCT_AT_MIN" > logs/selfdestruct.enabled
    echo "== self-destruct ARMED: this instance will be destroyed at T0 + $SELFDESTRUCT_AT_MIN min (after packing). Disarm: selfdestruct --cancel"
}

selfdestruct_check() {
    # Verifies the pieces only; contacts nothing and destroys nothing. Never prints the key.
    local ok=1
    if [ -n "${CONTAINER_ID:-}" ]; then echo "CONTAINER_ID: set"; else echo "CONTAINER_ID: MISSING"; ok=0; fi
    if [ -n "${CONTAINER_API_KEY:-}" ]; then echo "CONTAINER_API_KEY: set (${#CONTAINER_API_KEY} chars)"; else echo "CONTAINER_API_KEY: MISSING"; ok=0; fi
    if command -v vastai >/dev/null 2>&1; then echo "vastai CLI: $(command -v vastai)"
    elif command -v curl >/dev/null 2>&1; then echo "vastai CLI: missing; REST fallback via curl (DELETE console.vast.ai; undocumented for the instance key)"
    else echo "vastai CLI: missing and no curl"; ok=0; fi
    if [ "$ok" = 1 ]; then echo "selfdestruct check: OK (nothing was destroyed)"; return 0; fi
    echo "selfdestruct check: NOT AVAILABLE"; return 1
}

selfdestruct_now() {
    # Defensive: logs and returns non-zero if anything is missing or the call fails; never aborts the caller.
    { selfdestruct_check > /dev/null 2>&1; } || { log "self-destruct unavailable (CONTAINER_ID/CONTAINER_API_KEY/vastai missing): destroy the instance manually"; return 1; }
    log "SELF-DESTRUCT: destroying instance $CONTAINER_ID ($(elapsed_min) min since T0)"
    if command -v vastai >/dev/null 2>&1; then
        # Documented form (docs.vast.ai, Docker environment): vastai destroy instance $CONTAINER_ID [--api-key $CONTAINER_API_KEY]
        if vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; then log "destroy request accepted"; return 0; fi
        log "vastai destroy failed"
    fi
    if command -v curl >/dev/null 2>&1; then
        # REST fallback; the documented example uses an account key, so this may be refused for the instance key.
        if curl -fsS -X DELETE -H "Authorization: Bearer $CONTAINER_API_KEY" "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/"; then log "REST destroy request accepted"; return 0; fi
        log "REST destroy failed"
    fi
    log "self-destruct FAILED: destroy the instance manually"
    return 1
}

stage_selfdestruct() {
    ensure_t0
    case "${1:-}" in
        --check) selfdestruct_check ;;
        --cancel) rm -f logs/selfdestruct.enabled; echo "self-destruct disarmed" ;;
        --now) [ -f "$PACK_DIR/DONE" ] || echo "warning: pack_out/DONE missing, artifacts may not be packed"; selfdestruct_now ;;
        *) echo "usage: selfdestruct --check | --now | --cancel" >&2; return 2 ;;
    esac
}

wd_after_pack() {
    # With the backstop armed, keep watching until T0 + SELFDESTRUCT_AT_MIN, then destroy. Otherwise just end.
    sd_enabled || exit 0
    local at
    at="$(cat logs/selfdestruct.enabled)"
    log "pack done; self-destruct armed for T0 + ${at} min (now $(elapsed_min))"
    while sd_enabled && [ "$(elapsed_min)" -lt "$at" ]; do sleep "$WATCHDOG_POLL_S"; done
    if sd_enabled; then selfdestruct_now || true; else log "self-destruct disarmed"; fi
    exit 0
}

stage_watchdog() {
    ensure_t0
    log "watchdog up: limit ${TOTAL_LIMIT_MIN} min after T0, poll ${WATCHDOG_POLL_S}s, self-destruct $(sd_enabled && echo "armed at $(cat logs/selfdestruct.enabled) min" || echo off)"
    while true; do
        sleep "$WATCHDOG_POLL_S"
        export_new_epochs  # new epochN.pt -> stripped copies in pack_out/ (downloadable while training runs)
        if [ -f logs/train.exit ]; then
            log "training ended with exit $(cat logs/train.exit): packing"
            stage_pack
            wd_after_pack
        fi
        if [ "$(elapsed_min)" -ge "$TOTAL_LIMIT_MIN" ]; then
            log "TOTAL LIMIT REACHED ($(elapsed_min) min): stopping training"
            echo "$(date -u +%FT%TZ) elapsed $(elapsed_min) min >= ${TOTAL_LIMIT_MIN}" > logs/watchdog.fired
            if [ -f logs/train.pid ]; then
                kill -TERM -- "-$(cat logs/train.pid)" 2>/dev/null || true
                sleep 15
                kill -KILL -- "-$(cat logs/train.pid)" 2>/dev/null || true
            fi
            echo 143 > logs/train.exit
            date -u +%FT%TZ > logs/train.finished
            stage_pack
            wd_after_pack
        fi
    done
}

stage="${1:-}"
case "$stage" in
    setup) stage_setup 2>&1 | tee -a logs/setup.log ;;
    data) stage_data "${2:-}" 2>&1 | tee -a logs/data.log ;;
    train) stage_train "${2:-16}" "${3:-auto}" ;;
    resume) stage_resume "${2:-}" ;;
    status) stage_status ;;
    pack) stage_pack 2>&1 | tee -a logs/pack.log ;;
    watchdog) stage_watchdog ;;
    selfdestruct) stage_selfdestruct "${2:-}" 2>&1 | tee -a logs/selfdestruct.log ;;
    t0) echo "${2:-$(now)}" > logs/instance_t0; echo "T0 = $(cat logs/instance_t0)" ;;
    *) echo "usage: $0 setup|data <digest_sha256>|train [batch] [workers]|resume [hours]|status|pack|t0 [epoch]|selfdestruct --check|--now|--cancel" >&2; exit 2 ;;
esac
