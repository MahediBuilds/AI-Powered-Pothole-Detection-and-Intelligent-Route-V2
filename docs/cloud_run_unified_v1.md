# Cloud run runbook: unified_v1, YOLOv8s, imgsz 1280

One run, absolute total budget $3, displayed rate never above $0.70/h including disk. Everything below was prepared and smoke-tested locally; nothing here rents or contacts a cloud instance. Renting needs the owner's explicit approval and a live price check at the chosen disk size (owner's standing rules). Nothing on the instance stops charges by itself unless the optional self-destroy backstop (section 6) is armed: you must destroy the instance.

Files: `configs/train_unified_v1.yaml`, `src/roadsense/training/train.py`, `scripts/vast_remote_unified.sh` (on the instance), `scripts/cloud_bundle.py` (build, upload, verify), `scripts/export_checkpoints.py`, `scripts/select_checkpoint.py` (local, after the run), bundle in `data/cloud_upload/unified_v1/`.

## 1. Experiment as configured

| Item | Value |
|---|---|
| Model | YOLOv8s, 1 class, initialised from a copy of `model_archive/.../best.pt` (`pretrained/baseline_yolov8s_rdd2022_best.pt`, sha256 `2aabdd13...f0d798`, checked on the instance) |
| Data | `unified_v1`: train 12,193 images / 9,726 boxes (RDD2022 8,220 + HRP4K 3,973), pooled val 3,609 images / 1,399 boxes (RDD2022 2,784 + HRP4K 825). Test is never in the bundle |
| imgsz / seed | 1280 / 42, `deterministic: true` (kept: reproducibility is a project requirement, and the baseline timing used for the estimates was measured with it on, so the estimate already includes its cost) |
| Batch | 16 (x4 accumulation = nbs 64, same effective batch as the baseline). Memory estimate about 14 GB: local RTX 4060 used 3.55 GB at batch 4 (about 0.89 GB/image), the baseline used 12.4 GB at batch 64 and 640 px (4x pixels = about 0.78 GB/image) |
| Optimiser | AdamW, lr0 0.001, lrf 0.01, cosine, momentum 0.9, wd 5e-4, warmup 2 epochs, warmup_bias_lr 0 |
| Why not `auto` | `auto` picks by iteration count: the baseline (100 epochs x 8,220 images) got MuSGD lr 0.01; this run is about 9,400 iterations, which would flip it to AdamW lr 0.002 silently. Pinned instead. lr0 is half of that default because the weights are already trained for the task; AdamW is insensitive to the gradient-scale change from the new resolution and image mix. A judgement, not tested locally |
| Augmentation | identical to `train_baseline.yaml` (mosaic 1, scale 0.5, fliplr 0.5, hsv, close_mosaic 10) |
| Epochs / patience | 50 / 50. Patience equals epochs, so early stopping is effectively off: Ultralytics 8.4.210 stops on pooled mAP50-95 (its only fitness term), which is not our selection metric and could end the run before the close_mosaic phase. Selection is local, from the saved checkpoints |
| Checkpoints | `save_period: 5` writes when the 0-based epoch index is divisible by 5: `epoch0.pt` (epoch 1), `epoch5.pt` (epoch 6) ... `epoch45.pt` (epoch 46), 10 files, plus `best.pt` and `last.pt` (12 in all). With a shortened schedule the last multiple of 5 below the final epoch is the last `epochN.pt` |
| Time cap | Ultralytics `time` mode, passed by the remote `train` stage as `--time-hours = min(2.5, (200 - elapsed_minutes - 12) / 60)`. In time mode the trainer sets `epochs = ceil(time / mean epoch time)` and rebuilds the cosine schedule after every epoch. `train.py` makes that shrink-only (`min(50, ...)`): a late start gives a coherent shorter schedule that still anneals and reaches close_mosaic, while plenty of time does not stretch 50 epochs. Refused when under 0.5 h is left. Verified in the local smoke test (cap stops after the epoch in flight and keeps checkpoints); the shrink behaviour itself is covered by unit tests with a fake trainer, not by a real shortened run |
| Resume | `--resume` keeps the saved epoch count and schedule and uses its own wall-clock cap (Ultralytics' time mode would assign the remaining-time epoch count to the TOTAL epochs on resume). Run once locally: killed mid-run, resumed to the saved 4 epochs |
| Watchdog | independent, in the remote script: at T0 + 200 min it kills the training process group and packs. T0 is the first stage call (or `T0_EPOCH`). It starts Python only when a new `epochN.pt` appears (stripped copies for download), not every poll |
| Selection | `scripts/select_checkpoint.py`, pre-declared rule in its docstring (best.pt is placed at the LAST epoch tied on mAP50-95, the Ultralytics 8.4.210 behaviour) |

## 2. Measurements behind the plan

**Upload from this machine** (to `speed.cloudflare.com/__up`, nothing of the project sent):

| Streams | Aggregate |
|---|---|
| 1 (25 MB) | 0.46 MB/s (3.7 Mbit/s) |
| 1 (100 MB) | 0.57 MB/s |
| 4 x 10 MB | 1.84 MB/s |
| 8 x 10 MB | 2.34 MB/s |
| 16 x 10 MB | 3.99 MB/s |

Download (1 stream, 25 MB): 0.63 MB/s. The Wi-Fi link reports 867 Mbit/s, so the limit is the path or ISP, and several parallel streams scale it. The short test transfers include TCP ramp-up; the real rate to a Vast host is unmeasured.

**Upload tool tested against a real sshd** (an unprivileged OpenSSH 9.6 server in WSL on port 2222, default `MaxStartups 10:30:100`, no system configuration changed; the real 2.24 GB bundle, 16 streams): all 21 files arrived, the remote `sha256sum -c parts.sha256` matched, no connection was dropped. With sshd tightened to `MaxStartups 1:100:2` and no stagger, 5 connections were dropped; the retry-with-backoff recovered all of them and the remote check still matched. Speed on loopback (225 MB/s) says nothing about the internet. Not tested: a Vast host, a real internet path, ssh key and host-key handling on Vast.

**Bundle**: 2.22 GB of train+val images and labels (17 chunks of 128 MB) + 22.5 MB weights + 0.2 MB code (total 2.24 GB). Built in 50 s. Full verify (every part's sha256, the pinned digest, every file's sha256) and unpack: 26 s in Linux (WSL ext4) and 98 s on this Windows laptop.

| Upload rate | Time for 2.24 GB |
|---|---|
| 4.0 MB/s (16 streams) | 9.4 min |
| 2.3 MB/s | 16 min |
| 1.0 MB/s | 37 min |
| 0.5 MB/s (single stream) | 75 min |

**Decision: direct upload from this machine, in 16 parallel streams of 128 MB chunks (connection starts staggered 0.5 s apart, 5 attempts per part with 2/4/8/16 s backoff, resumable, remote sha256 check at the end), started the moment the instance accepts SSH, while `setup` runs** (setup needs only the 0.2 MB code, sent first). One TCP stream (0.5 MB/s) would cost about 75 min = $0.65; parallel streams cut it to an expected 10-20 min.

Alternatives considered:
- *Pre-stage the bundle on a free private store (for example a private Hugging Face dataset), instance pulls at 500+ Mbit/s in about 1 min.* Cheapest in paid time (saves the whole upload, about $0.1-0.3). Needs an account and the owner's approval to put the dataset on a third party. If approved, replace step 5.1's big upload by `curl -L -C - -o ... URL` of the same 17 chunks (same `data` stage and digest check).
- *Rebuild on the instance from public sources* (RDD2022 fetch took under 4 min in the baseline rental; HRP4K is a 12.8 GB Zenodo zip plus resize). Not implemented: it needs the builder to be deterministic across platforms, and the review-corrected labels would still have to be uploaded.

**Dataloader (this laptop, Ryzen 9 8945HS, imgsz 1280, mosaic on, HRP4K-style 1280x720 JPEGs)**: 23.4 img/s per worker (single process; 20.2 for native-size RDD2022), 85 img/s with 4 workers, 82 with 8 (the laptop saturated; shared with other jobs). A GPU-bound 5090 needs about 130-170 img/s, so 7-8 fast workers. A Vast vCPU may be slower than this laptop core; treat these as upper bounds.

**Local smoke on the RTX 4060**: init from baseline weights `Transferred 355/355`; steady training about 14-21 img/s at batch 4; validation of the 3,609 pooled val images at imgsz 1280 in 70 s (52 img/s, 13.7 ms/image inference). Checkpoints are 112 MB unstripped, 22.7 MB stripped.

**Linux pre-flight (WSL, free)**: the real bundle unpacked by the real `data` stage (digest verified, 31,610 files), `train.py --dry-run` with the real config, then the Ultralytics 8.4.210 train and val datasets built on CPU from the resolved lists and every label scanned: train 12,193 images / 9,726 boxes / 6,880 empty-label images, val 3,609 images / 1,399 boxes / 2,775 empty, 0 corrupt; equal to the local build. Scan time 20 s.

## 3. Time and cost budget

Per-epoch estimate on the RTX 5090 (train 12,193 images + validation 3,609):
- from the baseline timing (76 epochs in 1,212 s = 15.9 s/epoch for 8,220 + 2,784 images, deterministic, 6 workers, GPU 78-88% busy): train 12.7 s x (12,193/8,220) x 4 (pixels) = 75 s, validation 3.2 s x 1.3 x 4 = 17 s, total about 92 s;
- from the local run: 8x GPU speed-up assumed for the 5090 over the 4060 laptop: train about 90 s, validation about 9 s, total about 99 s;
- dataloader bound with 6 workers (about 21 img/s each, 126 img/s): train 97 s + validation about 20 s = about 117 s. With 10+ workers the GPU bound (about 92 s) applies.
Estimate: 92-117 s/epoch, 50 epochs = 77-98 min.

| Phase | Optimistic | Expected | Pessimistic |
|---|---|---|---|
| Rent to SSH ready, send code | 3 | 4 | 6 |
| Setup (runs during the upload) | 4 | 5 | 8 |
| Data upload (2.24 GB; starts at minute 2) | 9 | 16 | 37 |
| Verify + unpack (Linux, 2.2 GB) | 0.5 | 1 | 3 |
| Train start (dry run, label scan 20 s, model load) | 1 | 1.5 | 2 |
| Training 50 epochs | 77 | 98 | 125 |
| Pack + download of the tail (best, last, epoch45, run files, about 70 MB) | 3 | 5 | 8 |
| **Elapsed total, min** | **~97** | **~128** | **~190** |
| Cost at $0.557/h (24 vCPU, 63 GB host, listed 2026-11-11) | $0.90 | $1.19 | $1.76 |
| Cost at the $0.70 ceiling | $1.13 | $1.49 | $2.22 |

(Upload overlaps setup, so the elapsed total is rent + max(setup, upload) + verify + start + training + tail.) Add storage and bandwidth, which are NOT in the displayed hourly rate unless the offer card says so: confirm on the offer card the storage $/h at the chosen 40 GB (the 2026-11-11 shortlist had $0.333 per GB per month, about $0.019/h for 40 GB, included in `dph_total` there) and the bandwidth $/TB (that list showed $5.2/TB each way: 2.2 GB up plus about 0.4 GB down is about $0.014; a host at $20/TB would be about $0.05). Do not create a separate volume. Expected total spend about $1.2-1.3; the hard stops below keep the worst case under $2.7 even at $0.70/h.

Stored checkpoints are stripped on the instance (22.7 MB instead of 112 MB, so about 270 MB instead of 1.2 GB to download at 0.6 MB/s) and fetched while training runs, once each, so the final tail is small.

## 4. Instance requirements

| Need | Value | Why |
|---|---|---|
| GPU | RTX 5090 32 GB (any 24 GB+ Ada/Blackwell works) | batch 16 at 1280 needs about 14 GB; the earlier run used the 5090 with the "PyTorch (Vast)" CUDA 13 template, pins unchanged |
| vCPUs | 24 or more preferred, 16 minimum | 7-10 dataloader workers at about 21 img/s each to keep the GPU fed, plus the main process; with 16 threads and 6 workers the epoch is about 25% slower |
| RAM | 48 GB or more preferred, 31 GB minimum | the baseline measured about 3.5 GB per worker plus overhead: on 31 GB, 6 workers ran and 12 were OOM-killed (batch 64, 640 px). Batch 16 at 1280 buffers about the same bytes per worker. The remote script picks `workers = min(12, vCPUs-2, (available RAM or cgroup limit - 8 GB) / 3.5)`: 6 on a 31 GB host, 10+ on 63 GB. The per-worker figure is carried over from the baseline, not measured at 1280 |
| Disk | 40 GB | bundle parts 2.3 GB (kept) + unpacked data 2.3 GB + torch environment and caches about 15 GB + run about 2 GB; baseline used 32 GB. Priced into the displayed rate; do not go larger |
| Network | 500 Mbit/s or more down; inbound SSH from this machine; direct port | upload speed is the main variable cost |
| Reliability | 0.98 or more, on-demand, verified | as the earlier shortlist |
| Price | at most $0.70/h displayed, disk included | owner's limit |

From the 2026-11-11 shortlist (stale, recheck live): $0.508 (16 threads, 30 GB, 6 workers, about 117 s/epoch, about $1.1 total), $0.557 (24 threads, 63 GB, about 92 s/epoch, about $1.0 total), $0.609 (48 threads, 142 GB). The larger-RAM, more-CPU hosts are not more expensive per run; prefer the 63 GB / 24-thread class if its price is at or below about $0.60.

## 5. Command sequence

Local shell is Git Bash in `New Implementation`. Set once (the first line matters: without it Git Bash rewrites `/workspace/...` into `C:/Program Files/Git/workspace/...` and the upload fails; `cloud_bundle.py upload` refuses such a path):

```bash
export MSYS_NO_PATHCONV=1
KEY=~/.ssh/vast_id_rsa; H=<ip>; P=<port>
KH=<scratch>/known_hosts     # known-hosts file in the scratchpad
SSH="ssh -i $KEY -p $P -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KH root@$H"
SCP="scp -i $KEY -P $P -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KH"
R=/workspace/roadsense
UP="python scripts/cloud_bundle.py upload --out data/cloud_upload/unified_v1 --host $H --port $P --key $KEY --known-hosts $KH --remote-dir $R/upload"
```

### 5.0 Before renting (free, local)

```bash
.venv/Scripts/python.exe -m pytest tests/test_training_unified.py tests/test_training_config.py tests/test_cloud_bundle.py tests/test_select_checkpoint.py tests/test_remote_script.py tests/test_export_checkpoints.py -q
# Rebuild the data part only if data/processed/unified_v1 changed since the bundle was built:
rm -rf data/cloud_upload/unified_v1
.venv/Scripts/python.exe scripts/cloud_bundle.py build --dataset data/processed/unified_v1 --out data/cloud_upload/unified_v1 --source-digest data/processed/unified_v1/digest.json
# Always (re)build the code part right before renting; it normalises line endings to LF:
.venv/Scripts/python.exe scripts/cloud_bundle.py code --out data/cloud_upload/unified_v1
```
Write down the printed `PIN THIS: --expect-digest-sha256 <PIN>`. On the offer card note the hourly rate, storage $/h and bandwidth $/TB (section 3).

### 5.1 Instance reachable (minute 0 = billing start)

Set two local alarms now: **T0 + 185 min** (training should be done or nearly; start downloading) and **T0 + 215 min** (the instance must be gone; if not, destroy it).

```bash
T0=$(date +%s)                                   # or the rental start time, if known
$SSH "mkdir -p $R/upload && nvidia-smi -L && nproc && free -g | sed -n 2p && df -h $R | tail -1"
$UP --only code.tgz                              # 0.2 MB, seconds
$SSH "cd $R && tar xzf upload/code.tgz && bash scripts/vast_remote_unified.sh t0 $T0"
# big upload in the background (code.tgz is skipped: already complete), 16 staggered streams:
nohup $UP --streams 16 > upload.log 2>&1 &
# setup meanwhile (about 4-5 min):
$SSH "cd $R && nohup bash scripts/vast_remote_unified.sh setup > /dev/null 2>&1; tail -3 logs/setup.log"
# after setup: is the optional self-destroy backstop usable on this host? (checks variables and CLI only; destroys nothing)
$SSH "cd $R && bash scripts/vast_remote_unified.sh selfdestruct --check"
```
Check the setup result before going on: `setup ok` and torch/ultralytics versions in `logs/setup.log` equal the pins (torch 2.14.1+cu130, pillow 12.3.0, numpy 2.2.6, ultralytics 8.4.210).

### 5.2 Upload monitor (every 2-3 min)

`tail -2 upload.log` prints `done/total MB`, rate and ETA. See the abort rules. If the rate is below 2 MB/s after 3 min, restart with more streams (`--streams 24`; finished files are skipped). When the command ends it prints `remote sha256 check: all parts match` and exits 0; on exit 1 it lists the failed files: re-run the same command.

### 5.3 Verify and start training

```bash
$SSH "cd $R && bash scripts/vast_remote_unified.sh data <PIN>"     # fails loudly on any mismatch
$SSH "cd $R && SELFDESTRUCT=1 bash scripts/vast_remote_unified.sh train 16 auto"   # omit SELFDESTRUCT=1 if you do not want the backstop
$SSH "cd $R && bash scripts/vast_remote_unified.sh status"
```
`data` checks: initial-weights sha256; every part's sha256; the digest file against `<PIN>`; the unpacked file set and every file's sha256 against the digest; no `test` path anywhere. `train` prints the time cap it computed, refuses an existing run directory with output, a protected run directory, a second concurrent start, and a start with under 0.5 h of budget left.

First checks after about 5 min: `status` shows `epochs logged`, GPU utilisation above 75%, no `Reducing to batch` / `Killed`. Note the first epochs' time.

### 5.4 While training: status and progressive download

```bash
$SSH "cd $R && bash scripts/vast_remote_unified.sh status"        # every 5-10 min
mkdir -p runs/cloud/unified_v1/weights
# every 10 min: fetch each finished stripped epoch checkpoint exactly once (the index lists only completed files)
for f in $($SSH "python3 -c \"import json;print(' '.join(k for k in json.load(open('$R/pack_out/weights_index.json')) if k.startswith('epoch')))\""); do
  [ -f runs/cloud/unified_v1/weights/$f ] || $SCP root@$H:$R/pack_out/weights/$f runs/cloud/unified_v1/weights/
done
```
Stripped `epochN.pt` copies never change after the watchdog writes them (within about 30 s of the training checkpoint), so each is downloaded once. `best.pt` and `last.pt` change every epoch and are fetched only in 5.5.

### 5.5 Finish

Training ends by itself (exit file) or at the cap; the watchdog then runs `pack` automatically. Wait for `pack DONE` in `status` (or `pack_out/DONE`), then fetch only what is still missing (do not copy the whole `pack_out` again):

```bash
D=runs/cloud/unified_v1; mkdir -p $D/pack
# any epoch checkpoint the loop above has not fetched yet (the final one, epoch45.pt or the last multiple of 5):
for f in $($SSH "ls $R/pack_out/weights"); do [ -f $D/weights/$f ] || $SCP root@$H:$R/pack_out/weights/$f $D/weights/; done   # now also best.pt and last.pt
$SCP root@$H:$R/pack_out/SHA256SUMS root@$H:$R/pack_out/weights_index.json root@$H:$R/pack_out/run_files.tgz $D/pack/
cp $D/pack/SHA256SUMS $D/SHA256SUMS && cd $D && sed -i 's#  \./weights/#  weights/#' SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS     # every file present must say OK; best.pt, last.pt and all epoch*.pt must be among them
cd pack && sha256sum run_files.tgz | cut -c1-16 && cd ../..
```
`SHA256SUMS` lists `./weights/...`, `./weights_index.json` and `./run_files.tgz`; verify `run_files.tgz` and `weights_index.json` from `pack/` the same way if you prefer (`cd $D/pack && sha256sum -c --ignore-missing ../SHA256SUMS` after copying the two files next to a `weights` symlink or copy). `run_files.tgz` holds the run directory (results.csv, args.yaml, run_config.json, plots) and all logs; extract it next to the weights. Do not destroy the instance before every checksum is OK and `results.csv`, `args.yaml`, `run_config.json` are present. With the backstop armed, disarm it while you download: `$SSH "cd $R && bash scripts/vast_remote_unified.sh selfdestruct --cancel"`.

### 5.6 Destroy and verify

Checklist (all items, in this order):
1. All needed files copied and `sha256sum -c` all OK; every `epoch*.pt`, `best.pt`, `last.pt` and `results.csv` present locally.
2. `upload.log` shows no failed file; nothing else on the instance is needed (no unique data lives there).
3. Destroy the instance in the Vast UI (not stop: a stopped instance keeps billing storage).
4. Confirm in the UI that the instance list is empty and no storage is listed for it; open the billing page and RECORD what it shows (balance, current charges, auto-recharge on or off) without changing anything: the balance and the auto-recharge setting are the owner's, not ours to change.
5. Note the end time; compute elapsed minutes x displayed rate (+ storage and bandwidth) and record it with the run.
6. Locally: no leftover `ssh`/`scp` or `cloud_bundle.py upload` processes; stop the T0 + 185 and T0 + 215 alarms.

### 5.7 Local selection (after the instance is gone)

```bash
.venv/Scripts/python.exe scripts/select_checkpoint.py --weights-dir runs/cloud/unified_v1/weights \
    --results-csv <extracted run_files.tgz>/runs/train/unified_v1_y8s_1280/results.csv \
    --out runs/select/unified_v1 --batch 8
```
Evaluates every distinct checkpoint at imgsz 1280 on RDD2022 val and HRP4K val separately (about 12 checkpoints x 2 sources; identical weights evaluated once; on the 4060 a pooled-val pass is about 70 s, so roughly 20-30 min in total). The rule and tie-break are in the script docstring; `selection.json` records the choice. The script refuses a test yaml, and any list that has an image path with a `test` directory component. Test evaluation is a separate later step with `eval_checkpoint.py --mode apply`.

## 6. Billing safety: self-destroy backstop (optional) and alarms

Nothing on the instance stops charges by itself. Three layers: the watchdog (stops training and packs at T0 + 200 min), local alarms (T0 + 185, T0 + 215), and the optional self-destroy.

**Self-destroy.** Per the Vast.ai documentation (Docker environment page: "You can also stop or destroy the instance from within": `vastai destroy instance $CONTAINER_ID`, with `--api-key $CONTAINER_API_KEY` if the key is not configured automatically; the API reference describes `CONTAINER_API_KEY` as a restricted key that can only start, stop or destroy that one instance), the script runs `vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"`. If the `vastai` CLI is missing it tries a REST `DELETE https://console.vast.ai/api/v0/instances/$CONTAINER_ID/` with the container key as bearer token; the docs show that call only with an account key, so it may be refused. Everything is defensive: a missing variable or CLI, or a failed call, is logged to `logs/selfdestruct.log` and the script continues. Never tested against Vast (it would destroy a real instance): the check and the destroy command line were tested on Linux with a fake `vastai`.

- `selfdestruct --check` verifies `CONTAINER_ID`, `CONTAINER_API_KEY` and the CLI (or curl) are present and prints nothing secret; destroys nothing, contacts nothing. Run it right after `setup`.
- Arming: `SELFDESTRUCT=1 bash scripts/vast_remote_unified.sh train ...` (only armed if the check passes; refused if `SELFDESTRUCT_AT_MIN` is below `TOTAL_LIMIT_MIN`). The watchdog then keeps running after the pack and destroys the instance at T0 + `SELFDESTRUCT_AT_MIN` (default 205) minutes, never earlier, only after packing (including the normal finish: it waits for the deadline, it does not fire at the end of training).
- Whatever has not been downloaded by T0 + 205 min is lost with the instance. While downloading late, disarm: `selfdestruct --cancel`.
- `selfdestruct --now` destroys immediately (the operator's last resort from the instance shell). A plain `selfdestruct` does nothing.

**Alarms**: T0 + 185 min: training should be finished or about to be cut at 200; start the downloads and decide whether to disarm. T0 + 215 min: confirm the instance is gone in the Vast UI, and destroy it by hand if not.

## 7. Abort rules (elapsed minutes since billing start; rate x elapsed must stay below $2.5)

| When | Condition | Action |
|---|---|---|
| min 12 | upload below 1.5 MB/s after the restart with 24 streams (projected over 40 min) | stop the upload; decide: continue only if projected `data` completion is before minute 50, otherwise destroy |
| min 50 | `data` stage not yet OK | destroy (sunk cost under $0.55) |
| min 60 | training not started | destroy (the `train` stage already refuses to start with under 0.5 h left) |
| first 5 epochs | mean epoch time above 190 s (50 epochs would not finish before minute 190) or GPU below 50% with CPU saturated | fix workers only if the cause is clear (a restart costs the epochs done); otherwise let the time mode shorten the schedule (it does so coherently) and accept fewer epochs, or destroy if before epoch 3 |
| any time | training exits non-zero or `status` shows `Killed`/CUDA error | read `logs/train.log`; one `resume` attempt if `last.pt` exists (`bash scripts/vast_remote_unified.sh resume`, it computes the remaining cap); a second failure: pack, download, destroy |
| min 185 | alarm | training still running: start downloading everything available; the watchdog fires at min 200 and packs by about 202 |
| min 205 | self-destroy deadline (if armed) | everything must be downloaded by now, or cancelled with `selfdestruct --cancel` |
| **min 215** | alarm, always | **instance destroyed**, regardless of state (215 min x $0.70 = $2.51 plus disk, under $3) |

## 8. Smoke-test coverage and what is not covered

Tested locally (RTX 4060, Windows): initialisation from the baseline weights; explicit AdamW and `save_period`; `time=` hard cap stopping after one epoch with checkpoints kept; output-directory guard (existing run directory and `runs/train/baseline` refused); kill and resume from `last.pt`; the real `unified_v1` dataset with a 1-epoch 2% run and the full pooled val; the bundle (build, 17-chunk split, pinned digest, damaged part, tampering, test-path refusal, full verify and unpack of the real bundle); `select_checkpoint.py` end to end against `eval_checkpoint.py` on the stand-in dataset; the upload tool against a real sshd (16 streams, default and tight `MaxStartups`); `vast_remote_unified.sh` stages data (on the real bundle), train (time cap computed from elapsed time, refusal when late), status, pack, the watchdog (normal end, forced kill at the limit), self-destroy check/now/cancel and the timing of the armed destroy, on Ubuntu (WSL) with a fake trainer and a fake `vastai`; the real `train.py --dry-run` and an Ultralytics dataset scan of the unpacked real bundle on Linux (counts equal the local build).

Not covered: `setup` (pip installs and apt; the same lines as the baseline's) and anything on a real Vast host: real upload rate, real vCPU speed, real per-worker RAM at 1280, the 5090 epoch time, the real `vastai destroy` call, whether `CONTAINER_ID`/`CONTAINER_API_KEY` exist on the chosen template. A real shortened schedule (the shrink-only callback is unit-tested with a fake trainer). The watchdog's group kill was tested with a fake trainer, not with Ultralytics' dataloader workers.

## 9. Risks

- Upload rate to the instance is unmeasured; the plan assumes 2-4 MB/s with 16 streams. Below 1 MB/s it becomes the dominant cost.
- Epoch time is an estimate (92-117 s); a slowdown beyond the cap shortens the schedule coherently but gives fewer epochs.
- AdamW lr0 0.001 is untested. Starting from the baseline weights, the first epochs may dip in mAP before recovering (the 2% smoke run gave pooled-val mAP50 0.155 after one epoch of 244 images; that number says nothing about the full run).
- Worker RAM is extrapolated from the baseline; the script's rule is conservative, and a worker OOM kill would cost a resume.
- If `data/processed/unified_v1` changes after the bundle was built, rebuild the data part and use the new pin.
