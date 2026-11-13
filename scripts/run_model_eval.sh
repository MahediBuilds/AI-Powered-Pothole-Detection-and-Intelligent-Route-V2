#!/usr/bin/env bash
# Evaluate ANY single-class checkpoint under the same pre-declared protocol as run_baseline_eval_v4.sh.
#   scripts/run_model_eval.sh CHECKPOINT TAG "SIZE [SIZE ...]"
# Output root: runs/eval/<TAG>/<size>/...  (then: python scripts/summarize_model_eval.py runs/eval/<TAG> --tag TAG)
# Fixed: batch 8, conf 0.001, NMS IoU 0.7 (eval_checkpoint.py defaults), seed 0, group files below.
set -e
cd "$(dirname "$0")/.."
[ $# -ge 2 ] || { echo "usage: $0 CHECKPOINT TAG [\"SIZES\"]  (default sizes: 640 1280)" >&2; exit 2; }
CK=$1; TAG=$2; SIZES_ARG=${3:-"640 1280"}
PY=${PY:-.venv/Scripts/python.exe}
[ -f "$CK" ] || { echo "checkpoint not found: $CK" >&2; exit 2; }
case "$TAG" in runs*|*/*|*\*|"") echo "TAG must be a plain name" >&2; exit 2;; esac
U=data/processed/unified_v1; K=data/processed/kaggle_probe_v1
R=runs/eval/$TAG
mkdir -p $R/aux
# country key for the rdd2022 breakdown from the manifest original file names (China_MotorBike kept separate)
R=$R $PY - <<'PY'
import csv, os, re
R = os.environ["R"]
rows = [("image", "key")]
for r in csv.DictReader(open("data/processed/unified_v1/manifest.csv", newline="")):
    if r["source"] == "rdd2022" and r["status"] == "included":
        rows.append((r["output_path"].split("/")[-1], re.sub(r"_\d+\.\w+$", "", r["original_path"].split("/")[-1])))
csv.writer(open(R + "/aux/rdd2022_country.csv", "w", newline="")).writerows(rows)
PY
for SZ in $SIZES_ARG; do
  O=$R/$SZ; B=${BATCH:-8}
  C="$PY scripts/eval_checkpoint.py --checkpoint $CK --imgsz $SZ --batch $B"
  $C --mode select --data $U/eval/pooled_val.yaml --split val --groups $U/groups.csv --out $O/pooled_val
  OP=$O/pooled_val/operating_points.json
  for S in rdd2022 hrp4k; do
    $C --mode select --data $U/eval/${S}_val.yaml --split val --groups $U/groups_$S.csv --out $O/${S}_val
    # P/R/FP at the POOLED threshold on the per-source val: in-sample by construction, flagged
    $C --mode apply --allow-in-sample --operating-points $OP --data $U/eval/${S}_val.yaml --split val \
       --groups $U/groups_$S.csv --out $O/${S}_val_at_pooled
  done
  $C --mode apply --operating-points $OP --data $U/eval/rdd2022_test.yaml --split test --groups $U/groups_rdd2022.csv \
     --breakdown-csv $R/aux/rdd2022_country.csv --out $O/rdd2022_test
  $C --mode apply --operating-points $OP --data $U/eval/hrp4k_test.yaml --split test --groups $U/groups_hrp4k.csv --out $O/hrp4k_test
  $C --mode apply --operating-points $OP --data $K/eval/probe.yaml --split test --groups $K/groups_v2.csv \
     --manhole-boxes $K/manhole_boxes.csv --out $O/kaggle_probe
done
