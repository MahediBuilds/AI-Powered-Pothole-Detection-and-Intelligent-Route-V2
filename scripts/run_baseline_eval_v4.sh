#!/usr/bin/env bash
# Baseline evaluation under the pre-declared protocol (see eval_checkpoint.py docstring).
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
CK=model_archive/baseline_yolov8s_rdd2022/weights/best.pt
U=data/processed/unified_v1; K=data/processed/kaggle_probe_v1
R=runs/eval/baseline_yolov8s_rdd2022_v4
# country key for the rdd2022 breakdown taken from the manifest original file names (China_MotorBike kept separate)
mkdir -p $R/aux
$PY - <<'PY'
import csv, re
rows = [("image", "key")]
for r in csv.DictReader(open("data/processed/unified_v1/manifest.csv", newline="")):
    if r["source"] == "rdd2022" and r["status"] == "included":
        rows.append((r["output_path"].split("/")[-1], re.sub(r"_\d+\.\w+$", "", r["original_path"].split("/")[-1])))
csv.writer(open("runs/eval/baseline_yolov8s_rdd2022_v4/aux/rdd2022_country.csv", "w", newline="")).writerows(rows)
PY
for SZ in ${SIZES:-640 1280}; do
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
