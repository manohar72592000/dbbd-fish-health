#!/usr/bin/env bash
# Runs the main experiments, most important first (so a Colab time-out still leaves
# the core result).  ~1-1.5 h on a Colab T4.  Usage:
#   BDFF=/content/drive/MyDrive/data/BD-Freshwater-Fish \
#   MATSYA=/content/drive/MyDrive/data/MatsyaDx-BD \
#   [DET=/content/drive/MyDrive/yolo_dbbd/best.pt]  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
BDFF=${BDFF:?set BDFF to the BD-Freshwater-Fish folder}
MATSYA=${MATSYA:?set MATSYA to the MatsyaDx-BD folder}
DET=${DET:-}

# 0. software model cost (parameters, MFLOPs) -- no data needed
[ -f model_cost.json ] || python model_cost.py --out model_cost.json

# 1. data (once)
if [ ! -f data_cache/manifest.csv ]; then
  python prepare_data.py --bdff_dir "$BDFF" --matsya_dir "$MATSYA" --out data_cache ${DET:+--detector "$DET"}
fi

# 2. main runs: b=2 is the deployed setting, then the no-compression and stronger-compression ends
for b in 2 1 4; do
  if [ ! -f runs/b$b/metrics.json ]; then
    python train.py --cache data_cache --b $b --out runs/b$b
    python evaluate.py --cache data_cache --run runs/b$b
    python export_tflite.py --cache data_cache --run runs/b$b
  fi
done

# 3. unseen-disease test (leave one disease out), at b=2
for d in "EUS" "Bacterial red disease" "Bacterial gill disease"; do
  n="b2_lodo_${d// /_}"
  if [ ! -f runs/$n/metrics.json ]; then
    python train.py --cache data_cache --b 2 --holdout "$d" --out runs/$n
    python evaluate.py --cache data_cache --run runs/$n
  fi
done

# 4. numbers + figures into the manuscript
python model_cost.py --cache data_cache --out model_cost.json
python make_paper_numbers.py --cache data_cache --runs runs --cost model_cost.json --paper results
echo "Done. Tables and figures are in results/"
