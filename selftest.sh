#!/usr/bin/env bash
# 5-minute end-to-end check of the kit on FAKE data (no download, no GPU, no ImageNet weights).
# The numbers it prints are meaningless -- it only proves every script runs.
set -euo pipefail
cd "$(dirname "$0")"
T=${TMPDIR:-/tmp}/suras_selftest
rm -rf "$T"; mkdir -p "$T"
python dbbd.py
python make_synthetic_data.py --out "$T/fake" --n 24
python prepare_data.py --bdff_dir "$T/fake/BD-Freshwater-Fish" --matsya_dir "$T/fake/MatsyaDx-BD" --out "$T/cache" > /dev/null
for b in 1 2 4; do
  python train.py --cache "$T/cache" --b $b --out "$T/runs/b$b" --epochs1 4 --epochs2 2 --lr2 1e-4 \
                  --backbone tiny --batch 16 > /dev/null 2>&1
  python evaluate.py --cache "$T/cache" --run "$T/runs/b$b" 2>/dev/null | grep '^\['
  python export_tflite.py --cache "$T/cache" --run "$T/runs/b$b" --n_rep 20 > /dev/null 2>&1
done
python train.py --cache "$T/cache" --b 2 --holdout EUS --out "$T/runs/b2_lodo_EUS" --epochs1 4 --epochs2 2 \
                --lr2 1e-4 --backbone tiny --batch 16 > /dev/null 2>&1
python evaluate.py --cache "$T/cache" --run "$T/runs/b2_lodo_EUS" 2>/dev/null | grep -E '^\[|unseen'
mkdir -p "$T/paper"
python model_cost.py --cache "$T/cache" --out "$T/model_cost.json" 2>/dev/null | grep '^b='
python make_paper_numbers.py --cache "$T/cache" --runs "$T/runs" --cost "$T/model_cost.json" --paper "$T/paper"
echo "SELF-TEST PASSED (outputs in $T -- do not use these numbers)"
