#!/usr/bin/env bash
set -euo pipefail

CLAD_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLAD_OUTPUT_DIR="$CLAD_PROJECT_DIR/outputs/clad_stage1"
: "${LIBERO_DATASET_DIR:?Set LIBERO_DATASET_DIR to the absolute libero_10 directory}"

mkdir -p "$CLAD_OUTPUT_DIR"
cd "$CLAD_PROJECT_DIR"

echo "Console log: $CLAD_OUTPUT_DIR/train_console.log"
python scripts/train_clad_stage1.py \
  --data-config configs/data/libero_long.yaml \
  --model-config configs/model/clad_stage1.yaml \
  --train-config configs/train/stage1.yaml \
  --dataset-dir "$LIBERO_DATASET_DIR" \
  --cache-dir .cache/decisionnce/libero_long \
  --output-dir "$CLAD_OUTPUT_DIR" \
  --device cuda \
  "$@" 2>&1 | tee -a "$CLAD_OUTPUT_DIR/train_console.log"
