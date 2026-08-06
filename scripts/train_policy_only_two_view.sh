#!/usr/bin/env bash
set -euo pipefail

CLAD_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLAD_OUTPUT_DIR="${CLAD_POLICY_ONLY_TWO_VIEW_OUTPUT_DIR:-$CLAD_PROJECT_DIR/outputs/policy_only_two_view}"
CLAD_CACHE_DIR="${CLAD_TWO_VIEW_CACHE_DIR:-$CLAD_PROJECT_DIR/.cache/decisionnce/libero_long_two_view}"
: "${LIBERO_DATASET_DIR:?Set LIBERO_DATASET_DIR to the absolute libero_10 directory}"

mkdir -p "$CLAD_OUTPUT_DIR"
cd "$CLAD_PROJECT_DIR"

echo "Console log: $CLAD_OUTPUT_DIR/train_console.log"
python scripts/train_clad_stage2.py \
  --policy-variant policy_only \
  --data-config configs/data/libero_long_two_view.yaml \
  --model-config configs/model/policy_only_two_view.yaml \
  --train-config configs/train/stage2.yaml \
  --dataset-dir "$LIBERO_DATASET_DIR" \
  --cache-dir "$CLAD_CACHE_DIR" \
  --output-dir "$CLAD_OUTPUT_DIR" \
  --device cuda \
  "$@" 2>&1 | tee -a "$CLAD_OUTPUT_DIR/train_console.log"
