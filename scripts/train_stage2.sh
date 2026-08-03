#!/usr/bin/env bash
set -euo pipefail

CLAD_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLAD_OUTPUT_DIR="$CLAD_PROJECT_DIR/outputs/clad_stage2"
CLAD_FORESIGHT_CHECKPOINT="$CLAD_PROJECT_DIR/outputs/clad_stage1/stage1_foresight.pt"

if [[ ! -f "$CLAD_FORESIGHT_CHECKPOINT" ]]; then
  echo "Missing frozen foresight checkpoint: $CLAD_FORESIGHT_CHECKPOINT" >&2
  echo "Run scripts/export_stage1_foresight.py first." >&2
  exit 1
fi

mkdir -p "$CLAD_OUTPUT_DIR"
cd "$CLAD_PROJECT_DIR"

echo "Console log: $CLAD_OUTPUT_DIR/train_console.log"
python scripts/train_clad_stage2.py \
  --data-config configs/data/libero_long.yaml \
  --model-config configs/model/clad_stage2.yaml \
  --train-config configs/train/stage2.yaml \
  --dataset-dir /data/jack/libero_datasets/libero_10 \
  --cache-dir .cache/decisionnce/libero_long \
  --foresight-checkpoint "$CLAD_FORESIGHT_CHECKPOINT" \
  --output-dir "$CLAD_OUTPUT_DIR" \
  --device cuda \
  "$@" 2>&1 | tee -a "$CLAD_OUTPUT_DIR/train_console.log"
