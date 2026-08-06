#!/usr/bin/env bash
set -euo pipefail

CLAD_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLAD_CACHE_DIR="${CLAD_TWO_VIEW_CACHE_DIR:-$CLAD_PROJECT_DIR/.cache/decisionnce/libero_long_two_view}"
CLAD_PYTHON="${CLAD_PYTHON:-python}"
: "${LIBERO_DATASET_DIR:?Set LIBERO_DATASET_DIR to the absolute libero_10 directory}"

cd "$CLAD_PROJECT_DIR"
"$CLAD_PYTHON" scripts/cache_decisionnce_features.py \
  --dataset-dir "$LIBERO_DATASET_DIR" \
  --cache-dir "$CLAD_CACHE_DIR" \
  --model-name DecisionNCE-T \
  --device cuda \
  --batch-size 256 \
  --camera-key obs/agentview_rgb \
  --camera-key obs/eye_in_hand_rgb \
  "$@"
