#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/evaluate_libero.sh GPU_ID [evaluation options]

Examples:
  # Full LIBERO-10 evaluation: 50 rollouts per task
  ./scripts/evaluate_libero.sh 1

  # One-rollout smoke test with video
  CLAD_EVAL_OUTPUT_DIR=outputs/clad_evaluation_smoke \
    ./scripts/evaluate_libero.sh 1 \
      --task-ids 0 --rollouts-per-task 1 --max-steps 600 --save-videos

Optional path overrides:
  CLAD_STAGE2_CHECKPOINT
  CLAD_FORESIGHT_CHECKPOINT
  CLAD_DECISIONNCE_CACHE_DIR
  CLAD_LIBERO_CONFIG_DIR
  CLAD_EVAL_OUTPUT_DIR
  CLAD_PYTHON
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

CLAD_GPU_ID="$1"
shift
if [[ ! "$CLAD_GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer, got: $CLAD_GPU_ID" >&2
  exit 2
fi

CLAD_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLAD_STAGE2_CHECKPOINT="${CLAD_STAGE2_CHECKPOINT:-$CLAD_PROJECT_DIR/outputs/clad_stage2/stage2_latest.pt}"
CLAD_FORESIGHT_CHECKPOINT="${CLAD_FORESIGHT_CHECKPOINT:-$CLAD_PROJECT_DIR/outputs/clad_stage1/stage1_foresight.pt}"
CLAD_DECISIONNCE_CACHE_DIR="${CLAD_DECISIONNCE_CACHE_DIR:-$CLAD_PROJECT_DIR/.cache/decisionnce/libero_long}"
CLAD_LIBERO_CONFIG_DIR="${CLAD_LIBERO_CONFIG_DIR:-$CLAD_PROJECT_DIR/.cache/libero}"
CLAD_EVAL_OUTPUT_DIR="${CLAD_EVAL_OUTPUT_DIR:-$CLAD_PROJECT_DIR/outputs/clad_evaluation}"
CLAD_PYTHON="${CLAD_PYTHON:-python}"

cd "$CLAD_PROJECT_DIR"

for required_file in \
  "$CLAD_STAGE2_CHECKPOINT" \
  "$CLAD_FORESIGHT_CHECKPOINT" \
  "$CLAD_DECISIONNCE_CACHE_DIR/manifest.json" \
  "$CLAD_LIBERO_CONFIG_DIR/config.yaml"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done

mkdir -p "$CLAD_EVAL_OUTPUT_DIR"

echo "CLaD LIBERO evaluation launcher"
echo "  physical_gpu=$CLAD_GPU_ID -> torch_device=cuda:0"
echo "  checkpoint=$CLAD_STAGE2_CHECKPOINT"
echo "  output=$CLAD_EVAL_OUTPUT_DIR"
echo "  console_log=$CLAD_EVAL_OUTPUT_DIR/eval_console.log"

CUDA_VISIBLE_DEVICES="$CLAD_GPU_ID" \
LIBERO_CONFIG_PATH="$CLAD_LIBERO_CONFIG_DIR" \
MUJOCO_GL="${MUJOCO_GL:-egl}" \
  "$CLAD_PYTHON" scripts/evaluate_clad_libero.py \
    --checkpoint "$CLAD_STAGE2_CHECKPOINT" \
    --foresight-checkpoint "$CLAD_FORESIGHT_CHECKPOINT" \
    --cache-dir "$CLAD_DECISIONNCE_CACHE_DIR" \
    --libero-config-dir "$CLAD_LIBERO_CONFIG_DIR" \
    --output-dir "$CLAD_EVAL_OUTPUT_DIR" \
    --device cuda:0 \
    "$@" 2>&1 | tee -a "$CLAD_EVAL_OUTPUT_DIR/eval_console.log"
