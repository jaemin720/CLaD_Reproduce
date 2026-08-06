#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: ./scripts/evaluate_policy_only_two_view.sh GPU_ID [evaluation options]" >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: ./scripts/evaluate_policy_only_two_view.sh GPU_ID [evaluation options]"
  exit 0
fi

CLAD_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLAD_GPU_ID="$1"
shift

CLAD_STAGE2_CHECKPOINT="${CLAD_STAGE2_CHECKPOINT:-$CLAD_PROJECT_DIR/outputs/policy_only_two_view/stage2_latest.pt}" \
CLAD_DECISIONNCE_CACHE_DIR="${CLAD_TWO_VIEW_CACHE_DIR:-$CLAD_PROJECT_DIR/.cache/decisionnce/libero_long_two_view}" \
CLAD_EVAL_OUTPUT_DIR="${CLAD_EVAL_OUTPUT_DIR:-$CLAD_PROJECT_DIR/outputs/policy_only_two_view_evaluation}" \
  "$CLAD_PROJECT_DIR/scripts/evaluate_libero.sh" "$CLAD_GPU_ID" "$@"
