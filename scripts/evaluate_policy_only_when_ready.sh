#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/evaluate_policy_only_when_ready.sh GPU_ID [evaluation options]

Waits for the active 200K Policy-only training run, verifies the final
checkpoint step, and immediately starts LIBERO evaluation.

Examples:
  # Full evaluation: 10 tasks x 50 rollouts
  ./scripts/evaluate_policy_only_when_ready.sh 1

  # Short evaluation after training completes
  ./scripts/evaluate_policy_only_when_ready.sh 1 \
    --task-ids 0 --rollouts-per-task 10

Optional overrides:
  CLAD_POLICY_ONLY_OUTPUT_DIR       training output (default: outputs/policy_only_official)
  CLAD_POLICY_ONLY_EVAL_OUTPUT_DIR  evaluation output
  CLAD_POLICY_ONLY_EXPECTED_STEP    required final step (default: 200000)
  CLAD_WAIT_POLL_SECONDS            polling interval (default: 30)
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
CLAD_TRAIN_OUTPUT_DIR="${CLAD_POLICY_ONLY_OUTPUT_DIR:-$CLAD_PROJECT_DIR/outputs/policy_only_official}"
CLAD_CHECKPOINT="$CLAD_TRAIN_OUTPUT_DIR/stage2_latest.pt"
CLAD_TRAIN_LOG="$CLAD_TRAIN_OUTPUT_DIR/train_console.log"
CLAD_EVALUATION_DIR="${CLAD_POLICY_ONLY_EVAL_OUTPUT_DIR:-$CLAD_PROJECT_DIR/outputs/policy_only_official_evaluation}"
CLAD_EXPECTED_STEP="${CLAD_POLICY_ONLY_EXPECTED_STEP:-200000}"
CLAD_POLL_SECONDS="${CLAD_WAIT_POLL_SECONDS:-30}"
CLAD_COMPLETION_TEXT="Stage 2 finished | step=$CLAD_EXPECTED_STEP "

for value_name in CLAD_EXPECTED_STEP CLAD_POLL_SECONDS; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 2
  fi
done

training_is_running() {
  pgrep -f '[t]rain_clad_stage2.py.*--policy-variant[ =]policy_only' >/dev/null
}

training_is_complete() {
  [[ -f "$CLAD_TRAIN_LOG" ]] && grep -Fq "$CLAD_COMPLETION_TEXT" "$CLAD_TRAIN_LOG"
}

log_size_at_start=0
if [[ -f "$CLAD_TRAIN_LOG" ]]; then
  log_size_at_start="$(stat -c %s "$CLAD_TRAIN_LOG")"
fi

current_run_is_complete() {
  [[ -f "$CLAD_TRAIN_LOG" ]] || return 1
  tail -c "+$((log_size_at_start + 1))" "$CLAD_TRAIN_LOG" \
    | grep -Fq "$CLAD_COMPLETION_TEXT"
}

echo "Policy-only train-to-evaluation handoff"
echo "  training_log=$CLAD_TRAIN_LOG"
echo "  checkpoint=$CLAD_CHECKPOINT"
echo "  expected_step=$CLAD_EXPECTED_STEP"
echo "  evaluation_gpu=$CLAD_GPU_ID"

if training_is_running; then
  echo "Waiting for Policy-only training to finish..."
  elapsed=0
  next_status=600
  while training_is_running; do
    sleep "$CLAD_POLL_SECONDS"
    elapsed=$((elapsed + CLAD_POLL_SECONDS))
    if (( elapsed >= next_status )); then
      echo "  still waiting (${elapsed}s elapsed)"
      next_status=$((next_status + 600))
    fi
  done
  if ! current_run_is_complete; then
    echo "Policy-only training stopped without the expected completion record." >&2
    exit 1
  fi
elif ! training_is_complete; then
    echo "No active Policy-only trainer and no completed $CLAD_EXPECTED_STEP-step run." >&2
    exit 1
fi

if [[ ! -f "$CLAD_CHECKPOINT" ]]; then
  echo "Completed training log found, but checkpoint is missing: $CLAD_CHECKPOINT" >&2
  exit 1
fi

echo "Training completed; starting Policy-only LIBERO evaluation."
CLAD_STAGE2_CHECKPOINT="$CLAD_CHECKPOINT" \
CLAD_EVAL_OUTPUT_DIR="$CLAD_EVALUATION_DIR" \
  "$CLAD_PROJECT_DIR/scripts/evaluate_libero.sh" "$CLAD_GPU_ID" \
    --require-step "$CLAD_EXPECTED_STEP" \
    "$@"
