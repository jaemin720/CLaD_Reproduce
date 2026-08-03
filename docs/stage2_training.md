# Stage 2 training

## Position in the implementation plan

The Stage 2 model and trainer are implemented. The remaining prerequisite for
the paper-scale 200K-step run is a one-step GPU smoke test with the full U-Net
widths. After training, implementation moves to online LIBERO rollout and
evaluation.

## Prerequisites

Stage 2 uses the existing ten-task DecisionNCE cache and the compact Stage 1
foresight checkpoint:

```text
.cache/decisionnce/libero_long/manifest.json
outputs/clad_stage1/stage1_foresight.pt
```

The full `stage1_latest.pt` is accepted but not recommended. The compact file
avoids reading Stage 1 optimizer, target-encoder, and reconstruction state.

## Default paper-scale command

```bash
conda activate clad
cd /home/jack/practice/CLaD

./scripts/train_stage2.sh
```

The defaults come from `configs/model/clad_stage2.yaml` and
`configs/train/stage2.yaml`: 200,000 successful optimizer updates, batch size
128, the approximately 0.23B policy, fp16 AMP, and policy EMA. Complete
console output is also appended to `outputs/clad_stage2/train_console.log`.

Additional CLI overrides are passed through. For example, resume the same run
with:

```bash
./scripts/train_stage2.sh \
  --resume outputs/clad_stage2/stage2_latest.pt
```

## What happens before the first step

The entry point:

1. indexes all 132,090 valid LIBERO-LONG windows;
2. scans each of the 138,090 source actions once to compute per-dimension
   normalization bounds;
3. verifies and fingerprints the frozen foresight checkpoint with SHA256;
4. loads frozen CLaD in fp16 on CUDA;
5. omits cached `future` visual features because Stage 2 does not use a
   future-state target;
6. constructs the FiLM/U-Net optimizer and trainable-parameter EMA.

## Full-architecture GPU smoke test

Run this before the 200K-step job:

```bash
python scripts/train_clad_stage2.py \
  --dataset-dir /data/jack/libero_datasets/libero_10 \
  --cache-dir .cache/decisionnce/libero_long \
  --foresight-checkpoint outputs/clad_stage1/stage1_foresight.pt \
  --output-dir outputs/clad_stage2_full_smoke \
  --device cuda \
  --max-steps 1 \
  --batch-size 1 \
  --warmup-steps 0 \
  --num-workers 0 \
  --log-interval 1 \
  --checkpoint-interval 0 \
  --no-save-final-checkpoint
```

A successful result must report step 1, a finite loss/gradient norm, zero
consecutive skips, and `checkpoint=None`. The full U-Net widths remain enabled
in this smoke test.

For CPU plumbing or low-memory debugging only, `--down-dims 32 64 128` and
`--diffusion-timesteps 4` construct a much smaller non-reproduction policy.

## Logging and checkpointing

Console progress is compact and includes ETA, loss, gradient norm, learning
rate, AMP scale, and skipped updates. Complete records are appended to:

```text
outputs/clad_stage2/train_metrics.jsonl
```

Resolved dataset/model/trainer settings, action bounds, parameter counts, and
the frozen checkpoint fingerprint are written to a timestamped `run_config`
JSON file.

`stage2_latest.pt` is atomically replaced every 5,000 successful updates. It
does not duplicate the 1.24 GiB frozen CLaD state. It contains:

- trainable observation-FiLM and U-Net parameters;
- trainable-policy EMA parameters;
- action normalization buffers;
- AdamW, learning-rate scheduler, and AMP scaler;
- Python, NumPy, PyTorch, and CUDA RNG state;
- exact shuffled-data cursor and optimizer/skip counters;
- SHA256 identity of the required frozen foresight artifact.

The convenience script resumes with the identical configs and byte-identical
foresight artifact:

```bash
./scripts/train_stage2.sh \
  --resume outputs/clad_stage2/stage2_latest.pt
```

AMP-overflow attempts do not consume the 200K optimizer-step budget and do not
update the learning-rate scheduler or EMA. Persistent non-finite gradients
fail fast after the configured consecutive-skip limit.
