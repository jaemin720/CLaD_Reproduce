# Stage 2 conditioning bridge

## Position in the implementation plan

Stage 1 Cross-Modal Latent Dynamics pre-training is complete. Stage 2 trains a
diffusion policy for 200K steps while DecisionNCE and the trained CLaD model
remain frozen. The implemented bridge covers equations (20)--(21):

1. predict proprioceptive and semantic future latents from observation/action
   history with frozen CLaD;
2. encode the current proprioceptive and semantic observations;
3. condition each future latent on its corresponding current observation with
   a trainable FiLM layer;
4. concatenate `g_p` and `g_s` for the action denoiser.

The conditional denoiser, diffusion schedule, DDPM loss, and Stage 2 trainer
are the next implementation step.

## Reproduction assumptions

The paper does not publish the internal architecture of `e_p`, `e_s`, or the
FiLM layers. This implementation reuses the trained Stage 1 online state
encoders as `e_p` and `e_s`, freezes them, and mean-pools their state-token
dimension. This keeps current observations and predicted foresights in the
same `H=1024` latent space. A separate affine FiLM is learned per modality and
is initialized as an identity transform:

```text
o_p = mean(f_p(p_t))
o_s = mean(f_s(FiLM(v_t, l)))
g_p = z_hat_p * (1 + scale_p(o_p)) + shift_p(o_p)
g_s = z_hat_s * (1 + scale_s(o_s)) + shift_s(o_s)
```

The paper's equation (21) loses modality subscripts on `z_hat` in typesetting,
but the surrounding text says each foresight is paired with its corresponding
observation. The implementation therefore applies proprioception to
`z_hat_p` and semantics to `z_hat_s`.

During Stage 2, stochastic action-history masking is disabled. Future images
and future proprioception are also absent from this inference path; they were
needed only for Stage 1 targets and losses.

## Export the completed Stage 1 model

The full Stage 1 checkpoint contains AdamW state, EMA target encoders, and
reconstruction heads. They are unnecessary for Stage 2 and make the file much
larger. Export a compact frozen-foresight checkpoint once:

```bash
conda activate clad
cd /home/jack/practice/CLaD

python scripts/export_stage1_foresight.py \
  --source outputs/clad_stage1/stage1_latest.pt \
  --output outputs/clad_stage1/stage1_foresight.pt
```

The command is atomic and refuses to replace an existing output unless
`--overwrite` is supplied. It uses memory-mapped loading so the multi-gigabyte
optimizer tensors are not eagerly copied into RAM. Only use checkpoints that
you trust, because PyTorch checkpoint loading uses pickle metadata.

The backbone can load the full Stage 1 checkpoint directly, but the compact
artifact is recommended for all Stage 2 training and evaluation:

```python
import torch

from clad.models import CLaDForesightBackbone

backbone = CLaDForesightBackbone.from_checkpoint(
    "outputs/clad_stage1/stage1_foresight.pt",
    device="cuda",
    dtype=torch.float16,
)

print("parameters:", sum(p.numel() for p in backbone.parameters()))
print("frozen:", all(not p.requires_grad for p in backbone.parameters()))
print("training mode:", backbone.training)
```

Expected values for the last two lines are `True` and `False`. The history
batch preserves the existing per-view mapping, so the current single
`agentview_rgb` path is unchanged and a second cached view can be enabled
later without modifying the policy interface.
