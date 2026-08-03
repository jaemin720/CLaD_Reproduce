# Stage 2 foresight-conditioned diffusion policy

## Position in the implementation plan

The frozen CLaD backbone and observation-FiLM bridge produce two
`H=1024` vectors, `g_p` and `g_s`. This implementation completes the policy
model around equation (22):

```text
target actions a_0:       [B, 6, 7]
diffusion timestep k:     [B]
Gaussian noise epsilon:   [B, 6, 7]
global condition:         [g_p ; g_s] = [B, 2048]
predicted noise:           [B, 6, 7]
loss:                      MSE(predicted noise, epsilon)
```

The optimizer, trainable-policy EMA, checkpointing, logging, and 200K-step
Stage 2 trainer are implemented. The next step is a full-architecture GPU
smoke test before launching the paper-scale run.

## Architecture assumptions

The CLaD paper specifies only a diffusion policy, the DDPM objective, a
six-step action horizon, and an approximately 0.23B policy. It does not publish
the U-Net widths, diffusion schedule, timestep count, normalization, optimizer,
or sampling variance.

This reproduction follows the public
[Diffusion Policy repository](https://github.com/real-stanford/diffusion_policy)
at the architectural level:

- conditional 1D U-Net over the action trajectory;
- sinusoidal timestep embedding followed by a two-layer MLP;
- two GroupNorm/Mish convolutions per residual block;
- global scale-and-shift conditioning in every residual block;
- 100 DDPM training and inference steps;
- `squaredcos_cap_v2` beta schedule, fixed-small posterior variance, and
  clipped normalized sample prediction.

No Diffusion Policy package or source tree is vendored. The implementation in
`src/clad/models/clad_diffusion.py` is native Apache-2.0 project code and uses
only PyTorch. The upstream project is linked for methodological attribution.

The default widths are `[512, 1024, 1536]`, rather than the widths of any one
upstream vision experiment. They give exactly 227,412,743 denoiser parameters.
Adding the two Stage 2 observation-FiLM modules gives about 231.6M trainable
parameters, which matches the CLaD paper's rounded 0.23B policy budget.

## Six-step temporal padding

Two stride-2 downsampling operations require a temporal length divisible by
four. The paper's six-action trajectory is therefore padded internally to
length eight before the U-Net and cropped back to length six after the final
convolution. Padding is an implementation detail: the loss, returned tensor,
checkpoint contract, and executed action chunk all remain `[B, 6, 7]`.

## Action normalization

Diffusion operates in `[-1, 1]`. `LinearActionNormalizer` maps every action
dimension independently using training-set minima and maxima. It deliberately
raises an error until fitted, preventing accidental raw-action training. The
normalization parameters are registered buffers and will therefore be stored
in Stage 2 checkpoints.

The inspected LIBERO-LONG demonstrations are already globally bounded by
`[-1, 1]`, but several rotation dimensions occupy much narrower ranges.
Per-dimension fitting preserves their useful numerical resolution. The Stage 2
trainer computes these bounds from the source episodes before the first
optimizer step and persists the resulting normalizer in its checkpoint.

## Implemented interfaces

- `DDPMSchedule.add_noise()` implements the forward process in equation (22).
- `ConditionalUnet1D` predicts epsilon from the noisy action, timestep,
  `g_p`, and `g_s`.
- `CLaDDiffusionPolicy.forward()` returns the MSE loss and diagnostic tensors.
- `CLaDDiffusionPolicy.sample_actions()` performs all 100 reverse DDPM steps
  and converts the normalized result back to the environment action scale.
- `CLaDStage2Batch` requires online history and `target_actions`; future image
  and proprioceptive targets are no longer required.

All architecture and schedule settings are configurable under `diffusion` in
`configs/model/clad_stage2.yaml`. Changing the widths or schedule is a
reproduction deviation and changes the model/checkpoint shape.
