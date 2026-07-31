# CLaD implementation progress

This document tracks the implementation against the paper's two-stage
training procedure.

## 0. Project and data foundation — complete

- package, environment, and configuration layout;
- LIBERO task discovery and HDF5 validation;
- episode-safe windows at `t - tau`, `t`, and `t + tau`;
- one-camera default with a stable multi-view extension contract.

## 1. Frozen VLM and feature cache — complete

- frozen DecisionNCE adapter;
- official image preprocessing and text tokenization delegated to upstream;
- per-view image encoding without premature view fusion;
- task-wise, atomic HDF5 feature cache;
- dataset/model/checkpoint fingerprint and lazy cache reader.

The official DecisionNCE-T checkpoint was loaded and validated against one
real LIBERO-LONG task. Its observed image and text feature dimensions are both
1024. DecisionNCE source remains pinned as an MIT-licensed Git submodule, while
the downloaded checkpoint is not redistributed from this Apache-2.0 parent
repository.

## 2. Stage 1: Cross-Modal Latent Dynamics — in progress

- [x] semantic FiLM fusion of cached image and text features;
- [x] proprioceptive, semantic, and action tokenizers;
- [x] train-only stochastic action token masking;
- [x] modality-specific transition cross-attention;
- [x] asymmetric cross-attention: proprioceptive transition queries semantics;
- [x] learnable-query pooling into `z_dyn`;
- [x] future latent predictors, EMA target encoders, and reconstruction heads;
- [x] `L_latent + 0.1 * L_recon`;
- [x] cache-backed LIBERO dataset and composed Stage 1 model;
- [x] optimizer, EMA update, AMP, checkpointing, and 25K-step trainer.

The paper does not specify the input MLP depth, action positional encoding, or
multi-view fusion. This reproduction uses two-layer MLP tokenizers, learned
action position embeddings, and mean fusion when more than one cached camera
view is supplied. The default one-camera path performs no cross-view fusion.

The paper also omits cross-attention depth and head count. The configurable
default uses eight pre-norm layers per stack, 16 heads (64 dimensions per
head), and a `4H` feed-forward width. Applying the same stack depth to the two
modality transitions and the subsequent asymmetric transition makes the
completed Stage 1 model approach the reported 0.33B CLaD parameter budget.
Equation (10) follows the cited Perceiver readout: one learned `q_out` attends
over the proprioception-grounded semantic transition tokens and produces one
`H`-dimensional `z_dyn`.

Equations (11)--(19) use two-layer MLP predictors and reconstruction heads.
The paper does not explain how each EMA encoder's `N x H` state tokens become
the `H`-dimensional target in equations (15) and (16), so this reproduction
mean-pools the token dimension without adding target-only parameters. Equation
(17) is implemented as written by L2-normalizing the stopped-gradient EMA
targets, while equation (18) reconstructs raw future proprioception and the
future cached VLM visual feature `s_v`, before language FiLM.

`CachedLiberoWindowDataset` joins proprioception/actions from the source HDF5
with cached `prev`, `now`, and `future` VLM features using the same task,
episode, and anchor index. It requires complete task/camera cache coverage and
avoids loading raw images during Stage 1 training. `CLaDStage1Model` consumes
the collated batch, returns every intermediate diagnostic and loss component,
and exposes an explicit `update_ema()` call for use after each optimizer step.

`Stage1Trainer` performs step-based single-GPU training with configurable
gradient accumulation. Every successful optimizer update is followed by the
paper's 0.995 EMA target update. The checkpoint stores model/EMA parameters,
AdamW, learning-rate scheduler, fp16 scaler, process RNGs, and a consumed-batch
cursor. The custom shuffled batch sampler derives each epoch permutation from
the seed and epoch, so resuming is data-order exact even when DataLoader
workers had prefetched later batches.

Only the 25K steps, batch size 128, and EMA momentum are reported by the paper.
The default AdamW settings (`lr=1e-4`, weight decay 0.01, betas 0.9/0.95),
500-step warmup plus cosine decay, gradient norm 1.0, and CUDA fp16 AMP are
documented reproduction assumptions and remain configurable in
`configs/train/stage1.yaml`.

## 3. Stage 2: foresight-conditioned diffusion policy — pending

- freeze DecisionNCE and Stage 1 CLaD;
- observation-modulated foresight through FiLM;
- conditional 1D U-Net and DDPM action-noise objective;
- six-step, seven-dimensional action chunks.

## 4. LIBERO rollout and evaluation — pending

- online history buffer;
- checkpoint selection and 20/50-rollout protocols;
- task-level success metrics and videos;
- modality, reconstruction, and attention ablations.
