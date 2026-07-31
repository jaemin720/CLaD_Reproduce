# CLaD implementation progress

This document tracks the implementation against the paper's two-stage
training procedure.

## 0. Project and data foundation — complete

- package, environment, and configuration layout;
- LIBERO task discovery and HDF5 validation;
- episode-safe windows at `t - tau`, `t`, and `t + tau`;
- one-camera default with a stable multi-view extension contract.

## 1. Frozen VLM and feature cache — implemented, checkpoint validation pending

- frozen DecisionNCE adapter;
- official image preprocessing and text tokenization delegated to upstream;
- per-view image encoding without premature view fusion;
- task-wise, atomic HDF5 feature cache;
- dataset/model/checkpoint fingerprint and lazy cache reader.

DecisionNCE source is pinned as an MIT-licensed Git submodule. The remaining
external action is selecting the exact DecisionNCE or Robo-MUTUAL checkpoint.
The paper does not identify it, and checkpoints are deliberately not
redistributed from this Apache-2.0 parent repository.

## 2. Stage 1: Cross-Modal Latent Dynamics — next

- semantic FiLM fusion of cached image and text features;
- proprioceptive, semantic, and action tokenizers;
- modality-specific transition cross-attention;
- asymmetric cross-attention: proprioceptive transition queries semantics;
- learnable-query pooling into `z_dyn`;
- future latent predictors, EMA target encoders, and reconstruction heads;
- `L_latent + 0.1 * L_recon`.

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
