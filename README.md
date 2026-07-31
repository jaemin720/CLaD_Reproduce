# CLaD Reproduce

Unofficial reproduction of **CLaD: Planning with Grounded Foresight via
Cross-Modal Latent Dynamics**.

The implementation is intentionally split into the same two stages as the
paper:

1. pre-train cross-modal latent dynamics and grounded latent foresight;
2. freeze CLaD and train a foresight-conditioned diffusion policy.

## Current status

- [x] LIBERO HDF5 discovery and schema inspection
- [x] episode-safe `tau`-window sampling
- [x] frozen DecisionNCE adapter and versioned feature cache
- [x] official DecisionNCE-T checkpoint and real feature-cache smoke test
- [x] Stage 1 semantic/proprioceptive/action input encoders
- [x] Stage 1 modality-specific transition cross-attention
- [x] Stage 1 asymmetric cross-attention and learnable dynamics pooling
- [ ] Stage 1 latent foresight, EMA targets, and reconstruction
- [ ] Stage 2 diffusion policy
- [ ] LIBERO-LONG rollout evaluation

## Local data layout

The dataset is not copied into this repository. The default configuration
expects:

```text
/data/jack/libero_datasets/libero_10/
```

Each training window uses one anchor step `t` and contains:

```text
observations:  t - tau, t, t + tau
past actions:  [t - tau, t)
target actions:[t, t + tau)
```

Windows never cross HDF5 demonstration boundaries.

Images use a view-indexed structure even in the default single-camera mode:

```text
sample["images"]["agentview_rgb"]["prev" | "now" | "future"]
```

The default configuration contains only `obs/agentview_rgb`, matching the
single-image formulation in the paper. A future multi-view experiment can add
`obs/eye_in_hand_rgb` to `camera_keys` without changing the dataset or batch
interface. View fusion belongs to the DecisionNCE model adapter rather than the
HDF5 loader.

## Development setup

```bash
git clone --recurse-submodules https://github.com/jaemin720/CLaD_Reproduce.git
cd CLaD_Reproduce
conda env create -f environment.yml
conda activate clad
pip install -e ".[dev,train]"
```

LIBERO remains an external environment dependency. DecisionNCE is pinned as a
Git submodule so its source revision and MIT license are preserved explicitly.
The environment pins `setuptools=80.9.0` because DecisionNCE's `openai-clip`
dependency imports the legacy `pkg_resources` module, which was removed in
setuptools 81.
If this repository was cloned without submodules, initialize it first:

```bash
git submodule update --init --recursive
pip install -e third_party/DecisionNCE
```

Inspect the local LIBERO-LONG data:

```bash
python scripts/inspect_dataset.py \
  --dataset-dir /data/jack/libero_datasets/libero_10 \
  --horizon 6
```

Run the current tests:

```bash
pytest
```

## DecisionNCE feature cache

The official DecisionNCE package stays external and is loaded lazily. The
adapter is pinned by configuration to the inspected upstream source revision:

```text
ebdc585c5e6833ec3a2ba77f801b15c74d7a28f8
```

Install that source revision, place the selected checkpoint in DecisionNCE's
standard cache, and build features with:

```bash
python scripts/cache_decisionnce_features.py \
  --dataset-dir /data/jack/libero_datasets/libero_10 \
  --cache-dir .cache/decisionnce/libero_long \
  --model-name DecisionNCE-T
```

The upstream `DecisionNCE.load()` function downloads a missing checkpoint to
`~/.cache/DecisionNCE/<model-name>`. The cache script hashes that exact file
after loading, ensuring the feature fingerprint represents the weights that
were actually used. A custom Robo-MUTUAL checkpoint must be placed at the same
upstream cache path before running the script.

The cache is organized per task and stores one text feature per instruction
and one image feature per trajectory frame and camera view. Its fingerprint
contains the source dataset identity, camera keys, model variant, source
revision, checkpoint SHA-256, dtype, and schema version. Stale caches are
rejected unless overwrite is explicitly requested.

The paper does not state whether DecisionNCE-P, DecisionNCE-T, or a downstream
Robo-MUTUAL checkpoint was used. `DecisionNCE-T` is therefore an explicit
reproduction assumption rather than a confirmed paper detail.

## Licensing

Original code in this repository is Apache-2.0. DecisionNCE remains MIT
licensed inside its Git submodule; its license text and attribution are
preserved in `third_party/DecisionNCE/LICENSE`,
`LICENSES/DecisionNCE-MIT.txt`, and `THIRD_PARTY_NOTICES.md`.

Model checkpoints are not committed or redistributed by this repository.
Their terms must be reviewed separately from the source-code licenses.
