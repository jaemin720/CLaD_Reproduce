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
- [ ] frozen DecisionNCE feature adapter and cache
- [ ] Stage 1 CLaD
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

## Development setup

```bash
conda env create -f environment.yml
conda activate clad
pip install -e ".[dev]"
```

LIBERO and DecisionNCE remain external dependencies so their exact revisions
and checkpoints can be recorded independently.

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
