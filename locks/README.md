# Environment locks

These files complement the portable `environment.yml`:

- `conda-linux-64-explicit.txt` records exact conda artifact URLs from the
  validated Linux x86-64 environment.
- `pip-linux-py310.constraints.txt` records all 116 installed Python
  distributions by exact version.

The conda explicit lock is platform-specific. Do not use it on macOS, Windows,
or a different architecture. The pip file is a constraints file and must be
passed with `pip install -c`; installing it directly with `-r` would request
every development and transitive package and may replace conda-owned packages.

Neither file contains wheel hashes for PyPI artifacts, so they provide exact
version resolution rather than bit-for-bit artifact locking. Source identity is
provided separately by the parent Git revision and its submodule gitlinks.

Capture candidate snapshots only after complete training and LIBERO rollout
validation. Write them under ignored run artifacts first so the committed locks
are not accidentally destroyed:

```bash
mkdir -p outputs/environment_snapshot
conda list --explicit > outputs/environment_snapshot/conda-explicit.txt
python -m pip list --format=freeze > outputs/environment_snapshot/pip-freeze.txt
```

Review the diff, rerun import/unit/GPU rollout checks, and preserve the
explanatory headers when deliberately updating committed locks.
