# Third-party notices

The Apache License 2.0 in the repository root applies to the original
`CLaD Reproduce` source code. It does not replace the licenses of third-party
components.

## DecisionNCE

- Project: `2toinf/DecisionNCE`
- Source: <https://github.com/2toinf/DecisionNCE>
- Pinned revision: `ebdc585c5e6833ec3a2ba77f801b15c74d7a28f8`
- License: MIT
- Upstream copyright: `Copyright © 2024 Air`

DecisionNCE is included as a Git submodule under `third_party/DecisionNCE`.
It remains under its upstream MIT license. The license is preserved in:

- `third_party/DecisionNCE/LICENSE` when the submodule is initialized;
- `LICENSES/DecisionNCE-MIT.txt` in the parent repository.

The CLaD adapter imports and calls DecisionNCE's public API. No DecisionNCE
source file is copied into or relicensed as part of the Apache-2.0 parent
package.

## LIBERO

- Project: `Lifelong-Robot-Learning/LIBERO`
- Source: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- Pinned revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Source-code license: MIT
- Upstream copyright: `Copyright (c) 2023 Lifelong Robot Learning`

LIBERO is included as a Git submodule under `third_party/LIBERO`. Its source
code remains under the upstream MIT license; it is not relicensed under the
Apache-2.0 license of the parent project. The license is preserved in:

- `third_party/LIBERO/LICENSE` when the submodule is initialized;
- `LICENSES/LIBERO-MIT.txt` in the parent repository.

LIBERO benchmark data are not committed or redistributed by this repository.
The upstream project documents its benchmark data under CC BY 4.0. Dataset
users remain responsible for reviewing and complying with the applicable
upstream terms.

## Model checkpoints

DecisionNCE and underlying CLIP checkpoints are not distributed by this
repository. They are downloaded or installed separately from their upstream
locations. Model weights may have terms that differ from the source-code
license; users must review and comply with the applicable upstream terms
before use or redistribution.
