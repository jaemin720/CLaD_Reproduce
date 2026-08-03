#!/usr/bin/env python3
"""Export the frozen inference subset of a Stage 1 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from clad.models import export_frozen_foresight_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Full Stage 1 training checkpoint (for example stage1_latest.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the compact frozen-foresight checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = export_frozen_foresight_checkpoint(
        args.source,
        args.output,
        overwrite=args.overwrite,
    )
    size_gib = info.path.stat().st_size / 1024**3
    print(
        "Frozen CLaD foresight checkpoint exported "
        f"| step={info.global_step} "
        f"| tensors={info.tensor_count} "
        f"| size={size_gib:.2f} GiB "
        f"| path={info.path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
