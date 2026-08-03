#!/usr/bin/env python3
"""Create a non-interactive LIBERO path configuration for this checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

from clad.evaluation.libero_setup import configure_libero_paths, configure_robosuite_logging

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Parent directory containing suite folders such as libero_10.",
    )
    parser.add_argument(
        "--libero-source-root",
        type=Path,
        default=REPOSITORY_ROOT / "third_party" / "LIBERO",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=REPOSITORY_ROOT / ".cache" / "libero",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--configure-robosuite-logging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Install a private macro that disables robosuite's global /tmp log.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = configure_libero_paths(
        config_dir=args.config_dir,
        source_root=args.libero_source_root,
        dataset_root=args.dataset_root,
        overwrite=args.force,
    )
    print(f"LIBERO config: {config_path}")
    print(f"Use LIBERO_CONFIG_PATH={config_path.parent}")
    if args.configure_robosuite_logging:
        macros_path = configure_robosuite_logging(overwrite=args.force)
        print(f"robosuite private macros: {macros_path}")


if __name__ == "__main__":
    main()
