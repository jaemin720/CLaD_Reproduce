#!/usr/bin/env python3
"""Cache frozen DecisionNCE image/text features for LIBERO task files."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from clad.data.feature_cache import (
    DecisionNCEFeatureCacheBuilder,
    FeatureCacheSpec,
    sha256_file,
)
from clad.models import DecisionNCEAdapter, DecisionNCEAdapterConfig
from clad.models.decisionnce_adapter import OFFICIAL_DECISIONNCE_REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        choices=("DecisionNCE-P", "DecisionNCE-T"),
        default="DecisionNCE-T",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source-revision", default=OFFICIAL_DECISIONNCE_REVISION)
    parser.add_argument(
        "--camera-key",
        dest="camera_keys",
        action="append",
        help="Repeat for multiple views; defaults to obs/agentview_rgb.",
    )
    parser.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_keys = tuple(args.camera_keys or ("obs/agentview_rgb",))
    adapter_config = DecisionNCEAdapterConfig(
        model_name=args.model_name,
        device=args.device,
        source_revision=args.source_revision,
    )
    adapter = DecisionNCEAdapter.from_pretrained(adapter_config)

    # The upstream loader always resolves this exact cache path, downloading
    # the official checkpoint when absent. Hash only after load so the cache
    # fingerprint identifies the weights that were actually consumed.
    checkpoint_path = Path.home() / ".cache" / "DecisionNCE" / args.model_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DecisionNCE.load() returned but its checkpoint was not found at "
            f"{checkpoint_path}"
        )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    verified_config = replace(
        adapter.config,
        checkpoint_sha256=checkpoint_sha256,
    )
    adapter = DecisionNCEAdapter(backend=adapter.backend, config=verified_config)

    spec = FeatureCacheSpec(
        model_name=args.model_name,
        source_revision=args.source_revision,
        checkpoint_sha256=checkpoint_sha256,
        camera_keys=camera_keys,
        feature_dtype=args.feature_dtype,
    )
    builder = DecisionNCEFeatureCacheBuilder(
        adapter=adapter,
        spec=spec,
        batch_size=args.batch_size,
    )
    result = builder.build(
        dataset_dir=args.dataset_dir,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite,
        max_tasks=args.max_tasks,
    )
    print(f"built tasks: {len(result.built_tasks)}")
    print(f"skipped tasks: {len(result.skipped_tasks)}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint sha256: {checkpoint_sha256}")
    print(f"manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
