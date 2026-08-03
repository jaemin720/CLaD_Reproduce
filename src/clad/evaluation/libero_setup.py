"""Non-interactive path configuration for the vendored LIBERO runtime."""

from __future__ import annotations

import os
import tempfile
from collections.abc import MutableMapping
from importlib.util import find_spec
from pathlib import Path

import yaml

LIBERO_CONFIG_ENV = "LIBERO_CONFIG_PATH"
LIBERO_CONFIG_NAME = "config.yaml"
ROBOSUITE_PRIVATE_MACROS = '''"""CLaD-local robosuite runtime overrides."""

import robosuite.macros as _macros

# robosuite 1.4 otherwise writes to the process-global /tmp/robosuite.log.
_macros.FILE_LOGGING_LEVEL = None
'''


def configure_libero_paths(
    *,
    config_dir: Path,
    source_root: Path,
    dataset_root: Path,
    overwrite: bool = False,
) -> Path:
    """Write LIBERO's path file without triggering its interactive import hook."""

    resolved_config_dir = config_dir.expanduser().resolve()
    resolved_source_root = source_root.expanduser().resolve()
    resolved_dataset_root = dataset_root.expanduser().resolve()
    benchmark_root = resolved_source_root / "libero" / "libero"
    expected_source = benchmark_root / "__init__.py"
    if not expected_source.is_file():
        raise FileNotFoundError(
            f"LIBERO source is not initialized at {resolved_source_root}; "
            "run `git submodule update --init --recursive third_party/LIBERO`"
        )
    if not resolved_dataset_root.is_dir():
        raise FileNotFoundError(f"LIBERO dataset root does not exist: {resolved_dataset_root}")

    values = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(resolved_dataset_root),
        "assets": str(benchmark_root / "assets"),
    }
    config_path = resolved_config_dir / LIBERO_CONFIG_NAME
    if config_path.exists():
        existing = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if existing == values:
            return config_path
        if not overwrite:
            raise FileExistsError(
                f"LIBERO config already exists with different paths: {config_path}; "
                "pass --force to replace it"
            )

    resolved_config_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=resolved_config_dir,
        prefix=f".{LIBERO_CONFIG_NAME}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        yaml.safe_dump(values, stream, sort_keys=False)
        temporary_path = Path(stream.name)
    temporary_path.replace(config_path)
    return config_path


def activate_libero_config(
    config_dir: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    """Select a prepared LIBERO config and fail before its interactive import."""

    environment = os.environ if environ is None else environ
    requested_dir = config_dir.expanduser().resolve()
    configured_dir = environment.get(LIBERO_CONFIG_ENV)
    active_dir = (
        Path(configured_dir).expanduser().resolve()
        if configured_dir is not None
        else requested_dir
    )
    config_path = active_dir / LIBERO_CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(
            f"LIBERO config does not exist: {config_path}. Run "
            "`python scripts/configure_libero.py --dataset-root <dataset-root>` first."
        )
    environment[LIBERO_CONFIG_ENV] = str(active_dir)
    return config_path


def configure_robosuite_logging(
    *,
    package_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Disable robosuite 1.4's hard-coded global log via its private macros hook."""

    if package_dir is None:
        spec = find_spec("robosuite")
        locations = None if spec is None else spec.submodule_search_locations
        if not locations:
            raise ModuleNotFoundError(
                "robosuite is not installed; install the CLaD evaluation extra first"
            )
        resolved_package_dir = Path(next(iter(locations))).resolve()
    else:
        resolved_package_dir = package_dir.expanduser().resolve()
    if not (resolved_package_dir / "macros.py").is_file():
        raise FileNotFoundError(f"Invalid robosuite package directory: {resolved_package_dir}")

    private_path = resolved_package_dir / "macros_private.py"
    if private_path.exists():
        if private_path.read_text(encoding="utf-8") == ROBOSUITE_PRIVATE_MACROS:
            return private_path
        if not overwrite:
            raise FileExistsError(
                f"Existing robosuite private macros will not be replaced: {private_path}"
            )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=resolved_package_dir,
        prefix=".macros_private.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(ROBOSUITE_PRIVATE_MACROS)
        temporary_path = Path(stream.name)
    temporary_path.replace(private_path)
    return private_path
