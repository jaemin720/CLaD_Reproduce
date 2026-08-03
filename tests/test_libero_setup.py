from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from clad.evaluation.libero_setup import (
    ROBOSUITE_PRIVATE_MACROS,
    activate_libero_config,
    configure_libero_paths,
    configure_robosuite_logging,
)


def _source_tree(root: Path) -> Path:
    source_root = root / "LIBERO"
    package_root = source_root / "libero" / "libero"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").touch()
    return source_root


def test_configure_libero_paths_is_idempotent_and_uses_dataset_parent(tmp_path: Path) -> None:
    source_root = _source_tree(tmp_path)
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    config_dir = tmp_path / "config"

    config_path = configure_libero_paths(
        config_dir=config_dir,
        source_root=source_root,
        dataset_root=dataset_root,
    )
    second_path = configure_libero_paths(
        config_dir=config_dir,
        source_root=source_root,
        dataset_root=dataset_root,
    )

    assert second_path == config_path
    values = yaml.safe_load(config_path.read_text())
    assert values["datasets"] == str(dataset_root.resolve())
    assert values["benchmark_root"] == str((source_root / "libero" / "libero").resolve())


def test_configure_libero_paths_does_not_replace_an_unrelated_config(tmp_path: Path) -> None:
    source_root = _source_tree(tmp_path)
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("datasets: elsewhere\n")

    with pytest.raises(FileExistsError, match="--force"):
        configure_libero_paths(
            config_dir=config_dir,
            source_root=source_root,
            dataset_root=dataset_root,
        )


def test_activate_libero_config_respects_explicit_environment(tmp_path: Path) -> None:
    requested = tmp_path / "requested"
    active = tmp_path / "active"
    active.mkdir()
    config_path = active / "config.yaml"
    config_path.touch()
    environment = {"LIBERO_CONFIG_PATH": str(active)}

    assert activate_libero_config(requested, environ=environment) == config_path
    assert environment["LIBERO_CONFIG_PATH"] == str(active.resolve())


def test_activate_libero_config_fails_without_import_prompt(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="configure_libero.py"):
        activate_libero_config(tmp_path, environ={})


def test_configure_robosuite_logging_is_idempotent(tmp_path: Path) -> None:
    package_dir = tmp_path / "robosuite"
    package_dir.mkdir()
    (package_dir / "macros.py").touch()

    private_path = configure_robosuite_logging(package_dir=package_dir)
    second_path = configure_robosuite_logging(package_dir=package_dir)

    assert second_path == private_path
    assert private_path.read_text() == ROBOSUITE_PRIVATE_MACROS


def test_configure_robosuite_logging_preserves_existing_user_config(tmp_path: Path) -> None:
    package_dir = tmp_path / "robosuite"
    package_dir.mkdir()
    (package_dir / "macros.py").touch()
    private_path = package_dir / "macros_private.py"
    private_path.write_text("USER_SETTING = True\n")

    with pytest.raises(FileExistsError, match="will not be replaced"):
        configure_robosuite_logging(package_dir=package_dir)
    assert private_path.read_text() == "USER_SETTING = True\n"
