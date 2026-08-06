from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _config(relative_path: str) -> dict[str, Any]:
    with (ROOT / relative_path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    assert isinstance(values, dict)
    return values


def test_stage1_defaults_preserve_paper_reported_values() -> None:
    model = _config("configs/model/clad_stage1.yaml")
    train = _config("configs/train/stage1.yaml")
    data = _config("configs/data/libero_long.yaml")

    assert model["hidden_dim"] == 1024
    assert model["num_proprio_tokens"] == 4
    assert model["num_semantic_tokens"] == 4
    assert model["horizon"] == 6
    assert model["proprioception"] == "libero_joint_gripper"
    assert model["action_mask_ratio"] == 0.3
    assert model["foresight"]["ema_momentum"] == 0.995
    assert model["foresight"]["reconstruction_weight"] == 0.1
    assert data["horizon"] == model["horizon"]
    assert train["max_steps"] == 25_000
    assert train["batch_size"] == 128


def test_stage2_defaults_preserve_paper_reported_values() -> None:
    model = _config("configs/model/clad_stage2.yaml")
    train = _config("configs/train/stage2.yaml")

    assert model["diffusion"]["horizon"] == 6
    assert model["diffusion"]["condition_dim_per_modality"] == 1024
    assert train["max_steps"] == 200_000
    assert train["batch_size"] == 128


def test_policy_only_keeps_stage2_diffusion_contract() -> None:
    clad = _config("configs/model/clad_stage2.yaml")
    policy_only = _config("configs/model/policy_only.yaml")

    assert policy_only["diffusion"] == clad["diffusion"]
    assert policy_only["conditioning"]["hidden_dim"] == 1024
    assert policy_only["conditioning"]["horizon"] == 6
    assert policy_only["conditioning"]["action_dim"] == 7
    assert policy_only["conditioning"]["camera_views"] == ["agentview_rgb"]
    assert policy_only["conditioning"]["proprioception"] == "libero_joint_gripper"


def test_two_view_policy_only_changes_only_camera_contract() -> None:
    single_model = _config("configs/model/policy_only.yaml")
    two_view_model = _config("configs/model/policy_only_two_view.yaml")
    two_view_data = _config("configs/data/libero_long_two_view.yaml")

    assert two_view_model["diffusion"] == single_model["diffusion"]
    single_conditioning = dict(single_model["conditioning"])
    two_view_conditioning = dict(two_view_model["conditioning"])
    assert single_conditioning.pop("camera_views") == ["agentview_rgb"]
    assert two_view_conditioning.pop("camera_views") == [
        "agentview_rgb",
        "eye_in_hand_rgb",
    ]
    assert two_view_conditioning == single_conditioning
    assert two_view_data["camera_keys"] == [
        "obs/agentview_rgb",
        "obs/eye_in_hand_rgb",
    ]


def test_single_checkpoint_evaluation_preserves_reported_rollout_count() -> None:
    evaluation = _config("configs/eval/libero_long.yaml")

    assert evaluation["suite_name"] == "libero_10"
    assert evaluation["rollouts_per_task"] == 50
