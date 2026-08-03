"""Live DecisionNCE encoding, history buffering, and action-chunk inference."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clad.data import DecisionNCEFeatureCache
from clad.data.camera import camera_view_name
from clad.data.feature_cache import sha256_file
from clad.models import (
    CLaDDiffusionPolicy,
    CLaDHistoryBatch,
    DecisionNCEAdapter,
    DecisionNCEAdapterConfig,
)

DEFAULT_CAMERA_OBSERVATION_KEYS = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
}


def libero_proprioception(observation: Mapping[str, Any]) -> np.ndarray:
    """Recreate the 9D ``robot_states`` used in the LIBERO HDF5 files."""

    fields = (
        "robot0_gripper_qpos",
        "robot0_eef_pos",
        "robot0_eef_quat",
    )
    missing = [name for name in fields if name not in observation]
    if missing:
        raise KeyError(f"LIBERO observation is missing proprioception fields: {missing}")
    components = [np.asarray(observation[name], dtype=np.float32).reshape(-1) for name in fields]
    expected_sizes = (2, 3, 4)
    actual_sizes = tuple(component.size for component in components)
    if actual_sizes != expected_sizes:
        raise ValueError(
            "LIBERO proprioception components must have sizes (2, 3, 4), "
            f"got {actual_sizes}"
        )
    proprioception = np.concatenate(components)
    if not np.isfinite(proprioception).all():
        raise ValueError("LIBERO proprioception contains NaN or Inf")
    return proprioception


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    """One live observation represented in the Stage 1 input spaces."""

    vision_features: dict[str, torch.Tensor]
    proprioception: torch.Tensor


class OnlineDecisionNCEEncoder:
    """Encode live frames while reusing the exact cached task text feature."""

    def __init__(
        self,
        *,
        adapter: DecisionNCEAdapter,
        feature_cache: DecisionNCEFeatureCache,
        camera_observation_keys: Mapping[str, str],
    ) -> None:
        self.adapter = adapter
        self.feature_cache = feature_cache
        self.camera_observation_keys = dict(camera_observation_keys)
        expected_views = tuple(camera_view_name(key) for key in feature_cache.camera_keys)
        if set(self.camera_observation_keys) != set(expected_views):
            raise ValueError(
                "Live camera mapping must exactly match cached camera views: "
                f"expected={sorted(expected_views)}, "
                f"actual={sorted(self.camera_observation_keys)}"
            )
        if any(not key for key in self.camera_observation_keys.values()):
            raise ValueError("Live camera observation keys cannot be empty")
        self._task_instructions = {
            str(entry["task_id"]): str(entry["source"]["instruction"])
            for entry in feature_cache.manifest["tasks"]
        }

    @classmethod
    def from_feature_cache(
        cls,
        cache_dir: str | Path,
        *,
        device: str = "auto",
        camera_observation_keys: Mapping[str, str] | None = None,
        verify_checkpoint: bool = True,
    ) -> OnlineDecisionNCEEncoder:
        cache = DecisionNCEFeatureCache(cache_dir)
        spec = cache.manifest.get("spec")
        if not isinstance(spec, Mapping):
            cache.close()
            raise ValueError("DecisionNCE feature manifest must contain a spec mapping")
        model_name = str(spec["model_name"])
        expected_sha256 = str(spec["checkpoint_sha256"])
        # The official DecisionNCE.load() API always reads this standard path.
        checkpoint_path = Path.home() / ".cache" / "DecisionNCE" / model_name
        if verify_checkpoint:
            if not checkpoint_path.is_file():
                cache.close()
                raise FileNotFoundError(
                    f"DecisionNCE checkpoint does not exist: {checkpoint_path}"
                )
            actual_sha256 = sha256_file(checkpoint_path)
            if actual_sha256 != expected_sha256:
                cache.close()
                raise ValueError(
                    "Live DecisionNCE checkpoint does not match the training cache: "
                    f"expected sha256={expected_sha256}, actual sha256={actual_sha256}"
                )

        expected_views = tuple(camera_view_name(key) for key in cache.camera_keys)
        if camera_observation_keys is None:
            try:
                camera_observation_keys = {
                    view: DEFAULT_CAMERA_OBSERVATION_KEYS[view] for view in expected_views
                }
            except KeyError as error:
                cache.close()
                raise ValueError(
                    f"No default live observation key for cached view {error.args[0]!r}; "
                    "provide camera_observation_keys explicitly"
                ) from error
        try:
            adapter = DecisionNCEAdapter.from_pretrained(
                DecisionNCEAdapterConfig(
                    model_name=model_name,
                    device=device,
                    source_revision=str(spec["source_revision"]),
                    checkpoint_sha256=expected_sha256,
                )
            )
            return cls(
                adapter=adapter,
                feature_cache=cache,
                camera_observation_keys=camera_observation_keys,
            )
        except BaseException:
            cache.close()
            raise

    @property
    def device(self) -> torch.device:
        return self.adapter.device

    def text_feature(self, task_id: str, instruction: str) -> torch.Tensor:
        expected = self._task_instructions.get(task_id)
        if expected is None:
            raise KeyError(f"Task {task_id!r} is not present in the DecisionNCE cache")
        if instruction.strip() != expected.strip():
            raise ValueError(
                f"LIBERO instruction does not match cached task {task_id!r}: "
                f"{instruction!r} != {expected!r}"
            )
        return self.feature_cache.text_feature(task_id).to(self.device)

    @torch.inference_mode()
    def encode_observation(self, observation: Mapping[str, Any]) -> EncodedObservation:
        images: dict[str, torch.Tensor] = {}
        for view_name, observation_key in self.camera_observation_keys.items():
            if observation_key not in observation:
                raise KeyError(
                    f"LIBERO observation is missing camera field {observation_key!r}"
                )
            image = np.asarray(observation[observation_key])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"Camera {observation_key!r} must have shape [H,W,3], got {image.shape}"
                )
            images[view_name] = torch.from_numpy(
                np.ascontiguousarray(image, dtype=np.uint8)
            ).unsqueeze(0)
        encoded = {
            name: feature.squeeze(0)
            for name, feature in self.adapter.encode_views(images).items()
        }
        proprioception = torch.from_numpy(libero_proprioception(observation)).to(self.device)
        return EncodedObservation(
            vision_features=encoded,
            proprioception=proprioception,
        )

    def close(self) -> None:
        self.feature_cache.close()


class OnlineHistoryBuffer:
    """Fixed-width history with repeat/zero padding at episode reset."""

    def __init__(self, *, horizon: int, action_dim: int) -> None:
        if horizon <= 0 or action_dim <= 0:
            raise ValueError("horizon and action_dim must be positive")
        self.horizon = horizon
        self.action_dim = action_dim
        self._observations: deque[EncodedObservation] = deque(maxlen=horizon + 1)
        self._actions: deque[torch.Tensor] = deque(maxlen=horizon)

    def reset(self, initial_observation: EncodedObservation) -> None:
        self._observations.clear()
        self._actions.clear()
        self._observations.extend([initial_observation] * (self.horizon + 1))
        action_device = initial_observation.proprioception.device
        self._actions.extend(
            torch.zeros(self.action_dim, device=action_device) for _ in range(self.horizon)
        )

    def append(self, action: np.ndarray | torch.Tensor, observation: EncodedObservation) -> None:
        if len(self._observations) != self.horizon + 1:
            raise RuntimeError("History buffer must be reset before append()")
        action_tensor = torch.as_tensor(
            action,
            device=observation.proprioception.device,
            dtype=torch.float32,
        )
        if action_tensor.shape != (self.action_dim,):
            raise ValueError(
                f"action must have shape ({self.action_dim},), got {tuple(action_tensor.shape)}"
            )
        if not torch.isfinite(action_tensor).all():
            raise ValueError("action contains NaN or Inf")
        previous_views = set(self._observations[-1].vision_features)
        if set(observation.vision_features) != previous_views:
            raise ValueError("Camera views changed inside an online episode")
        self._actions.append(action_tensor)
        self._observations.append(observation)

    def history(self, text_feature: torch.Tensor) -> CLaDHistoryBatch:
        if (
            len(self._observations) != self.horizon + 1
            or len(self._actions) != self.horizon
        ):
            raise RuntimeError("History buffer must be reset before history()")
        previous = self._observations[0]
        current = self._observations[-1]
        text = text_feature if text_feature.ndim == 2 else text_feature.unsqueeze(0)
        if text.ndim != 2 or text.shape[0] != 1:
            raise ValueError(f"text_feature must have shape [D] or [1,D], got {text.shape}")
        return CLaDHistoryBatch(
            vision_prev={
                name: value.unsqueeze(0)
                for name, value in previous.vision_features.items()
            },
            vision_now={
                name: value.unsqueeze(0)
                for name, value in current.vision_features.items()
            },
            text_features=text,
            proprio_prev=previous.proprioception.unsqueeze(0),
            proprio_now=current.proprioception.unsqueeze(0),
            past_actions=torch.stack(tuple(self._actions)).unsqueeze(0),
        )


@dataclass(frozen=True, slots=True)
class PolicyPlan:
    """One environment-scale action chunk and its wall-clock sampling time."""

    actions: np.ndarray
    inference_seconds: float


class CLaDOnlinePolicy:
    """Stateful receding-horizon wrapper around the EMA diffusion policy."""

    def __init__(
        self,
        *,
        model: CLaDDiffusionPolicy,
        encoder: OnlineDecisionNCEEncoder,
        execution_steps: int,
        amp_enabled: bool = True,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        if not 1 <= execution_steps <= model.config.horizon:
            raise ValueError(
                f"execution_steps must be in [1, {model.config.horizon}], "
                f"got {execution_steps}"
            )
        parameter = next(model.denoiser.parameters())
        self.model = model
        self.encoder = encoder
        self.device = parameter.device
        if encoder.device != self.device:
            raise ValueError(
                f"DecisionNCE and policy must share a device: {encoder.device} != {self.device}"
            )
        self.execution_steps = execution_steps
        self.amp_enabled = amp_enabled and self.device.type == "cuda"
        self.amp_dtype = amp_dtype
        self.history_buffer = OnlineHistoryBuffer(
            horizon=model.config.horizon,
            action_dim=model.config.action_dim,
        )
        self._text_feature: torch.Tensor | None = None
        self._generator = torch.Generator(device=self.device)

    def reset(
        self,
        *,
        task_id: str,
        instruction: str,
        observation: Mapping[str, Any],
        seed: int,
    ) -> None:
        self._generator.manual_seed(seed)
        self._text_feature = self.encoder.text_feature(task_id, instruction)
        self.history_buffer.reset(self.encoder.encode_observation(observation))

    def observe(self, action: np.ndarray, observation: Mapping[str, Any]) -> None:
        self.history_buffer.append(action, self.encoder.encode_observation(observation))

    @torch.inference_mode()
    def plan(self) -> PolicyPlan:
        if self._text_feature is None:
            raise RuntimeError("Online policy must be reset before plan()")
        history = self.history_buffer.history(self._text_feature)
        started = time.perf_counter()
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            sample = self.model.sample_actions(history, generator=self._generator)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        actions = sample.actions[0, : self.execution_steps].float().cpu().numpy()
        if not np.isfinite(actions).all():
            raise FloatingPointError("Sampled policy actions contain NaN or Inf")
        return PolicyPlan(actions=actions, inference_seconds=elapsed)
