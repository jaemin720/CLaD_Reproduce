"""Single-GPU trainer for Stage 2 foresight-conditioned action diffusion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from clad.models import CLaDDiffusionPolicy, CLaDStage2Batch, DiffusionPolicyLossOutput
from clad.training.stage1_trainer import ResumableRandomBatchSampler

STAGE2_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Stage2TrainerConfig:
    """Optimization, EMA, runtime, and checkpoint settings for Stage 2."""

    max_steps: int = 200_000
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    beta1: float = 0.95
    beta2: float = 0.999
    optimizer_epsilon: float = 1e-8
    warmup_steps: int = 500
    min_lr_ratio: float = 0.01
    max_grad_norm: float = 1.0
    amp_enabled: bool = True
    amp_dtype: str = "float16"
    amp_init_scale: float = 2_048.0
    max_consecutive_optimizer_skips: int = 16
    ema_enabled: bool = True
    ema_update_after_step: int = 0
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75
    ema_min_decay: float = 0.0
    ema_max_decay: float = 0.9999
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    log_interval: int = 10
    checkpoint_interval: int = 5_000
    save_final_checkpoint: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        for name, value in {
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW beta1 and beta2 must be in [0, 1)")
        if self.optimizer_epsilon <= 0.0:
            raise ValueError(f"optimizer_epsilon must be positive, got {self.optimizer_epsilon}")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError(
                "warmup_steps must be in [0, max_steps), "
                f"got {self.warmup_steps} for {self.max_steps} steps"
            )
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        if self.max_grad_norm <= 0.0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError(f"amp_dtype must be 'float16' or 'bfloat16', got {self.amp_dtype!r}")
        if self.amp_init_scale <= 0.0:
            raise ValueError(f"amp_init_scale must be positive, got {self.amp_init_scale}")
        if self.max_consecutive_optimizer_skips <= 0:
            raise ValueError(
                "max_consecutive_optimizer_skips must be positive, "
                f"got {self.max_consecutive_optimizer_skips}"
            )
        if self.ema_update_after_step < 0:
            raise ValueError("ema_update_after_step must be non-negative")
        if self.ema_inv_gamma <= 0.0 or self.ema_power <= 0.0:
            raise ValueError("ema_inv_gamma and ema_power must be positive")
        if not 0.0 <= self.ema_min_decay <= self.ema_max_decay < 1.0:
            raise ValueError("EMA decays must satisfy 0 <= min <= max < 1")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        if self.log_interval < 0 or self.checkpoint_interval < 0:
            raise ValueError("log_interval and checkpoint_interval must be non-negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Stage2TrainerConfig:
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown Stage 2 trainer settings: {unknown}")
        return cls(**dict(values))


@dataclass(frozen=True, slots=True)
class ForesightCheckpointIdentity:
    """Stable identity of the frozen Stage 1 artifact used by Stage 2."""

    path: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> ForesightCheckpointIdentity:
        resolved = Path(path).expanduser().resolve()
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        return cls(
            path=str(resolved),
            size_bytes=resolved.stat().st_size,
            sha256=digest.hexdigest(),
        )

    def matches(self, other: ForesightCheckpointIdentity) -> bool:
        """Allow relocation while requiring byte-identical frozen weights."""

        return self.size_bytes == other.size_bytes and self.sha256 == other.sha256

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ForesightCheckpointIdentity:
        return cls(
            path=str(values["path"]),
            size_bytes=int(values["size_bytes"]),
            sha256=str(values["sha256"]),
        )


class TrainableParameterEMA:
    """EMA shadow for trainable FiLM/U-Net parameters, excluding frozen CLaD."""

    def __init__(
        self,
        model: nn.Module,
        *,
        update_after_step: int,
        inv_gamma: float,
        power: float,
        min_decay: float,
        max_decay: float,
    ) -> None:
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.optimization_step = 0
        self.decay = 0.0
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not parameters:
            raise ValueError("EMA model has no trainable parameters")
        self.shadow = {name: parameter.detach().clone() for name, parameter in parameters.items()}

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        return max(self.min_decay, min(value, self.max_decay))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if parameters.keys() != self.shadow.keys():
            raise ValueError("EMA and model trainable parameter names do not match")
        self.decay = self.get_decay(self.optimization_step)
        for name, parameter in parameters.items():
            shadow = self.shadow[name]
            shadow.mul_(self.decay).add_(
                parameter.detach().to(device=shadow.device, dtype=shadow.dtype),
                alpha=1.0 - self.decay,
            )
        self.optimization_step += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if parameters.keys() != self.shadow.keys():
            raise ValueError("EMA and model trainable parameter names do not match")
        for name, parameter in parameters.items():
            parameter.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimization_step": self.optimization_step,
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping):
            raise ValueError("EMA state must contain a shadow mapping")
        if shadow.keys() != self.shadow.keys():
            raise ValueError("EMA checkpoint parameter names do not match")
        for name, destination in self.shadow.items():
            source = shadow[name]
            if not isinstance(source, torch.Tensor) or source.shape != destination.shape:
                raise ValueError(f"Invalid EMA tensor for {name!r}")
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
        self.optimization_step = int(state["optimization_step"])
        self.decay = float(state["decay"])


@dataclass(frozen=True, slots=True)
class Stage2TrainingResult:
    global_step: int
    attempt_step: int
    skipped_optimizer_steps: int
    latest_metrics: dict[str, float]
    checkpoint_path: Path | None


@dataclass(frozen=True, slots=True)
class _TrainStepResult:
    metrics: dict[str, float]
    optimizer_ran: bool


def build_stage2_dataloader(
    dataset: Dataset[dict[str, Any]],
    config: Stage2TrainerConfig,
    *,
    generator: torch.Generator | None = None,
) -> DataLoader[dict[str, Any]]:
    """Build the checkpointable shuffled loader used for Stage 2."""

    persistent_workers = config.persistent_workers and config.num_workers > 0
    sampler = ResumableRandomBatchSampler(
        dataset_size=len(dataset),
        batch_size=config.batch_size,
        seed=generator.initial_seed() if generator is not None else config.seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent_workers,
        generator=generator,
    )


class Stage2Trainer:
    """Optimize trainable policy parameters while keeping CLaD frozen."""

    def __init__(
        self,
        *,
        model: CLaDDiffusionPolicy,
        dataloader: DataLoader[dict[str, Any]],
        config: Stage2TrainerConfig,
        device: torch.device | str,
        output_dir: str | Path,
        foresight_checkpoint: ForesightCheckpointIdentity,
        metric_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        self.model = model.to(self.device)
        if not bool(self.model.action_normalizer.fitted.item()):
            raise ValueError("Stage 2 action normalizer must be fitted before training")
        self.dataloader = dataloader
        if len(dataloader) == 0:
            raise ValueError("Dataloader has no complete batches; reduce batch_size or add data")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.foresight_checkpoint = foresight_checkpoint
        self.metric_callback = metric_callback or self._print_metrics
        self.global_step = 0
        self.attempt_step = 0
        self.skipped_optimizer_steps = 0
        self.consecutive_optimizer_skips = 0
        self.latest_metrics: dict[str, float] = {}
        self._iterator: Iterator[dict[str, Any]] | None = None
        batch_sampler = dataloader.batch_sampler
        self._resumable_sampler = (
            batch_sampler if isinstance(batch_sampler, ResumableRandomBatchSampler) else None
        )

        self._trainable_parameters = tuple(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )
        if not self._trainable_parameters:
            raise ValueError("Stage 2 model has no trainable parameters")
        self.optimizer = AdamW(
            self._trainable_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.optimizer_epsilon,
            weight_decay=config.weight_decay,
        )
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._lr_multiplier)
        self.amp_enabled = config.amp_enabled and self.device.type == "cuda"
        self.amp_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.amp_dtype]
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=config.amp_init_scale,
            enabled=self.amp_enabled and self.amp_dtype == torch.float16,
        )
        self.ema = (
            TrainableParameterEMA(
                self.model,
                update_after_step=config.ema_update_after_step,
                inv_gamma=config.ema_inv_gamma,
                power=config.ema_power,
                min_decay=config.ema_min_decay,
                max_decay=config.ema_max_decay,
            )
            if config.ema_enabled
            else None
        )

    @staticmethod
    def seed_everything(seed: int) -> torch.Generator:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _lr_multiplier(self, scheduler_step: int) -> float:
        if self.config.warmup_steps > 0 and scheduler_step < self.config.warmup_steps:
            return (scheduler_step + 1) / self.config.warmup_steps
        decay_steps = self.config.max_steps - self.config.warmup_steps
        progress = min(
            1.0,
            max(0.0, (scheduler_step - self.config.warmup_steps) / decay_steps),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

    def _next_batch(self) -> CLaDStage2Batch:
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        try:
            raw_batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            raw_batch = next(self._iterator)
        batch = CLaDStage2Batch.from_mapping(raw_batch).to(
            self.device,
            non_blocking=self.config.pin_memory,
        )
        if self._resumable_sampler is not None:
            self._resumable_sampler.mark_consumed()
        return batch

    @staticmethod
    def _print_metrics(metrics: dict[str, float]) -> None:
        print(json.dumps(metrics, sort_keys=True), flush=True)

    @staticmethod
    def _loss_metrics(output: DiffusionPolicyLossOutput) -> dict[str, float]:
        values = torch.stack(
            (
                output.total.detach(),
                output.predicted_noise.detach().square().mean().sqrt(),
                output.target_noise.detach().square().mean().sqrt(),
                output.normalized_actions.detach().abs().max(),
                output.timesteps.detach().float().mean(),
            )
        ).float()
        names = (
            "loss",
            "predicted_noise_rms",
            "target_noise_rms",
            "normalized_action_abs_max",
            "diffusion_timestep_mean",
        )
        return dict(zip(names, values.cpu().tolist(), strict=True))

    def _train_step(self) -> _TrainStepResult:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        accumulation_steps = self.config.gradient_accumulation_steps
        for _ in range(accumulation_steps):
            batch = self._next_batch()
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                output = self.model(batch)
                scaled_loss = output.total / accumulation_steps
            self.scaler.scale(scaled_loss).backward()
            for name, value in self._loss_metrics(output).items():
                accumulated[name] = accumulated.get(name, 0.0) + value / accumulation_steps

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            self._trainable_parameters,
            max_norm=self.config.max_grad_norm,
        )
        previous_scale = float(self.scaler.get_scale())
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_ran = float(self.scaler.get_scale()) >= previous_scale
        else:
            optimizer_ran = bool(torch.isfinite(gradient_norm).item())
            if optimizer_ran:
                self.optimizer.step()
        if optimizer_ran:
            if self.ema is not None:
                self.ema.update(self.model)
            self.scheduler.step()

        accumulated["gradient_norm"] = float(gradient_norm.detach())
        accumulated["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        accumulated["amp_scale"] = float(self.scaler.get_scale())
        accumulated["ema_decay"] = self.ema.decay if self.ema is not None else 0.0
        accumulated["optimizer_step_skipped"] = float(not optimizer_ran)
        return _TrainStepResult(metrics=accumulated, optimizer_ran=optimizer_ran)

    def train(self, *, max_steps: int | None = None) -> Stage2TrainingResult:
        """Train until an absolute number of successful optimizer updates."""

        target_step = self.config.max_steps if max_steps is None else max_steps
        if not self.global_step <= target_step <= self.config.max_steps:
            raise ValueError(
                f"max_steps must be in [{self.global_step}, {self.config.max_steps}], "
                f"got {target_step}"
            )

        checkpoint_path: Path | None = None
        last_log_time = time.perf_counter()
        while self.global_step < target_step:
            step_result = self._train_step()
            self.attempt_step += 1
            if step_result.optimizer_ran:
                self.global_step += 1
                self.consecutive_optimizer_skips = 0
            else:
                self.skipped_optimizer_steps += 1
                self.consecutive_optimizer_skips += 1

            metrics = step_result.metrics
            metrics["step"] = float(self.global_step)
            metrics["attempt_step"] = float(self.attempt_step)
            metrics["skipped_optimizer_steps"] = float(self.skipped_optimizer_steps)
            metrics["consecutive_optimizer_skips"] = float(self.consecutive_optimizer_skips)
            self.latest_metrics = metrics
            should_log = self.config.log_interval > 0 and (
                self.attempt_step % self.config.log_interval == 0 or not step_result.optimizer_ran
            )
            if should_log:
                current_time = time.perf_counter()
                metrics["seconds_per_log_interval"] = current_time - last_log_time
                last_log_time = current_time
                self.metric_callback(dict(metrics))

            if self.consecutive_optimizer_skips >= self.config.max_consecutive_optimizer_skips:
                raise FloatingPointError(
                    "Optimizer update was skipped "
                    f"{self.consecutive_optimizer_skips} consecutive times; "
                    f"latest gradient_norm={metrics['gradient_norm']}, "
                    f"amp_scale={metrics['amp_scale']}. Check inputs/losses or "
                    "lower amp_init_scale."
                )
            if (
                step_result.optimizer_ran
                and self.config.checkpoint_interval > 0
                and self.global_step % self.config.checkpoint_interval == 0
            ):
                checkpoint_path = self.save_checkpoint()

        final_step_was_saved = (
            checkpoint_path is not None
            and self.config.checkpoint_interval > 0
            and self.global_step % self.config.checkpoint_interval == 0
        )
        if self.config.save_final_checkpoint and not final_step_was_saved:
            checkpoint_path = self.save_checkpoint()
        return Stage2TrainingResult(
            global_step=self.global_step,
            attempt_step=self.attempt_step,
            skipped_optimizer_steps=self.skipped_optimizer_steps,
            latest_metrics=dict(self.latest_metrics),
            checkpoint_path=checkpoint_path,
        )

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "stage2_latest.pt"

    def _trainable_model_state(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def _load_trainable_model_state(self, state: Mapping[str, Any]) -> None:
        parameters = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        if state.keys() != parameters.keys():
            missing = sorted(parameters.keys() - state.keys())
            unexpected = sorted(state.keys() - parameters.keys())
            raise ValueError(
                "Stage 2 trainable state names do not match: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for name, parameter in parameters.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise ValueError(f"Invalid Stage 2 trainable tensor for {name!r}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    def _data_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self._resumable_sampler is not None:
            state["batch_sampler"] = self._resumable_sampler.state_dict()
        if self.dataloader.generator is not None:
            state["loader_generator"] = self.dataloader.generator.get_state()
        return state

    def _restore_data_state(self, state: Mapping[str, Any]) -> None:
        if "batch_sampler" in state:
            if self._resumable_sampler is None:
                raise ValueError(
                    "Checkpoint requires a resumable batch sampler; construct the "
                    "DataLoader with build_stage2_dataloader()"
                )
            self._resumable_sampler.load_state_dict(state["batch_sampler"])
        if "loader_generator" in state:
            if self.dataloader.generator is None:
                raise ValueError(
                    "Checkpoint contains DataLoader RNG state but this loader has none"
                )
            self.dataloader.generator.set_state(state["loader_generator"])

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Atomically save Stage 2 state without duplicating frozen CLaD."""

        destination = self.checkpoint_path if path is None else Path(path)
        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        payload = {
            "schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
            "global_step": self.global_step,
            "attempt_step": self.attempt_step,
            "skipped_optimizer_steps": self.skipped_optimizer_steps,
            "consecutive_optimizer_skips": self.consecutive_optimizer_skips,
            "model_trainable": self._trainable_model_state(),
            "action_normalizer": self.model.action_normalizer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "trainer_config": asdict(self.config),
            "policy_config": asdict(self.model.config),
            "conditioner_config": asdict(self.model.conditioner.config),
            "foresight_checkpoint": asdict(self.foresight_checkpoint),
            "latest_metrics": self.latest_metrics,
            "rng_state": self._rng_state(),
            "data_state": self._data_state(),
        }
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore trainable policy, EMA, optimizer, AMP, RNG, and data cursor."""

        checkpoint_path = Path(path).expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location="cpu")
        if int(payload.get("schema_version", -1)) != STAGE2_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Stage 2 checkpoint schema: {payload.get('schema_version')!r}"
            )
        if payload.get("trainer_config") != asdict(self.config):
            raise ValueError("Stage 2 trainer config does not match the checkpoint")
        if payload.get("policy_config") != asdict(self.model.config):
            raise ValueError("Stage 2 policy config does not match the checkpoint")
        if payload.get("conditioner_config") != asdict(self.model.conditioner.config):
            raise ValueError("Stage 2 conditioner config does not match the checkpoint")
        source_identity = ForesightCheckpointIdentity.from_mapping(payload["foresight_checkpoint"])
        if not self.foresight_checkpoint.matches(source_identity):
            raise ValueError("Frozen foresight checkpoint does not match the Stage 2 checkpoint")

        global_step = int(payload["global_step"])
        attempt_step = int(payload["attempt_step"])
        skipped_optimizer_steps = int(payload["skipped_optimizer_steps"])
        consecutive_optimizer_skips = int(payload["consecutive_optimizer_skips"])
        if not 0 <= global_step <= self.config.max_steps:
            raise ValueError(
                f"Checkpoint step {global_step} is outside this run's valid range "
                f"[0, {self.config.max_steps}]"
            )
        if not 0 <= global_step <= attempt_step:
            raise ValueError("Checkpoint counters require 0 <= global_step <= attempt_step")
        if skipped_optimizer_steps != attempt_step - global_step:
            raise ValueError("Checkpoint optimizer skip count is inconsistent")
        if not 0 <= consecutive_optimizer_skips <= skipped_optimizer_steps:
            raise ValueError("Checkpoint consecutive optimizer skip count is invalid")

        self._load_trainable_model_state(payload["model_trainable"])
        self.model.action_normalizer.load_state_dict(payload["action_normalizer"])
        self.optimizer.load_state_dict(payload["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for name, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[name] = value.to(self.device)
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        ema_state = payload.get("ema")
        if self.ema is None and ema_state is not None:
            raise ValueError("Checkpoint contains EMA state but this run disables EMA")
        if self.ema is not None:
            if not isinstance(ema_state, Mapping):
                raise ValueError("Checkpoint is missing required EMA state")
            self.ema.load_state_dict(ema_state)
            if self.ema.optimization_step != global_step:
                raise ValueError(
                    "EMA optimization step does not match global step: "
                    f"{self.ema.optimization_step} != {global_step}"
                )
        self.global_step = global_step
        self.attempt_step = attempt_step
        self.skipped_optimizer_steps = skipped_optimizer_steps
        self.consecutive_optimizer_skips = consecutive_optimizer_skips
        self.latest_metrics = dict(payload.get("latest_metrics", {}))
        self._restore_rng_state(payload["rng_state"])
        self._restore_data_state(payload.get("data_state", {}))
        self._iterator = None
        return self.global_step
