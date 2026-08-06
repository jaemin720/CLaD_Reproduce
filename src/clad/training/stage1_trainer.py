"""Single-GPU trainer for Stage 1 Cross-Modal Latent Dynamics."""

from __future__ import annotations

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
from torch.utils.data import DataLoader, Dataset, Sampler

from clad.models import (
    CLaDStage1Batch,
    CLaDStage1Config,
    CLaDStage1Model,
    CLaDStage1Output,
)

STAGE1_CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class Stage1TrainerConfig:
    """Optimization, runtime, and checkpoint settings for Stage 1."""

    max_steps: int = 25_000
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 500
    min_lr_ratio: float = 0.01
    max_grad_norm: float = 1.0
    amp_enabled: bool = True
    amp_dtype: str = "float16"
    amp_init_scale: float = 2_048.0
    max_consecutive_optimizer_skips: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    log_interval: int = 10
    checkpoint_interval: int = 1_000
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
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        if self.log_interval < 0 or self.checkpoint_interval < 0:
            raise ValueError("log_interval and checkpoint_interval must be non-negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Stage1TrainerConfig:
        known = cls.__dataclass_fields__
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"Unknown Stage 1 trainer settings: {unknown}")
        return cls(**dict(values))


@dataclass(frozen=True, slots=True)
class Stage1TrainingResult:
    """Final trainer state returned to scripts and tests."""

    global_step: int
    attempt_step: int
    skipped_optimizer_steps: int
    latest_metrics: dict[str, float]
    checkpoint_path: Path | None


@dataclass(frozen=True, slots=True)
class _TrainStepResult:
    metrics: dict[str, float]
    optimizer_ran: bool


class ResumableRandomBatchSampler(Sampler[list[int]]):
    """Deterministic shuffled batches whose consumed position is checkpointable.

    DataLoader workers may prefetch sampler outputs. The trainer therefore
    advances ``batches_consumed`` only after it actually receives a batch,
    rather than when this sampler yields indices.
    """

    def __init__(self, *, dataset_size: int, batch_size: int, seed: int) -> None:
        if dataset_size <= 0:
            raise ValueError(f"dataset_size must be positive, got {dataset_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.batches_consumed = 0

    @property
    def batches_per_epoch(self) -> int:
        return self.dataset_size // self.batch_size

    def __len__(self) -> int:
        if self.batches_per_epoch == 0:
            return 0
        start_batch = self.batches_consumed % self.batches_per_epoch
        return self.batches_per_epoch - start_batch

    def __iter__(self) -> Iterator[list[int]]:
        if self.batches_per_epoch == 0:
            return
        epoch, start_batch = divmod(
            self.batches_consumed,
            self.batches_per_epoch,
        )
        generator = torch.Generator()
        generator.manual_seed((self.seed + epoch) % (2**63 - 1))
        indices = torch.randperm(self.dataset_size, generator=generator)
        for batch_index in range(start_batch, self.batches_per_epoch):
            start = batch_index * self.batch_size
            stop = start + self.batch_size
            yield indices[start:stop].tolist()

    def mark_consumed(self) -> None:
        self.batches_consumed += 1

    def state_dict(self) -> dict[str, int]:
        return {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "batches_consumed": self.batches_consumed,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }
        observed = {name: int(state[name]) for name in expected}
        if observed != expected:
            raise ValueError(
                f"Checkpoint DataLoader settings do not match this run: {observed} != {expected}"
            )
        batches_consumed = int(state["batches_consumed"])
        if batches_consumed < 0:
            raise ValueError("batches_consumed must be non-negative")
        self.batches_consumed = batches_consumed


def build_stage1_dataloader(
    dataset: Dataset[dict[str, Any]],
    config: Stage1TrainerConfig,
    *,
    generator: torch.Generator | None = None,
) -> DataLoader[dict[str, Any]]:
    """Create the shuffled, drop-last loader used by the step-based trainer."""

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


class Stage1Trainer:
    """Optimize a composed CLaD Stage 1 model and maintain its EMA targets."""

    def __init__(
        self,
        *,
        model: CLaDStage1Model,
        dataloader: DataLoader[dict[str, Any]],
        config: Stage1TrainerConfig,
        device: torch.device | str,
        output_dir: str | Path,
        metric_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")

        self.model = model.to(self.device)
        self.dataloader = dataloader
        if len(dataloader) == 0:
            raise ValueError("Dataloader has no complete batches; reduce batch_size or add data")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_callback = metric_callback or self._print_metrics
        # ``global_step`` counts successful optimizer updates, matching the
        # paper's 25K training-step budget. AMP overflows only advance the
        # diagnostic attempt counter.
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
            raise ValueError("Stage 1 model has no trainable parameters")
        self.optimizer = AdamW(
            self._trainable_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_multiplier,
        )
        self.amp_enabled = config.amp_enabled and self.device.type == "cuda"
        self.amp_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.amp_dtype]
        scaler_enabled = self.amp_enabled and self.amp_dtype == torch.float16
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=config.amp_init_scale,
            enabled=scaler_enabled,
        )

    @staticmethod
    def seed_everything(seed: int) -> torch.Generator:
        """Seed Python, NumPy, PyTorch, and return a DataLoader generator."""

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

    def _next_batch(self) -> CLaDStage1Batch:
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        try:
            raw_batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            raw_batch = next(self._iterator)
        batch = CLaDStage1Batch.from_mapping(raw_batch).to(
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
    def _loss_metrics(output: CLaDStage1Output) -> dict[str, float]:
        losses = output.losses
        names = (
            "loss",
            "loss_latent",
            "loss_reconstruction",
            "loss_latent_proprio",
            "loss_latent_semantic",
            "loss_reconstruction_proprio",
            "loss_reconstruction_semantic",
            "action_mask_ratio",
        )
        values = torch.stack(
            (
                losses.total.detach(),
                losses.latent.detach(),
                losses.reconstruction.detach(),
                losses.latent_proprio.detach(),
                losses.latent_semantic.detach(),
                losses.reconstruction_proprio.detach(),
                losses.reconstruction_semantic.detach(),
                output.actions.mask.float().mean().detach(),
            )
        ).float()
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
                scaled_loss = output.losses.total / accumulation_steps
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
            self.model.update_ema()
            self.scheduler.step()

        accumulated["gradient_norm"] = float(gradient_norm.detach())
        accumulated["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        accumulated["amp_scale"] = float(self.scaler.get_scale())
        accumulated["optimizer_step_skipped"] = float(not optimizer_ran)
        return _TrainStepResult(
            metrics=accumulated,
            optimizer_ran=optimizer_ran,
        )

    def train(self, *, max_steps: int | None = None) -> Stage1TrainingResult:
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
        return Stage1TrainingResult(
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
        return self.output_dir / "stage1_latest.pt"

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Atomically save all state required to resume optimization."""

        destination = self.checkpoint_path if path is None else Path(path)
        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        payload = {
            "schema_version": STAGE1_CHECKPOINT_SCHEMA_VERSION,
            "global_step": self.global_step,
            "attempt_step": self.attempt_step,
            "skipped_optimizer_steps": self.skipped_optimizer_steps,
            "consecutive_optimizer_skips": self.consecutive_optimizer_skips,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "trainer_config": asdict(self.config),
            "model_config": asdict(self.model.config),
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
                    "DataLoader with build_stage1_dataloader()"
                )
            self._resumable_sampler.load_state_dict(state["batch_sampler"])
        if "loader_generator" in state:
            if self.dataloader.generator is None:
                raise ValueError(
                    "Checkpoint contains DataLoader RNG state but this loader has none"
                )
            self.dataloader.generator.set_state(state["loader_generator"])

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore model, optimizer, scheduler, AMP, RNG, and global step."""

        checkpoint_path = Path(path).expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location="cpu")
        schema_version = int(payload.get("schema_version", -1))
        if schema_version not in {1, STAGE1_CHECKPOINT_SCHEMA_VERSION}:
            raise ValueError(
                f"Unsupported Stage 1 checkpoint schema: {payload.get('schema_version')!r}"
            )
        stored_model_config = payload.get("model_config")
        if not isinstance(stored_model_config, Mapping):
            raise ValueError("Stage 1 checkpoint is missing its model config")
        checkpoint_model_config = CLaDStage1Config.from_checkpoint_mapping(
            stored_model_config
        )
        if checkpoint_model_config != self.model.config:
            raise ValueError("Stage 1 model config does not match the checkpoint")
        if schema_version == 1:
            # Schema 1 counted AMP-skipped attempts as global steps. The
            # scheduler advanced only after real optimizer updates, so its
            # last_epoch recovers the successful update count.
            attempt_step = int(payload["global_step"])
            global_step = int(payload["scheduler"]["last_epoch"])
            skipped_optimizer_steps = max(0, attempt_step - global_step)
            consecutive_optimizer_skips = 0
        else:
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
            raise ValueError(
                "Checkpoint counters must satisfy 0 <= global_step <= attempt_step, "
                f"got {global_step} and {attempt_step}"
            )
        if skipped_optimizer_steps != attempt_step - global_step:
            raise ValueError(
                "Checkpoint skip count is inconsistent with attempt/global steps: "
                f"{skipped_optimizer_steps} != {attempt_step} - {global_step}"
            )
        if not 0 <= consecutive_optimizer_skips <= skipped_optimizer_steps:
            raise ValueError(
                "Checkpoint consecutive optimizer skip count is invalid: "
                f"{consecutive_optimizer_skips}"
            )
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for name, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[name] = value.to(self.device)
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.global_step = global_step
        self.attempt_step = attempt_step
        self.skipped_optimizer_steps = skipped_optimizer_steps
        self.consecutive_optimizer_skips = consecutive_optimizer_skips
        self.latest_metrics = dict(payload.get("latest_metrics", {}))
        self._restore_rng_state(payload["rng_state"])
        self._restore_data_state(payload.get("data_state", {}))
        self._iterator = None
        return self.global_step
