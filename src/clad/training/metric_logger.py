"""Persistent metrics and compact console output for Stage 1 training."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


class Stage1MetricLogger:
    """Append complete JSONL metrics while printing one compact status line."""

    stage_label = "Stage1"

    def __init__(
        self,
        *,
        output_dir: str | Path,
        max_steps: int,
        console: TextIO | None = None,
        run_id: str | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.console = console or sys.stdout
        self.run_id = run_id or self._new_run_id()
        self.metrics_path = self.output_dir / "train_metrics.jsonl"
        self.config_path = self.output_dir / f"run_config_{self.run_id}.json"
        self._metrics_stream = self.metrics_path.open("a", encoding="utf-8")
        self._last_step = 0
        self._last_update_time = time.perf_counter()
        self._seconds_per_step: float | None = None

    @staticmethod
    def _new_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}_{os.getpid()}"

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _number(value: float, *, precision: int = 4) -> str:
        if not math.isfinite(value):
            return str(value)
        return f"{value:.{precision}f}"

    @staticmethod
    def _duration(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(seconds):
            return "--:--:--"
        seconds = max(0, round(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def start(self, step: int) -> None:
        """Start or reset ETA estimation from a fresh or resumed step."""

        if not 0 <= step <= self.max_steps:
            raise ValueError(f"step must be in [0, {self.max_steps}], got {step}")
        self._last_step = step
        self._last_update_time = time.perf_counter()
        self._seconds_per_step = None

    def _progress_estimate(self, step: int) -> tuple[float | None, float | None]:
        now = time.perf_counter()
        step_delta = step - self._last_step
        if step_delta > 0:
            observed = (now - self._last_update_time) / step_delta
            if self._seconds_per_step is None:
                self._seconds_per_step = observed
            else:
                self._seconds_per_step = 0.8 * self._seconds_per_step + 0.2 * observed
            self._last_step = step
            self._last_update_time = now
        eta_seconds = (
            (self.max_steps - step) * self._seconds_per_step
            if self._seconds_per_step is not None
            else None
        )
        return self._seconds_per_step, eta_seconds

    def write_run_config(self, values: Mapping[str, Any]) -> Path:
        """Save one resolved configuration snapshot for this process."""

        payload = {
            "created_at": self._timestamp(),
            "run_id": self.run_id,
            **dict(values),
        }
        self.config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return self.config_path

    def __call__(self, metrics: dict[str, float]) -> None:
        step = int(metrics.get("step", 0.0))
        seconds_per_step, eta_seconds = self._progress_estimate(step)
        record = {
            "logged_at": self._timestamp(),
            "run_id": self.run_id,
            "estimated_seconds_per_step": seconds_per_step,
            "estimated_eta_seconds": eta_seconds,
            **metrics,
        }
        self._metrics_stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._metrics_stream.flush()

        attempt = int(metrics.get("attempt_step", step))
        skips = int(metrics.get("skipped_optimizer_steps", 0.0))
        status = "skip" if metrics.get("optimizer_step_skipped", 0.0) else "ok"
        progress = 100.0 * step / self.max_steps
        rate = f"{seconds_per_step:.3f}s/step" if seconds_per_step is not None else "--s/step"
        attempt_text = f" | try {attempt}" if attempt != step else ""
        line = (
            f"[{self.stage_label}] {step:>5}/{self.max_steps} ({progress:>5.1f}%) | "
            f"ETA {self._duration(eta_seconds)} | {rate}{attempt_text} | "
            f"loss {self._number(metrics.get('loss', math.nan))} | "
            f"grad {metrics.get('gradient_norm', math.nan):.3g} | "
            f"lr {metrics.get('learning_rate', math.nan):.3e} | "
            f"amp {metrics.get('amp_scale', 1.0):.0f} | "
            f"skips {skips} | {status}"
        )
        print(line, file=self.console, flush=True)

    def close(self) -> None:
        self._metrics_stream.close()

    def __enter__(self) -> Stage1MetricLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class Stage2MetricLogger(Stage1MetricLogger):
    """Stage 2 variant with the same durable JSONL and ETA behavior."""

    stage_label = "Stage2"
