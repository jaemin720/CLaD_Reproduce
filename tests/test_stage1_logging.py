from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from clad.training import Stage1MetricLogger, Stage2MetricLogger


def test_stage1_metric_logger_saves_jsonl_and_prints_compact_line(
    tmp_path: Path,
) -> None:
    console = StringIO()
    logger = Stage1MetricLogger(
        output_dir=tmp_path,
        max_steps=25_000,
        console=console,
        run_id="test-run",
    )
    logger.start(0)
    config_path = logger.write_run_config(
        {
            "device": "cuda",
            "output_dir": tmp_path,
            "trainer_config": {"batch_size": 128},
        }
    )
    logger(
        {
            "step": 10.0,
            "attempt_step": 11.0,
            "loss": 12.34567,
            "loss_latent": 8.5,
            "loss_reconstruction": 38.4567,
            "gradient_norm": 91.9,
            "learning_rate": 5e-5,
            "amp_scale": 2048.0,
            "skipped_optimizer_steps": 1.0,
            "optimizer_step_skipped": 0.0,
            "seconds_per_log_interval": 1.25,
        }
    )
    logger.close()

    console_line = console.getvalue().strip()
    assert console_line.startswith("[Stage1]    10/25000 (  0.0%) | ETA ")
    assert "s/step | try 11" in console_line
    assert "loss 12.3457" in console_line
    assert "amp 2048 | skips 1 | ok" in console_line
    assert "{" not in console_line

    records = [
        json.loads(line) for line in (tmp_path / "train_metrics.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["run_id"] == "test-run"
    assert records[0]["loss_reconstruction"] == 38.4567
    assert records[0]["logged_at"]
    assert records[0]["estimated_seconds_per_step"] >= 0.0
    assert records[0]["estimated_eta_seconds"] >= 0.0

    saved_config = json.loads(config_path.read_text())
    assert saved_config["run_id"] == "test-run"
    assert saved_config["trainer_config"]["batch_size"] == 128
    assert saved_config["output_dir"] == str(tmp_path)


def test_stage2_metric_logger_uses_stage2_prefix(tmp_path: Path) -> None:
    console = StringIO()
    logger = Stage2MetricLogger(
        output_dir=tmp_path,
        max_steps=200_000,
        console=console,
        run_id="stage2-test",
    )
    logger.start(0)
    logger(
        {
            "step": 10.0,
            "attempt_step": 10.0,
            "loss": 0.75,
            "gradient_norm": 2.0,
            "learning_rate": 1e-4,
            "amp_scale": 2048.0,
            "skipped_optimizer_steps": 0.0,
            "optimizer_step_skipped": 0.0,
        }
    )
    logger.close()

    assert console.getvalue().startswith("[Stage2]    10/200000")
