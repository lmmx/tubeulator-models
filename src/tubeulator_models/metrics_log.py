"""JSONL metrics logging for training runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .evaluate import RouteMetrics


__all__ = ["MetricsLogger"]


class MetricsLogger:
    """Append-only JSONL logger for per-epoch training metrics."""

    def __init__(self, model_type: str, hp_tag: str, base_dir: Path):
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = logs_dir / f"{model_type}_{timestamp}_{hp_tag}.jsonl"

    def log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: RouteMetrics,
        beam_ran: bool,
    ) -> None:
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "beam_ran": beam_ran,
            **{
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in asdict(metrics).items()
            },
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
