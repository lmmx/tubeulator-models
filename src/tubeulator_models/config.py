"""Training configuration — single source of truth after TOML resolution."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, computed_field

from .defaults import default_model_type, repo_root, resolve, resolve_data


__all__ = ["TrainConfig"]


class TrainConfig(BaseModel):
    """All tuneable values for a training run. Built from TOML, not code."""

    model_config = {"frozen": True}

    # ── identity ──────────────────────────────────────────────
    model_type: str = "change"

    # ── encoder ───────────────────────────────────────────────
    d_model: int
    n_heads: int
    n_enc_layers: int
    dropout: float

    # ── decoder ───────────────────────────────────────────────
    max_seq: int

    # ── training ──────────────────────────────────────────────
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    val_split: float
    seed: int
    log_every: int
    warmup_ratio: float
    label_smoothing: float
    scheduled_sampling: float

    # ── inference ─────────────────────────────────────────────
    beam_width: int
    beam_eval_interval: int
    beam_eval_sample: float

    # ── route enumeration ─────────────────────────────────────
    max_transfers: int
    max_routes_per_od: int
    transfer_penalty: float

    @computed_field
    @property
    def hp_tag(self) -> str:
        """Short hashable string for logging / checkpoint naming."""
        parts = [
            f"model={self.model_type}",
            f"d={self.d_model}",
            f"lr={self.lr:.0e}",
            f"bs={self.batch_size}",
        ]
        defaults = resolve(self.model_type)
        if self.dropout != defaults.get("dropout"):
            parts.append(f"do={self.dropout}")
        if self.n_enc_layers != defaults.get("n_enc_layers"):
            parts.append(f"L={self.n_enc_layers}")
        if self.max_seq != defaults.get("max_seq"):
            parts.append(f"seq={self.max_seq}")
        if self.label_smoothing > 0:
            parts.append(f"ls={self.label_smoothing}")
        if self.scheduled_sampling > 0:
            parts.append(f"ss={self.scheduled_sampling}")
        return "_".join(parts)

    # ── derived paths ─────────────────────────────────────────

    @property
    def gtfs_path(self) -> Path:
        return repo_root() / resolve_data()["gtfs_path"]

    @property
    def routes_path(self) -> Path:
        return repo_root() / resolve_data()["routes_path"]

    @property
    def checkpoint_dir(self) -> Path:
        return repo_root() / resolve_data()["checkpoint_dir"]

    # ── constructor ───────────────────────────────────────────

    @classmethod
    def from_defaults(
        cls,
        model_type: str | None = None,
        profile: str | None = None,
        **overrides,
    ) -> TrainConfig:
        """TOML defaults ← explicit overrides. The only way to build this."""
        mt = model_type or default_model_type()
        defaults = resolve(mt, profile=profile)
        defaults.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**defaults)
