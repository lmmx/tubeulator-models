"""Layered HP defaults from TOML: base → model type → profile → overrides."""

from __future__ import annotations

import tomllib
from pathlib import Path


__all__ = [
    "resolve",
    "resolve_data",
    "resolve_filter",
    "resolve_analysis",
    "resolve_plot",
    "repo_root",
]

_TOML = Path(__file__).with_name("defaults.toml")
_REPO_ROOT = Path(__file__).parents[2]

MODEL_TYPES = ("line", "change", "station", "nexthop")


def _scalars(d: dict) -> dict:
    """Keep only scalar (non-table) values from a TOML section."""
    return {k: v for k, v in d.items() if not isinstance(v, dict)}


def _raw() -> dict:
    return tomllib.loads(_TOML.read_text())


def resolve(
    model_type: str,
    profile: str | None = None,
) -> dict:
    """
    Merge base → model.<type> → profile → profile.model.<type>.

    Returns a flat dict suitable for unpacking into TrainConfig.
    """
    raw = _raw()

    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type {model_type!r}, expected one of {MODEL_TYPES}"
        )

    # 1. Base scalars
    merged: dict = {}
    merged.update(_scalars(raw.get("base", {})))

    # 2. Model-type overrides
    merged.update(raw.get("model", {}).get(model_type, {}))

    # 3. Resolve active profile: explicit arg > TOML base > none
    active_profile = profile or merged.pop("profile", None)
    merged.pop("profile", None)  # don't pass through to TrainConfig

    if active_profile:
        p = raw.get("profiles", {}).get(active_profile, {})
        merged.update(_scalars(p))
        merged.update(p.get("model", {}).get(model_type, {}))

    merged["model_type"] = model_type
    return merged


def resolve_data() -> dict:
    """Return the [data] section."""
    return _raw().get("data", {})


def resolve_filter() -> dict:
    """Return the [filter] section."""
    return _raw().get("filter", {})


def resolve_analysis() -> dict:
    """Return the [analysis] section."""
    return _raw().get("analysis", {})


def resolve_plot() -> dict:
    """Return the [plot] section."""
    return _raw().get("plot", {})


def repo_root() -> Path:
    return _REPO_ROOT


def resolve_hub() -> dict:
    """Return the [hub] section — model name → HF repo ID."""
    return _raw().get("hub", {})


def default_model_type() -> str:
    """Return the default model type from TOML [base]."""
    return _raw().get("base", {})["default_model"]
