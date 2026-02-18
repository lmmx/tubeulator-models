"""Resolve configuration from defaults.toml."""

from __future__ import annotations

from pathlib import Path
import tomllib

__all__ = ["resolve", "repo_root"]

_TOML = Path(__file__).with_name("defaults.toml")

# src/tubeulator_models -> src -> repo root
_REPO_ROOT = Path(__file__).parents[2]


def resolve(section: str | None = None) -> dict:
    """Return the full config dict, or a specific top-level section."""
    raw = tomllib.loads(_TOML.read_text())
    return raw if section is None else raw.get(section, {})


def repo_root() -> Path:
    return _REPO_ROOT
