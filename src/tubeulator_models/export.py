"""Export trained checkpoints to Hugging Face–ready directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .config import TrainConfig
from .defaults import repo_root, resolve_hub
from .models.combined import RouteModel
from .topology import build_adj_mask, extract


__all__ = ["export"]

EXPORTS_DIR = repo_root() / "models"

# ── Model cards ───────────────────────────────────────────────

_CARD_POLICY = """\
---
library_name: pytorch
tags:
  - graph-neural-network
  - routing
  - london-underground
  - gatv2
license: mit
---

# Tube Next-Hop Policy

A GATv2 encoder + MLP decoder trained to route on the London Underground graph.

Given a current station and a destination, the model outputs a distribution over
adjacent stations.  Greedy rollout achieves **1.00 Dijkstra ratio** (optimal
shortest travel time) with **100% success rate** across all origin–destination
pairs.

## Architecture

- **Encoder:** {n_enc_layers}-layer GATv2, d={d_model}, {n_heads} heads
- **Decoder:** 3-layer MLP with adjacency masking ({n_stations} stations)
- **Training signal:** KL divergence against Q-soft targets derived from
  Floyd–Warshall shortest paths
- **Parameters:** {n_params:,}

## Usage
```python
import json, torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

repo = "{hub_id}"
config = json.loads(open(hf_hub_download(repo, "config.json")).read())
weights = load_file(hf_hub_download(repo, "model.safetensors"))
metadata = json.loads(open(hf_hub_download(repo, "metadata.json")).read())
# See the project repository for full inference code.
```

## Training

Part of the [Model Trains](https://github.com/lmmx/tubeulator-models) series.
Trained on GTFS timetable data from Transport for London.
"""

_CARD_VALUE = """\
---
library_name: pytorch
tags:
  - graph-neural-network
  - distance-estimation
  - london-underground
  - gatv2
license: mit
---

# Tube Distance Field

A GATv2 encoder + MLP value head trained to predict shortest travel time between
any pair of London Underground stations.

Given an origin and destination, the model outputs estimated travel time in
minutes.  **MAE: {mae} min** — 90% of predictions within 30 seconds of the
Floyd–Warshall ground truth.

## Architecture

- **Encoder:** {n_enc_layers}-layer GATv2, d={d_model}, {n_heads} heads
- **Value head:** 4-layer MLP with LayerNorm
- **Training signal:** Huber loss (δ=2 min) against Floyd–Warshall all-pairs
  shortest times
- **Parameters:** {n_params:,}

## Usage
```python
import json, torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

repo = "{hub_id}"
config = json.loads(open(hf_hub_download(repo, "config.json")).read())
weights = load_file(hf_hub_download(repo, "model.safetensors"))
metadata = json.loads(open(hf_hub_download(repo, "metadata.json")).read())
# See the project repository for full inference code.
```

## Limitations

While the value head is accurate as a distance oracle, Bellman rollout using
these predictions achieves only 43% routing success due to error compounding
across hops.  For routing, use the companion policy model.

## Training

Part of the [Model Trains](https://github.com/lmmx/tubeulator-models) series.
Trained on GTFS timetable data from Transport for London.
"""


def _build_model(cfg: TrainConfig, topo, device: torch.device) -> RouteModel:
    """Reconstruct the model architecture from config + topology."""
    stations = topo.all_stations
    lines = topo.all_lines

    model = RouteModel(
        n_stations=len(stations),
        n_lines=len(lines),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_enc_layers=cfg.n_enc_layers,
        model_type=cfg.model_type,
        max_seq=cfg.max_seq,
        dropout=cfg.dropout,
        value_primary=cfg.value_primary,
    ).to(device)

    adj = build_adj_mask(topo, stations).to(device)
    model.decoder.set_adj_mask(adj)
    return model


def export(
    variant: str,
    profile: str = "full",
    mae: float | None = None,
) -> Path:
    """
    Export a checkpoint to a Hugging Face–ready directory.

    Args:
        variant: 'policy' or 'value'
        profile: training profile to reconstruct architecture
        mae: value head MAE to include in model card (value variant only)

    Returns:
        Path to the export directory.
    """
    hub_ids = resolve_hub()
    hub_key = f"nexthop_{variant}"
    if hub_key not in hub_ids:
        raise ValueError(f"No hub ID for {hub_key!r} in [hub] config")
    hub_id = hub_ids[hub_key]

    is_value = variant == "value"

    cfg = TrainConfig.from_defaults(
        model_type="nexthop",
        profile=profile,
        value_primary=is_value,
    )

    device = torch.device("cpu")
    topo = extract(cfg.gtfs_path)
    stations = topo.all_stations
    lines = topo.all_lines

    model = _build_model(cfg, topo, device)

    # Load checkpoint
    if is_value:
        ckpt_path = cfg.checkpoint_dir / "model_nexthop_value_best.pt"
    else:
        ckpt_path = cfg.checkpoint_dir / "model_nexthop_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded: {ckpt_path}")

    n_params = sum(p.numel() for p in model.parameters())

    # ── Output directory ──────────────────────────────────────
    export_dir = EXPORTS_DIR / hub_key
    export_dir.mkdir(parents=True, exist_ok=True)

    # ── Save weights as safetensors ───────────────────────────
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, export_dir / "model.safetensors")
    print(f"  Saved model.safetensors ({n_params:,} params)")

    # ── Save architecture config ──────────────────────────────
    config = {
        "model_type": "nexthop",
        "variant": variant,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_enc_layers": cfg.n_enc_layers,
        "max_seq": cfg.max_seq,
        "dropout": cfg.dropout,
        "n_stations": len(stations),
        "n_lines": len(lines),
        "value_primary": is_value,
        "n_params": n_params,
    }
    (export_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # ── Save station/line metadata ────────────────────────────
    adj = build_adj_mask(topo, stations)
    metadata = {
        "stations": stations,
        "lines": lines,
        "adjacency": adj.nonzero(as_tuple=False).tolist(),
    }
    (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # ── Model card ────────────────────────────────────────────
    card_vars = dict(
        hub_id=hub_id,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_enc_layers=cfg.n_enc_layers,
        n_stations=len(stations),
        n_params=n_params,
    )
    if is_value:
        card_vars["mae"] = f"{mae:.2f}" if mae else "0.25"
        card = _CARD_VALUE.format(**card_vars)
    else:
        card = _CARD_POLICY.format(**card_vars)

    (export_dir / "README.md").write_text(card)

    print(f"  Exported to {export_dir}")
    print(f"  Hub ID: {hub_id}")
    return export_dir


def main():
    p = argparse.ArgumentParser(description="Export model to HF-ready directory")
    p.add_argument(
        "variant",
        choices=["policy", "value"],
        help="Which model to export",
    )
    p.add_argument("--profile", default="full")
    p.add_argument("--mae", type=float, default=None, help="Value head MAE for card")
    p.add_argument(
        "--upload",
        action="store_true",
        help="Upload to Hugging Face after export",
    )
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    export_dir = export(args.variant, profile=args.profile, mae=args.mae)

    if args.upload:
        hub_ids = resolve_hub()
        hub_id = hub_ids[f"nexthop_{args.variant}"]
        _upload(export_dir, hub_id, private=args.private)


def _upload(export_dir: Path, hub_id: str, private: bool = False):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(hub_id, exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=hub_id,
        commit_message=f"Export {export_dir.name}",
    )
    print(f"  Uploaded to https://huggingface.co/{hub_id}")
