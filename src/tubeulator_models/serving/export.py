"""Export trained checkpoints to Hugging Face–ready directories."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..config import TrainConfig
from ..defaults import repo_root, resolve_hub
from ..graph.topology import build_adj_mask, extract
from ..models.combined import RouteModel


__all__ = ["export"]

EXPORTS_DIR = repo_root() / "models"


@dataclass
class PolicyMetrics:
    """Metrics from a policy training run."""

    success_rate: float
    dijkstra_ratio: float
    step_acc: float
    len_ratio: float
    n_od_pairs: int
    n_steps: int
    n_train_od: int
    n_val_od: int
    n_train_steps: int
    n_val_steps: int
    batch_size: int
    train_time_s: float


@dataclass
class ValueMetrics:
    """Metrics from a value training run."""

    mae: float


# ── Model cards ───────────────────────────────────────────────

_CARD_POLICY = """\
---
library_name: pytorch
license: mit
tags:
  - graph-neural-network
  - gatv2
  - routing
  - shortest-path
  - london-underground
  - transport
  - pytorch-geometric
metrics:
  - accuracy
model-index:
  - name: {hub_id_short}
    results:
      - task:
          type: graph-routing
          name: Next-Hop Routing
        dataset:
          type: custom
          name: TfL GTFS Timetable (London Underground)
        metrics:
          - type: accuracy
            name: Rollout Success Rate
            value: {success_rate}
          - type: custom
            name: Dijkstra Ratio
            value: {dijkstra_ratio}
          - type: custom
            name: Step Accuracy
            value: {step_acc}
          - type: custom
            name: Length Ratio
            value: {len_ratio}
---

# Tube Next-Hop Policy

A GATv2 encoder + MLP decoder trained to predict optimal next-hop routing
decisions on the London Underground graph. Given a current station and a
destination, the model outputs a probability distribution over adjacent
stations.

Greedy rollout achieves **{dijkstra_ratio:.2f}× Dijkstra ratio** (optimal
shortest travel time) with **{success_pct} success rate** across all
{n_od_pairs:,} origin–destination pairs.

## Intended Use

This model is a research artifact and demonstration of learned graph routing.
It predicts the next station to visit on a shortest-time path between any two
London Underground stations, using only the graph topology and travel-time edge
weights from TfL's GTFS timetable feed.

Potential applications include learned routing components in transit planning
systems, GNN routing benchmarks, and educational demonstrations of
graph-attention-based policy learning.

## Architecture

| Component | Details |
|---|---|
| Encoder | {n_enc_layers}-layer GATv2, d={d_model}, {n_heads} heads |
| Decoder | 3-layer MLP with adjacency masking |
| Graph | {n_stations} stations, {n_lines} lines, {n_edges} directed edges |
| Parameters | {n_params:,} |
| Training signal | KL divergence vs. Q-soft targets from Floyd–Warshall |
| Label smoothing | {label_smoothing} |
| Scheduled sampling | {scheduled_sampling} |

## Evaluation Results

Evaluated on a held-out set of {n_val_od:,} OD pairs ({n_val_steps:,} next-hop
steps), with greedy rollout to completion.

| Metric | Value |
|---|---|
| Rollout success rate | {success_pct} |
| Dijkstra ratio (travel time vs. optimal) | {dijkstra_ratio:.2f} |
| Step accuracy (single-step top-1) | {step_acc_pct} |
| Length ratio (hops vs. optimal hops) | {len_ratio:.2f} |

> **Note on step accuracy:** The {step_acc_pct} top-1 step accuracy reflects
> that many nodes have multiple equally-optimal next hops. The model distributes
> probability across these alternatives, which is correct behavior — the
> rollout metrics confirm optimal routing.

## Training Data

The graph topology and edge travel times are extracted from Transport for
London's [GTFS timetable feed](https://tfl.gov.uk/info-for/open-data-users/).
Floyd–Warshall all-pairs shortest paths provide the Q-value supervision signal.

- {n_od_pairs:,} OD pairs → {n_steps:,} next-hop training steps
- 90/10 OD-pair split ({n_train_od:,} train / {n_val_od:,} val)
- Batch size: {batch_size:,} steps

## Training Details

- Optimizer: AdamW, lr={lr}
- Compiled with `torch.compile(mode='default')`
- Training time: ~{train_time_min:.0f} minutes
- Hardware: single GPU

## Limitations

- The model is specific to the London Underground graph topology at the time
  of the GTFS snapshot. It will not generalize to other transit networks
  without retraining.
- Edge weights represent scheduled travel times, not real-time conditions.
- The adjacency mask is fixed at inference time — the model cannot handle
  station closures or service disruptions without mask modification.

## Usage

```bash
pip install tubeulator-models[inference]
```

```python
from tubeulator_models import TubeRouter

router = TubeRouter.from_pretrained("{hub_id}")
route = router.route("West Ham", "Shoreditch")
print(route)
# West Ham
#   → [district] Bromley-by-Bow (2.0m)
#   ...
# ✓ 8 hops · 2 lines · 1 transfer · 18.0 min

# With waypoints
route = router.route("Camden Town", "Canary Wharf", via=["King's Cross"])
```

For CLI usage:

```bash
pip install tubeulator-models[cli]
tm-infer policy --model {hub_id} -o "West Ham" -d Shoreditch
```

## Links

- **Code:** [tubeulator-models](https://github.com/lmmx/tubeulator-models)
- **Companion model:** Distance field (value head) for travel-time estimation

Part of the [Model Trains](https://github.com/lmmx/tubeulator-models) series.
Trained on GTFS timetable data from Transport for London.
"""

_CARD_VALUE = """\
---
library_name: pytorch
license: mit
tags:
  - graph-neural-network
  - gatv2
  - distance-estimation
  - shortest-path
  - london-underground
  - transport
  - pytorch-geometric
metrics:
  - mae
model-index:
  - name: {hub_id_short}
    results:
      - task:
          type: graph-distance-estimation
          name: Travel Time Prediction
        dataset:
          type: custom
          name: TfL GTFS Timetable (London Underground)
        metrics:
          - type: mae
            name: Mean Absolute Error (minutes)
            value: {mae}
---

# Tube Distance Field

A GATv2 encoder + MLP value head trained to predict shortest travel time between
any pair of London Underground stations.

Given an origin and destination, the model outputs estimated travel time in
minutes.  **MAE: {mae} min** — 90% of predictions within 30 seconds of the
Floyd–Warshall ground truth.

## Intended Use

This model is a research artifact demonstrating learned distance estimation on
a transit graph. It predicts the shortest travel time in minutes between any
two London Underground stations using the graph topology and scheduled edge
weights from TfL's GTFS timetable feed.

## Architecture

| Component | Details |
|---|---|
| Encoder | {n_enc_layers}-layer GATv2, d={d_model}, {n_heads} heads |
| Value head | 4-layer MLP with LayerNorm |
| Graph | {n_stations} stations |
| Parameters | {n_params:,} |
| Training signal | Huber loss (δ=2 min) vs. Floyd–Warshall all-pairs shortest times |

## Evaluation Results

| Metric | Value |
|---|---|
| Mean Absolute Error | {mae} min |

## Limitations

While the value head is accurate as a distance oracle, Bellman rollout using
these predictions achieves only 43% routing success due to error compounding
across hops. For routing, use the companion policy model.

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

## Links

- **Code:** [tubeulator-models](https://github.com/lmmx/tubeulator-models)
- **Companion model:** Next-hop policy for optimal routing

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
    metrics: PolicyMetrics | ValueMetrics | None = None,
) -> Path:
    """
    Export a checkpoint to a Hugging Face–ready directory.

    Args:
        variant: 'policy' or 'value'
        profile: training profile to reconstruct architecture
        metrics: training metrics to embed in model card

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

    # Load metrics: CLI override > sidecar file
    if metrics is None:
        metrics = _load_metrics(ckpt_path, variant)

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
    n_edges = int(adj.sum().item())
    metadata = {
        "stations": stations,
        "lines": lines,
        "adjacency": adj.nonzero(as_tuple=False).tolist(),
        "stop_names": topo.stop_names,
    }
    (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # ── Save topology for standalone inference ────────────────
    edge_times_export: dict[str, list] = {}
    for line_id, edges in topo.edge_time.items():
        edge_times_export[line_id] = [
            [from_sid, to_sid, t] for (from_sid, to_sid), t in edges.items()
        ]

    line_edges_export: dict[str, list] = {}
    for line_id, edges in topo.edge_time.items():
        line_edges_export[line_id] = [[a, b] for (a, b) in edges.keys()]

    # Build transfer lookup
    from ..defaults import resolve_data
    from ..graph.topology import build_transfer_lookup, load_interchange_data

    data_cfg = resolve_data()
    ic_rel = data_cfg.get("interchange_path", "")
    ic_path = repo_root() / ic_rel if ic_rel else None
    transfers_export: dict[str, list] = {}
    if ic_path is not None and ic_path.is_file():
        ic_data = load_interchange_data(ic_path)
        tl = build_transfer_lookup(topo, stations, ic_data, discount=1.0)
        for (station, from_line, to_line), secs in tl.items():
            transfers_export.setdefault(station, []).append([from_line, to_line, secs])

    topo_export = {
        "edge_times": edge_times_export,
        "line_edges": line_edges_export,
        "hub_members": {k: list(v) for k, v in topo.hub_members.items()},
        "transfers": transfers_export,
    }
    (export_dir / "topology.json").write_text(json.dumps(topo_export) + "\n")
    print("  Saved topology.json")

    # ── Save graph tensors for standalone inference ───────────
    from ..graph.enriched import build_enriched_graph
    from .infer import _read_stop_coords

    node_coords = _read_stop_coords(cfg.gtfs_path)
    graph = build_enriched_graph(topo, node_coords=node_coords)
    save_file(
        {
            "x": graph.x.contiguous(),
            "edge_index": graph.edge_index.contiguous(),
            "edge_attr": graph.edge_attr.contiguous(),
        },
        export_dir / "graph.safetensors",
    )
    print(
        f"  Saved graph.safetensors ({graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges)"
    )

    # ── Model card ────────────────────────────────────────────
    card_vars = dict(
        hub_id=hub_id,
        hub_id_short=hub_id.split("/")[-1],
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_enc_layers=cfg.n_enc_layers,
        n_stations=len(stations),
        n_lines=len(lines),
        n_edges=n_edges,
        n_params=n_params,
    )

    if is_value:
        if not isinstance(metrics, ValueMetrics):
            raise ValueError("Value export requires ValueMetrics (pass --mae)")
        card_vars["mae"] = f"{metrics.mae:.2f}"
        card = _CARD_VALUE.format(**card_vars)
    else:
        if not isinstance(metrics, PolicyMetrics):
            raise ValueError(
                "Policy export requires PolicyMetrics "
                "(pass --success-rate and other metric flags)"
            )
        card_vars.update(
            success_rate=metrics.success_rate,
            success_pct=f"{metrics.success_rate * 100:.1f}%",
            dijkstra_ratio=metrics.dijkstra_ratio,
            step_acc=metrics.step_acc,
            step_acc_pct=f"{metrics.step_acc * 100:.1f}%",
            len_ratio=metrics.len_ratio,
            n_od_pairs=metrics.n_od_pairs,
            n_steps=metrics.n_steps,
            n_train_od=metrics.n_train_od,
            n_val_od=metrics.n_val_od,
            n_train_steps=metrics.n_train_steps,
            n_val_steps=metrics.n_val_steps,
            batch_size=metrics.batch_size,
            lr=f"{cfg.lr:.0e}",
            label_smoothing=cfg.label_smoothing,
            scheduled_sampling=cfg.scheduled_sampling,
            train_time_min=metrics.train_time_s / 60,
        )
        card = _CARD_POLICY.format(**card_vars)

    (export_dir / "README.md").write_text(card)

    print(f"  Exported to {export_dir}")
    print(f"  Hub ID: {hub_id}")
    return export_dir


def _load_metrics(ckpt_path: Path, variant: str) -> PolicyMetrics | ValueMetrics:
    """Load metrics saved by the trainer alongside the checkpoint."""
    metrics_path = ckpt_path.with_suffix(".metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No metrics file at {metrics_path} — "
            f"re-run training to generate it, or pass metrics via CLI flags"
        )
    raw = json.loads(metrics_path.read_text())
    if variant == "value":
        return ValueMetrics(mae=raw["mae"])
    return PolicyMetrics(**raw)


def main():
    p = argparse.ArgumentParser(description="Export model to HF-ready directory")
    p.add_argument(
        "variant",
        choices=["policy", "value"],
        help="Which model to export",
    )
    p.add_argument("--profile", default="full")

    # Value metrics
    p.add_argument("--mae", type=float, default=None, help="Value head MAE for card")

    # Policy metrics
    p.add_argument("--success-rate", type=float, default=None)
    p.add_argument("--dijkstra-ratio", type=float, default=None)
    p.add_argument("--step-acc", type=float, default=None)
    p.add_argument("--len-ratio", type=float, default=None)
    p.add_argument("--n-od-pairs", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--n-train-od", type=int, default=None)
    p.add_argument("--n-val-od", type=int, default=None)
    p.add_argument("--n-train-steps", type=int, default=None)
    p.add_argument("--n-val-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--train-time", type=float, default=None, help="Training time in seconds"
    )

    p.add_argument(
        "--upload",
        action="store_true",
        help="Upload to Hugging Face after export",
    )
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    metrics = None
    if args.variant == "policy" and args.success_rate is not None:
        metrics = PolicyMetrics(
            success_rate=args.success_rate,
            dijkstra_ratio=args.dijkstra_ratio or 1.0,
            step_acc=args.step_acc or 0.0,
            len_ratio=args.len_ratio or 1.0,
            n_od_pairs=args.n_od_pairs or 0,
            n_steps=args.n_steps or 0,
            n_train_od=args.n_train_od or 0,
            n_val_od=args.n_val_od or 0,
            n_train_steps=args.n_train_steps or 0,
            n_val_steps=args.n_val_steps or 0,
            batch_size=args.batch_size or 0,
            train_time_s=args.train_time or 0.0,
        )
    elif args.variant == "value" and args.mae is not None:
        metrics = ValueMetrics(mae=args.mae)

    export_dir = export(args.variant, profile=args.profile, metrics=metrics)

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
