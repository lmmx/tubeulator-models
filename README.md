# Tubeulator Models

Graph neural network models of the London Underground trained on TfL timetable data.

Two models share a GATv2 encoder over the station topology graph and differ in their decoder head:

- **nexthop policy** — given a current station and destination, predicts the optimal next station to visit. Greedy rollout achieves 1.09× Dijkstra ratio with 100% success across all 239,610 origin–destination pairs.
- **nexthop value** — predicts shortest travel time in minutes between any pair of stations. MAE of ~0.5 min vs. Floyd–Warshall ground truth.

## Installation
```bash
pip install tubeulator-models[inference]
```

For the CLI:
```bash
pip install tubeulator-models[cli]
```

## Usage

### Python API
```python
from tubeulator_models import TubeRouter

router = TubeRouter.from_pretrained("permutans/tube-nexthop-policy")
route = router.route("West Ham", "Shoreditch")
print(route)
# West Ham
#   → [district] Bromley-by-Bow (2.0m)
#   ...
# ✓ 8 hops · 2 lines · 1 transfer · 16.0 min
```

Route through waypoints:
```python
route = router.route("Camden Town", "Canary Wharf", via=["King's Cross"])
```

The `Route` object has `.steps`, `.total_minutes`, `.lines_used`, `.n_transfers`, `.success`, and a `.to_dict()` method.

### CLI
```bash
tm-infer policy --model permutans/tube-nexthop-policy -o "West Ham" -d Shoreditch
tm-infer policy --model permutans/tube-nexthop-policy -o "Camden Town" -d "Canary Wharf" -v "King's Cross"
tm-infer value  --model permutans/tube-nexthop-value  -o "West Ham" -d Shoreditch
```

## Training from source

Clone the repo and use the justfile recipes:
```bash
just all        # fetch data + build graph + train
```

Or step by step:
```bash
just fetch      # pull timetables from TfL API → GTFS zip
just graph      # GTFS → GeoParquet → PyG graph objects
just train nexthop full
just train nexthop full --value-primary --batch-size 1024 --epochs 200
```

All hyperparameters live in `defaults.toml`. CLI flags override the TOML but never replace it:
```bash
tm-train --model nexthop --profile full --lr 5e-5
```

Profiles control the training regime:

| Profile | Epochs | Batch size | LR | Notes |
|---------|--------|------------|----|-------|
| `dev` | 20 | 512 | 5e-4 | Fast iteration |
| `full` | 200 | 256 | 1e-4 | Production, `d_model=512`, 16-layer encoder |

## Data pipeline

| Step | CLI | Recipe | Output |
|------|-----|--------|--------|
| Fetch timetables | `tm-build-gtfs` | `just fetch` | `data/tfl_station_data_gtfs.zip` |
| Build graph | `tm-gtfs2pyg` | `just graph` | `data/graph/`, `data/pyg/` |
| Plot network | `tm-plot` | `just plot` | `data/graph/network.png` |
| Train | `tm-train` | `just train` | `checkpoints/` |
| Export | `tm-export` | `just upload` | `models/` → HuggingFace |

Staged graph conversion is also available:
```bash
uv run --group prep    tm-gtfs2graph   # GTFS → GeoParquet
uv run --group pyg     tm-graph2pyg    # GeoParquet → PyG .pt
```

## Graph structure visualisation

The Underground is modelled as a weighted spatial graph:

- nodes = stations
- edges = directed travel segments (GTFS-derived)
- weights = normalised travel times + transfer structure

Besides using it for shortest-path routing, we can also visualise this graph's
structural properties relevant to routing through
a spectral measure derived from the graph Laplacian pseudoinverse,
specifically effective resistance curvature.

This measure provides a *geometry of redundancy*, as a scalar value for each station,
as described by [Devriendt and Lambiotte][dev22] (2022)
_Discrete curvature on graphs from the effective resistance_:

[dev22]: https://arxiv.org/abs/2201.06385

- low values mean a station is structurally redundant, having multiple routing alternatives
- high values indicate structurally constrained regions with limited path diversity (the "bottlenecks")

<img src="https://raw.githubusercontent.com/lmmx/tubeulator-models/master/data/graph/network_annotated.png" />

## Configuration

`defaults.toml` is the single source of truth for all tuneable values. The merge order is:
```
[base] → [model.<type>] → [profiles.<name>] → [profiles.<name>.model.<type>] → CLI overrides
```

## Development
```bash
just lint       # ruff check
just clean      # remove data/ and checkpoints/
```

## Links

- [Policy model on HuggingFace](https://huggingface.co/permutans/tube-nexthop-policy)
- [Value model on HuggingFace](https://huggingface.co/permutans/tube-nexthop-value)
- [tubeulator-models on PyPI](https://pypi.org/project/tubeulator-models/)
