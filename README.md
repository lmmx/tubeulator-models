# Tubeulator Models

Three graph neural network models of the London Underground, each predicting routes at a different granularity:

- **line** — sequence of (line, direction) pairs: "take the Jubilee westbound, then the Northern northbound"
- **change** — adds interchange stations: where exactly to transfer between lines
- **station** — full station-by-station path from origin to destination
- **nexthop** 

All three share a GATv2 encoder over the station topology graph and differ only in their decoder head.

## Quickstart
```bash
just all        # fetch data + build graph + enumerate routes + train all models
```

Or step by step:
```bash
just fetch      # pull timetables from TfL API → GTFS zip
just graph      # GTFS → GeoParquet → PyG graph objects
just routes     # enumerate routes for all origin-destination pairs
just train-all  # train line, change, and station models sequentially
```

## Training
```bash
just train change           # single model, default profile (dev)
just train station full     # single model, full profile
just train-all              # all three, default profile
just train-full             # all three, full profile
just dev                    # rebuild routes + train change model (fast iteration)
```

All hyperparameters live in `defaults.toml`. CLI flags override the TOML but never replace it:
```bash
tm-train --model line --profile full --lr 5e-5
```

Profiles control the training regime:

| Profile | Epochs | Batch size | LR | Notes |
|---------|--------|------------|----|-------|
| `dev` | 20 | 512 | 5e-4 | Fast iteration |
| `full` | 200 | 256 | 1e-4 | Production, `d_model=256`, deeper encoder |

Note: to train the value primary use

```bash
just train nexthop full --value-primary --batch-size 1024 --epochs 200
```

## Data pipeline

Each step has a CLI entry point and a corresponding justfile recipe:

| Step | CLI | Recipe | Output |
|------|-----|--------|--------|
| Fetch timetables | `tm-build-gtfs` | `just fetch` | `data/tfl_station_data_gtfs.zip` |
| Build graph | `tm-gtfs2pyg` | `just graph` | `data/graph/`, `data/pyg/` |
| Enumerate routes | `tm-build-routes` | `just routes` | `data/routes.json` |
| Plot network | `tm-plot` | `just plot` | `data/graph/network.png` |
| Train | `tm-train` | `just train` | `checkpoints/` |

Staged graph conversion is also available if you only need to re-run part of the pipeline:
```bash
uv run --group prep    tm-gtfs2graph   # GTFS → GeoParquet
uv run --group pyg     tm-graph2pyg    # GeoParquet → PyG .pt
```

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
