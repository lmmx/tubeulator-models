"""CLI entrypoints for the tubeulator-models pipeline."""

from __future__ import annotations

from pathlib import Path

from .defaults import repo_root, resolve_data, resolve_plot


__all__ = [
    "build_gtfs",
    "build_routes",
    "plot_graph",
    "gtfs2graph",
    "graph2pyg",
    "gtfs2pyg",
]


def _graph_paths() -> tuple[Path, Path]:
    cfg = resolve_data()
    base = repo_root() / cfg["graph_dir"]
    base.mkdir(parents=True, exist_ok=True)
    return base / "nodes.parquet", base / "edges.parquet"


def _pyg_path() -> Path:
    cfg = resolve_data()
    base = repo_root() / cfg["pyg_dir"]
    base.mkdir(parents=True, exist_ok=True)
    return base / "london_transit.pt"


def sync_timetables() -> None:
    """Download and extract PDF timetables for lines without API support."""
    from .pipeline.timetable_pdf import sync_all

    sync_all()


def build_gtfs() -> None:
    """Fetch timetables from TfL API and write a GTFS zip."""
    from .pipeline.gtfs_builder import build_gtfs as _build

    cfg = resolve_data()
    out = repo_root() / cfg["gtfs_path"]
    print(f"Building GTFS from TfL API → {out}")
    _build(out)


def gtfs2graph() -> None:
    """Load GTFS → build travel-summary graph → filter to London → save GeoParquet."""
    from .pipeline.gtfs import load_london_graph

    print("Loading GTFS and building London travel graph…")
    nodes, edges = load_london_graph()

    nodes_path, edges_path = _graph_paths()
    nodes.to_parquet(nodes_path)
    edges.reset_index().to_parquet(edges_path)
    print(f"Saved {len(nodes):,} nodes → {nodes_path}")
    print(f"Saved {len(edges):,} edges → {edges_path}")


def graph2pyg() -> None:
    """Load saved GeoParquet → convert to PyG Data → save .pt file."""
    import geopandas as gpd
    import torch

    from .pipeline.convert import gdf_to_pyg

    nodes_path, edges_path = _graph_paths()
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f"Graph parquet files not found in {nodes_path.parent}. "
            "Run `tm-gtfs2graph` first."
        )

    print("Loading GeoParquet…")
    nodes = gpd.read_parquet(nodes_path)
    edges = gpd.read_parquet(edges_path).set_index(["from_stop_id", "to_stop_id"])

    print("Converting to PyG Data…")
    data = gdf_to_pyg(nodes, edges)

    out = _pyg_path()
    torch.save(data, out)
    print(f"Saved PyG Data → {out}")
    print(f"  nodes : {data.num_nodes:,}  (features: {data.x.shape[1]})")
    print(f"  edges : {data.num_edges:,}  (features: {data.edge_attr.shape[1]})")


def gtfs2pyg() -> None:
    """Full pipeline: GTFS → GeoParquet → PyG .pt  (requires both dep groups)."""
    gtfs2graph()
    graph2pyg()


def build_routes() -> None:
    """Enumerate routes for all OD pairs and save training data."""
    from .config import TrainConfig
    from .training.routes import build_dataset
    from .graph.topology import extract

    cfg = TrainConfig.from_defaults()
    topo = extract(cfg.gtfs_path)
    print(f"Topology: {topo.n_stations} stations, {topo.n_lines} lines")
    build_dataset(
        topo,
        max_transfers=cfg.max_transfers,
        max_results=cfg.max_routes_per_od,
        transfer_penalty=cfg.transfer_penalty,
        output_path=cfg.routes_path,
    )


def plot_graph() -> None:
    """Visualise the saved transit graph."""
    from .viz.visualise import plot_transit_graph

    nodes_path, edges_path = _graph_paths()
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError("Run tm-gtfs2graph first.")

    cfg = resolve_plot()
    out = repo_root() / cfg["output"]
    plot_transit_graph(nodes_path, edges_path, output_path=out)
