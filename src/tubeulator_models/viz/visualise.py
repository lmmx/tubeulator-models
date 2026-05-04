"""Visualise the London transit graph from saved GeoParquet."""

from __future__ import annotations

from pathlib import Path

import city2graph as c2g
import contextily as ctx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.text import Text
from torch_geometric.utils import to_undirected

from ..defaults import resolve_plot
from ..graph.resistance import (
    edge_curvature,
    effective_resistance,
    node_curvature,
)


__all__ = ["plot_transit_graph"]


def plot_transit_graph(
    nodes_path: Path,
    edges_path: Path,
    output_path: Path | None = None,
) -> None:
    cfg = resolve_plot()

    nodes = gpd.read_parquet(nodes_path)
    edges = gpd.read_parquet(edges_path).set_index(["from_stop_id", "to_stop_id"])

    # --- build PyG-like tensors ---
    node_map = {nid: i for i, nid in enumerate(nodes.index)}

    src = torch.tensor(
        [node_map[f] for f in edges.index.get_level_values("from_stop_id")],
        dtype=torch.long,
    )
    dst = torch.tensor(
        [node_map[t] for t in edges.index.get_level_values("to_stop_id")],
        dtype=torch.long,
    )

    edge_index = torch.stack([src, dst], dim=0)
    edge_weight = 1.0 / torch.clamp(
        torch.tensor(edges["travel_time_sec"].values, dtype=torch.float),
        min=1.0,
    )

    # --- CLEANUP
    # 1. Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    edge_weight = edge_weight[mask]

    # 2. Symmetrize: add reverse edges, keep min weight (max conductance) for duplicates
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="max")

    # 3. Verify
    print(
        f"After cleanup: {edge_index.size(1)} edges, "
        f"{(edge_index[0] == edge_index[1]).sum().item()} self-loops"
    )
    # --- FIN CLEANUP

    # --- resistance geometry ---
    R = effective_resistance(edge_index, edge_weight)

    ncurv = node_curvature(edge_index, edge_weight, R)
    ecurv = edge_curvature(edge_index, ncurv, R)

    # --- CLEANUP ROUND 2
    # Build lookup from (src, dst) → curvature value
    curv_lookup = {}
    res_lookup = {}
    for k in range(edge_index.size(1)):
        pair = (edge_index[0, k].item(), edge_index[1, k].item())
        curv_lookup[pair] = ecurv[k].item()
        res_lookup[pair] = R[k].item()

    # Map back to original edges DataFrame
    edges = edges.copy()
    nodes = nodes.copy()

    orig_src = [node_map[f] for f in edges.index.get_level_values("from_stop_id")]
    orig_dst = [node_map[t] for t in edges.index.get_level_values("to_stop_id")]

    edges["curvature"] = [
        curv_lookup.get((s, d), curv_lookup.get((d, s), 0.0))
        for s, d in zip(orig_src, orig_dst)
    ]
    nodes["curvature"] = ncurv.cpu().numpy()
    # --- FIN CLEANUP DEUX

    # --- debug: lowest / highest curvature stations ---
    console = Console()

    k = 20
    nodes_debug = nodes.copy()

    name_col = (
        "stop_name"
        if "stop_name" in nodes_debug.columns
        else ("name" if "name" in nodes_debug.columns else None)
    )

    lowest = nodes_debug.sort_values("curvature", ascending=True).head(k)
    highest = nodes_debug.sort_values("curvature", ascending=False).head(k)

    console.print("\n[bold]Lowest curvature stations (most connected):[/bold]")
    for i, row in lowest.iterrows():
        label = row[name_col] if name_col else str(i)
        console.print(Text(f"{label}: {row['curvature']:.6f}", style="magenta1"))

    console.print("\n[bold]Highest curvature stations (most bottlenecked):[/bold]")
    for i, row in highest.iterrows():
        label = row[name_col] if name_col else str(i)
        console.print(Text(f"{label}: {row['curvature']:.6f}", style="bright_yellow"))
    # Should equal the number of connected components = 91.0
    # print(f"sum(p) = {ncurv.sum().item():.2f} (should be close to #CC)")
    # --- FIN DEBUG

    # --- diagnostics ---
    def stats(x, name):
        x = x[torch.isfinite(x)]
        print(f"\n{name}")
        print(f"min:  {x.min().item():.6f}")
        print(f"max:  {x.max().item():.6f}")
        print(f"mean: {x.mean().item():.6f}")
        print(f"std:  {x.std().item():.6f}")

    stats(ncurv, "node_curvature")
    stats(ecurv, "edge_curvature")
    stats(R, "effective_resistance")

    print("\nnode curvature range:", (ncurv.max() - ncurv.min()).item())
    print("edge curvature range:", (ecurv.max() - ecurv.min()).item())

    print(f"Plotting {len(nodes):,} nodes, {len(edges):,} edges...")

    cmap_name = "spring"

    ax = c2g.plot_graph(
        figsize=(20, 20),
        nodes=nodes,
        edges=edges,
        node_color="curvature",
        edge_color="curvature",
        edge_linewidth=(edges["frequency"] / edges["frequency"].max() * 3),
        edge_alpha=0.2,
        node_alpha=1.0,
        markersize=30.0,
        cmap=cmap_name,
    )

    sm = mpl.cm.ScalarMappable(
        cmap=mpl.colormaps[cmap_name],
        norm=mpl.colors.Normalize(
            vmin=min(edges["curvature"].min(), nodes["curvature"].min()),
            vmax=max(edges["curvature"].max(), nodes["curvature"].max()),
        ),
    )

    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("Resistance curvature (higher = more bottlenecked)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")

    ctx.add_basemap(
        ax,
        crs=nodes.crs,
        source=ctx.providers.CartoDB.DarkMatter,
    )

    ax.set_title(cfg["title"])

    if output_path:
        plt.savefig(output_path, dpi=cfg["dpi"], bbox_inches="tight")
        print(f"Saved → {output_path}")
    else:
        plt.show()
