"""Visualise the London transit graph from saved GeoParquet."""

from __future__ import annotations

from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import city2graph as c2g

__all__ = ["plot_transit_graph"]


def plot_transit_graph(
    nodes_path: Path,
    edges_path: Path,
    output_path: Path | None = None,
) -> None:
    nodes = gpd.read_parquet(nodes_path)
    edges = gpd.read_parquet(edges_path).set_index(["from_stop_id", "to_stop_id"])

    print(f"Plotting {len(nodes):,} nodes, {len(edges):,} edges...")

    ax = c2g.plot_graph(
        nodes=nodes,
        edges=edges,
        edge_color="travel_time_sec",
        edge_linewidth=edges["frequency"] / edges["frequency"].max() * 3,
        edge_alpha=0.8,
    )
    ctx.add_basemap(ax, crs=nodes.crs, source=ctx.providers.CartoDB.DarkMatter)

    ax.set_title("London Tube Network — travel time (colour) & frequency (width)")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {output_path}")
    else:
        plt.show()