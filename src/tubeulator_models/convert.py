"""Convert travel-graph GeoDataFrames to PyTorch Geometric Data objects."""

from __future__ import annotations

import geopandas as gpd
import torch
from torch_geometric.data import Data

__all__ = ["gdf_to_pyg"]


def gdf_to_pyg(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> Data:
    """
    Build a PyG Data object from the travel-summary GeoDataFrames.

    Node features (x):  [easting, northing]  — BNG metres, already projected
    Edge features:      [travel_time_sec, frequency]
    Edge index:         from/to stop positions in the node index
    """
    # --- node features ---
    coords = nodes.geometry.apply(lambda g: [g.x, g.y]).tolist()
    x = torch.tensor(coords, dtype=torch.float)

    # positional lookup: stop_id → integer position
    stop_to_idx = {stop_id: i for i, stop_id in enumerate(nodes.index)}

    # --- edge index ---
    from_ids = edges.index.get_level_values("from_stop_id")
    to_ids = edges.index.get_level_values("to_stop_id")

    # drop any edges whose endpoints fell outside the node set (safety)
    mask = [f in stop_to_idx and t in stop_to_idx for f, t in zip(from_ids, to_ids)]
    edges = edges[mask]
    from_ids = from_ids[mask]
    to_ids = to_ids[mask]

    src = [stop_to_idx[s] for s in from_ids]
    dst = [stop_to_idx[s] for s in to_ids]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # --- edge features ---
    edge_attr = torch.tensor(
        edges[["travel_time_sec", "frequency"]].values,
        dtype=torch.float,
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
