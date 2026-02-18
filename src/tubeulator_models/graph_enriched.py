"""Build an enriched PyG graph with line-identity edge features."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from .topology import Topology


__all__ = ["build_enriched_graph"]


def build_enriched_graph(
    topo: Topology,
    node_coords: dict[str, tuple[float, float]] | None = None,
) -> Data:
    """
    Build a PyG Data object from the topology.

    Node features (4):
        [norm_x, norm_y, n_lines_serving, is_interchange]

    Edge features (n_lines):
        one-hot line identity

    If node_coords is None, uses index-based placeholder coords
    (fine for training — the GNN gets structural info from topology anyway).
    """
    stations = topo.all_stations
    lines = topo.all_lines
    st2i = {s: i for i, s in enumerate(stations)}
    ln2i = {ln: i for i, ln in enumerate(lines)}
    # N = len(stations)
    L = len(lines)

    # --- node features ---
    feats = []
    for s in stations:
        n_lines = len(topo.station_lines.get(s, set()))
        is_ix = 1.0 if s in topo.interchanges else 0.0
        if node_coords and s in node_coords:
            x, y = node_coords[s]
        else:
            x, y = 0.0, 0.0
        feats.append([x, y, float(n_lines), is_ix])

    x = torch.tensor(feats, dtype=torch.float)
    # normalise coords if they're real
    if node_coords:
        x[:, :2] = (x[:, :2] - x[:, :2].mean(0)) / (x[:, :2].std(0) + 1e-8)

    # --- edges with line identity ---
    src, dst, attrs = [], [], []
    for line in lines:
        li = ln2i[line]
        one_hot = [0.0] * L
        one_hot[li] = 1.0
        adj = topo.line_adj.get(line, {})
        for station_a, neighbors in adj.items():
            if station_a not in st2i:
                continue
            for station_b in neighbors:
                if station_b not in st2i:
                    continue
                src.append(st2i[station_a])
                dst.append(st2i[station_b])
                attrs.append(one_hot)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
