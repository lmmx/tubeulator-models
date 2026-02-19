"""Build an enriched PyG graph with line identity + travel time edge features."""

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

    Edge features (n_lines + 1):
        [one_hot_line..., normalised_travel_time]
    """
    stations = topo.all_stations
    lines = topo.all_lines
    st2i = {s: i for i, s in enumerate(stations)}
    ln2i = {ln: i for i, ln in enumerate(lines)}
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
    if node_coords:
        x[:, :2] = (x[:, :2] - x[:, :2].mean(0)) / (x[:, :2].std(0) + 1e-8)

    # --- edges with line identity + travel time ---
    src, dst, attrs = [], [], []
    all_times = []

    # First pass: collect all travel times for normalisation
    for line in lines:
        adj = topo.line_adj.get(line, {})
        for station_a, neighbors in adj.items():
            if station_a not in st2i:
                continue
            for station_b in neighbors:
                if station_b not in st2i:
                    continue
                tt = topo.travel_time(line, station_a, station_b)
                all_times.append(tt)

    time_mean = sum(all_times) / max(len(all_times), 1)
    time_std = (
        sum((t - time_mean) ** 2 for t in all_times) / max(len(all_times), 1)
    ) ** 0.5
    time_std = max(time_std, 1e-8)

    # Second pass: build edge tensors
    for line in lines:
        li = ln2i[line]
        adj = topo.line_adj.get(line, {})
        for station_a, neighbors in adj.items():
            if station_a not in st2i:
                continue
            for station_b in neighbors:
                if station_b not in st2i:
                    continue
                src.append(st2i[station_a])
                dst.append(st2i[station_b])

                one_hot = [0.0] * L
                one_hot[li] = 1.0

                tt = topo.travel_time(line, station_a, station_b)
                norm_tt = (tt - time_mean) / time_std

                attrs.append(one_hot + [norm_tt])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
