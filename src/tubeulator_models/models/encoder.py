"""Shared GATv2 encoder over the station topology graph."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


__all__ = ["GATEncoder"]


class GATEncoder(nn.Module):
    """
    Multi-layer GATv2 encoder.

    Node features:  [norm_easting, norm_northing, n_lines, is_interchange]  (4-dim)
    Edge features:  one-hot line id  (n_lines-dim)

    Produces H ∈ R^{N × d_model}.
    """

    def __init__(
        self,
        d_node: int = 4,
        d_edge: int = 11,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_proj = nn.Linear(d_node, d_model)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                GATv2Conv(
                    d_model,
                    d_model // n_heads,
                    heads=n_heads,
                    edge_dim=d_edge,
                    concat=True,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = self.node_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(h + h_res)  # residual
            h = torch.relu(h)
            h = self.dropout(h)
        return h
