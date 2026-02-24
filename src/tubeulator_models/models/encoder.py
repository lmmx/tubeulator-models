"""Shared GATv2 encoder over the station topology graph."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


__all__ = ["GATEncoder"]


class GATEncoder(nn.Module):
    """
    Iterative weight-tied GATv2 encoder.

    Runs a single GATv2Conv layer K times with shared weights,
    giving a K-hop receptive field (Bellman-Ford-style).
    """

    def __init__(
        self,
        d_node: int = 4,
        d_edge: int = 11,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,  # now means n_iterations
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_proj = nn.Linear(d_node, d_model)
        self.n_iters = n_layers

        # Single shared conv + norm (weight-tied)
        self.conv = GATv2Conv(
            d_model,
            d_model // n_heads,
            heads=n_heads,
            edge_dim=d_edge,
            concat=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = self.node_proj(x)
        for _ in range(self.n_iters):
            h_res = h
            h = self.conv(h, edge_index, edge_attr=edge_attr)
            h = self.norm(h + h_res)
            h = torch.relu(h)
            h = self.dropout(h)
        return h
