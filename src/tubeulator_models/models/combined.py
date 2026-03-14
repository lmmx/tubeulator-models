"""Combined encoder-decoder route model."""

from __future__ import annotations

import torch.nn as nn

from .decoders import NextHopDecoder
from .encoder import GATEncoder


__all__ = ["RouteModel"]

_DECODERS = {
    "nexthop": NextHopDecoder,
}


class RouteModel(nn.Module):
    def __init__(
        self,
        n_stations: int,
        n_lines: int,
        d_model: int,
        n_heads: int,
        n_enc_layers: int,
        model_type: str,
        max_seq: int,
        dropout: float,
        value_primary: bool = False,
    ):
        super().__init__()
        if model_type not in _DECODERS:
            raise ValueError(
                f"Unknown model_type {model_type!r}, expected one of {tuple(_DECODERS)}"
            )
        self.model_type = model_type
        self.n_stations = n_stations
        self.n_lines = n_lines

        self.encoder = GATEncoder(
            d_node=4,
            d_edge=n_lines + 1,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            dropout=dropout,
        )

        if model_type == "nexthop":
            self.decoder = NextHopDecoder(
                d_model=d_model,
                n_stations=n_stations,
                dropout=dropout,
                value_primary=value_primary,
            )

    def forward(
        self,
        graph_x,
        graph_edge_index,
        graph_edge_attr,
        origins,
        dests,
    ):
        H = self.encoder(graph_x, graph_edge_index, graph_edge_attr)
        h_o = H[origins]
        h_d = H[dests]

        if self.model_type == "nexthop":
            # origins = current station IDs, dests = destination station IDs
            return self.decoder(h_o, h_d, current_ids=origins)
