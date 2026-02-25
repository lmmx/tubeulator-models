"""Combined encoder-decoder route model."""

from __future__ import annotations

import torch.nn as nn

from .decoders import (
    HybridStationDecoder,
    InterchangeDecoder,
    LineSeqDecoder,
    NextHopDecoder,
    StationSeqDecoder,
    TransformerStationDecoder,
)
from .encoder import GATEncoder


__all__ = ["RouteModel"]

_DECODERS = {
    "line": LineSeqDecoder,
    "change": InterchangeDecoder,
    "station": HybridStationDecoder,
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
        n_dec_layers: int = 4,
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

        if model_type == "line":
            self.decoder = LineSeqDecoder(d_model, n_lines, max_legs=max_seq)
        elif model_type == "change":
            self.decoder = InterchangeDecoder(
                d_model,
                n_lines,
                n_stations,
                max_legs=max_seq,
            )
        elif model_type == "nexthop":
            self.decoder = NextHopDecoder(
                d_model=d_model,
                n_stations=n_stations,
                dropout=dropout,
                value_primary=value_primary,
            )
        elif model_type == "station":
            decoder_cls = _DECODERS["station"]
            if decoder_cls is HybridStationDecoder:
                self.decoder = HybridStationDecoder(
                    d_model=d_model,
                    n_stations=n_stations,
                    max_len=max_seq,
                    n_heads=n_heads,
                    dropout=dropout,
                    # window_size=cfg.window_size if hasattr(cfg, "window_size") else 0,
                    window_size=12,
                )
            elif decoder_cls is TransformerStationDecoder:
                self.decoder = TransformerStationDecoder(
                    d_model=d_model,
                    n_stations=n_stations,
                    max_len=max_seq,
                    n_heads=n_heads,
                    n_dec_layers=n_dec_layers,
                    dropout=dropout,
                )
            else:
                # GRU fallback
                self.decoder = StationSeqDecoder(
                    d_model,
                    n_stations,
                    max_len=max_seq,
                )

    def forward(
        self,
        graph_x,
        graph_edge_index,
        graph_edge_attr,
        origins,
        dests,
        labels=None,
        sampling_p: float = 0.0,
    ):
        H = self.encoder(graph_x, graph_edge_index, graph_edge_attr)
        h_o = H[origins]
        h_d = H[dests]

        if self.model_type == "nexthop":
            # origins = current station IDs, dests = destination station IDs
            return self.decoder(h_o, h_d, current_ids=origins)

        if self.model_type == "station":
            return self.decoder(
                h_o,
                h_d,
                H,
                origins,
                dests,
                labels=labels,
                sampling_p=sampling_p,
            )
        else:
            return self.decoder(h_o, h_d, labels=labels, sampling_p=sampling_p)
