"""Decoder head."""

from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["NextHopDecoder"]


class NextHopDecoder(nn.Module):
    """
    Next-hop policy decoder with optional value head.

    Policy: (h_current, h_dest) → logits over adjacent stations
    Value:  (h_current, h_dest) → predicted remaining travel time to destination

    When value_primary=True, the value head is the main output and gets
    significantly more capacity. The policy MLP is retained but not
    trained — inference uses Bellman rollout instead.
    """

    MASK_VALUE = -1e4

    def __init__(
        self,
        d_model: int,
        n_stations: int,
        dropout: float = 0.1,
        value_primary: bool = False,
    ):
        super().__init__()
        self.n_stations = n_stations
        self.value_primary = value_primary

        # Policy MLP — present in both modes for checkpoint compat,
        # but only trained when value_primary=False
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_stations),
        )

        if value_primary:
            # 4-layer MLP with more capacity for learning the full distance field
            self.value_head = nn.Sequential(
                nn.Linear(2 * d_model, 2 * d_model),
                nn.ReLU(),
                nn.LayerNorm(2 * d_model),
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
            )
        else:
            self.value_head = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        self.register_buffer("adj_mask", None)

    def set_adj_mask(self, mask: torch.Tensor) -> None:
        self.adj_mask = mask

    def forward(
        self,
        h_current: torch.Tensor,  # (B, d)
        h_dest: torch.Tensor,  # (B, d)
        current_ids: torch.Tensor | None = None,  # (B,) station indices for adj masking
    ) -> dict[str, torch.Tensor]:
        combined = torch.cat([h_current, h_dest], dim=-1)
        value = self.value_head(combined).squeeze(-1)  # (B,)

        if current_ids is not None:
            logits = self.mlp(combined)
            if self.adj_mask is not None:
                mask = self.adj_mask[current_ids]
                logits = logits.masked_fill(~mask, self.MASK_VALUE)
        else:
            # Value-only forward — no policy logits needed
            logits = torch.zeros(
                h_current.size(0), self.n_stations, device=h_current.device
            )

        return {"next_station": logits, "value": value}
