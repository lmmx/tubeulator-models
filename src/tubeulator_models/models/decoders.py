"""Three decoder heads: line-seq (A), interchange (B), station-seq (C)."""

from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["LineSeqDecoder", "InterchangeDecoder", "StationSeqDecoder"]


class LineSeqDecoder(nn.Module):
    """
    Model A — predicts [(line, direction), ...] autoregressively.
    Each GRU step emits (line_logits, dir_logits).
    """

    def __init__(self, d_model: int, n_lines: int, max_legs: int = 4):
        super().__init__()
        self.d_model = d_model
        self.max_legs = max_legs
        self.n_lines = n_lines

        self.init_proj = nn.Linear(2 * d_model, d_model)
        self.gru = nn.GRUCell(d_model, d_model)

        self.line_head = nn.Linear(d_model, n_lines)
        self.dir_head = nn.Linear(d_model, 2)

        # feedback embeddings
        self.line_emb = nn.Embedding(n_lines, d_model // 2)
        self.dir_emb = nn.Embedding(2, d_model // 2)
        self.stop_token = nn.Linear(d_model, 1)

        # learned start token — first GRU input before any predictions exist
        self.start_input = nn.Parameter(torch.zeros(1, d_model))

    def forward(
        self,
        h_origin: torch.Tensor,
        h_dest: torch.Tensor,
        labels=None,
        sampling_p: float = 0.0,
    ):
        device = h_origin.device
        B = h_origin.size(0)
        h = torch.relu(self.init_proj(torch.cat([h_origin, h_dest], dim=-1)))

        if self.training and sampling_p > 0.0:
            use_own = torch.rand(self.max_legs, device=device) < sampling_p
        else:
            use_own = torch.zeros(self.max_legs, dtype=torch.bool, device=device)

        all_line_logits = []
        all_dir_logits = []
        all_stop_logits = []

        gru_input = self.start_input.expand(B, -1)

        for step in range(self.max_legs):
            h = self.gru(gru_input, h)
            line_logits = self.line_head(h)
            dir_logits = self.dir_head(h)
            all_line_logits.append(line_logits)
            all_dir_logits.append(dir_logits)
            all_stop_logits.append(self.stop_token(h))

            own_ln = line_logits.argmax(-1)
            own_dir = dir_logits.argmax(-1)

            if labels is not None and step * 2 + 1 < labels.size(1):
                teacher_ln = labels[:, step * 2].clamp(min=0)
                teacher_dir = labels[:, step * 2 + 1].clamp(min=0)
                ln_tok = torch.where(use_own[step], own_ln, teacher_ln)
                dir_tok = torch.where(use_own[step], own_dir, teacher_dir)
            else:
                ln_tok = own_ln
                dir_tok = own_dir

            # feedback becomes next step's GRU input (not residual-added to h)
            gru_input = torch.cat(
                [self.line_emb(ln_tok), self.dir_emb(dir_tok)],
                dim=-1,
            )

        return {
            "line": torch.stack(all_line_logits, dim=1),
            "dir": torch.stack(all_dir_logits, dim=1),
            "stop": torch.stack(all_stop_logits, dim=1),
        }


class InterchangeDecoder(nn.Module):
    """
    Model B — predicts [(line, direction, station), ...] per leg.
    The station prediction is the exit/interchange station for that leg.

    Supports an optional (n_lines, n_stations) mask that constrains the
    station head to only stations served by the predicted line — same
    principle as the station model's adjacency mask.
    """

    def __init__(self, d_model: int, n_lines: int, n_stations: int, max_legs: int = 4):
        super().__init__()
        self.d_model = d_model
        self.max_legs = max_legs

        self.init_proj = nn.Linear(2 * d_model, d_model)
        self.gru = nn.GRUCell(d_model, d_model)

        self.line_head = nn.Linear(d_model, n_lines)
        self.dir_head = nn.Linear(d_model, 2)
        self.station_head = nn.Linear(d_model, n_stations)

        d_fb = d_model // 3
        self.line_emb = nn.Embedding(n_lines, d_fb)
        self.dir_emb = nn.Embedding(2, d_fb)
        self.station_emb = nn.Embedding(n_stations, d_model - 2 * d_fb)
        self.fb_proj = nn.Linear(d_model, d_model)

        self.start_input = nn.Parameter(torch.zeros(1, d_model))

        # (n_lines, n_stations) boolean mask, set via set_line_station_mask
        self.register_buffer("line_station_mask", None)

    def set_line_station_mask(self, mask: torch.Tensor) -> None:
        """Set the line→station mask. mask[l, s] = True iff station s is on line l."""
        self.line_station_mask = mask

    def forward(self, h_origin, h_dest, labels=None, sampling_p: float = 0.0):
        device = h_origin.device
        B = h_origin.size(0)
        h = torch.relu(self.init_proj(torch.cat([h_origin, h_dest], dim=-1)))

        if self.training and sampling_p > 0.0:
            use_own = torch.rand(self.max_legs, device=device) < sampling_p
        else:
            use_own = torch.zeros(self.max_legs, dtype=torch.bool, device=device)

        all_ln, all_dir, all_st = [], [], []

        gru_input = self.start_input.expand(B, -1)

        for step in range(self.max_legs):
            h = self.gru(gru_input, h)
            ln_logits = self.line_head(h)
            dir_logits = self.dir_head(h)
            st_logits = self.station_head(h)

            own_ln = ln_logits.argmax(-1)
            own_dir = dir_logits.argmax(-1)

            if labels is not None and step * 3 + 2 < labels.size(1):
                teacher_ln = labels[:, step * 3].clamp(min=0)
                teacher_dir = labels[:, step * 3 + 1].clamp(min=0)
                teacher_st = labels[:, step * 3 + 2].clamp(min=0)
                ln_tok = torch.where(use_own[step], own_ln, teacher_ln)
                dir_tok = torch.where(use_own[step], own_dir, teacher_dir)
                # mask follows teacher's line during training (same principle
                # as adjacency masking: teacher sequence is always valid, so
                # the correct station is always unmasked)
                mask_ln = teacher_ln
            else:
                ln_tok = own_ln
                dir_tok = own_dir
                mask_ln = own_ln

            # constrain station head to stations on the selected line
            if self.line_station_mask is not None:
                line_mask = self.line_station_mask[mask_ln]  # (B, n_stations)
                st_logits = st_logits.masked_fill(~line_mask, -1e4)

            all_ln.append(ln_logits)
            all_dir.append(dir_logits)
            all_st.append(st_logits)

            own_st = st_logits.argmax(-1)
            if labels is not None and step * 3 + 2 < labels.size(1):
                st_tok = torch.where(use_own[step], own_st, teacher_st)
            else:
                st_tok = own_st

            fb = torch.cat(
                [
                    self.line_emb(ln_tok),
                    self.dir_emb(dir_tok),
                    self.station_emb(st_tok),
                ],
                dim=-1,
            )
            gru_input = self.fb_proj(fb)

        return {
            "line": torch.stack(all_ln, dim=1),
            "dir": torch.stack(all_dir, dim=1),
            "station": torch.stack(all_st, dim=1),
        }


class StationSeqDecoder(nn.Module):
    """
    Model C — predicts full station sequence autoregressively.
    Uses a pointer-style mechanism with adjacency masking: at each step,
    only stations adjacent to the current station are valid candidates.
    """

    def __init__(self, d_model: int, n_stations: int, max_len: int = 40):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.n_stations = n_stations

        self.init_proj = nn.Linear(2 * d_model, d_model)
        self.gru = nn.GRUCell(d_model, d_model)
        self.station_emb = nn.Embedding(n_stations, d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

        self.start_input = nn.Parameter(torch.zeros(1, d_model))

        # (N, N) boolean mask, set via set_adj_mask before training
        self.register_buffer("adj_mask", None)

    def set_adj_mask(self, mask: torch.Tensor) -> None:
        self.adj_mask = mask

    def forward(
        self,
        h_origin,
        h_dest,
        H_all,
        origins,
        labels=None,
        sampling_p: float = 0.0,
    ):
        device = h_origin.device
        B = h_origin.size(0)
        h = torch.relu(self.init_proj(torch.cat([h_origin, h_dest], dim=-1)))

        keys = self.key_proj(H_all)  # (N, d)

        if self.training and sampling_p > 0.0:
            use_own = torch.rand(self.max_len, device=device) < sampling_p
        else:
            use_own = torch.zeros(self.max_len, dtype=torch.bool, device=device)

        current = origins  # (B,)
        all_logits = []

        gru_input = self.start_input.expand(B, -1)

        for step in range(self.max_len):
            h = self.gru(gru_input, h)
            q = self.query_proj(h)
            logits = torch.matmul(q, keys.t())

            if self.adj_mask is not None:
                mask = self.adj_mask[current]
                fill_val = -1e4
                logits = logits.masked_fill(~mask, fill_val)

            all_logits.append(logits)

            own_tok = logits.argmax(-1)

            if labels is not None and step < labels.size(1):
                teacher_tok = labels[:, step].clamp(min=0)
                tok = torch.where(use_own[step], own_tok, teacher_tok)
                current = teacher_tok  # always follow teacher for masking
            else:
                tok = own_tok
                current = tok

            gru_input = self.station_emb(tok)

        return {"station": torch.stack(all_logits, dim=1)}