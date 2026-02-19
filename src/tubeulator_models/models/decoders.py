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
        self.stop_token = nn.Linear(d_model, 1)  # predict whether to stop

    def forward(
        self,
        h_origin: torch.Tensor,
        h_dest: torch.Tensor,
        labels=None,
        sampling_p: float = 0.0,
    ):
        """
        h_origin, h_dest: (B, d_model)
        labels: (B, max_steps*2) flattened [line, dir, line, dir, ...] or None for inference.
        Returns dict of logits tensors.
        """
        device = h_origin.device
        h = torch.relu(self.init_proj(torch.cat([h_origin, h_dest], dim=-1)))

        # Generate mask directly on device — no CPU→GPU copy inside compiled region
        if self.training and sampling_p > 0.0:
            use_own = torch.rand(self.max_legs, device=device) < sampling_p
        else:
            use_own = torch.zeros(self.max_legs, dtype=torch.bool, device=device)

        all_line_logits = []
        all_dir_logits = []
        all_stop_logits = []

        for step in range(self.max_legs):
            h = self.gru(h, h)
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

            feedback = torch.cat(
                [
                    self.line_emb(ln_tok),
                    self.dir_emb(dir_tok),
                ],
                dim=-1,
            )
            h = h + feedback

        return {
            "line": torch.stack(all_line_logits, dim=1),
            "dir": torch.stack(all_dir_logits, dim=1),
            "stop": torch.stack(all_stop_logits, dim=1),
        }


class InterchangeDecoder(nn.Module):
    """
    Model B — predicts [(line, direction, station), ...] per leg.
    The station prediction is the exit/interchange station for that leg.
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

    def forward(self, h_origin, h_dest, labels=None, sampling_p: float = 0.0):
        device = h_origin.device
        h = torch.relu(self.init_proj(torch.cat([h_origin, h_dest], dim=-1)))

        if self.training and sampling_p > 0.0:
            use_own = torch.rand(self.max_legs, device=device) < sampling_p
        else:
            use_own = torch.zeros(self.max_legs, dtype=torch.bool, device=device)

        all_ln, all_dir, all_st = [], [], []

        for step in range(self.max_legs):
            h = self.gru(h, h)
            ln_logits = self.line_head(h)
            dir_logits = self.dir_head(h)
            st_logits = self.station_head(h)
            all_ln.append(ln_logits)
            all_dir.append(dir_logits)
            all_st.append(st_logits)

            own_ln = ln_logits.argmax(-1)
            own_dir = dir_logits.argmax(-1)
            own_st = st_logits.argmax(-1)

            if labels is not None and step * 3 + 2 < labels.size(1):
                teacher_ln = labels[:, step * 3].clamp(min=0)
                teacher_dir = labels[:, step * 3 + 1].clamp(min=0)
                teacher_st = labels[:, step * 3 + 2].clamp(min=0)
                ln_tok = torch.where(use_own[step], own_ln, teacher_ln)
                dir_tok = torch.where(use_own[step], own_dir, teacher_dir)
                st_tok = torch.where(use_own[step], own_st, teacher_st)
            else:
                ln_tok = own_ln
                dir_tok = own_dir
                st_tok = own_st

            fb = torch.cat(
                [
                    self.line_emb(ln_tok),
                    self.dir_emb(dir_tok),
                    self.station_emb(st_tok),
                ],
                dim=-1,
            )
            h = h + self.fb_proj(fb)

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
        h = torch.relu(self.init_proj(torch.cat([h_origin, h_dest], dim=-1)))

        keys = self.key_proj(H_all)  # (N, d)

        if self.training and sampling_p > 0.0:
            use_own = torch.rand(self.max_len, device=device) < sampling_p
        else:
            use_own = torch.zeros(self.max_len, dtype=torch.bool, device=device)

        current = origins  # (B,)
        all_logits = []

        for step in range(self.max_len):
            h = self.gru(h, h)
            q = self.query_proj(h)
            logits = torch.matmul(q, keys.t())

            if self.adj_mask is not None:
                mask = self.adj_mask[current]
                logits = logits.masked_fill(~mask, float("-inf"))

            all_logits.append(logits)

            own_tok = logits.argmax(-1)

            if labels is not None and step < labels.size(1):
                teacher_tok = labels[:, step].clamp(min=0)
                tok = torch.where(use_own[step], own_tok, teacher_tok)
                current = teacher_tok  # always follow teacher for masking
            else:
                tok = own_tok
                current = tok

            h = h + self.station_emb(tok)

        return {"station": torch.stack(all_logits, dim=1)}
