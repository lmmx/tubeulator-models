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
        # B = h_origin.size(0)
        h = torch.relu(self.init_proj(torch.cat([h_origin, h_dest], dim=-1)))

        all_line_logits = []
        all_dir_logits = []
        all_stop_logits = []

        for step in range(self.max_legs):
            h = self.gru(h, h)
            all_line_logits.append(self.line_head(h))
            all_dir_logits.append(self.dir_head(h))
            all_stop_logits.append(self.stop_token(h))

            use_own = self.training and torch.rand(1).item() < sampling_p

            if labels is not None and step * 2 + 1 < labels.size(1) and not use_own:
                ln_tok = labels[:, step * 2].clamp(min=0)
                dir_tok = labels[:, step * 2 + 1].clamp(min=0)
            else:
                ln_tok = all_line_logits[-1].argmax(-1)
                dir_tok = all_dir_logits[-1].argmax(-1)

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
        # B = h_origin.size(0)
        h = torch.relu(self.init_proj(torch.cat([h_origin, h_dest], dim=-1)))

        all_ln, all_dir, all_st = [], [], []

        for step in range(self.max_legs):
            h = self.gru(h, h)
            all_ln.append(self.line_head(h))
            all_dir.append(self.dir_head(h))
            all_st.append(self.station_head(h))

            use_own = self.training and torch.rand(1).item() < sampling_p

            if labels is not None and step * 3 + 2 < labels.size(1) and not use_own:
                ln_tok = labels[:, step * 3].clamp(min=0)
                dir_tok = labels[:, step * 3 + 1].clamp(min=0)
                st_tok = labels[:, step * 3 + 2].clamp(min=0)
            else:
                ln_tok = all_ln[-1].argmax(-1)
                dir_tok = all_dir[-1].argmax(-1)
                st_tok = all_st[-1].argmax(-1)

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
    Uses a pointer-style mechanism: scores over all station embeddings H.
    """

    def __init__(self, d_model: int, n_stations: int, max_len: int = 40):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

        self.init_proj = nn.Linear(2 * d_model, d_model)
        self.gru = nn.GRUCell(d_model, d_model)
        self.station_emb = nn.Embedding(n_stations, d_model)

        # pointer attention: query from GRU, keys from encoder H
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

    def forward(self, h_origin, h_dest, H_all, labels=None, sampling_p: float = 0.0):
        """
        H_all: (N, d_model) — all node embeddings from encoder (shared across batch).
        We compute pointer logits over these.
        """
        # B = h_origin.size(0)
        h = torch.relu(self.init_proj(torch.cat([h_origin, h_dest], dim=-1)))

        keys = self.key_proj(H_all)  # (N, d)

        all_logits = []
        for step in range(self.max_len):
            h = self.gru(h, h)
            q = self.query_proj(h)  # (B, d)
            # pointer scores
            logits = torch.matmul(q, keys.t())  # (B, N)
            all_logits.append(logits)

            use_own = self.training and torch.rand(1).item() < sampling_p

            if labels is not None and step < labels.size(1) and not use_own:
                tok = labels[:, step].clamp(min=0)
            else:
                tok = logits.argmax(-1)

            h = h + self.station_emb(tok)

        return {"station": torch.stack(all_logits, dim=1)}  # (B, max_len, N)
