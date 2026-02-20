"""Decoder heads: line-seq (A), interchange (B), station-seq GRU (C), station-seq Transformer (D)."""

from __future__ import annotations

import torch
import torch.nn as nn


__all__ = [
    "LineSeqDecoder",
    "InterchangeDecoder",
    "StationSeqDecoder",
    "TransformerStationDecoder",
]


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
                mask_ln = teacher_ln
            else:
                ln_tok = own_ln
                dir_tok = own_dir
                mask_ln = own_ln

            if self.line_station_mask is not None:
                line_mask = self.line_station_mask[mask_ln]
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
    Model C — GRU pointer decoder for full station sequence.
    Kept for comparison; TransformerStationDecoder is the replacement.
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

        self.register_buffer("adj_mask", None)

    def set_adj_mask(self, mask: torch.Tensor) -> None:
        self.adj_mask = mask

    def forward(
        self,
        h_origin,
        h_dest,
        H_all,
        origins,
        dests=None,
        labels=None,
        sampling_p: float = 0.0,
    ):
        device = h_origin.device
        B = h_origin.size(0)
        h = torch.relu(self.init_proj(torch.cat([h_origin, h_dest], dim=-1)))

        keys = self.key_proj(H_all)

        if self.training and sampling_p > 0.0:
            use_own = torch.rand(self.max_len, device=device) < sampling_p
        else:
            use_own = torch.zeros(self.max_len, dtype=torch.bool, device=device)

        current = origins
        all_logits = []

        gru_input = self.start_input.expand(B, -1)

        for step in range(self.max_len):
            h = self.gru(gru_input, h)
            q = self.query_proj(h)
            logits = torch.matmul(q, keys.t())

            if self.adj_mask is not None:
                mask = self.adj_mask[current]
                logits = logits.masked_fill(~mask, -1e4)

            all_logits.append(logits)

            own_tok = logits.argmax(-1)

            if labels is not None and step < labels.size(1):
                teacher_tok = labels[:, step].clamp(min=0)
                tok = torch.where(use_own[step], own_tok, teacher_tok)
                current = teacher_tok
            else:
                tok = own_tok
                current = tok

            gru_input = self.station_emb(tok)

        return {"station": torch.stack(all_logits, dim=1)}


class TransformerStationDecoder(nn.Module):
    """
    Model D — Transformer decoder for full station sequence prediction.

    Replaces the GRU pointer decoder.  Cross-attends to the full set of
    GATv2 encoder station embeddings; self-attends over the autoregressive
    station sequence with causal masking.  Adjacency mask constrains output
    logits identically to the GRU version.
    """

    MASK_VALUE = -1e4

    def __init__(
        self,
        d_model: int,
        n_stations: int,
        max_len: int = 50,
        n_heads: int = 8,
        n_dec_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_stations = n_stations
        self.max_len = max_len

        # Token, positional, and destination embeddings
        self.station_emb = nn.Embedding(n_stations, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dest_proj = nn.Linear(d_model, d_model, bias=False)

        # Transformer decoder: cross-attends to encoder graph embeddings
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.tf_decoder = nn.TransformerDecoder(layer, num_layers=n_dec_layers)

        # Output projection to station vocabulary
        self.out_proj = nn.Linear(d_model, n_stations)

        # (N, N) boolean adjacency mask, set via set_adj_mask before training
        self.register_buffer("adj_mask", None)

    def set_adj_mask(self, adj: torch.Tensor) -> None:
        self.adj_mask = adj

    # ── internal helpers ──────────────────────────────────────

    def _embed_tokens(
        self,
        tokens: torch.Tensor,  # (B, T) station indices
        h_dest: torch.Tensor,  # (B, d) destination embedding from encoder
    ) -> torch.Tensor:
        """Station embedding + positional embedding + destination bias → (B, T, d)."""
        B, T = tokens.shape
        tok = self.station_emb(tokens)  # (B, T, d)
        pos = self.pos_emb(torch.arange(T, device=tokens.device))  # (T, d)
        dst = self.dest_proj(h_dest)  # (B, d)
        return tok + pos.unsqueeze(0) + dst.unsqueeze(1)

    def _causal_mask(
        self, T: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """(T, T) upper-triangular mask for causal self-attention."""
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        return mask.to(dtype=dtype)

    def _apply_adj_mask(
        self,
        logits: torch.Tensor,  # (B, T, V) or (B, V)
        current_stations: torch.Tensor,  # (B, T) or (B,) station indices
    ) -> torch.Tensor:
        if self.adj_mask is None:
            return logits
        adj_rows = self.adj_mask[current_stations]  # same shape as logits
        return logits.masked_fill(~adj_rows, self.MASK_VALUE)

    # ── forward ───────────────────────────────────────────────

    def forward(
        self,
        h_origin: torch.Tensor,
        h_dest: torch.Tensor,
        H_all: torch.Tensor,
        origins: torch.Tensor,
        dests: torch.Tensor,
        labels: torch.Tensor | None = None,
        sampling_p: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if labels is not None:
            if self.training and sampling_p > 0.0:
                return self._forward_scheduled(
                    h_dest, H_all, origins, labels, sampling_p
                )
            return self._forward_teacher_forced(h_dest, H_all, origins, labels)
        return self._forward_greedy(h_dest, H_all, origins, dests)

    def _forward_scheduled(
        self,
        h_dest: torch.Tensor,
        H_all: torch.Tensor,
        origins: torch.Tensor,
        labels: torch.Tensor,
        sampling_p: float,
    ) -> dict[str, torch.Tensor]:
        """Teacher forcing with scheduled sampling — step-by-step.

        At each step, with probability sampling_p, use the model's own
        prediction instead of the teacher token.  This forces the model
        to learn recovery from its own errors and maintain a diverse
        output distribution (critical for beam search).
        """
        if labels.size(1) > self.max_len:
            labels = labels[:, : self.max_len]

        B, T = labels.shape
        device = labels.device
        memory = H_all.unsqueeze(0).expand(B, -1, -1)

        # Build input sequence token by token, mixing teacher and own predictions
        use_own = torch.rand(T, device=device) < sampling_p
        # First token is always the origin (teacher), never sampled
        use_own[0] = False

        tokens = origins.unsqueeze(1)  # (B, 1) — start with origin
        all_logits = []

        for step in range(T):
            tgt = self._embed_tokens(tokens, h_dest)
            causal = self._causal_mask(tokens.size(1), device, tgt.dtype)

            out = self.tf_decoder(tgt, memory, tgt_mask=causal)
            step_logits = self.out_proj(out[:, -1, :])  # (B, V)

            # Mask by the token we're "at" (teacher position for masking,
            # same principle as the GRU: mask always follows ground truth)
            if step < T:
                mask_station = labels[:, step].clamp(min=0)
            else:
                mask_station = tokens[:, -1]
            step_logits = self._apply_adj_mask(step_logits, mask_station)

            all_logits.append(step_logits)

            # Next input token: teacher or own prediction
            if step < T - 1:
                own_tok = step_logits.argmax(-1)  # (B,)
                teacher_tok = labels[:, step]  # (B,)
                if use_own[step + 1]:
                    next_tok = own_tok
                else:
                    next_tok = teacher_tok.clamp(min=0)
                tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)

        logits = torch.stack(all_logits, dim=1)  # (B, T, V)
        return {"station": logits}

    def _forward_teacher_forced(
        self,
        h_dest: torch.Tensor,
        H_all: torch.Tensor,
        origins: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Parallel teacher-forced forward pass for training and eval loss."""
        # Truncate labels to max decoder length (matches GRU behavior)
        if labels.size(1) > self.max_len:
            labels = labels[:, : self.max_len]

        B, T = labels.shape
        device = labels.device

        # Decoder input: shift labels right, prepend origin
        # labels  = [s0, s1, s2, ..., s_{T-1}]   (s0 == origin)
        # dec_in  = [origin, s0, s1, ..., s_{T-2}]
        # target  = [s0, s1, s2, ..., s_{T-1}]
        dec_input = torch.cat([origins.unsqueeze(1), labels[:, :-1]], dim=1)  # (B, T)

        tgt = self._embed_tokens(dec_input, h_dest)  # (B, T, d)
        memory = H_all.unsqueeze(0).expand(B, -1, -1)  # (B, N, d)
        causal = self._causal_mask(T, device, tgt.dtype)

        out = self.tf_decoder(tgt, memory, tgt_mask=causal)  # (B, T, d)
        logits = self.out_proj(out)  # (B, T, V)

        # Adjacency mask: at position t, we're "at" dec_input[:, t]
        logits = self._apply_adj_mask(logits, dec_input)

        return {"station": logits}

    def _forward_greedy(
        self,
        h_dest: torch.Tensor,
        H_all: torch.Tensor,
        origins: torch.Tensor,
        dests: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Autoregressive greedy decode (used when labels=None during eval)."""
        B = origins.size(0)
        device = origins.device
        memory = H_all.unsqueeze(0).expand(B, -1, -1)  # (B, N, d)

        tokens = origins.unsqueeze(1)  # (B, 1)
        all_logits = []

        for step in range(self.max_len):
            T = tokens.size(1)
            tgt = self._embed_tokens(tokens, h_dest)
            causal = self._causal_mask(T, device, tgt.dtype)

            out = self.tf_decoder(tgt, memory, tgt_mask=causal)
            step_logits = self.out_proj(out[:, -1, :])  # (B, V)
            step_logits = self._apply_adj_mask(step_logits, tokens[:, -1])
            all_logits.append(step_logits.unsqueeze(1))

            nxt = step_logits.argmax(-1, keepdim=True)  # (B, 1)
            tokens = torch.cat([tokens, nxt], dim=1)

            if (nxt.squeeze(-1) == dests).all():
                break

        logits = torch.cat(all_logits, dim=1)  # (B, steps, V)

        # Pad to max_len for consistent shape downstream
        if logits.size(1) < self.max_len:
            pad = logits.new_full(
                (B, self.max_len - logits.size(1), self.n_stations),
                self.MASK_VALUE,
            )
            logits = torch.cat([logits, pad], dim=1)

        return {"station": logits}
