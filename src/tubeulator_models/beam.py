"""Beam search decoding for route-prediction models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["beam_decode"]


@torch.no_grad()
def beam_decode(
    model,
    graph_x: torch.Tensor,
    graph_edge_index: torch.Tensor,
    graph_edge_attr: torch.Tensor,
    origins: torch.Tensor,
    dests: torch.Tensor,
    beam_width: int = 5,
) -> list[list[tuple[torch.Tensor, float]]]:
    """
    Beam search decoding. Returns one list per example in the batch,
    each containing (sequence_tensor, log_prob) tuples sorted best-first.

    For line/change models, sequence_tensor is flat [line, dir, ...] or
    [line, dir, station, ...].
    For station model, it's [station_0, station_1, ...].
    """
    model.eval()
    H = model.encoder(graph_x, graph_edge_index, graph_edge_attr)
    h_o = H[origins]
    h_d = H[dests]

    B = origins.size(0)
    device = origins.device
    mt = model.model_type
    dec = model.decoder

    results: list[list[tuple[torch.Tensor, float]]] = []

    for b in range(B):
        ho = h_o[b].unsqueeze(0)  # (1, d)
        hd = h_d[b].unsqueeze(0)

        h_init = torch.relu(dec.init_proj(torch.cat([ho, hd], dim=-1)))  # (1, d)

        if mt in ("line", "change"):
            beams = _beam_structured(dec, h_init, mt, beam_width, device, H if mt == "station" else None)
        else:
            beams = _beam_pointer(dec, h_init, H, beam_width, device)

        results.append(beams)

    return results


def _beam_structured(
    dec,
    h_init: torch.Tensor,
    model_type: str,
    beam_width: int,
    device: torch.device,
    H_all=None,
) -> list[tuple[torch.Tensor, float]]:
    """Beam search for line/change decoders (multi-head per step)."""
    max_legs = dec.max_legs

    # Each beam: (hidden_state, token_sequence, cumulative_log_prob)
    beams: list[tuple[torch.Tensor, list[int], float]] = [
        (h_init, [], 0.0)
    ]

    for step in range(max_legs):
        candidates = []

        for h, seq, cum_lp in beams:
            h_next = dec.gru(h, h)

            line_logits = dec.line_head(h_next)           # (1, n_lines)
            dir_logits = dec.dir_head(h_next)             # (1, 2)
            line_lp = F.log_softmax(line_logits, dim=-1)  # (1, n_lines)
            dir_lp = F.log_softmax(dir_logits, dim=-1)    # (1, 2)

            # Top-k per head
            k_line = min(beam_width, line_lp.size(-1))
            top_lines = line_lp.topk(k_line, dim=-1)
            top_dirs = dir_lp.topk(2, dim=-1)  # only 2 directions

            if model_type == "change":
                st_logits = dec.station_head(h_next)
                st_lp = F.log_softmax(st_logits, dim=-1)
                k_st = min(beam_width, st_lp.size(-1))
                top_sts = st_lp.topk(k_st, dim=-1)

                for li in range(k_line):
                    for di in range(2):
                        for si in range(k_st):
                            ln_tok = top_lines.indices[0, li]
                            dir_tok = top_dirs.indices[0, di]
                            st_tok = top_sts.indices[0, si]
                            lp = (
                                top_lines.values[0, li].item()
                                + top_dirs.values[0, di].item()
                                + top_sts.values[0, si].item()
                            )

                            fb = torch.cat([
                                dec.line_emb(ln_tok.unsqueeze(0)),
                                dec.dir_emb(dir_tok.unsqueeze(0)),
                                dec.station_emb(st_tok.unsqueeze(0)),
                            ], dim=-1)
                            h_fb = h_next + dec.fb_proj(fb)

                            new_seq = seq + [
                                ln_tok.item(), dir_tok.item(), st_tok.item(),
                            ]
                            candidates.append((h_fb, new_seq, cum_lp + lp))
            else:
                # line model
                for li in range(k_line):
                    for di in range(2):
                        ln_tok = top_lines.indices[0, li]
                        dir_tok = top_dirs.indices[0, di]
                        lp = (
                            top_lines.values[0, li].item()
                            + top_dirs.values[0, di].item()
                        )

                        fb = torch.cat([
                            dec.line_emb(ln_tok.unsqueeze(0)),
                            dec.dir_emb(dir_tok.unsqueeze(0)),
                        ], dim=-1)
                        h_fb = h_next + fb

                        new_seq = seq + [ln_tok.item(), dir_tok.item()]
                        candidates.append((h_fb, new_seq, cum_lp + lp))

        # Keep top beam_width
        candidates.sort(key=lambda x: x[2], reverse=True)
        beams = candidates[:beam_width]

    return [
        (torch.tensor(seq, dtype=torch.long, device=device), lp)
        for _, seq, lp in beams
    ]


def _beam_pointer(
    dec,
    h_init: torch.Tensor,
    H_all: torch.Tensor,
    beam_width: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, float]]:
    """Beam search for station (pointer) decoder."""
    max_len = dec.max_len
    keys = dec.key_proj(H_all)  # (N, d)

    beams: list[tuple[torch.Tensor, list[int], float]] = [
        (h_init, [], 0.0)
    ]

    for step in range(max_len):
        candidates = []

        for h, seq, cum_lp in beams:
            h_next = dec.gru(h, h)
            q = dec.query_proj(h_next)          # (1, d)
            logits = torch.matmul(q, keys.t())  # (1, N)
            lp = F.log_softmax(logits, dim=-1)  # (1, N)

            k = min(beam_width, lp.size(-1))
            top = lp.topk(k, dim=-1)

            for i in range(k):
                tok = top.indices[0, i]
                h_fb = h_next + dec.station_emb(tok.unsqueeze(0))
                new_seq = seq + [tok.item()]
                candidates.append((h_fb, new_seq, cum_lp + top.values[0, i].item()))

        candidates.sort(key=lambda x: x[2], reverse=True)
        beams = candidates[:beam_width]

    return [
        (torch.tensor(seq, dtype=torch.long, device=device), lp)
        for _, seq, lp in beams
    ]