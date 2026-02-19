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
    """
    model.eval()
    H = model.encoder(graph_x, graph_edge_index, graph_edge_attr)
    h_o = H[origins]
    h_d = H[dests]

    mt = model.model_type
    dec = model.decoder

    if mt == "station":
        return _beam_pointer_batched(
            dec, h_o, h_d, H, beam_width, origins.device, origins
        )
    else:
        return _beam_structured_batched(dec, h_o, h_d, mt, beam_width, origins.device)


def _beam_structured_batched(
    dec,
    h_o: torch.Tensor,
    h_d: torch.Tensor,
    model_type: str,
    beam_width: int,
    device: torch.device,
) -> list[list[tuple[torch.Tensor, float]]]:
    """
    Beam search for line/change decoders.

    Still loops over batch elements (combinatorial head explosion makes true
    batching impractical), but vectorises the per-beam work within each example.
    """
    B = h_o.size(0)
    max_legs = dec.max_legs
    results: list[list[tuple[torch.Tensor, float]]] = []

    for b in range(B):
        ho = h_o[b].unsqueeze(0)
        hd = h_d[b].unsqueeze(0)
        h_init = torch.relu(dec.init_proj(torch.cat([ho, hd], dim=-1)))

        # beams: (K, d), sequences: list of K lists, log_probs: (K,)
        h = h_init  # (1, d)
        seqs: list[list[int]] = [[]]
        lps = torch.zeros(1, device=device)

        for step in range(max_legs):
            K = h.size(0)
            h_next = dec.gru(h, h)  # (K, d)

            line_lp = F.log_softmax(dec.line_head(h_next), dim=-1)  # (K, n_lines)
            dir_lp = F.log_softmax(dec.dir_head(h_next), dim=-1)  # (K, 2)

            k_line = min(beam_width, line_lp.size(-1))

            if model_type == "change":
                st_lp = F.log_softmax(dec.station_head(h_next), dim=-1)
                k_st = min(beam_width, st_lp.size(-1))

                # For each beam, compute joint scores over top-k combinations
                top_ln = line_lp.topk(k_line, dim=-1)  # vals/idx: (K, k_line)
                top_dir = dir_lp.topk(2, dim=-1)  # vals/idx: (K, 2)
                top_st = st_lp.topk(k_st, dim=-1)  # vals/idx: (K, k_st)

                # Enumerate all combos: K * k_line * 2 * k_st candidates
                cand_h = []
                cand_seqs = []
                cand_lps = []

                for ki in range(K):
                    for li in range(k_line):
                        for di in range(2):
                            for si in range(k_st):
                                ln_tok = top_ln.indices[ki, li]
                                dir_tok = top_dir.indices[ki, di]
                                st_tok = top_st.indices[ki, si]
                                score = (
                                    lps[ki].item()
                                    + top_ln.values[ki, li].item()
                                    + top_dir.values[ki, di].item()
                                    + top_st.values[ki, si].item()
                                )

                                fb = torch.cat(
                                    [
                                        dec.line_emb(ln_tok.unsqueeze(0)),
                                        dec.dir_emb(dir_tok.unsqueeze(0)),
                                        dec.station_emb(st_tok.unsqueeze(0)),
                                    ],
                                    dim=-1,
                                )
                                h_fb = h_next[ki].unsqueeze(0) + dec.fb_proj(fb)

                                cand_h.append(h_fb)
                                cand_seqs.append(
                                    seqs[ki]
                                    + [ln_tok.item(), dir_tok.item(), st_tok.item()]
                                )
                                cand_lps.append(score)

            else:
                # line model
                top_ln = line_lp.topk(k_line, dim=-1)
                top_dir = dir_lp.topk(2, dim=-1)

                cand_h = []
                cand_seqs = []
                cand_lps = []

                for ki in range(K):
                    for li in range(k_line):
                        for di in range(2):
                            ln_tok = top_ln.indices[ki, li]
                            dir_tok = top_dir.indices[ki, di]
                            score = (
                                lps[ki].item()
                                + top_ln.values[ki, li].item()
                                + top_dir.values[ki, di].item()
                            )

                            fb = torch.cat(
                                [
                                    dec.line_emb(ln_tok.unsqueeze(0)),
                                    dec.dir_emb(dir_tok.unsqueeze(0)),
                                ],
                                dim=-1,
                            )
                            h_fb = h_next[ki].unsqueeze(0) + fb

                            cand_h.append(h_fb)
                            cand_seqs.append(seqs[ki] + [ln_tok.item(), dir_tok.item()])
                            cand_lps.append(score)

            # Prune to top beam_width
            if len(cand_lps) > beam_width:
                top_indices = sorted(
                    range(len(cand_lps)), key=lambda i: cand_lps[i], reverse=True
                )[:beam_width]
            else:
                top_indices = list(range(len(cand_lps)))

            h = torch.cat([cand_h[i] for i in top_indices], dim=0)
            seqs = [cand_seqs[i] for i in top_indices]
            lps = torch.tensor([cand_lps[i] for i in top_indices], device=device)

        results.append(
            [
                (torch.tensor(seq, dtype=torch.long, device=device), lps[i].item())
                for i, seq in enumerate(seqs)
            ]
        )

    return results


def _beam_pointer_batched(
    dec,
    h_o: torch.Tensor,
    h_d: torch.Tensor,
    H_all: torch.Tensor,
    beam_width: int,
    device: torch.device,
    origins: torch.Tensor,
) -> list[list[tuple[torch.Tensor, float]]]:
    """
    Fully batched beam search for the station (pointer) decoder.
    Tracks current station per beam and applies adjacency masking.
    """
    B = h_o.size(0)
    max_len = dec.max_len
    K = beam_width
    N = H_all.size(0)

    keys = dec.key_proj(H_all)  # (N, d)
    h_init = torch.relu(dec.init_proj(torch.cat([h_o, h_d], dim=-1)))  # (B, d)

    h = h_init  # (B, d)
    cum_lps = torch.zeros(B, 1, device=device)  # (B, 1)
    token_seqs = torch.zeros(B, 1, max_len, dtype=torch.long, device=device)
    # Track current station: (B, n_beams)
    current = origins.unsqueeze(1)  # (B, 1)
    n_beams = 1

    for step in range(max_len):
        # h: (B * n_beams, d)
        h_next = dec.gru(h, h)
        q = dec.query_proj(h_next)  # (B * n_beams, d)
        logits = torch.matmul(q, keys.t())  # (B * n_beams, N)

        # Apply adjacency mask
        if dec.adj_mask is not None:
            current_flat = current.reshape(-1)  # (B * n_beams,)
            mask = dec.adj_mask[current_flat]  # (B * n_beams, N)
            logits = logits.masked_fill(~mask, float("-inf"))

        lp = F.log_softmax(logits, dim=-1)

        # Reshape to (B, n_beams, N)
        lp = lp.view(B, n_beams, N)
        h_next = h_next.view(B, n_beams, -1)

        scores = cum_lps.unsqueeze(-1) + lp  # (B, n_beams, N)
        scores_flat = scores.view(B, -1)  # (B, n_beams * N)

        actual_k = min(K, scores_flat.size(-1))
        top_scores, top_flat_idx = scores_flat.topk(actual_k, dim=-1)

        beam_idx = top_flat_idx // N
        tok_idx = top_flat_idx % N

        # Gather hidden states
        d_model = h_next.size(-1)
        h_selected = h_next.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, d_model))

        # Feedback
        tok_flat = tok_idx.reshape(-1)
        fb = dec.station_emb(tok_flat).view(B, actual_k, -1)
        h = (h_selected + fb).view(B * actual_k, -1)

        # Update sequences
        if step == 0 and n_beams == 1:
            new_seqs = token_seqs.expand(-1, actual_k, -1).clone()
        else:
            new_seqs = token_seqs.gather(
                1, beam_idx.unsqueeze(-1).expand(-1, -1, max_len)
            )
        new_seqs[:, :, step] = tok_idx
        token_seqs = new_seqs

        # Update current station per beam
        current = tok_idx  # (B, K)

        cum_lps = top_scores
        n_beams = actual_k

    results: list[list[tuple[torch.Tensor, float]]] = []
    for b in range(B):
        beams = []
        for k in range(n_beams):
            seq = token_seqs[b, k]
            score = cum_lps[b, k].item()
            beams.append((seq, score))
        beams.sort(key=lambda x: x[1], reverse=True)
        results.append(beams)

    return results
