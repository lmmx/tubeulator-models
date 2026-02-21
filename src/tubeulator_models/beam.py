"""Beam search decoding for route-prediction models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .models.decoders import HybridStationDecoder, TransformerStationDecoder


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
    Beam search decoding.  Returns one list per example in the batch,
    each containing (sequence_tensor, log_prob) tuples sorted best-first.

    Dispatches to the appropriate implementation based on decoder type.
    """
    model.eval()
    H = model.encoder(graph_x, graph_edge_index, graph_edge_attr)
    h_o = H[origins]
    h_d = H[dests]

    mt = model.model_type
    dec = model.decoder

    if isinstance(dec, TransformerStationDecoder):
        return _beam_transformer_station(
            dec,
            h_d,
            H,
            origins,
            dests,
            beam_width,
            origins.device,
        )
    elif isinstance(dec, HybridStationDecoder):
        return _beam_hybrid(dec, h_o, h_d, H, beam_width, origins.device, origins)
    elif mt == "station":
        return _beam_pointer_gru(dec, h_o, h_d, H, beam_width, origins.device, origins)
    else:
        return _beam_structured(dec, h_o, h_d, mt, beam_width, origins.device)


# ══════════════════════════════════════════════════════════════
#  Transformer station decoder beam search
# ══════════════════════════════════════════════════════════════


def _beam_hybrid(
    dec: HybridStationDecoder,
    h_o: torch.Tensor,
    h_d: torch.Tensor,
    H_all: torch.Tensor,
    beam_width: int,
    device: torch.device,
    origins: torch.Tensor,
) -> list[list[tuple[torch.Tensor, float]]]:
    """Beam search for HybridStationDecoder (GRU + cross-attention)."""
    B = h_o.size(0)
    max_len = dec.max_len
    K = beam_width
    N = H_all.size(0)

    memory = H_all.unsqueeze(0)  # (1, N, d)
    h_init = torch.relu(dec.init_proj(torch.cat([h_o, h_d], dim=-1)))

    h = h_init
    gru_input = dec.start_input.expand(B, -1)
    cum_lps = torch.zeros(B, 1, device=device)
    token_seqs = torch.zeros(B, 1, max_len, dtype=torch.long, device=device)
    current = origins.unsqueeze(1)
    n_beams = 1

    for step in range(max_len):
        # GRU step
        h_next = dec.gru(gru_input, h)

        # Cross-attention
        BK = h_next.size(0)
        mem = memory.expand(BK, -1, -1)
        query = h_next.unsqueeze(1)
        attended, _ = dec.cross_attn(query, mem, mem)
        h_out = dec.cross_norm(h_next + attended.squeeze(1))

        # Output logits
        logits = dec.out_proj(h_out)

        # Adjacency mask
        if dec.adj_mask is not None:
            current_flat = current.reshape(-1)
            mask = dec.adj_mask[current_flat]
            logits = logits.masked_fill(~mask, dec.MASK_VALUE)

        lp = F.log_softmax(logits, dim=-1)
        lp = lp.view(B, n_beams, N)
        h_next = h_next.view(B, n_beams, -1)

        scores = cum_lps.unsqueeze(-1) + lp
        scores_flat = scores.view(B, -1)

        actual_k = min(K, scores_flat.size(-1))
        top_scores, top_flat_idx = scores_flat.topk(actual_k, dim=-1)

        beam_idx = top_flat_idx // N
        tok_idx = top_flat_idx % N

        d_model = h_next.size(-1)
        h_selected = h_next.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, d_model))

        tok_flat = tok_idx.reshape(-1)
        fb = dec.station_emb(tok_flat).view(B, actual_k, -1)
        h = h_selected.view(B * actual_k, -1)
        gru_input = fb.view(B * actual_k, -1)

        if step == 0 and n_beams == 1:
            new_seqs = token_seqs.expand(-1, actual_k, -1).clone()
        else:
            new_seqs = token_seqs.gather(
                1, beam_idx.unsqueeze(-1).expand(-1, -1, max_len)
            )
        new_seqs[:, :, step] = tok_idx
        token_seqs = new_seqs

        current = tok_idx
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


def _beam_transformer_station(
    dec: TransformerStationDecoder,
    h_dest: torch.Tensor,  # (B, d)
    H_all: torch.Tensor,  # (N, d)
    origins: torch.Tensor,  # (B,)
    dests: torch.Tensor,  # (B,)
    beam_width: int,
    device: torch.device,
) -> list[list[tuple[torch.Tensor, float]]]:
    """
    Beam search for TransformerStationDecoder.

    All live beams at a given step have identical token length, so they
    batch cleanly without padding.  With adjacency masking the effective
    branching factor is 3–4, making beam_width=20 near-exhaustive.
    """
    B = origins.size(0)
    K = beam_width
    memory_base = H_all.unsqueeze(0)  # (1, N, d)

    all_results: list[list[tuple[torch.Tensor, float]]] = []

    for b in range(B):
        orig = origins[b]
        dest = dests[b]
        h_d_b = h_dest[b].unsqueeze(0)  # (1, d)

        # Each beam: (token_tensor_of_length_T, cumulative_log_prob)
        live: list[tuple[torch.Tensor, float]] = [
            (orig.unsqueeze(0), 0.0)  # start with origin token
        ]
        finished: list[tuple[torch.Tensor, float]] = []

        for step in range(dec.max_len - 1):
            if not live:
                break

            n = len(live)
            seqs = torch.stack([s for s, _ in live])  # (n, T)
            scores = [sc for _, sc in live]
            T = seqs.size(1)

            # Expand single-example tensors to beam count
            h_d_exp = h_d_b.expand(n, -1)  # (n, d)
            memory = memory_base.expand(n, -1, -1)  # (n, N, d)

            tgt = dec._embed_tokens(seqs, h_d_exp)
            causal = dec._causal_mask(T, device, tgt.dtype)

            out = dec.tf_decoder(tgt, memory, tgt_mask=causal)
            logits = dec.out_proj(out[:, -1, :])  # (n, V)

            # Adjacency mask based on last token in each beam
            logits = dec._apply_adj_mask(logits, seqs[:, -1])

            temperature = 3.0  # >1 flattens distribution, exposes alternatives
            log_probs = F.log_softmax(logits / temperature, dim=-1)  # (n, V)

            # Expand candidates
            candidates: list[tuple[torch.Tensor, float]] = []

            for i in range(n):
                # Only consider unmasked tokens
                valid_count = (logits[i] > dec.MASK_VALUE + 1).sum().item()
                topk_k = min(K, max(1, int(valid_count)))
                topk_lp, topk_idx = log_probs[i].topk(topk_k)

                for j in range(topk_lp.size(0)):
                    tok = topk_idx[j]
                    new_seq = torch.cat([live[i][0], tok.unsqueeze(0)])
                    new_score = scores[i] + topk_lp[j].item()

                    if tok.item() == dest.item() and new_seq.size(0) > 1:
                        finished.append((new_seq, new_score))
                    else:
                        candidates.append((new_seq, new_score))

            # Prune to top K
            candidates.sort(key=lambda x: x[1], reverse=True)
            live = candidates[:K]

            # Early exit: enough finished beams
            finished.sort(key=lambda x: x[1], reverse=True)
            finished = finished[:K]
            if len(finished) >= K:
                break

        merged = finished + live
        merged.sort(key=lambda x: x[1], reverse=True)
        all_results.append(merged[:K])

    return all_results


# ══════════════════════════════════════════════════════════════
#  GRU station (pointer) decoder beam search
# ══════════════════════════════════════════════════════════════


def _beam_pointer_gru(
    dec,
    h_o: torch.Tensor,
    h_d: torch.Tensor,
    H_all: torch.Tensor,
    beam_width: int,
    device: torch.device,
    origins: torch.Tensor,
) -> list[list[tuple[torch.Tensor, float]]]:
    """
    Fully batched beam search for the GRU station (pointer) decoder.
    Tracks current station per beam and applies adjacency masking.
    """
    B = h_o.size(0)
    max_len = dec.max_len
    K = beam_width
    N = H_all.size(0)

    keys = dec.key_proj(H_all)
    h_init = torch.relu(dec.init_proj(torch.cat([h_o, h_d], dim=-1)))

    h = h_init
    gru_input = dec.start_input.expand(B, -1)
    cum_lps = torch.zeros(B, 1, device=device)
    token_seqs = torch.zeros(B, 1, max_len, dtype=torch.long, device=device)
    current = origins.unsqueeze(1)
    n_beams = 1

    for step in range(max_len):
        h_next = dec.gru(gru_input, h)
        q = dec.query_proj(h_next)
        logits = torch.matmul(q, keys.t())

        if dec.adj_mask is not None:
            current_flat = current.reshape(-1)
            mask = dec.adj_mask[current_flat]
            logits = logits.masked_fill(~mask, -1e4)

        lp = F.log_softmax(logits, dim=-1)
        lp = lp.view(B, n_beams, N)
        h_next = h_next.view(B, n_beams, -1)

        scores = cum_lps.unsqueeze(-1) + lp
        scores_flat = scores.view(B, -1)

        actual_k = min(K, scores_flat.size(-1))
        top_scores, top_flat_idx = scores_flat.topk(actual_k, dim=-1)

        beam_idx = top_flat_idx // N
        tok_idx = top_flat_idx % N

        d_model = h_next.size(-1)
        h_selected = h_next.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, d_model))

        tok_flat = tok_idx.reshape(-1)
        fb = dec.station_emb(tok_flat).view(B, actual_k, -1)
        h = h_selected.view(B * actual_k, -1)
        gru_input = fb.view(B * actual_k, -1)

        if step == 0 and n_beams == 1:
            new_seqs = token_seqs.expand(-1, actual_k, -1).clone()
        else:
            new_seqs = token_seqs.gather(
                1, beam_idx.unsqueeze(-1).expand(-1, -1, max_len)
            )
        new_seqs[:, :, step] = tok_idx
        token_seqs = new_seqs

        current = tok_idx
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


# ══════════════════════════════════════════════════════════════
#  Line / change (structured) decoder beam search
# ══════════════════════════════════════════════════════════════


def _beam_structured(
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

        h = h_init
        gru_input = dec.start_input.expand(1, -1)
        seqs: list[list[int]] = [[]]
        lps = torch.zeros(1, device=device)

        for step in range(max_legs):
            K = h.size(0)
            h_next = dec.gru(gru_input, h)

            line_lp = F.log_softmax(dec.line_head(h_next), dim=-1)
            dir_lp = F.log_softmax(dec.dir_head(h_next), dim=-1)

            k_line = min(beam_width, line_lp.size(-1))

            if model_type == "change":
                st_raw = dec.station_head(h_next)
                k_st = min(beam_width, st_raw.size(-1))

                top_ln = line_lp.topk(k_line, dim=-1)
                top_dir = dir_lp.topk(2, dim=-1)

                cand_h = []
                cand_fb = []
                cand_seqs = []
                cand_lps = []

                for ki in range(K):
                    for li in range(k_line):
                        ln_tok = top_ln.indices[ki, li]

                        masked = st_raw[ki].clone()
                        if dec.line_station_mask is not None:
                            masked[~dec.line_station_mask[ln_tok]] = -1e4
                        st_lp = F.log_softmax(masked, dim=-1)
                        local_top_st = st_lp.topk(k_st, dim=-1)

                        for di in range(2):
                            for si in range(k_st):
                                dir_tok = top_dir.indices[ki, di]
                                st_tok = local_top_st.indices[si]
                                score = (
                                    lps[ki].item()
                                    + top_ln.values[ki, li].item()
                                    + top_dir.values[ki, di].item()
                                    + local_top_st.values[si].item()
                                )

                                fb = torch.cat(
                                    [
                                        dec.line_emb(ln_tok.unsqueeze(0)),
                                        dec.dir_emb(dir_tok.unsqueeze(0)),
                                        dec.station_emb(st_tok.unsqueeze(0)),
                                    ],
                                    dim=-1,
                                )
                                fb_out = dec.fb_proj(fb)

                                cand_h.append(h_next[ki].unsqueeze(0))
                                cand_fb.append(fb_out)
                                cand_seqs.append(
                                    seqs[ki]
                                    + [ln_tok.item(), dir_tok.item(), st_tok.item()]
                                )
                                cand_lps.append(score)

            else:
                top_ln = line_lp.topk(k_line, dim=-1)
                top_dir = dir_lp.topk(2, dim=-1)

                cand_h = []
                cand_fb = []
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

                            cand_h.append(h_next[ki].unsqueeze(0))
                            cand_fb.append(fb)
                            cand_seqs.append(seqs[ki] + [ln_tok.item(), dir_tok.item()])
                            cand_lps.append(score)

            if len(cand_lps) > beam_width:
                top_indices = sorted(
                    range(len(cand_lps)), key=lambda i: cand_lps[i], reverse=True
                )[:beam_width]
            else:
                top_indices = list(range(len(cand_lps)))

            h = torch.cat([cand_h[i] for i in top_indices], dim=0)
            gru_input = torch.cat([cand_fb[i] for i in top_indices], dim=0)
            seqs = [cand_seqs[i] for i in top_indices]
            lps = torch.tensor([cand_lps[i] for i in top_indices], device=device)

        results.append(
            [
                (torch.tensor(seq, dtype=torch.long, device=device), lps[i].item())
                for i, seq in enumerate(seqs)
            ]
        )

    return results
