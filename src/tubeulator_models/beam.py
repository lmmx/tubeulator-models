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
    model.eval()
    device = origins.device
    use_amp = device.type == "cuda"

    with torch.amp.autocast("cuda", enabled=use_amp):
        H = model.encoder(graph_x, graph_edge_index, graph_edge_attr)
        h_o = H[origins]
        h_d = H[dests]

        mt = model.model_type
        dec = model.decoder

        if isinstance(dec, TransformerStationDecoder):
            return _beam_transformer_station(
                dec, h_d, H, origins, dests, beam_width, device
            )
        elif isinstance(dec, HybridStationDecoder):
            return _beam_hybrid(dec, h_o, h_d, H, beam_width, device, origins)
        elif mt == "station":
            return _beam_pointer_gru(dec, h_o, h_d, H, beam_width, device, origins)
        else:
            return _beam_structured(dec, h_o, h_d, mt, beam_width, device)


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
    d = dec.d_model
    W = dec.window_size

    memory = H_all.unsqueeze(0)  # (1, N, d)
    h_init = torch.relu(dec.init_proj(torch.cat([h_o, h_d], dim=-1)))

    h = h_init
    gru_input = dec.start_input.expand(B, -1)
    cum_lps = torch.zeros(B, 1, device=device)
    token_seqs = torch.zeros(B, 1, max_len, dtype=torch.long, device=device)
    current = origins.unsqueeze(1)
    n_beams = 1

    # Window buffer: (B, n_beams, 0, d) — grows each step, capped at W
    if W > 0:
        win_buf = torch.zeros(B, 1, 0, d, device=device)

    for step in range(max_len):
        # GRU step
        h_next = dec.gru(gru_input, h)

        # Windowed self-attention
        if W > 0 and win_buf.size(2) > 0:
            BK = h_next.size(0)
            # Reshape for attention: (B*n_beams, W_cur, d)
            wb = win_buf.view(BK, -1, d)
            query = h_next.unsqueeze(1)
            attended, _ = dec.self_attn(query, wb, wb)
            h_next = dec.self_attn_norm(h_next + attended.squeeze(1))

        # Append to window buffer (before beam expansion changes n_beams)
        if W > 0:
            new_entry = h_next.view(B, n_beams, 1, d)
            win_buf = torch.cat([win_buf, new_entry], dim=2)
            if win_buf.size(2) > W:
                win_buf = win_buf[:, :, -W:, :]

        # Cross-attention
        BK = h_next.size(0)
        mem = memory.expand(BK, -1, -1)
        h_out = h_next.unsqueeze(1)
        for attn, norm in zip(dec.cross_layers, dec.cross_norms):
            attended, _ = attn(h_out, mem, mem)
            h_out = norm(h_out + attended)
        h_out = h_out.squeeze(1)

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

        h_selected = h_next.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, d))

        # Gather window buffer by selected beams
        if W > 0:
            W_cur = win_buf.size(2)
            win_buf = win_buf.gather(
                1, beam_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, W_cur, d)
            )

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
    h_dest: torch.Tensor,
    H_all: torch.Tensor,
    origins: torch.Tensor,
    dests: torch.Tensor,
    beam_width: int,
    device: torch.device,
) -> list[list[tuple[torch.Tensor, float]]]:
    """Fully batched beam search — all B examples × K beams in one forward pass per step."""
    B = origins.size(0)
    K = beam_width
    N = dec.n_stations
    max_len = dec.max_len

    # Token buffer: (B, K, max_len+1), position 0 = origin
    tokens = torch.zeros(B, K, max_len + 1, dtype=torch.long, device=device)
    tokens[:, :, 0] = origins.unsqueeze(1)

    cum_scores = torch.full((B, K), -1e9, device=device)
    cum_scores[:, 0] = 0.0  # only first beam active at start

    # Finished beams stored separately so they can't be evicted by live beams
    fin_tokens = torch.zeros(B, K, max_len + 1, dtype=torch.long, device=device)
    fin_scores = torch.full((B, K), -1e9, device=device)
    fin_lengths = torch.zeros(B, K, dtype=torch.long, device=device)

    temperature = 3.0

    for step in range(max_len):
        T = step + 1

        flat_tokens = tokens[:, :, :T].reshape(B * K, T)
        h_d_flat = h_dest.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        mem = H_all.unsqueeze(0).expand(B * K, -1, -1)

        tgt = dec._embed_tokens(flat_tokens, h_d_flat)
        causal = dec._causal_mask(T, device, tgt.dtype)
        out = dec.tf_decoder(tgt, mem, tgt_mask=causal)
        logits = dec.out_proj(out[:, -1, :]).view(B, K, N)

        # Adjacency mask
        last_tok = tokens[:, :, step]
        if dec.adj_mask is not None:
            adj = dec.adj_mask[last_tok.reshape(-1)].view(B, K, N)
            logits = logits.masked_fill(~adj, dec.MASK_VALUE)

        lp = F.log_softmax(logits / temperature, dim=-1)
        scores = cum_scores.unsqueeze(-1) + lp  # (B, K, N)
        scores_flat = scores.view(B, K * N)

        top_scores, top_flat = scores_flat.topk(K, dim=-1)
        beam_idx = top_flat // N
        tok_idx = top_flat % N

        # Reorder parent sequences and append new token
        prev = tokens[:, :, :T].gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, T))
        tokens[:, :, :T] = prev
        tokens[:, :, T] = tok_idx
        cum_scores = top_scores

        # Store newly finished beams (reached destination)
        reached = tok_idx == dests.unsqueeze(1)  # (B, K)
        if reached.any():
            for b in range(B):
                for k in range(K):
                    if reached[b, k]:
                        score = cum_scores[b, k].item()
                        worst = fin_scores[b].argmin().item()
                        if score > fin_scores[b, worst].item():
                            fin_scores[b, worst] = score
                            fin_lengths[b, worst] = T + 1
                            fin_tokens[b, worst, : T + 1] = tokens[b, k, : T + 1]

        # Early exit: all examples have K finished beams beating all live beams
        if (fin_scores.min(dim=1).values > cum_scores.max(dim=1).values).all():
            break

    # Merge finished + live, sort best-first
    results: list[list[tuple[torch.Tensor, float]]] = []
    for b in range(B):
        beams: list[tuple[torch.Tensor, float]] = []
        for k in range(K):
            if fin_scores[b, k] > -1e8:
                L = fin_lengths[b, k].item()
                beams.append((fin_tokens[b, k, :L].clone(), fin_scores[b, k].item()))
        for k in range(K):
            beams.append((tokens[b, k, : max_len + 1].clone(), cum_scores[b, k].item()))
        beams.sort(key=lambda x: x[1], reverse=True)
        results.append(beams[:K])

    return results


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


@torch.no_grad()
def rollout_nexthop(
    model,
    graph_x: torch.Tensor,
    graph_edge_index: torch.Tensor,
    graph_edge_attr: torch.Tensor,
    origins: torch.Tensor,  # (B,)
    dests: torch.Tensor,  # (B,)
    max_steps: int = 60,
) -> list[list[int]]:
    """
    Greedy rollout for next-hop policy model.

    Returns list of B station-index lists (including origin, excluding
    steps after destination is reached or max_steps exhausted).
    """
    model.eval()
    device = origins.device
    B = origins.size(0)
    N = model.n_stations
    use_amp = device.type == "cuda"

    with torch.amp.autocast("cuda", enabled=use_amp):
        H = model.encoder(graph_x, graph_edge_index, graph_edge_attr)

    h_d = H[dests]  # (B, d) — fixed for entire rollout

    current = origins.clone()  # (B,)
    # Track routes as padded tensor for efficiency
    route_buf = torch.full((B, max_steps + 1), -1, dtype=torch.long, device=device)
    route_buf[:, 0] = origins
    route_len = torch.ones(B, dtype=torch.long, device=device)  # starts at 1 (origin)

    # Visited mask: (B, N) — prevent cycles
    visited = torch.zeros(B, N, dtype=torch.bool, device=device)
    visited.scatter_(1, origins.unsqueeze(1), True)

    # Track which examples are still active
    active = torch.ones(B, dtype=torch.bool, device=device)

    adj_mask = model.decoder.adj_mask  # (N, N) or None

    for step in range(max_steps):
        if not active.any():
            break

        with torch.amp.autocast("cuda", enabled=use_amp):
            h_c = H[current]
            out = model.decoder(h_c, h_d, current_ids=current)

        logits = out["next_station"]  # (B, N)

        # Soft-block visited stations
        logits = logits.masked_fill(visited, -1e4)

        # Hard-block non-adjacent moves
        if adj_mask is not None:
            adj = adj_mask[current]
            logits = logits.masked_fill(~adj, float("-inf"))

        nxt = logits.argmax(dim=-1)  # (B,)

        # Only update active examples
        current = torch.where(active, nxt, current)
        step_idx = route_len.clamp(max=max_steps)
        route_buf.scatter_(1, step_idx.unsqueeze(1), current.unsqueeze(1))
        route_len += active.long()

        visited.scatter_(1, current.unsqueeze(1), True)

        # Deactivate examples that reached destination
        reached = current == dests
        active = active & ~reached

    # Convert to lists
    routes: list[list[int]] = []
    for b in range(B):
        length = route_len[b].item()
        routes.append(route_buf[b, :length].tolist())

    return routes


@torch.no_grad()
def beam_rollout_nexthop(
    model,
    graph_x: torch.Tensor,
    graph_edge_index: torch.Tensor,
    graph_edge_attr: torch.Tensor,
    origins: torch.Tensor,
    dests: torch.Tensor,
    beam_width: int = 8,
    max_steps: int = 60,
) -> list[list[tuple[list[int], float]]]:
    """
    Beam search rollout for next-hop policy.

    Returns list of B lists of (route, cum_log_prob) tuples, best-first.
    Completed routes (reached dest) are preferred; live beams appear
    only as fallback when no completion exists.
    """
    model.eval()
    device = origins.device
    B = origins.size(0)
    N = model.n_stations
    K = beam_width
    use_amp = device.type == "cuda"

    with torch.amp.autocast("cuda", enabled=use_amp):
        H = model.encoder(graph_x, graph_edge_index, graph_edge_attr)

    h_d = H[dests]  # (B, d)
    d = h_d.size(-1)

    # ── State tensors ─────────────────────────────────────────
    current = origins.unsqueeze(1).expand(-1, K).contiguous()  # (B, K)

    cum_scores = torch.full((B, K), -1e9, device=device)
    cum_scores[:, 0] = 0.0  # only beam 0 active initially

    route_buf = torch.full((B, K, max_steps + 1), -1, dtype=torch.long, device=device)
    route_buf[:, :, 0] = origins.unsqueeze(1)
    route_len = torch.ones(B, K, dtype=torch.long, device=device)

    visited = torch.zeros(B, K, N, dtype=torch.bool, device=device)
    visited.scatter_(2, origins.view(B, 1, 1).expand(-1, K, -1), True)

    # Finished beams — fixed-size buffer, replace worst on improvement
    fin_routes = torch.full((B, K, max_steps + 1), -1, dtype=torch.long, device=device)
    fin_scores = torch.full((B, K), -1e9, device=device)
    fin_lengths = torch.zeros(B, K, dtype=torch.long, device=device)

    for step in range(max_steps):
        # Forward pass: (B*K,) positions through the decoder MLP
        current_flat = current.reshape(B * K)
        h_d_exp = h_d.unsqueeze(1).expand(-1, K, -1).reshape(B * K, d)

        with torch.amp.autocast("cuda", enabled=use_amp):
            h_c = H[current_flat]
            out = model.decoder(h_c, h_d_exp, current_ids=current_flat)

        logits = out["next_station"].view(B, K, N)
        values = out["value"].view(B, K)

        # Visited is a softer penalty — prefers unvisited but allows backtrack
        logits = logits.masked_fill(visited, -1e4)

        # Hard-block non-adjacent — must dominate visited mask
        adj_for_beam = model.decoder.adj_mask[current_flat].view(B, K, N)
        logits = logits.masked_fill(~adj_for_beam, float("-inf"))

        lp = F.log_softmax(logits, dim=-1)

        # A* scoring: path log-prob minus estimated remaining cost
        # Lower value = closer to destination = better
        # We need value estimates for the *next* stations, not current ones
        # Approximate: use current value as a proxy (one-step lookahead
        # would require N forward passes per beam, not worth it)
        value_bonus = -0.1 * values.unsqueeze(-1).expand_as(lp)
        scores = cum_scores.unsqueeze(-1) + lp + value_bonus  # (B, K, N)

        topk_scores, topk_flat = scores.view(B, K * N).topk(K, dim=-1)
        beam_idx = topk_flat // N
        tok_idx = topk_flat % N

        # Gather parent state by selected beams
        route_buf = route_buf.gather(
            1, beam_idx.unsqueeze(-1).expand(-1, -1, max_steps + 1)
        )
        visited = visited.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, N))
        route_len = route_len.gather(1, beam_idx)

        # Append new hop
        route_buf.scatter_(2, route_len.unsqueeze(-1), tok_idx.unsqueeze(-1))
        route_len = route_len + 1
        visited.scatter_(2, tok_idx.unsqueeze(-1), True)

        current = tok_idx
        cum_scores = topk_scores

        # Store completed beams, kill them for future expansion
        reached = tok_idx == dests.unsqueeze(1)  # (B, K)
        if reached.any():
            for b in range(B):
                for k in range(K):
                    if reached[b, k]:
                        worst = fin_scores[b].argmin().item()
                        if cum_scores[b, k] > fin_scores[b, worst]:
                            L = route_len[b, k].item()
                            fin_scores[b, worst] = cum_scores[b, k].item()
                            fin_lengths[b, worst] = L
                            fin_routes[b, worst, :L] = route_buf[b, k, :L]
            cum_scores = cum_scores.masked_fill(reached, -1e9)

        # Early exit: all live beams worse than all finished beams
        if (cum_scores.max(dim=1).values < fin_scores.min(dim=1).values).all():
            break

    # ── Assemble results ──────────────────────────────────────
    results: list[list[tuple[list[int], float]]] = []
    for b in range(B):
        beams: list[tuple[list[int], float]] = []
        for k in range(K):
            if fin_scores[b, k] > -1e8:
                L = fin_lengths[b, k].item()
                beams.append((fin_routes[b, k, :L].tolist(), fin_scores[b, k].item()))
        if not beams:
            # No completion — return best live beam as fallback
            best_k = cum_scores[b].argmax().item()
            L = route_len[b, best_k].item()
            beams.append(
                (route_buf[b, best_k, :L].tolist(), cum_scores[b, best_k].item())
            )
        beams.sort(key=lambda x: x[1], reverse=True)
        results.append(beams[:K])

    return results
