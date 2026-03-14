"""Beam search decoding for route-prediction models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


__all__ = ["rollout_nexthop", "beam_rollout_nexthop", "bellman_rollout_nexthop"]


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


@torch.no_grad()
def bellman_rollout_nexthop(
    model,
    graph_x: torch.Tensor,
    graph_edge_index: torch.Tensor,
    graph_edge_attr: torch.Tensor,
    origins: torch.Tensor,
    dests: torch.Tensor,
    edge_time_matrix: torch.Tensor,
    max_steps: int = 60,
) -> list[list[int]]:
    """
    Bellman-optimal rollout using value head as cost-to-go estimate.

    Action rule: argmin_n [ edge_time(s, n) + V(n, d) ]
    No policy logits, no beam search.
    """
    model.eval()
    device = origins.device
    B = origins.size(0)
    N = model.n_stations
    use_amp = device.type == "cuda"

    with torch.amp.autocast("cuda", enabled=use_amp):
        H = model.encoder(graph_x, graph_edge_index, graph_edge_attr)

    h_d = H[dests]  # (B, d)
    adj_mask = model.decoder.adj_mask  # (N, N)

    # Precompute V(n, d) for all stations and all examples in batch.
    # V_all[b, n] = value_head([H[n], h_d[b]])
    # Single forward pass through value head on (B*N, 2d) tensor.
    d = H.size(1)
    H_exp = H.unsqueeze(0).expand(B, -1, -1)  # (B, N, d)
    h_d_exp = h_d.unsqueeze(1).expand(-1, N, -1)  # (B, N, d)
    combined = torch.cat([H_exp, h_d_exp], dim=-1)  # (B, N, 2d)

    with torch.amp.autocast("cuda", enabled=use_amp):
        V_all = model.decoder.value_head(combined.view(B * N, 2 * d)).view(
            B, N
        )  # (B, N) predicted remaining time for each station

    # Edge times from current station: will index per step
    edge_times = edge_time_matrix.to(device)  # (N, N)

    current = origins.clone()
    route_buf = torch.full((B, max_steps + 1), -1, dtype=torch.long, device=device)
    route_buf[:, 0] = origins
    route_len = torch.ones(B, dtype=torch.long, device=device)

    visited = torch.zeros(B, N, dtype=torch.bool, device=device)
    visited.scatter_(1, origins.unsqueeze(1), True)

    active = torch.ones(B, dtype=torch.bool, device=device)

    for step in range(max_steps):
        if not active.any():
            break

        # Cost for each neighbor: edge_time(current, n) + V(n, d)
        costs = edge_times[current] + V_all  # (B, N)

        # Soft-penalize visited (prefer unvisited, allow backtrack)
        costs = costs + visited.float() * 1e6

        # Hard-block non-adjacent
        if adj_mask is not None:
            costs = costs.masked_fill(~adj_mask[current], float("inf"))

        nxt = costs.argmin(dim=-1)

        current = torch.where(active, nxt, current)
        step_idx = route_len.clamp(max=max_steps)
        route_buf.scatter_(1, step_idx.unsqueeze(1), current.unsqueeze(1))
        route_len += active.long()

        visited.scatter_(1, current.unsqueeze(1), True)

        reached = current == dests
        active = active & ~reached

    routes: list[list[int]] = []
    for b in range(B):
        length = route_len[b].item()
        routes.append(route_buf[b, :length].tolist())
    return routes
