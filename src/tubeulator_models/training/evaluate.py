"""Evaluation metrics for route-prediction models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


__all__ = [
    "compute_nexthop_step_metrics",
    "compute_nexthop_rollout_metrics",
]


@dataclass
class NextHopMetrics:
    step_acc: float  # fraction of correct next-hop predictions
    rollout_success: float  # fraction of rollouts reaching destination
    avg_length_ratio: float  # mean(rollout_len / best_gt_len) for successful rollouts
    avg_dijkstra_ratio: float
    n_steps: int
    n_rollouts: int
    stratified: dict[int, tuple[float, float, int]] | None = (
        None  # bucket -> (success, len_ratio, n)
    )

    def __str__(self) -> str:
        parts = [
            f"step_acc={self.step_acc:.1%}",
            f"success={self.rollout_success:.1%}",
            f"len_ratio={self.avg_length_ratio:.2f}",
            f"vs_dijkstra={self.avg_dijkstra_ratio:.2f}",
        ]
        return " | ".join(parts)


def compute_nexthop_step_metrics(
    logits: torch.Tensor,  # (B, V)
    targets: torch.Tensor,  # (B,)
) -> tuple[int, int]:
    """Returns (correct, total) for a batch of next-hop predictions."""
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    return correct, targets.size(0)


def _route_travel_time(route: list[int], edge_time_matrix: torch.Tensor) -> float:
    """Sum of edge travel times (seconds) for a station sequence."""
    total = 0.0
    for i in range(len(route) - 1):
        total += edge_time_matrix[route[i], route[i + 1]].item()
    return total


def compute_nexthop_rollout_metrics(
    rollouts: list[list[int]],
    dests: torch.Tensor,
    gt_routes: list[list[list[int]]],
    edge_time_matrix: torch.Tensor | None = None,
    optimal_times: torch.Tensor | None = None,
    origins: torch.Tensor | None = None,
    strat_keys: list[int] | None = None,
) -> NextHopMetrics:
    from collections import defaultdict

    B = len(rollouts)
    n_success = 0
    length_ratios: list[float] = []
    dijkstra_ratios: list[float] = []
    bucket_data: dict[int, list[tuple[bool, float, float]]] = defaultdict(list)

    for b in range(B):
        route = rollouts[b]
        dest = dests[b].item()
        reached = len(route) > 1 and route[-1] == dest

        if edge_time_matrix is not None:
            rollout_cost = _route_travel_time(route, edge_time_matrix)
            best_gt_cost = min(
                _route_travel_time(r, edge_time_matrix) for r in gt_routes[b]
            )
        else:
            rollout_cost = float(len(route))
            best_gt_cost = float(min(len(r) for r in gt_routes[b]))

        ratio = (
            rollout_cost / best_gt_cost
            if reached and best_gt_cost > 0
            else float("inf")
        )

        dij_ratio = float("inf")
        if reached and optimal_times is not None and origins is not None:
            opt = optimal_times[origins[b].item(), dest].item()
            if opt > 0:
                dij_ratio = rollout_cost / opt

        if reached:
            n_success += 1
            length_ratios.append(ratio)
            if dij_ratio != float("inf"):
                dijkstra_ratios.append(dij_ratio)

        if strat_keys is not None:
            bucket_data[strat_keys[b]].append((reached, ratio, dij_ratio))

    avg_ratio = (
        sum(length_ratios) / len(length_ratios) if length_ratios else float("inf")
    )
    avg_dijkstra = (
        sum(dijkstra_ratios) / len(dijkstra_ratios) if dijkstra_ratios else float("inf")
    )

    stratified = None
    if strat_keys is not None:
        stratified = {}
        for k, entries in sorted(bucket_data.items()):
            succ = sum(1 for r, _, _ in entries if r)
            ratios = [r for ok, r, _ in entries if ok]
            dij_ratios = [d for ok, _, d in entries if ok and d != float("inf")]
            avg_r = sum(ratios) / len(ratios) if ratios else float("inf")
            avg_d = sum(dij_ratios) / len(dij_ratios) if dij_ratios else float("inf")
            stratified[k] = (succ / len(entries), avg_r, avg_d, len(entries))

    return NextHopMetrics(
        step_acc=0.0,
        rollout_success=n_success / max(B, 1),
        avg_length_ratio=avg_ratio,
        avg_dijkstra_ratio=avg_dijkstra,
        n_steps=0,
        n_rollouts=B,
        stratified=stratified,
    )
