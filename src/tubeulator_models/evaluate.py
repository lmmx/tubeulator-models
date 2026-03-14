"""Evaluation metrics for route-prediction models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dataset import PAD


__all__ = [
    "RouteMetrics",
    "compute_nexthop_step_metrics",
    "compute_nexthop_rollout_metrics",
]


@dataclass
class RouteMetrics:
    exact_match: float
    any_in_beam: float
    line_acc: float | None
    dir_acc: float | None
    station_acc: float | None
    topologically_valid: float
    n_examples: int
    stratified: dict[int, tuple[float, int]] | None = None

    def __str__(self) -> str:
        parts = [
            f"exact={self.exact_match:.1%}",
            f"beam={self.any_in_beam:.1%}",
        ]
        if self.line_acc is not None:
            parts.append(f"line={self.line_acc:.1%}")
        if self.dir_acc is not None:
            parts.append(f"dir={self.dir_acc:.1%}")
        if self.station_acc is not None:
            parts.append(f"station={self.station_acc:.1%}")
        parts.append(f"valid={self.topologically_valid:.1%}")
        return " | ".join(parts)


def _sequences_match(pred: torch.Tensor, label: torch.Tensor) -> bool:
    """Check if predicted sequence matches a label, ignoring PAD."""
    label_len = (label != PAD).sum().item()
    if label_len == 0:
        return True
    if pred.size(0) < label_len:
        return False
    return torch.equal(pred[:label_len], label[:label_len])


def _best_matching_label(
    pred: torch.Tensor,
    all_labels: list[torch.Tensor],
) -> torch.Tensor:
    """Return the valid label with the most matching tokens."""
    best_label = all_labels[0]
    best_overlap = -1
    for label in all_labels:
        label_len = (label != PAD).sum().item()
        compare_len = min(pred.size(0), label_len)
        if compare_len == 0:
            continue
        overlap = (pred[:compare_len] == label[:compare_len]).sum().item()
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    return best_label


def _per_head_accuracy(
    pred: torch.Tensor,
    label: torch.Tensor,
    stride: int,
    offset: int,
) -> tuple[int, int]:
    correct = 0
    total = 0
    max_steps = pred.size(0) // stride if stride > 0 else pred.size(0)
    for step in range(max_steps):
        pred_col = step * stride + offset
        label_col = step * stride + offset
        if label_col >= label.size(0) or label[label_col] == PAD:
            break
        if pred_col >= pred.size(0):
            break
        total += 1
        if pred[pred_col] == label[label_col]:
            correct += 1
    return correct, total


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
