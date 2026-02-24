"""Evaluation metrics for route-prediction models."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .dataset import PAD


__all__ = ["RouteMetrics", "compute_metrics"]


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


def _is_valid(
    pred: torch.Tensor,
    model_type: str,
    n_lines: int,
    n_stations: int,
    ref_label: torch.Tensor,
) -> bool:
    """Check predicted tokens are in valid ranges."""
    stride = {"line": 2, "change": 3, "station": 1}[model_type]
    label_len = (ref_label != PAD).sum().item()
    n_steps = label_len // stride if stride > 1 else label_len

    for step in range(n_steps):
        if model_type in ("line", "change"):
            li = step * stride
            di = step * stride + 1
            if li >= pred.size(0) or not (0 <= pred[li] < n_lines):
                return False
            if di >= pred.size(0) or not (0 <= pred[di] < 2):
                return False
            if model_type == "change":
                si = step * stride + 2
                if si >= pred.size(0) or not (0 <= pred[si] < n_stations):
                    return False
        elif model_type == "station":
            if step >= pred.size(0) or not (0 <= pred[step] < n_stations):
                return False
    return True


def compute_metrics(
    beam_results: list[list[tuple[torch.Tensor, float]]],
    model_type: str,
    n_lines: int,
    n_stations: int,
    all_valid_labels: list[list[torch.Tensor]],
    strat_keys: torch.Tensor | None = None,
) -> RouteMetrics:
    """
    Compute metrics from beam search output.

    beam_results: list (length B) of lists of (sequence, log_prob) tuples.
    all_valid_labels: list (length B) of lists of all valid route labels.
    """
    B = len(all_valid_labels)
    stride = {"line": 2, "change": 3, "station": 1}[model_type]

    em_correct = 0
    beam_correct = 0
    valid_correct = 0
    line_c, line_t = 0, 0
    dir_c, dir_t = 0, 0
    station_c, station_t = 0, 0
    top1_matches: list[bool] = []

    for b in range(B):
        routes = all_valid_labels[b]
        beams = beam_results[b]

        if not beams:
            top1_matches.append(False)
            continue

        top_pred = beams[0][0]  # best beam = top-1 prediction

        # Exact match: does top-1 prediction match any valid route?
        top1_match = any(_sequences_match(top_pred, lbl) for lbl in routes)
        top1_matches.append(top1_match)
        if top1_match:
            em_correct += 1

        # Beam match: does ANY beam match any valid route?
        beam_match = any(
            _sequences_match(pred, lbl) for pred, _lp in beams for lbl in routes
        )
        if beam_match:
            beam_correct += 1

        # Per-head accuracy on top-1 vs best-matching label
        best_label = _best_matching_label(top_pred, routes)

        if model_type in ("line", "change"):
            lc, lt = _per_head_accuracy(top_pred, best_label, stride, 0)
            dc, dt = _per_head_accuracy(top_pred, best_label, stride, 1)
            line_c += lc
            line_t += lt
            dir_c += dc
            dir_t += dt

        if model_type == "change":
            sc, st = _per_head_accuracy(top_pred, best_label, stride, 2)
            station_c += sc
            station_t += st
        elif model_type == "station":
            sc, st = _per_head_accuracy(top_pred, best_label, 1, 0)
            station_c += sc
            station_t += st

        # Validity of top-1
        if _is_valid(top_pred, model_type, n_lines, n_stations, routes[0]):
            valid_correct += 1

        stratified = None
        if strat_keys is not None:
            from collections import defaultdict

            buckets: dict[int, list[bool]] = defaultdict(list)
            for match, key in zip(top1_matches, strat_keys.tolist()):
                buckets[key].append(match)
            stratified = {
                k: (sum(v) / len(v), len(v)) for k, v in sorted(buckets.items())
            }

    return RouteMetrics(
        exact_match=em_correct / max(B, 1),
        any_in_beam=beam_correct / max(B, 1),
        line_acc=line_c / max(line_t, 1) if line_t > 0 else None,
        dir_acc=dir_c / max(dir_t, 1) if dir_t > 0 else None,
        station_acc=station_c / max(station_t, 1) if station_t > 0 else None,
        topologically_valid=valid_correct / max(B, 1),
        n_examples=B,
        stratified=stratified,
    )


@dataclass
class NextHopMetrics:
    step_acc: float  # fraction of correct next-hop predictions
    rollout_success: float  # fraction of rollouts reaching destination
    avg_length_ratio: float  # mean(rollout_len / best_gt_len) for successful rollouts
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


def compute_nexthop_rollout_metrics(
    rollouts: list[list[int]],
    dests: torch.Tensor,
    gt_routes: list[list[list[int]]],
    strat_keys: list[int] | None = None,
) -> NextHopMetrics:
    """
    Evaluate rollout quality against ground-truth routes.

    gt_routes[b] is a list of valid station sequences for OD pair b.
    """
    from collections import defaultdict

    B = len(rollouts)
    n_success = 0
    length_ratios: list[float] = []
    bucket_data: dict[int, list[tuple[bool, float]]] = defaultdict(list)

    for b in range(B):
        route = rollouts[b]
        dest = dests[b].item()
        reached = len(route) > 1 and route[-1] == dest

        # Best ground-truth length for this OD pair
        best_gt_len = min(len(r) for r in gt_routes[b])
        ratio = (
            len(route) / best_gt_len if reached and best_gt_len > 0 else float("inf")
        )

        if reached:
            n_success += 1
            length_ratios.append(ratio)

        if strat_keys is not None:
            bucket_data[strat_keys[b]].append((reached, ratio))

    avg_ratio = (
        sum(length_ratios) / max(len(length_ratios), 1)
        if length_ratios
        else float("inf")
    )

    stratified = None
    if strat_keys is not None:
        stratified = {}
        for k, entries in sorted(bucket_data.items()):
            succ = sum(1 for r, _ in entries if r)
            ratios = [r for ok, r in entries if ok]
            avg_r = sum(ratios) / max(len(ratios), 1) if ratios else float("inf")
            stratified[k] = (succ / len(entries), avg_r, len(entries))

    return NextHopMetrics(
        step_acc=0.0,  # filled in by caller
        rollout_success=n_success / max(B, 1),
        avg_length_ratio=avg_ratio,
        n_steps=0,
        n_rollouts=B,
        stratified=stratified,
    )
