"""Evaluation metrics for route-prediction models."""

from __future__ import annotations

from dataclasses import dataclass

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
