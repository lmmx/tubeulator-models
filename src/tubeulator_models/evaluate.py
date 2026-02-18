"""Evaluation metrics for route-prediction models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dataset import PAD


__all__ = ["RouteMetrics", "compute_metrics"]


@dataclass
class RouteMetrics:
    exact_match: float
    line_acc: float | None
    dir_acc: float | None
    station_acc: float | None
    topologically_valid: float
    n_examples: int

    def __str__(self) -> str:
        parts = [f"exact={self.exact_match:.1%}"]
        if self.line_acc is not None:
            parts.append(f"line={self.line_acc:.1%}")
        if self.dir_acc is not None:
            parts.append(f"dir={self.dir_acc:.1%}")
        if self.station_acc is not None:
            parts.append(f"station={self.station_acc:.1%}")
        parts.append(f"valid={self.topologically_valid:.1%}")
        return " | ".join(parts)


def _decode_predictions(logits: dict, model_type: str) -> dict[str, torch.Tensor]:
    """Argmax each head to get predicted tokens."""
    preds = {}
    if model_type in ("line", "change"):
        preds["line"] = logits["line"].argmax(-1)  # (B, max_legs)
        preds["dir"] = logits["dir"].argmax(-1)  # (B, max_legs)
    if model_type in ("change", "station"):
        preds["station"] = logits["station"].argmax(-1)  # (B, max_legs) or (B, max_len)
    return preds


def _per_head_accuracy(
    pred: torch.Tensor,
    labels: torch.Tensor,
    stride: int,
    offset: int,
) -> tuple[int, int]:
    """Compare pred[:, step] vs labels[:, step*stride + offset], ignoring PAD."""
    correct = 0
    total = 0
    max_steps = pred.size(1)
    for step in range(max_steps):
        col = step * stride + offset
        if col >= labels.size(1):
            break
        mask = labels[:, col] != PAD
        if mask.any():
            correct += (pred[:, step][mask] == labels[:, col][mask]).sum().item()
            total += mask.sum().item()
    return correct, total


def _exact_match(preds: dict, labels: torch.Tensor, model_type: str) -> tuple[int, int]:
    """Check if the entire predicted sequence matches the label."""
    B = labels.size(0)
    stride = {"line": 2, "change": 3, "station": 1}[model_type]
    correct = 0

    for b in range(B):
        match = True

        if model_type in ("line", "change"):
            max_legs = preds["line"].size(1)
            for step in range(max_legs):
                li = step * stride
                di = step * stride + 1
                if li >= labels.size(1) or labels[b, li] == PAD:
                    break
                if preds["line"][b, step] != labels[b, li]:
                    match = False
                    break
                if di < labels.size(1) and preds["dir"][b, step] != labels[b, di]:
                    match = False
                    break
                if model_type == "change":
                    si = step * stride + 2
                    if si < labels.size(1) and labels[b, si] != PAD:
                        if preds["station"][b, step] != labels[b, si]:
                            match = False
                            break

        elif model_type == "station":
            max_len = preds["station"].size(1)
            for step in range(min(max_len, labels.size(1))):
                if labels[b, step] == PAD:
                    break
                if preds["station"][b, step] != labels[b, step]:
                    match = False
                    break

        if match:
            correct += 1

    return correct, B


def _validity_rate(
    preds: dict,
    labels: torch.Tensor,
    model_type: str,
    n_lines: int,
    n_stations: int,
) -> tuple[int, int]:
    """Check predicted tokens are in valid ranges (basic structural validity)."""
    B = labels.size(0)
    valid = 0

    for b in range(B):
        ok = True
        if model_type in ("line", "change"):
            max_legs = preds["line"].size(1)
            for step in range(max_legs):
                col = step * (2 if model_type == "line" else 3)
                if col >= labels.size(1) or labels[b, col] == PAD:
                    break
                if not (0 <= preds["line"][b, step] < n_lines):
                    ok = False
                    break
                if not (0 <= preds["dir"][b, step] < 2):
                    ok = False
                    break
                if model_type == "change":
                    if not (0 <= preds["station"][b, step] < n_stations):
                        ok = False
                        break
        elif model_type == "station":
            for step in range(min(preds["station"].size(1), labels.size(1))):
                if labels[b, step] == PAD:
                    break
                if not (0 <= preds["station"][b, step] < n_stations):
                    ok = False
                    break
        if ok:
            valid += 1

    return valid, B


def compute_metrics(
    logits: dict,
    labels: torch.Tensor,
    model_type: str,
    n_lines: int,
    n_stations: int,
) -> RouteMetrics:
    """Compute all metrics for a batch of predictions."""
    preds = _decode_predictions(logits, model_type)
    stride = {"line": 2, "change": 3, "station": 1}[model_type]

    # per-head accuracy
    line_c, line_t = 0, 0
    dir_c, dir_t = 0, 0
    station_c, station_t = 0, 0

    if model_type in ("line", "change"):
        line_c, line_t = _per_head_accuracy(preds["line"], labels, stride, 0)
        dir_c, dir_t = _per_head_accuracy(preds["dir"], labels, stride, 1)

    if model_type == "change":
        station_c, station_t = _per_head_accuracy(
            preds["station"],
            labels,
            stride,
            2,
        )
    elif model_type == "station":
        station_c, station_t = _per_head_accuracy(
            preds["station"],
            labels,
            stride,
            0,
        )

    em_c, em_t = _exact_match(preds, labels, model_type)
    val_c, val_t = _validity_rate(
        preds,
        labels,
        model_type,
        n_lines,
        n_stations,
    )

    return RouteMetrics(
        exact_match=em_c / max(em_t, 1),
        line_acc=line_c / max(line_t, 1) if line_t > 0 else None,
        dir_acc=dir_c / max(dir_t, 1) if dir_t > 0 else None,
        station_acc=(station_c / max(station_t, 1)) if station_t > 0 else None,
        topologically_valid=val_c / max(val_t, 1),
        n_examples=em_t,
    )
