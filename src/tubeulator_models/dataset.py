"""PyTorch datasets for the three route-prediction models."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


__all__ = ["RouteDataset", "collate_routes"]

PAD = -1


class RouteDataset(Dataset):
    """
    Each item yields (idx, origin, destination, label).

    When multiple routes exist for an OD pair, __getitem__ samples
    one uniformly at random each time it's called.

    All label tensors are pre-computed at init to avoid per-call Python overhead.
    """

    def __init__(self, data_path: Path, model: str = "change"):
        with open(data_path) as f:
            blob = json.load(f)
        self.stations: list[str] = blob["stations"]
        self.lines: list[str] = blob["lines"]
        self.model = model

        # Pre-compute everything into tensors at init time
        raw_examples = blob["examples"]
        self._origins = torch.zeros(len(raw_examples), dtype=torch.long)
        self._dests = torch.zeros(len(raw_examples), dtype=torch.long)
        # Each element: list of pre-built label tensors (one per valid route)
        self._all_labels: list[list[torch.Tensor]] = []

        for i, ex in enumerate(raw_examples):
            self._origins[i] = ex["origin"]
            self._dests[i] = ex["destination"]

            routes = ex.get("routes", [ex])
            labels = [self._make_label(r) for r in routes]
            self._all_labels.append(labels)

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    def __len__(self) -> int:
        return len(self._origins)

    def _make_label(self, route: dict) -> torch.Tensor:
        """Build a label tensor from a single route dict."""
        if self.model == "line":
            flat = []
            for ln, d in route["label_line"]:
                flat.extend([ln, d])
            return torch.tensor(flat, dtype=torch.long)
        elif self.model == "change":
            flat = []
            for ln, d, st in route["label_change"]:
                flat.extend([ln, d, st])
            return torch.tensor(flat, dtype=torch.long)
        elif self.model == "station":
            return torch.tensor(route["label_station"], dtype=torch.long)
        else:
            raise ValueError(f"Unknown model: {self.model!r}")

    def get_all_labels(self, idx: int) -> list[torch.Tensor]:
        """Return label tensors for ALL valid routes at this OD pair."""
        return self._all_labels[idx]

    def __getitem__(self, idx: int):
        label = random.choice(self._all_labels[idx])
        return idx, self._origins[idx].item(), self._dests[idx].item(), label


def collate_routes(batch):
    """Collate variable-length labels into padded batch."""
    indices, origins, dests, labels = zip(*batch)
    indices = list(indices)
    origins = torch.tensor(origins, dtype=torch.long)
    dests = torch.tensor(dests, dtype=torch.long)
    labels = pad_sequence(labels, batch_first=True, padding_value=PAD)
    return indices, origins, dests, labels
