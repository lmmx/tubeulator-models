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
    """

    def __init__(self, data_path: Path, model: str = "change"):
        with open(data_path) as f:
            blob = json.load(f)
        self.stations: list[str] = blob["stations"]
        self.lines: list[str] = blob["lines"]
        self.examples: list[dict] = blob["examples"]
        self.model = model

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    def __len__(self) -> int:
        return len(self.examples)

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

    def _pick_route(self, ex: dict) -> dict:
        """Pick a route from available options. Handles both old and new format."""
        if "routes" in ex:
            return random.choice(ex["routes"])
        return ex

    def get_all_labels(self, idx: int) -> list[torch.Tensor]:
        """Return label tensors for ALL valid routes at this OD pair."""
        ex = self.examples[idx]
        if "routes" in ex:
            return [self._make_label(r) for r in ex["routes"]]
        return [self._make_label(ex)]

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        origin = ex["origin"]
        dest = ex["destination"]
        label = self._make_label(self._pick_route(ex))
        return idx, origin, dest, label


def collate_routes(batch):
    """Collate variable-length labels into padded batch."""
    indices, origins, dests, labels = zip(*batch)
    indices = list(indices)
    origins = torch.tensor(origins, dtype=torch.long)
    dests = torch.tensor(dests, dtype=torch.long)
    labels = pad_sequence(labels, batch_first=True, padding_value=PAD)
    return indices, origins, dests, labels