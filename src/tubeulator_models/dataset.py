"""PyTorch datasets for the three route-prediction models."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


__all__ = ["RouteDataset", "collate_routes"]

PAD = -1


class RouteDataset(Dataset):
    """
    Each item yields (origin, destination, label).

    `model` selects which label to serve: 'line', 'change', or 'station'.
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

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        origin = ex["origin"]
        dest = ex["destination"]

        if self.model == "line":
            flat = []
            for ln, d in ex["label_line"]:
                flat.extend([ln, d])
            label = torch.tensor(flat, dtype=torch.long)
        elif self.model == "change":
            flat = []
            for ln, d, st in ex["label_change"]:
                flat.extend([ln, d, st])
            label = torch.tensor(flat, dtype=torch.long)
        elif self.model == "station":
            label = torch.tensor(ex["label_station"], dtype=torch.long)
        else:
            raise ValueError(f"Unknown model: {self.model!r}")

        return origin, dest, label


def collate_routes(batch):
    """Collate variable-length labels into padded batch."""
    origins, dests, labels = zip(*batch)
    origins = torch.tensor(origins, dtype=torch.long)
    dests = torch.tensor(dests, dtype=torch.long)
    labels = pad_sequence(labels, batch_first=True, padding_value=PAD)
    return origins, dests, labels
