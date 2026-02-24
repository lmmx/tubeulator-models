"""PyTorch datasets for the three route-prediction models."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


__all__ = ["RouteDataset", "GPURouteDataset", "collate_routes"]

PAD = -1


def _make_label(route: dict, model: str) -> torch.Tensor:
    if model == "line":
        flat = []
        for ln, d in route["label_line"]:
            flat.extend([ln, d])
        return torch.tensor(flat, dtype=torch.long)
    elif model == "change":
        flat = []
        for ln, d, st in route["label_change"]:
            flat.extend([ln, d, st])
        return torch.tensor(flat, dtype=torch.long)
    elif model == "station":
        return torch.tensor(route["label_station"], dtype=torch.long)
    else:
        raise ValueError(f"Unknown model: {model!r}")


class GPURouteDataset:
    """
    Entire dataset pre-loaded and padded on GPU.

    All labels stored as a single (n_examples, max_routes, max_label_len) tensor
    so resampling is a single vectorized gather — no Python loops.
    """

    def __init__(self, data_path: Path, model: str, device: torch.device):
        with open(data_path) as f:
            blob = json.load(f)

        self.stations: list[str] = blob["stations"]
        self.lines: list[str] = blob["lines"]
        self.model = model
        self.device = device

        examples = blob["examples"]
        n = len(examples)

        origins_list = []
        dests_list = []
        all_label_sets: list[list[torch.Tensor]] = []
        max_len = 0
        max_routes = 0

        for ex in examples:
            origins_list.append(ex["origin"])
            dests_list.append(ex["destination"])
            routes = ex.get("routes", [ex])
            labels = [_make_label(r, model) for r in routes]
            all_label_sets.append(labels)
            max_routes = max(max_routes, len(labels))
            for lab in labels:
                max_len = max(max_len, lab.size(0))

        self.origins = torch.tensor(origins_list, dtype=torch.long, device=device)
        self.dests = torch.tensor(dests_list, dtype=torch.long, device=device)
        self.n = n
        self.max_label_len = max_len
        self.max_routes = max_routes

        # (n, max_routes, max_label_len) — all routes padded into one tensor
        self._label_bank = torch.full(
            (n, max_routes, max_len), PAD, dtype=torch.long, device=device
        )
        # (n,) — number of valid routes per example
        self._n_routes = torch.ones(n, dtype=torch.long, device=device)

        for i, labels in enumerate(all_label_sets):
            self._n_routes[i] = len(labels)
            for j, lab in enumerate(labels):
                self._label_bank[i, j, : lab.size(0)] = lab

        # Pre-allocate active labels — filled by resample_labels()
        self.labels = torch.full((n, max_len), PAD, dtype=torch.long, device=device)
        # Arange for advanced indexing — allocated once
        self._arange = torch.arange(n, device=device)
        self.resample_labels()

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    def resample_labels(self) -> None:
        """Randomly pick one route per example. Fully vectorized on GPU."""
        rand = torch.randint(0, 2**31, (self.n,), device=self.device, dtype=torch.long)
        route_idx = rand % self._n_routes  # (n,)
        self.labels = self._label_bank[self._arange, route_idx]  # (n, max_len)

    def get_all_labels_batch(self, indices: torch.Tensor) -> list[list[torch.Tensor]]:
        """
        Return all valid route labels for a batch of indices.
        All tensors stay on GPU to match beam decode outputs.
        """
        bank_slice = self._label_bank[indices]  # (batch, max_routes, max_len)
        n_routes_slice = self._n_routes[indices]  # (batch,)
        # Single CPU sync just for the counts
        counts = n_routes_slice.tolist()

        result = []
        for i, nr in enumerate(counts):
            result.append([bank_slice[i, j] for j in range(nr)])
        return result

    def get_batch(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice a batch by indices. Pure GPU indexing."""
        return indices, self.origins[indices], self.dests[indices], self.labels[indices]


class RouteDataset(Dataset):
    """CPU-based dataset kept for compatibility."""

    def __init__(self, data_path: Path, model: str = "change"):
        with open(data_path) as f:
            blob = json.load(f)
        self.stations: list[str] = blob["stations"]
        self.lines: list[str] = blob["lines"]
        self.model = model

        raw_examples = blob["examples"]
        self._origins = torch.zeros(len(raw_examples), dtype=torch.long)
        self._dests = torch.zeros(len(raw_examples), dtype=torch.long)
        self._all_labels: list[list[torch.Tensor]] = []

        for i, ex in enumerate(raw_examples):
            self._origins[i] = ex["origin"]
            self._dests[i] = ex["destination"]
            routes = ex.get("routes", [ex])
            labels = [_make_label(r, self.model) for r in routes]
            self._all_labels.append(labels)

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    def __len__(self) -> int:
        return len(self._origins)

    def get_all_labels(self, idx: int) -> list[torch.Tensor]:
        return self._all_labels[idx]

    def __getitem__(self, idx: int):
        label = random.choice(self._all_labels[idx])
        return idx, self._origins[idx].item(), self._dests[idx].item(), label


def collate_routes(batch):
    indices, origins, dests, labels = zip(*batch)
    indices = list(indices)
    origins = torch.tensor(origins, dtype=torch.long)
    dests = torch.tensor(dests, dtype=torch.long)
    labels = pad_sequence(labels, batch_first=True, padding_value=PAD)
    return indices, origins, dests, labels


class NextHopGPUDataset:
    """
    Dataset for next-hop policy training.

    Explodes each route [s0, s1, ..., sN] with destination D into
    per-step samples: (current=si, dest=D, target=s_{i+1}).

    Stores an OD-pair index per step so train/val splitting happens
    at the OD level (no leakage).
    """

    def __init__(self, data_path: Path, device: torch.device):
        with open(data_path) as f:
            blob = json.load(f)

        self.stations: list[str] = blob["stations"]
        self.lines: list[str] = blob["lines"]
        self.device = device

        step_current: list[int] = []
        step_dest: list[int] = []
        step_target: list[int] = []
        step_od: list[int] = []

        # OD-level data for rollout eval
        od_origins: list[int] = []
        od_dests: list[int] = []
        self.od_routes: list[list[list[int]]] = []  # ground truth station seqs
        step_remaining: list[int] = []

        for od_idx, ex in enumerate(blob["examples"]):
            origin = ex["origin"]
            dest = ex["destination"]
            od_origins.append(origin)
            od_dests.append(dest)

            routes = ex.get("routes", [ex])
            seen_triples: set[tuple[int, int, int]] = set()
            route_seqs: list[list[int]] = []

            for route in routes:
                stations = route["label_station"]
                route_seqs.append(stations)
                for i in range(len(stations) - 1):
                    triple = (stations[i], dest, stations[i + 1])
                    if triple not in seen_triples:
                        seen_triples.add(triple)
                        step_current.append(stations[i])
                        step_dest.append(dest)
                        step_target.append(stations[i + 1])
                        step_od.append(od_idx)
                        step_remaining.append(len(stations) - 1 - i)

            self.od_routes.append(route_seqs)

        self.step_current = torch.tensor(step_current, dtype=torch.long, device=device)
        self.step_dest = torch.tensor(step_dest, dtype=torch.long, device=device)
        self.step_target = torch.tensor(step_target, dtype=torch.long, device=device)
        self.step_od = torch.tensor(step_od, dtype=torch.long, device=device)
        self.n_steps = len(step_current)
        self.step_remaining = torch.tensor(
            step_remaining, dtype=torch.float32, device=device
        )

        self.od_origins = torch.tensor(od_origins, dtype=torch.long, device=device)
        self.od_dests = torch.tensor(od_dests, dtype=torch.long, device=device)
        self.n_od = len(od_origins)

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    def steps_for_ods(self, od_indices: torch.Tensor) -> torch.Tensor:
        """Return step-level indices belonging to the given OD pairs."""
        mask = torch.isin(self.step_od, od_indices)
        return mask.nonzero(as_tuple=False).squeeze(1)

    def get_step_batch(
        self, step_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (current, dest, target, remaining_hops) for given step indices."""
        return (
            self.step_current[step_indices],
            self.step_dest[step_indices],
            self.step_target[step_indices],
            self.step_remaining[step_indices],
        )

    def get_od_batch(
        self, od_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[list[int]]]]:
        """Return (origins, dests, ground_truth_routes) for rollout eval."""
        origins = self.od_origins[od_indices]
        dests = self.od_dests[od_indices]
        routes = [self.od_routes[i] for i in od_indices.tolist()]
        return origins, dests, routes
