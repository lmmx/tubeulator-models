"""PyTorch datasets for the three route-prediction models."""

from __future__ import annotations

import json
from pathlib import Path

import torch


__all__ = ["NextHopGPUDataset"]

PAD = -1


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
        step_remaining: list[float] = []

        for od_idx, ex in enumerate(blob["examples"]):
            origin = ex["origin"]
            dest = ex["destination"]
            od_origins.append(origin)
            od_dests.append(dest)

            routes = ex.get("routes", [ex])
            seen_triples: dict[tuple[int, int, int], int] = {}
            route_seqs: list[list[int]] = []

            for route in routes:
                stations = route["label_station"]
                cum_times = route.get("cum_times")
                route_seqs.append(stations)
                for i in range(len(stations) - 1):
                    triple = (stations[i], dest, stations[i + 1])
                    if cum_times is not None:
                        remaining = (cum_times[-1] - cum_times[i]) / 60.0
                    else:
                        remaining = float(len(stations) - 1 - i)

                    if triple in seen_triples:
                        idx = seen_triples[triple]
                        step_remaining[idx] = min(step_remaining[idx], remaining)
                    else:
                        seen_triples[triple] = len(step_current)
                        step_current.append(stations[i])
                        step_dest.append(dest)
                        step_target.append(stations[i + 1])
                        step_od.append(od_idx)
                        step_remaining.append(remaining)

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
