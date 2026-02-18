"""Enumerate topologically valid routes and produce training labels."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .topology import Topology


__all__ = ["Route", "find_routes", "build_dataset"]


@dataclass
class Route:
    stations: list[str]
    legs: list[tuple[str, int, list[str]]]  # (line, direction, stations_this_leg)

    @property
    def n_transfers(self) -> int:
        return max(0, len(self.legs) - 1)

    def label_line(self) -> list[tuple[str, int]]:
        """Line-sequence model: [(line, direction), ...]"""
        return [(ln, d) for ln, d, _ in self.legs]

    def label_change(self) -> list[tuple[str, int, str]]:
        """Interchange model: [(line, direction, exit_station), ...]"""
        return [(ln, d, sts[-1]) for ln, d, sts in self.legs]

    def label_station(self) -> list[str]:
        """Full station-sequence model."""
        return list(self.stations)


def find_routes(
    topo: Topology,
    origin: str,
    dest: str,
    max_transfers: int,
    max_results: int,
) -> list[Route]:
    if origin == dest:
        return []

    results: list[Route] = []
    queue: deque = deque()

    for line in topo.station_lines.get(origin, set()):
        queue.append(
            (origin, line, 0, frozenset([origin]), [origin], [(line, [origin])])
        )

    seen: set[tuple[str, str, int]] = set()

    while queue and len(results) < max_results:
        station, line, xfers, visited, path, legs = queue.popleft()

        key = (station, line, xfers)
        if key in seen:
            continue
        seen.add(key)

        if station == dest and len(path) > 1:
            route_legs = []
            for leg_line, leg_sts in legs:
                d = (
                    topo.direction_of(leg_line, leg_sts[0], leg_sts[-1])
                    if len(leg_sts) >= 2
                    else 0
                )
                route_legs.append((leg_line, d, leg_sts))
            results.append(Route(stations=path, legs=route_legs))
            continue

        for nxt in topo.neighbors(station, line):
            if nxt not in visited:
                new_legs = legs[:-1] + [(legs[-1][0], legs[-1][1] + [nxt])]
                queue.append(
                    (
                        nxt,
                        line,
                        xfers,
                        visited | {nxt},
                        path + [nxt],
                        new_legs,
                    )
                )

        if station in topo.interchanges and xfers < max_transfers:
            for other in topo.station_lines[station]:
                if other != line:
                    queue.append(
                        (
                            station,
                            other,
                            xfers + 1,
                            visited,
                            path,
                            legs + [(other, [station])],
                        )
                    )

    results.sort(key=lambda r: (r.n_transfers, len(r.stations)))
    return results


def build_dataset(
    topo: Topology,
    max_transfers: int,
    max_results: int,
    output_path: Path | None = None,
) -> list[dict]:
    """Build training examples for all OD pairs. Returns list of dicts."""
    stations = topo.all_stations
    lines = topo.all_lines
    st2i = {s: i for i, s in enumerate(stations)}
    ln2i = {ln: i for i, ln in enumerate(lines)}

    examples = []
    total = len(stations)
    for idx, origin in enumerate(stations):
        if idx % 25 == 0:
            print(f"  routing from {idx}/{total} stations...")
        for dest in stations:
            if origin == dest:
                continue
            routes = find_routes(topo, origin, dest, max_transfers, max_results)
            if not routes:
                continue
            best = routes[0]
            examples.append(
                {
                    "origin": st2i[origin],
                    "destination": st2i[dest],
                    "label_line": [(ln2i[ln], d) for ln, d in best.label_line()],
                    "label_change": [
                        (ln2i[ln], d, st2i[st]) for ln, d, st in best.label_change()
                    ],
                    "label_station": [st2i[s] for s in best.label_station()],
                }
            )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {"stations": stations, "lines": lines, "examples": examples},
                f,
            )
        print(f"Saved {len(examples):,} examples → {output_path}")

    return examples
