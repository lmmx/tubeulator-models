"""Extract line-aware topology from GTFS."""

from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


__all__ = ["Topology", "extract"]


@dataclass
class Topology:
    # line → station → {adjacent stations on that line}
    line_adj: dict[str, dict[str, set[str]]]
    # station → {lines serving it}
    station_lines: dict[str, set[str]]
    interchanges: set[str]
    stop_names: dict[str, str]
    # line → longest observed trip sequence (canonical order)
    line_order: dict[str, list[str]]

    def neighbors(self, station: str, line: str) -> set[str]:
        return self.line_adj.get(line, {}).get(station, set())

    def direction_of(self, line: str, from_st: str, to_st: str) -> int:
        """0 = toward higher canonical index, 1 = toward lower."""
        order = self.line_order.get(line, [])
        try:
            return 0 if order.index(to_st) > order.index(from_st) else 1
        except ValueError:
            return 0

    @property
    def all_stations(self) -> list[str]:
        return sorted(self.station_lines)

    @property
    def all_lines(self) -> list[str]:
        return sorted(self.line_adj)

    @property
    def n_stations(self) -> int:
        return len(self.station_lines)

    @property
    def n_lines(self) -> int:
        return len(self.line_adj)


def extract(gtfs_path: Path) -> Topology:
    with zipfile.ZipFile(gtfs_path) as zf:
        stop_names = {}
        with zf.open("stops.txt") as f:
            for row in csv.DictReader(io.TextIOWrapper(f)):
                stop_names[row["stop_id"]] = row["stop_name"]

        trip_line: dict[str, str] = {}
        with zf.open("trips.txt") as f:
            for row in csv.DictReader(io.TextIOWrapper(f)):
                trip_line[row["trip_id"]] = row["route_id"]

        trip_stops: dict[str, list[tuple[int, str]]] = defaultdict(list)
        with zf.open("stop_times.txt") as f:
            for row in csv.DictReader(io.TextIOWrapper(f)):
                trip_stops[row["trip_id"]].append(
                    (int(row["stop_sequence"]), row["stop_id"])
                )

    line_adj: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    station_lines: dict[str, set[str]] = defaultdict(set)
    line_seqs: dict[str, list[list[str]]] = defaultdict(list)

    for trip_id, stops in trip_stops.items():
        line = trip_line.get(trip_id)
        if not line:
            continue
        ordered = [s for _, s in sorted(stops)]
        if len(ordered) < 2:
            continue
        line_seqs[line].append(ordered)
        for a, b in zip(ordered, ordered[1:]):
            line_adj[line][a].add(b)
            line_adj[line][b].add(a)
            station_lines[a].add(line)
            station_lines[b].add(line)

    line_order = {ln: max(seqs, key=len) for ln, seqs in line_seqs.items()}
    interchanges = {s for s, ls in station_lines.items() if len(ls) >= 2}

    return Topology(
        line_adj=dict(line_adj),
        station_lines=dict(station_lines),
        interchanges=interchanges,
        stop_names=stop_names,
        line_order=line_order,
    )
