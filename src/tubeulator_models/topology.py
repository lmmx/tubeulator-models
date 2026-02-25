"""Extract line-aware topology with travel times from GTFS."""

from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median


__all__ = ["Topology", "extract"]


def _parse_gtfs_time(s: str) -> int:
    """Parse HH:MM:SS to seconds since midnight. GTFS allows H > 23."""
    parts = s.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


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
    # line -> (from_stop, to_stop) -> median travel time in seconds
    edge_time: dict[str, dict[tuple[str, str], float]]
    # (from_stop, to_stop) -> frozenset of lines serving this directed edge
    edge_lines: dict[tuple[str, str], frozenset[str]]

    def neighbors(self, station: str, line: str) -> set[str]:
        return self.line_adj.get(line, {}).get(station, set())

    def travel_time(self, line: str, from_st: str, to_st: str) -> float:
        """Seconds between adjacent stations on a line. Falls back to 120s."""
        return self.edge_time.get(line, {}).get((from_st, to_st), 120.0)

    def direction_of(self, line: str, from_st: str, to_st: str) -> int:
        """0 = toward higher canonical index, 1 = toward lower."""
        order = self.line_order.get(line, [])
        try:
            return 0 if order.index(to_st) > order.index(from_st) else 1
        except ValueError:
            return 0

    def lines_on_edge(self, from_st: str, to_st: str) -> frozenset[str]:
        """All lines that serve this directed edge."""
        return self.edge_lines.get((from_st, to_st), frozenset())

    def equivalent_lines_for_leg(
        self, line: str, stations: list[str]
    ) -> frozenset[str]:
        """All lines that serve every edge in this leg's station sequence."""
        if len(stations) < 2:
            return frozenset({line})
        valid: set[str] | None = None
        for s1, s2 in zip(stations, stations[1:]):
            el = self.edge_lines.get((s1, s2), frozenset())
            if valid is None:
                valid = set(el)
            else:
                valid &= el
        return frozenset(valid) if valid else frozenset({line})

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

        # Now also parse arrival_time
        trip_stops: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        with zf.open("stop_times.txt") as f:
            for row in csv.DictReader(io.TextIOWrapper(f)):
                trip_stops[row["trip_id"]].append(
                    (
                        int(row["stop_sequence"]),
                        row["stop_id"],
                        row["arrival_time"],
                    )
                )

    line_adj: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    station_lines: dict[str, set[str]] = defaultdict(set)
    line_seqs: dict[str, list[list[str]]] = defaultdict(list)

    # Collect raw travel times: line -> (from, to) -> [seconds, ...]
    raw_times: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for trip_id, stops in trip_stops.items():
        line = trip_line.get(trip_id)
        if not line:
            continue
        ordered = sorted(stops)  # sort by stop_sequence
        if len(ordered) < 2:
            continue

        station_seq = [stop_id for _, stop_id, _ in ordered]
        line_seqs[line].append(station_seq)

        for (_, s1, t1), (_, s2, t2) in zip(ordered, ordered[1:]):
            line_adj[line][s1].add(s2)
            line_adj[line][s2].add(s1)
            station_lines[s1].add(line)
            station_lines[s2].add(line)

            dt = _parse_gtfs_time(t2) - _parse_gtfs_time(t1)
            if 0 < dt < 3600:  # sanity: positive and under 60 minutes
                raw_times[line][(s1, s2)].append(float(dt))

    # Aggregate to median
    edge_time: dict[str, dict[tuple[str, str], float]] = {}
    for line, edges in raw_times.items():
        edge_time[line] = {pair: median(times) for pair, times in edges.items()}

    line_order = {ln: max(seqs, key=len) for ln, seqs in line_seqs.items()}
    interchanges = {s for s, ls in station_lines.items() if len(ls) >= 2}

    # Build edge → lines equivalence map
    edge_lines_raw: dict[tuple[str, str], set[str]] = defaultdict(set)
    for line, adj in line_adj.items():
        for station, neighbors in adj.items():
            for neighbor in neighbors:
                edge_lines_raw[(station, neighbor)].add(line)
    edge_lines = {pair: frozenset(lines) for pair, lines in edge_lines_raw.items()}

    n_timed = sum(len(e) for e in edge_time.values())
    n_total = sum(sum(len(nbrs) for nbrs in adj.values()) for adj in line_adj.values())
    n_shared = sum(1 for lines in edge_lines.values() if len(lines) > 1)
    print(f"  travel times: {n_timed} edges timed out of {n_total} adjacencies")
    print(f"  shared edges: {n_shared} edges served by multiple lines")

    return Topology(
        line_adj=dict(line_adj),
        station_lines=dict(station_lines),
        interchanges=interchanges,
        stop_names=stop_names,
        line_order=line_order,
        edge_time=edge_time,
        edge_lines=edge_lines,
    )


def build_adj_mask(topo: Topology, stations: list[str]) -> torch.Tensor:
    import torch

    st2i = {s: i for i, s in enumerate(stations)}
    N = len(stations)
    mask = torch.zeros(N, N, dtype=torch.bool)
    for i in range(N):
        mask[i, i] = True  # self-loop: sequence starts at origin
    for adj in topo.line_adj.values():
        for station, neighbors in adj.items():
            i = st2i.get(station)
            if i is None:
                continue
            for neighbor in neighbors:
                j = st2i.get(neighbor)
                if j is not None:
                    mask[i, j] = True
    return mask


def build_line_station_mask(
    topo: Topology, stations: list[str], lines: list[str]
) -> torch.Tensor:
    """(n_lines, n_stations) boolean: mask[l, s] = True iff station s is on line l."""
    import torch

    st2i = {s: i for i, s in enumerate(stations)}
    ln2i = {ln: i for i, ln in enumerate(lines)}
    N = len(stations)
    L = len(lines)
    mask = torch.zeros(L, N, dtype=torch.bool)
    for station, serving_lines in topo.station_lines.items():
        si = st2i.get(station)
        if si is None:
            continue
        for line in serving_lines:
            li = ln2i.get(line)
            if li is not None:
                mask[li, si] = True
    return mask


def build_edge_time_matrix(topo: Topology, stations: list[str]) -> torch.Tensor:
    """(N, N) matrix: minimum travel time in seconds between adjacent stations."""
    import torch

    N = len(stations)
    st2i = {s: i for i, s in enumerate(stations)}
    matrix = torch.full((N, N), float("inf"))
    matrix.fill_diagonal_(0.0)

    # All adjacencies get 120s fallback — guarantees no inf for any traversable edge
    for adj in topo.line_adj.values():
        for station, neighbors in adj.items():
            i = st2i.get(station)
            if i is None:
                continue
            for neighbor in neighbors:
                j = st2i.get(neighbor)
                if j is not None:
                    matrix[i, j] = 120.0

    # Overwrite with observed times (min across lines)
    for line, edges in topo.edge_time.items():
        for (s1, s2), time in edges.items():
            i, j = st2i.get(s1), st2i.get(s2)
            if i is not None and j is not None:
                matrix[i, j] = min(matrix[i, j].item(), time)

    return matrix


def floyd_warshall_times(topo: Topology, stations: list[str]) -> torch.Tensor:
    """(N, N) all-pairs shortest travel time in seconds.

    Runs Floyd-Warshall on the raw adjacency graph, ignoring lines
    and transfer penalties. This is the true unconstrained optimum.
    """
    import torch

    matrix = build_edge_time_matrix(topo, stations)
    N = matrix.size(0)

    # Self-loops = 0
    matrix.fill_diagonal_(0.0)

    for k in range(N):
        through_k = matrix[:, k].unsqueeze(1) + matrix[k, :].unsqueeze(0)
        matrix = torch.minimum(matrix, through_k)

    return matrix
