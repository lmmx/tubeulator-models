"""Extract line-aware topology with travel times from GTFS."""

from __future__ import annotations

import csv
import io
import json as _json
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch

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
    route_names: dict[str, str] = field(default_factory=dict)

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
        route_names: dict[str, str] = {}
        with zf.open("stops.txt") as f:
            for row in csv.DictReader(io.TextIOWrapper(f)):
                stop_names[row["stop_id"]] = row["stop_name"]

            with zf.open("routes.txt") as f:
                for row in csv.DictReader(io.TextIOWrapper(f)):
                    name = row.get("route_long_name") or row.get("route_short_name", "")
                    if name:
                        route_names[row["route_id"]] = name

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

        # Detect interleaved branches: any duplicate sequence numbers?
        seqs = [seq for seq, _, _ in ordered]
        has_branches = len(seqs) != len(set(seqs))

        station_seq = [stop_id for _, stop_id, _ in ordered]
        line_seqs[line].append(station_seq)

        if has_branches:
            continue  # skip for adjacency + timing, keep for line_seqs

        for (_, s1, t1), (_, s2, t2) in zip(ordered, ordered[1:]):
            if s1 == s2:
                continue  # self-loop from duplicate stop entries
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
        route_names=route_names,
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


# ── Transfer-aware shortest paths ─────────────────────────────

_SLUG_OVERRIDES: dict[str, str] = {
    "docklands-light-railway": "dlr",
}


def _slugify_line(name: str) -> str:
    """Convert a GTFS route name to a TfL API-style line slug."""
    s = name.lower().strip()
    for suffix in (" line",):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    s = s.replace(" & ", "-").replace(" ", "-")
    return _SLUG_OVERRIDES.get(s, s)


def _normalize_station_name(name: str) -> str:
    s = name.lower().strip()
    for suffix in (
        " underground station",
        " rail station",
        " dlr station",
        " station",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.strip()


def load_interchange_data(path: Path) -> list[dict]:
    """Load interchange-times JSON (array of station objects)."""
    with open(path) as f:
        raw = _json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("data", "rows", "train"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
    raise ValueError(f"Unexpected interchange data format in {path}")


def floyd_warshall_with_transfers(
    topo: Topology,
    stations: list[str],
    lines: list[str],
    interchange_data: list[dict],
    discount: float = 0.65,
    default_transfer_s: float = 240.0,
) -> "torch.Tensor":
    """All-pairs shortest time (seconds) with line-aware transfer penalties.

    Builds a compact (station, line) expanded graph containing only valid
    pairs, runs Floyd-Warshall, then projects back to (N, N) by
    minimising over line pairs at both endpoints.

    Args:
        topo: Network topology from ``extract()``.
        stations: Ordered station list (stop_ids) — same order as the
            training dataset.
        lines: Ordered line list (GTFS route_ids).
        interchange_data: Parsed interchange JSON from
            ``load_interchange_data()``.
        discount: Multiply raw interchange minutes by this before
            converting to seconds.  The raw values are *maximums*
            including waiting; 0.65 is a reasonable starting point
            for the walk-plus-typical-wait component.
        default_transfer_s: Fallback transfer time (seconds, *before*
            discount) for interchange pairs not in the dataset
            (e.g. Elizabeth line, which post-dates the FOI data).
    """
    import torch

    N = len(stations)
    L = len(lines)
    st2i = {s: i for i, s in enumerate(stations)}
    ln2i = {ln: i for i, ln in enumerate(lines)}

    # ── Slug → route_id mapping ───────────────────────────────
    slug_to_routes: dict[str, list[str]] = defaultdict(list)
    for route_id in lines:
        # Route ID itself as slug — handles GTFS feeds with slug-style IDs
        slug_to_routes[route_id.lower()].append(route_id)
        name = topo.route_names.get(route_id, "")
        if name:
            slug = _slugify_line(name)
            if route_id not in slug_to_routes[slug]:
                slug_to_routes[slug].append(route_id)

    def _resolve_slug(slug: str) -> list[int]:
        routes = slug_to_routes.get(slug, [])
        return [ln2i[r] for r in routes if r in ln2i]

    # ── Station matching ──────────────────────────────────────
    name_to_stop: dict[str, str] = {}
    for stop_id in stations:
        name = topo.stop_names.get(stop_id, "")
        if name:
            name_to_stop[_normalize_station_name(name)] = stop_id

    def _match_station(sd: dict) -> int | None:
        uid = sd.get("station_unique_id", "")
        if uid in st2i:
            return st2i[uid]
        tb = sd.get("station_name_tb", "")
        if tb:
            sid = name_to_stop.get(_normalize_station_name(tb))
            if sid is not None:
                return st2i.get(sid)
        return None

    # ── Parse interchange entries ─────────────────────────────
    transfer_cost: dict[tuple[int, int, int], float] = {}  # (si, li_from, li_to) → s
    cross_cost: dict[tuple[int, int], float] = {}  # (si, sj) → s
    unmatched_slugs: set[str] = set()
    n_matched = 0
    n_unmatched = 0

    for sd in interchange_data:
        si = _match_station(sd)
        if si is None:
            n_unmatched += 1
            continue
        n_matched += 1

        for ic in sd.get("interchanges", []):
            mins = ic.get("minutes")
            if mins is None:
                continue
            cost_s = mins * 60.0 * discount

            if ic.get("branch_interchange"):
                slug = ic.get("line_slug", "")
                if not slug:
                    continue
                lis = _resolve_slug(slug)
                if not lis:
                    unmatched_slugs.add(slug)
                for a in range(len(lis)):
                    for b in range(a + 1, len(lis)):
                        key_ab = (si, lis[a], lis[b])
                        key_ba = (si, lis[b], lis[a])
                        if (
                            key_ab not in transfer_cost
                            or cost_s < transfer_cost[key_ab]
                        ):
                            transfer_cost[key_ab] = cost_s
                        if (
                            key_ba not in transfer_cost
                            or cost_s < transfer_cost[key_ba]
                        ):
                            transfer_cost[key_ba] = cost_s

            elif "cross_station" in ic:
                other = _normalize_station_name(ic["cross_station"])
                other_sid = name_to_stop.get(other)
                if other_sid is not None and other_sid in st2i:
                    sj = st2i[other_sid]
                    if (si, sj) not in cross_cost or cost_s < cross_cost[(si, sj)]:
                        cross_cost[(si, sj)] = cost_s
                    if (sj, si) not in cross_cost or cost_s < cross_cost[(sj, si)]:
                        cross_cost[(sj, si)] = cost_s

            else:
                fs = ic.get("from_line_slug", "")
                ts = ic.get("to_line_slug", "")
                if not fs or not ts:
                    continue
                from_lis = _resolve_slug(fs)
                to_lis = _resolve_slug(ts)
                if not from_lis:
                    unmatched_slugs.add(fs)
                if not to_lis:
                    unmatched_slugs.add(ts)
                for lf in from_lis:
                    for lt in to_lis:
                        if lf != lt:
                            key = (si, lf, lt)
                            if key not in transfer_cost or cost_s < transfer_cost[key]:
                                transfer_cost[key] = cost_s

    # Fill symmetric pairs the data didn't list explicitly
    reverse: dict[tuple[int, int, int], float] = {}
    for (si, lf, lt), cost in transfer_cost.items():
        rev = (si, lt, lf)
        if rev not in transfer_cost:
            reverse[rev] = cost
    transfer_cost.update(reverse)

    print(f"  interchange: {n_matched} stations matched, " f"{n_unmatched} unmatched")
    print(
        f"  {len(transfer_cost)} directed transfer edges, "
        f"{len(cross_cost)} cross-station pairs"
    )
    if unmatched_slugs:
        print(f"  WARNING unmatched slugs: {sorted(unmatched_slugs)}")

    # ── Build compact expanded graph ──────────────────────────
    # Only include (station, line) where station is actually on that line.
    valid_nodes: list[tuple[int, int]] = []  # (station_idx, line_idx)
    node_idx: dict[tuple[int, int], int] = {}

    for si, station_id in enumerate(stations):
        for line_id in topo.station_lines.get(station_id, set()):
            li = ln2i.get(line_id)
            if li is not None:
                node_idx[(si, li)] = len(valid_nodes)
                valid_nodes.append((si, li))

    M = len(valid_nodes)
    print(
        f"  expanded graph: {M} (station,line) nodes "
        f"(from {N}×{L}={N * L}, pruned {N * L - M})"
    )

    matrix = torch.full((M, M), float("inf"))
    matrix.fill_diagonal_(0.0)

    # ── Same-line travel edges ────────────────────────────────
    for line_id, edges in topo.edge_time.items():
        li = ln2i.get(line_id)
        if li is None:
            continue
        for (s1, s2), t in edges.items():
            si, sj = st2i.get(s1), st2i.get(s2)
            if si is None or sj is None:
                continue
            ni = node_idx.get((si, li))
            nj = node_idx.get((sj, li))
            if ni is not None and nj is not None:
                cur = matrix[ni, nj].item()
                if t < cur:
                    matrix[ni, nj] = t

    # 120s fallback for adjacencies without observed times
    for line_id, adj in topo.line_adj.items():
        li = ln2i.get(line_id)
        if li is None:
            continue
        for station, neighbors in adj.items():
            si = st2i.get(station)
            if si is None:
                continue
            ni = node_idx.get((si, li))
            if ni is None:
                continue
            for nb in neighbors:
                sj = st2i.get(nb)
                if sj is None:
                    continue
                nj = node_idx.get((sj, li))
                if nj is not None and matrix[ni, nj].item() == float("inf"):
                    matrix[ni, nj] = 120.0

    # ── Transfer edges at interchanges ────────────────────────
    default_cost = default_transfer_s * discount
    for station_id in topo.interchanges:
        si = st2i.get(station_id)
        if si is None:
            continue
        serving = [
            ln2i[ln] for ln in topo.station_lines.get(station_id, set()) if ln in ln2i
        ]
        for lf in serving:
            for lt in serving:
                if lf == lt:
                    continue
                ni = node_idx.get((si, lf))
                nj = node_idx.get((si, lt))
                if ni is None or nj is None:
                    continue
                cost = transfer_cost.get((si, lf, lt), default_cost)
                cur = matrix[ni, nj].item()
                if cost < cur:
                    matrix[ni, nj] = cost

    # ── Cross-station interchange edges ───────────────────────
    for (si, sj), cost in cross_cost.items():
        lines_i = [
            ln2i[ln] for ln in topo.station_lines.get(stations[si], set()) if ln in ln2i
        ]
        lines_j = [
            ln2i[ln] for ln in topo.station_lines.get(stations[sj], set()) if ln in ln2i
        ]
        for li in lines_i:
            for lj in lines_j:
                ni = node_idx.get((si, li))
                nj = node_idx.get((sj, lj))
                if ni is not None and nj is not None:
                    cur = matrix[ni, nj].item()
                    if cost < cur:
                        matrix[ni, nj] = cost

    # ── Floyd-Warshall ────────────────────────────────────────
    print(f"  Floyd-Warshall on {M} nodes...")
    through_k = torch.empty(M, M)
    for k in range(M):
        torch.add(
            matrix[:, k].unsqueeze(1),
            matrix[k, :].unsqueeze(0),
            out=through_k,
        )
        torch.minimum(matrix, through_k, out=matrix)

    # ── Project back to (N, N) ────────────────────────────────
    station_nodes: list[list[int]] = [[] for _ in range(N)]
    for ni, (si, _) in enumerate(valid_nodes):
        station_nodes[si].append(ni)

    projected = torch.full((N, N), float("inf"))
    for s1 in range(N):
        n1 = station_nodes[s1]
        if not n1:
            continue
        # min over all line variants at the origin
        min_from = matrix[n1].min(dim=0).values  # (M,)
        # min over all line variants at each destination
        for s2 in range(N):
            n2 = station_nodes[s2]
            if n2:
                projected[s1, s2] = min_from[n2].min()

    projected.fill_diagonal_(0.0)

    n_inf = (projected == float("inf")).sum().item() - N  # exclude diagonal
    n_reachable = N * (N - 1) - n_inf
    print(f"  {n_reachable} reachable OD pairs, {n_inf} unreachable")

    return projected
