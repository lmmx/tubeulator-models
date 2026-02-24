"""Enumerate topologically valid routes using travel-time-weighted Dijkstra."""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from .topology import Topology


__all__ = ["Route", "find_routes", "build_dataset"]


@dataclass
class Route:
    stations: list[str]
    legs: list[tuple[str, int, list[str]]]  # (line, direction, stations_this_leg)
    travel_time: float  # total seconds including transfer penalties

    @property
    def n_transfers(self) -> int:
        return max(0, len(self.legs) - 1)

    def label_line(self) -> list[tuple[str, int]]:
        return [(ln, d) for ln, d, _ in self.legs]

    def label_change(self) -> list[tuple[str, int, str]]:
        return [(ln, d, sts[-1]) for ln, d, sts in self.legs]

    def label_station(self) -> list[str]:
        return list(self.stations)

    def signature(self) -> tuple:
        """Station sequence is the canonical route identity.

        Two routes visiting the same stations in the same order are the same
        physical journey regardless of which shared-track line they name.
        """
        return tuple(self.stations)


def find_routes(
    topo: Topology,
    origin: str,
    dest: str,
    max_transfers: int,
    max_results: int,
    transfer_penalty: float,
) -> list[Route]:
    """Dijkstra over (station, line, n_transfers) state space."""
    if origin == dest:
        return []

    # Priority queue: (cost, tiebreak, station, line, n_transfers, path, legs)
    pq: list = []
    counter = 0
    seen_signatures: set[tuple] = set()
    results: list[Route] = []

    for line in topo.station_lines.get(origin, set()):
        heapq.heappush(
            pq, (0.0, counter, origin, line, 0, [origin], [(line, [origin])])
        )
        counter += 1

    # (station, line, n_transfers) -> best cost seen
    best_cost: dict[tuple[str, str, int], float] = {}

    while pq and len(results) < max_results:
        cost, _, station, line, xfers, path, legs = heapq.heappop(pq)

        key = (station, line, xfers)
        if key in best_cost and best_cost[key] < cost:
            continue
        best_cost[key] = cost

        if station == dest and len(path) > 1:
            route_legs = []
            for leg_line, leg_sts in legs:
                d = (
                    topo.direction_of(leg_line, leg_sts[0], leg_sts[-1])
                    if len(leg_sts) >= 2
                    else 0
                )
                route_legs.append((leg_line, d, leg_sts))

            route = Route(stations=path, legs=route_legs, travel_time=cost)
            sig = route.signature()
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                results.append(route)
            continue

        # Continue on same line
        for nxt in topo.neighbors(station, line):
            if nxt in path:  # prevent cycles
                continue
            edge_cost = topo.travel_time(line, station, nxt)
            new_cost = cost + edge_cost
            nxt_key = (nxt, line, xfers)
            if nxt_key not in best_cost or new_cost < best_cost[nxt_key]:
                new_legs = legs[:-1] + [(legs[-1][0], legs[-1][1] + [nxt])]
                heapq.heappush(
                    pq,
                    (
                        new_cost,
                        counter,
                        nxt,
                        line,
                        xfers,
                        path + [nxt],
                        new_legs,
                    ),
                )
                counter += 1

        # Transfer at interchange
        if station in topo.interchanges and xfers < max_transfers:
            for other in topo.station_lines[station]:
                if other != line:
                    new_cost = cost + transfer_penalty
                    xfer_key = (station, other, xfers + 1)
                    if xfer_key not in best_cost or new_cost < best_cost[xfer_key]:
                        heapq.heappush(
                            pq,
                            (
                                new_cost,
                                counter,
                                station,
                                other,
                                xfers + 1,
                                path,
                                legs + [(other, [station])],
                            ),
                        )
                        counter += 1

    results.sort(key=lambda r: r.travel_time)
    return results


def _expand_line_variants(
    route: Route,
    topo: Topology,
    ln2i: dict[str, int],
    st2i: dict[str, int],
    transfer_penalty: float = 240.0,
) -> list[dict]:
    """Generate label dicts for all line-equivalent variants of a route."""
    leg_line_options: list[list[str]] = []
    for line, _dir, stations in route.legs:
        equiv = topo.equivalent_lines_for_leg(line, stations)
        leg_line_options.append(sorted(equiv))

    station_label = [st2i[s] for s in route.stations]
    variants: list[dict] = []
    seen: set[tuple] = set()

    for line_combo in product(*leg_line_options):
        label_line = []
        label_change = []

        cum_times = [0.0]
        t = 0.0
        for leg_idx, ((_, _, sts), new_ln) in enumerate(zip(route.legs, line_combo)):
            d = topo.direction_of(new_ln, sts[0], sts[-1]) if len(sts) >= 2 else 0
            label_line.append((ln2i[new_ln], d))
            label_change.append((ln2i[new_ln], d, st2i[sts[-1]]))

            if leg_idx > 0:
                t += transfer_penalty
            for i in range(len(sts) - 1):
                t += topo.travel_time(new_ln, sts[i], sts[i + 1])
                cum_times.append(t)

        key = (tuple(tuple(x) for x in label_line),)
        if key in seen:
            continue
        seen.add(key)

        variants.append(
            {
                "label_line": label_line,
                "label_change": label_change,
                "label_station": station_label,
                "travel_time": route.travel_time,
                "cum_times": cum_times,
            }
        )

    return variants


def build_dataset(
    topo: Topology,
    max_transfers: int,
    max_results: int,
    transfer_penalty: float,
    output_path: Path | None = None,
) -> list[dict]:
    """Build training examples for all OD pairs.

    Stores all valid routes per pair, expanded with line-equivalent variants
    for shared-track segments.
    """
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    stations = topo.all_stations
    lines = topo.all_lines
    st2i = {s: i for i, s in enumerate(stations)}
    ln2i = {ln: i for i, ln in enumerate(lines)}

    examples = []
    n_routes = 0
    n_variants = 0

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("[cyan]{task.completed}/{task.total} origins"),
        TextColumn("·"),
        TextColumn("[green]{task.fields[pairs]:,} pairs"),
        TextColumn("[green]{task.fields[routes]:,} routes"),
        TextColumn("[green]{task.fields[variants]:,} variants"),
        refresh_per_second=10,
    ) as progress:
        task = progress.add_task(
            "Routing",
            total=len(stations),
            pairs=0,
            routes=0,
            variants=0,
        )

        for origin in stations:
            for dest in stations:
                if origin == dest:
                    continue
                routes = find_routes(
                    topo,
                    origin,
                    dest,
                    max_transfers,
                    max_results,
                    transfer_penalty,
                )
                if not routes:
                    continue

                route_labels = []
                for route in routes:
                    variants = _expand_line_variants(
                        route, topo, ln2i, st2i, transfer_penalty
                    )
                    route_labels.extend(variants)

                # Dedup across routes (different Dijkstra paths might produce
                # the same label after line expansion)
                seen_labels: set[tuple] = set()
                deduped: list[dict] = []
                for rl in route_labels:
                    key = (
                        tuple(tuple(x) for x in rl["label_line"]),
                        tuple(tuple(x) for x in rl["label_change"]),
                    )
                    if key not in seen_labels:
                        seen_labels.add(key)
                        deduped.append(rl)

                n_routes += len(routes)
                n_variants += len(deduped)
                examples.append(
                    {
                        "origin": st2i[origin],
                        "destination": st2i[dest],
                        "routes": deduped,
                    }
                )
            progress.update(
                task,
                advance=1,
                pairs=len(examples),
                routes=n_routes,
                variants=n_variants,
            )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {"stations": stations, "lines": lines, "examples": examples},
                f,
            )
        avg_r = n_routes / max(len(examples), 1)
        avg_v = n_variants / max(len(examples), 1)
        print(
            f"Saved {len(examples):,} OD pairs, "
            f"{n_routes:,} routes ({avg_r:.1f} avg), "
            f"{n_variants:,} with line variants ({avg_v:.1f} avg) → {output_path}"
        )

    return examples
