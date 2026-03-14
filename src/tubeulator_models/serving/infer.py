# src/tubeulator_models/infer.py
"""Inference CLI for trained tube routing models."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path
from typing import NamedTuple

import torch
from rich.console import Console
from rich.table import Table
from rich.text import Text
from safetensors.torch import load_file

from ..config import TrainConfig
from ..defaults import repo_root, resolve_data, resolve_hub
from ..graph.enriched import build_enriched_graph
from ..graph.topology import (
    Topology,
    build_adj_mask,
    build_transfer_lookup,
    extract,
    load_interchange_data,
)
from ..models.combined import RouteModel


__all__ = ["load_model", "rollout", "predict_time"]

console = Console()


# ── GTFS coordinate extraction ────────────────────────────────


def _read_stop_coords(gtfs_path: Path) -> dict[str, tuple[float, float]]:
    """Read stop_id → (lon, lat) from GTFS stops.txt."""
    coords = {}
    with zipfile.ZipFile(gtfs_path) as zf:
        with zf.open("stops.txt") as f:
            for row in csv.DictReader(io.TextIOWrapper(f)):
                sid = row["stop_id"]
                lat = float(row.get("stop_lat", 0.0) or 0.0)
                lon = float(row.get("stop_lon", 0.0) or 0.0)
                coords[sid] = (lon, lat)
    return coords


# ── Network topology export helper ─────────────────────────────


class _ExportedTopology:
    """Minimal Topology-compatible object reconstructed from exported JSON."""

    def __init__(self, topo_data: dict) -> None:
        # edge_time: {line: {(from, to): seconds}}
        self.edge_time: dict[str, dict[tuple[str, str], float]] = {}
        for line_id, triples in topo_data["edge_times"].items():
            self.edge_time[line_id] = {(f, t): secs for f, t, secs in triples}

        # _lines_on_edge: {(from, to): set of line_ids}
        self._lines_on_edge: dict[tuple[str, str], set[str]] = {}
        for line_id, pairs in topo_data["line_edges"].items():
            for f, t in pairs:
                self._lines_on_edge.setdefault((f, t), set()).add(line_id)

        # hub_members: {hub_id: set of member station ids}
        self.hub_members: dict[str, set[str]] = {
            k: set(v) for k, v in topo_data["hub_members"].items()
        }

        # transfer_lookup: {(station, from_line, to_line): seconds}
        self.transfer_lookup: dict[tuple[str, str, str], float] = {}
        for station, entries in topo_data.get("transfers", {}).items():
            for from_line, to_line, secs in entries:
                self.transfer_lookup[(station, from_line, to_line)] = secs

    def lines_on_edge(self, from_sid: str, to_sid: str) -> set[str]:
        return self._lines_on_edge.get((from_sid, to_sid), set())

    def travel_time(self, line_id: str, from_sid: str, to_sid: str) -> float:
        return self.edge_time.get(line_id, {}).get((from_sid, to_sid), 120.0)


# ── Model loading ─────────────────────────────────────────────


class LoadedModel(NamedTuple):
    model: RouteModel
    config: dict
    stations: list[str]
    lines: list[str]
    stop_names: dict[str, str]
    adj: torch.Tensor
    H: torch.Tensor  # precomputed encoder output
    topo: Topology | None  # for line attribution
    transfer_lookup: dict[tuple[str, str, str], float] | None


def _download(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename))


def _infer_config_from_checkpoint(state_dict: dict[str, torch.Tensor]) -> dict:
    """Derive architecture params directly from weight tensor shapes."""
    d_model = state_dict["encoder.node_proj.weight"].shape[0]

    # Count encoder layers
    n_enc_layers = sum(
        1 for k in state_dict if k.startswith("encoder.convs.") and k.endswith(".att")
    )

    # n_heads from attention shape: [1, n_heads, d_model // n_heads]
    att = state_dict["encoder.convs.0.att"]
    n_heads = att.shape[1]

    # n_stations from policy head output
    n_stations = state_dict["decoder.mlp.6.weight"].shape[0]

    # value_primary: small value head has indices 0,3 only; large has 0,2,4,6,8,10
    value_primary = "decoder.value_head.4.weight" in state_dict

    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_enc_layers": n_enc_layers,
        "n_stations": n_stations,
        "value_primary": value_primary,
    }


def _load_from_checkpoint(variant: str, profile: str = "full") -> LoadedModel:
    is_value = variant == "value"

    # We only need TrainConfig for paths — not architecture
    cfg = TrainConfig.from_defaults(
        model_type="nexthop",
        profile=profile,
        value_primary=is_value,
    )

    device = torch.device("cpu")

    # Load checkpoint first, infer architecture from it
    if is_value:
        ckpt_path = cfg.checkpoint_dir / "model_nexthop_value_best.pt"
    else:
        ckpt_path = cfg.checkpoint_dir / "model_nexthop_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    inferred = _infer_config_from_checkpoint(state_dict)

    topo = extract(cfg.gtfs_path)
    stations = topo.all_stations
    lines = topo.all_lines

    node_coords = _read_stop_coords(cfg.gtfs_path)
    graph = build_enriched_graph(topo, node_coords=node_coords)

    model = RouteModel(
        n_stations=len(stations),
        n_lines=len(lines),
        d_model=inferred["d_model"],
        n_heads=inferred["n_heads"],
        n_enc_layers=inferred["n_enc_layers"],
        model_type="nexthop",
        max_seq=cfg.max_seq,
        dropout=0.0,  # doesn't matter for eval
        value_primary=inferred["value_primary"],
    ).to(device)

    adj = build_adj_mask(topo, stations).to(device)
    model.decoder.set_adj_mask(adj)
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    # Build transfer cost lookup for display
    data_cfg = resolve_data()
    ic_rel = data_cfg.get("interchange_path", "")
    ic_path = repo_root() / ic_rel if ic_rel else None
    transfer_lookup = None
    if ic_path is not None and ic_path.is_file():
        ic_data = load_interchange_data(ic_path)
        transfer_lookup = build_transfer_lookup(
            topo,
            stations,
            ic_data,
            discount=1.0,  # cfg.transfer_discount,
        )

    with torch.no_grad():
        H = model.encoder(graph.x, graph.edge_index, graph.edge_attr)

    return LoadedModel(
        model=model,
        config={
            **inferred,
            "max_seq": cfg.max_seq,
            "n_params": n_params,
        },
        stations=stations,
        lines=lines,
        stop_names=topo.stop_names,
        adj=adj,
        H=H,
        topo=topo,
        transfer_lookup=transfer_lookup,
    )


def _load_from_export(source: str | Path) -> LoadedModel:
    """
    Load from an export directory or HF repo.
    Requires graph.safetensors in the export, or falls back to GTFS rebuild.
    """
    source_path = Path(source)

    if source_path.is_dir():
        config = json.loads((source_path / "config.json").read_text())
        metadata = json.loads((source_path / "metadata.json").read_text())
        weights = load_file(str(source_path / "model.safetensors"))
        graph_path = source_path / "graph.safetensors"
        saved_graph = load_file(str(graph_path)) if graph_path.exists() else None
    else:
        repo = str(source)
        config = json.loads(_download(repo, "config.json").read_text())
        metadata = json.loads(_download(repo, "metadata.json").read_text())
        weights = load_file(str(_download(repo, "model.safetensors")))
        try:
            saved_graph = load_file(str(_download(repo, "graph.safetensors")))
        except Exception:
            saved_graph = None

    if source_path.is_dir():
        topo_path = source_path / "topology.json"
        topo_data = json.loads(topo_path.read_text()) if topo_path.exists() else None
    else:
        try:
            topo_data = json.loads(_download(repo, "topology.json").read_text())
        except Exception:
            topo_data = None

    topo = _ExportedTopology(topo_data) if topo_data is not None else None

    stations = metadata["stations"]
    lines = metadata["lines"]
    stop_names = metadata.get("stop_names", {})
    is_value = config.get("value_primary", False)

    model = RouteModel(
        n_stations=config["n_stations"],
        n_lines=len(lines),
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_enc_layers=config["n_enc_layers"],
        model_type=config.get("model_type", "nexthop"),
        max_seq=config["max_seq"],
        dropout=config.get("dropout", 0.0),
        value_primary=is_value,
    )

    adj = torch.zeros(config["n_stations"], config["n_stations"], dtype=torch.bool)
    for i, j in metadata["adjacency"]:
        adj[i, j] = True
    model.decoder.set_adj_mask(adj)

    model.load_state_dict(weights)
    model.eval()

    if saved_graph is not None:
        graph_x = saved_graph["x"]
        edge_index = saved_graph["edge_index"]
        edge_attr = saved_graph["edge_attr"]
    else:
        # Fallback: rebuild from GTFS
        cfg = TrainConfig.from_defaults(model_type="nexthop")
        topo = extract(cfg.gtfs_path)
        node_coords = _read_stop_coords(cfg.gtfs_path)
        graph = build_enriched_graph(topo, node_coords=node_coords)
        graph_x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        if not stop_names:
            stop_names = topo.stop_names

    with torch.no_grad():
        H = model.encoder(graph_x, edge_index, edge_attr)

    return LoadedModel(
        model=model,
        config=config,
        stations=stations,
        lines=lines,
        stop_names=stop_names,
        adj=adj,
        H=H,
        topo=topo,
        transfer_lookup=topo.transfer_lookup if topo is not None else None,
    )


def load_model(
    variant: str,
    source: str | None = None,
    profile: str = "full",
) -> LoadedModel:
    """
    Load a trained model for inference.

    source: HF repo ID, local export dir, or None (load from checkpoint).
    """
    if source is not None:
        return _load_from_export(source)

    # No source given — try checkpoint first, then resolve from hub config
    try:
        return _load_from_checkpoint(variant, profile=profile)
    except FileNotFoundError:
        hub_ids = resolve_hub()
        hub_key = f"nexthop_{variant}"
        if hub_key in hub_ids:
            return _load_from_export(hub_ids[hub_key])
        raise


# ── Station lookup ────────────────────────────────────────────


def resolve_station(name: str, stations: list[str], stop_names: dict[str, str]) -> int:
    """
    Resolve a human station name to a station index.

    Tries: exact stop_id → exact display name → substring of display name.
    """
    if name in stations:
        return stations.index(name)

    # Reverse map: lowercase display name → list of stop_ids
    name_to_ids: dict[str, list[str]] = {}
    for sid, display in stop_names.items():
        name_to_ids.setdefault(display.lower(), []).append(sid)

    lower = name.lower()

    # Exact display name match
    if lower in name_to_ids:
        for sid in name_to_ids[lower]:
            if sid in stations:
                return stations.index(sid)

    # Substring match
    matches = []
    for display_lower, sids in name_to_ids.items():
        if lower in display_lower:
            for sid in sids:
                if sid in stations:
                    matches.append((stations.index(sid), stop_names[sid]))

    if len(matches) == 1:
        return matches[0][0]
    if len(matches) > 1:
        names = ", ".join(f"'{display}'" for _, display in matches[:10])
        extra = f" (+{len(matches) - 10} more)" if len(matches) > 10 else ""
        raise SystemExit(f"Ambiguous station '{name}', matches: {names}{extra}")

    raise SystemExit(f"Unknown station: '{name}'")


def _display_name(idx: int, stations: list[str], stop_names: dict[str, str]) -> str:
    sid = stations[idx]
    return stop_names.get(sid, sid)


def _destination_set(dest_idx: int, stations: list[str], topo: Topology) -> set[int]:
    """All station indices that count as 'arrived' for a destination."""
    targets = {dest_idx}
    if topo is None:
        return targets
    dest_sid = stations[dest_idx]
    # Find which hub this station belongs to
    for hub, members in topo.hub_members.items():
        if dest_sid in members or dest_sid == hub:
            for member in members:
                if member in stations:
                    targets.add(stations.index(member))
            if hub in stations:
                targets.add(stations.index(hub))
    return targets


# ── Inference ─────────────────────────────────────────────────


@torch.no_grad()
def rollout(
    lm: LoadedModel,
    origin: str,
    destination: str,
    max_hops: int = 60,
) -> tuple[list[int], bool]:
    """
    Greedy policy rollout from origin to destination.

    Returns (path as station indices, success flag).
    """
    orig_idx = resolve_station(origin, lm.stations, lm.stop_names)
    dest_idx = resolve_station(destination, lm.stations, lm.stop_names)

    path = [orig_idx]
    current = orig_idx
    visited = {orig_idx}

    dest_set = _destination_set(dest_idx, lm.stations, lm.topo)

    for _ in range(max_hops):
        if current in dest_set:
            break

        h_current = lm.H[current].unsqueeze(0)
        h_dest = lm.H[dest_idx].unsqueeze(0)
        current_ids = torch.tensor([current], dtype=torch.long)

        out = lm.model.decoder(h_current, h_dest, current_ids=current_ids)
        logits = out["next_station"].squeeze(0)  # (N,)

        # Prefer unvisited adjacent stations
        adj_mask = lm.adj[current].clone()
        unvisited = adj_mask.clone()
        for v in visited:
            unvisited[v] = False

        if unvisited.any():
            logits[~unvisited] = float("-inf")
        else:
            # All neighbours visited — allow backtrack but block non-adjacent
            logits[~adj_mask] = float("-inf")

        next_hop = logits.argmax().item()
        path.append(next_hop)
        visited.add(next_hop)
        current = next_hop

    return path, current in dest_set


@torch.no_grad()
def rollout_via(
    lm: LoadedModel,
    origin: str,
    destination: str,
    via: list[str],
    max_hops: int = 60,
) -> tuple[list[int], bool]:
    """
    Rollout through waypoints: origin → via[0] → via[1] → ... → destination.

    Returns (full path, overall success).
    """
    waypoints = [origin] + via + [destination]
    full_path: list[int] = []
    all_success = True

    for i in range(len(waypoints) - 1):
        segment, success = rollout(
            lm, waypoints[i], waypoints[i + 1], max_hops=max_hops
        )
        if not success:
            all_success = False
        if full_path:
            segment = segment[1:]  # drop duplicate junction station
        full_path.extend(segment)

    return full_path, all_success


@torch.no_grad()
def predict_time(
    lm: LoadedModel,
    origin: str,
    destination: str,
) -> float:
    """Predict travel time in minutes using the value head."""
    orig_idx = resolve_station(origin, lm.stations, lm.stop_names)
    dest_idx = resolve_station(destination, lm.stations, lm.stop_names)

    h_o = lm.H[orig_idx].unsqueeze(0)
    h_d = lm.H[dest_idx].unsqueeze(0)
    combined = torch.cat([h_o, h_d], dim=-1)
    return lm.model.decoder.value_head(combined).squeeze().item()


# ── CLI rendering ─────────────────────────────────────────────


LINE_STYLES: dict[str, str] = {
    "bakerloo": "bold rgb(179,99,5)",
    "central": "bold red",
    "circle": "bold yellow",
    "district": "bold green",
    "hammersmith-city": "bold rgb(243,169,187)",
    "jubilee": "bold rgb(160,165,169)",
    "metropolitan": "bold rgb(155,0,86)",
    "northern": "bold white on black",
    "piccadilly": "bold rgb(0,54,136)",
    "victoria": "bold rgb(0,152,212)",
    "waterloo-city": "bold rgb(149,205,186)",
    "elizabeth": "bold rgb(105,80,161)",
    "dlr": "bold rgb(0,164,167)",
}


def _assign_lines(
    path: list[int],
    stations: list[str],
    topo: Topology,
) -> list[tuple[str | None, list[str]]]:
    """
    Assign tube lines to each segment of a path.

    Returns list of (line_id or None, [shared_lines]) per hop.
    Greedy: stays on current line as long as possible.
    """
    if len(path) < 2:
        return []

    segments = []
    current_line: str | None = None

    for i in range(len(path) - 1):
        from_sid = stations[path[i]]
        to_sid = stations[path[i + 1]]
        available = topo.lines_on_edge(from_sid, to_sid)
        available_list = sorted(available)

        if current_line and current_line in available:
            # Stay on the same line
            segments.append((current_line, available_list))
        elif available_list:
            # Pick a line — prefer one that also serves the next edge (lookahead)
            chosen = available_list[0]
            if i + 2 < len(path):
                next_from = stations[path[i + 1]]
                next_to = stations[path[i + 2]]
                next_available = topo.lines_on_edge(next_from, next_to)
                continuable = available & next_available
                if continuable:
                    chosen = sorted(continuable)[0]
            current_line = chosen
            segments.append((current_line, available_list))
        else:
            current_line = None
            segments.append((None, []))

    return segments


def _line_style(line_id: str | None) -> str:
    if line_id is None:
        return "dim"
    return LINE_STYLES.get(line_id, "bold")


def _compute_cumulative_times(
    path: list[int],
    segments: list[tuple[str | None, list[str]]],
    stations: list[str],
    topo: Topology,
    transfer_lookup: dict[tuple[str, str, str], float] | None = None,
) -> tuple[list[float], list[bool], list[float]]:
    """
    Returns (cumulative_seconds, estimated_flags, transfer_seconds_per_hop).
    """
    cum = [0.0]
    estimated = [False]
    transfers = [0.0]  # one per station, index 0 = origin

    prev_line: str | None = None

    for i, (line_id, _alts) in enumerate(segments):
        from_sid = stations[path[i]]
        to_sid = stations[path[i + 1]]

        transfer_t = 0.0

        # Transfer penalty when line changes
        if prev_line is not None and line_id is not None and line_id != prev_line:
            transfer_station = stations[path[i]]
            if transfer_lookup is not None:
                transfer_t = transfer_lookup.get(
                    (transfer_station, prev_line, line_id), 0.0
                )

        if line_id:
            edge_t = topo.travel_time(line_id, from_sid, to_sid)
            real = (from_sid, to_sid) in topo.edge_time.get(line_id, {})
        else:
            edge_t = 120.0
            real = False

        cum.append(cum[-1] + transfer_t + edge_t)
        estimated.append(not real)
        transfers.append(transfer_t)
        prev_line = line_id

    return cum, estimated, transfers


def _render_route(path: list[int], success: bool, lm: LoadedModel) -> None:
    has_lines = lm.topo is not None and len(path) >= 2
    segments = _assign_lines(path, lm.stations, lm.topo) if has_lines else []
    cum_times, estimated, transfer_times = (
        _compute_cumulative_times(
            path,
            segments,
            lm.stations,
            lm.topo,
            transfer_lookup=lm.transfer_lookup,
        )
        if has_lines
        else ([], [], [])
    )

    table = Table(
        title="Route",
        show_header=True,
        title_style="bold",
        border_style="dim",
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Station")
    if has_lines:
        table.add_column("Line", min_width=18)
        table.add_column("Time", justify="right", min_width=8)

    prev_line: str | None = None

    for i, idx in enumerate(path):
        name = _display_name(idx, lm.stations, lm.stop_names)
        for suffix in (" Underground Station", " DLR Station", " Station"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break

        station_style = (
            "bold green" if i == 0 else "bold cyan" if i == len(path) - 1 else ""
        )

        if has_lines:
            # Time column
            if cum_times:
                mins = cum_times[i] / 60.0
                marker = "*" if estimated[i] else " "
                xfer = transfer_times[i] if i < len(transfer_times) else 0.0
                if xfer > 0:
                    time_str = f"(+{xfer / 60:.1f}m) {marker}{mins:.1f}m"
                else:
                    time_str = f"{marker}{mins:.1f}m"
            else:
                time_str = ""

            if i < len(segments):
                line_id, _alts = segments[i]
                is_transfer = prev_line is not None and line_id != prev_line
                style = _line_style(line_id)

                line_display = Text()
                if is_transfer:
                    line_display.append("↳ ", style="bold yellow")
                line_display.append(line_id or "?", style=style)

                table.add_row(
                    str(i),
                    Text(name, style=station_style),
                    line_display,
                    time_str,
                )
                prev_line = line_id
            else:
                table.add_row(
                    str(i), Text(name, style=station_style), Text(""), time_str
                )
        else:
            table.add_row(str(i), name, style=station_style)

    console.print(table)
    if estimated and any(estimated):
        n_est = sum(estimated)
        console.print(
            f"[dim]* {n_est} edge{'s' if n_est != 1 else ''} "
            f"using 120s fallback (no GTFS timing data)[/dim]"
        )

    status = Text()
    if success:
        status.append("✓ ", style="bold green")
        status.append(f"Arrived in {len(path) - 1} hops")
    else:
        status.append("✗ ", style="bold red")
        status.append(f"Failed after {len(path) - 1} hops")

    if segments:
        lines_used = []
        seen = set()
        for line_id, _ in segments:
            if line_id and line_id not in seen:
                lines_used.append(line_id)
                seen.add(line_id)
        n_transfers = sum(
            1 for i in range(1, len(segments)) if segments[i][0] != segments[i - 1][0]
        )
        status.append(f" · {len(lines_used)} line{'s' if len(lines_used) != 1 else ''}")
        if n_transfers:
            status.append(f" · {n_transfers} transfer{'s' if n_transfers != 1 else ''}")

    if cum_times:
        total_mins = cum_times[-1] / 60.0
        status.append(f" · {total_mins:.1f} min")

    console.print(status)


def _render_time(origin: str, destination: str, minutes: float) -> None:
    console.print(
        f"[bold]{origin}[/bold] → [bold]{destination}[/bold]: "
        f"[cyan]{minutes:.1f} min[/cyan]"
    )


# ── CLI ───────────────────────────────────────────────────────


def run_policy(args: argparse.Namespace) -> None:
    console.print("[dim]Loading policy model...[/dim]")
    lm = load_model("policy", source=args.model, profile=args.profile)
    console.print(f"[dim]Loaded ({lm.config['n_params']:,} params)[/dim]\n")

    if args.via:
        path, success = rollout_via(lm, args.origin, args.destination, args.via)
    else:
        path, success = rollout(lm, args.origin, args.destination)

    _render_route(path, success, lm)

    if not success:
        raise SystemExit(1)


def run_value(args: argparse.Namespace) -> None:
    console.print("[dim]Loading value model...[/dim]")
    lm = load_model("value", source=args.model, profile=args.profile)
    console.print(f"[dim]Loaded ({lm.config['n_params']:,} params)[/dim]\n")

    minutes = predict_time(lm, args.origin, args.destination)
    orig_display = _display_name(
        resolve_station(args.origin, lm.stations, lm.stop_names),
        lm.stations,
        lm.stop_names,
    )
    dest_display = _display_name(
        resolve_station(args.destination, lm.stations, lm.stop_names),
        lm.stations,
        lm.stop_names,
    )
    _render_time(orig_display, dest_display, minutes)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--origin", required=True, help="Origin station name")
    parser.add_argument(
        "-d", "--destination", required=True, help="Destination station name"
    )
    parser.add_argument(
        "-v", "--via", action="append", default=[], help="Waypoint (repeatable)"
    )
    parser.add_argument(
        "--profile", default="dev", help="Config profile (default: dev)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tm-infer",
        description="Run inference with trained tube routing models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_policy = sub.add_parser("policy", help="Greedy policy rollout.")
    _add_common_args(p_policy)
    p_policy.add_argument(
        "--model", default=None, help="HF repo ID or local export dir."
    )
    p_policy.set_defaults(func=run_policy)

    p_value = sub.add_parser("value", help="Predict travel time.")
    _add_common_args(p_value)
    p_value.add_argument(
        "--model", default=None, help="HF repo ID or local export dir."
    )
    p_value.set_defaults(func=run_value)

    p_route = sub.add_parser(
        "route", help="Full route: policy rollout + time estimate."
    )
    _add_common_args(p_route)
    p_route.add_argument("--policy-model", default=None)
    p_route.add_argument("--value-model", default=None)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
