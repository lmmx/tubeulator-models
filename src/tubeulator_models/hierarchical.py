"""Hierarchical decoding: change model legs → deterministic station fill."""

from __future__ import annotations

import argparse
from collections import deque

import torch

from .topology import Topology


__all__ = ["expand_legs_to_stations", "hierarchical_decode", "evaluate_hierarchical"]


def _walk_line(
    topo: Topology,
    line: str,
    from_station: str,
    to_station: str,
) -> list[str] | None:
    """
    BFS along a single line's adjacency from from_station to to_station.
    Returns the station sequence including both endpoints, or None if
    no path exists on that line.
    """
    if from_station == to_station:
        return [from_station]

    adj = topo.line_adj.get(line, {})
    if from_station not in adj:
        return None

    visited = {from_station}
    queue: deque[list[str]] = deque([[from_station]])

    while queue:
        path = queue.popleft()
        current = path[-1]
        for neighbor in adj.get(current, set()):
            if neighbor == to_station:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])

    return None


def expand_legs_to_stations(
    topo: Topology,
    origin: str,
    destination: str,
    legs: list[tuple[str, int, str]],
) -> list[str] | None:
    if not legs:
        return None

    full_sequence: list[str] = []
    current = origin

    for line, _direction, end_station in legs:
        segment = _walk_line(topo, line, current, end_station)
        if segment is None:
            return None
        if full_sequence:
            segment = segment[1:]
        full_sequence.extend(segment)
        current = end_station

    if full_sequence and full_sequence[-1] != destination:
        return None

    return full_sequence


def hierarchical_decode(
    topo: Topology,
    stations: list[str],
    lines: list[str],
    beam_results: list[list[tuple[torch.Tensor, float]]],
    origins: list[int],
    destinations: list[int],
) -> list[tuple[list[str] | None, int]]:
    """
    Take batched beam output from the change model and expand each
    hypothesis into a full station sequence. If the predicted line doesn't
    serve the current station, attempts to repair by finding a line that
    serves both the current station and the predicted interchange.
    """
    results: list[tuple[list[str] | None, int]] = []

    for b, beams in enumerate(beam_results):
        origin_name = stations[origins[b]]
        dest_name = stations[destinations[b]]
        expanded = None
        rank = -1

        for beam_idx, (seq_tensor, _score) in enumerate(beams):
            tokens = (
                seq_tensor.tolist() if hasattr(seq_tensor, "tolist") else seq_tensor
            )

            legs: list[tuple[str, int, str]] = []
            current = origin_name
            valid = True

            for i in range(0, len(tokens) - 2, 3):
                ln_idx, dir_idx, st_idx = tokens[i], tokens[i + 1], tokens[i + 2]
                if ln_idx < 0 or st_idx < 0:
                    break
                if ln_idx >= len(lines) or st_idx >= len(stations):
                    break

                pred_line = lines[ln_idx]
                interchange = stations[st_idx]

                serving = topo.station_lines.get(current, set())
                if pred_line in serving:
                    legs.append((pred_line, dir_idx, interchange))
                else:
                    interchange_serving = topo.station_lines.get(interchange, set())
                    shared = serving & interchange_serving
                    if shared:
                        fixed = False
                        for alt_line in sorted(shared):
                            test = _walk_line(topo, alt_line, current, interchange)
                            if test is not None:
                                legs.append((alt_line, dir_idx, interchange))
                                fixed = True
                                break
                        if not fixed:
                            valid = False
                            break
                    else:
                        valid = False
                        break

                current = interchange
                if interchange == dest_name:
                    break

            if not valid or not legs:
                continue

            candidate = expand_legs_to_stations(topo, origin_name, dest_name, legs)
            if candidate is not None:
                expanded = candidate
                rank = beam_idx
                break

        results.append((expanded, rank))

    return results


def evaluate_hierarchical(
    topo: Topology,
    cfg_profile: str | None = None,
) -> None:
    """Diagnostic evaluation of hierarchical decode vs station model."""
    from rich import print as rprint

    from .beam import beam_decode
    from .config import TrainConfig
    from .dataset import PAD, GPURouteDataset
    from .graph_enriched import build_enriched_graph
    from .models.combined import RouteModel
    from .topology import build_adj_mask, build_line_station_mask, extract

    cfg = TrainConfig.from_defaults(model_type="change", profile=cfg_profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    rprint("[bold]Extracting topology...")
    topo_data = extract(cfg.gtfs_path)
    graph = build_enriched_graph(topo_data).to(device)

    ds_change = GPURouteDataset(cfg.routes_path, model="change", device=device)
    ds_station = GPURouteDataset(cfg.routes_path, model="station", device=device)

    n_total = ds_change.n
    n_val = max(1, int(cfg.val_split * n_total))
    n_train = n_total - n_val
    split_gen = torch.Generator(device=device).manual_seed(cfg.seed)
    perm = torch.randperm(n_total, device=device, generator=split_gen)
    val_idx = perm[n_train:]

    stations = ds_change.stations
    lines = ds_change.lines
    st2i = {s: i for i, s in enumerate(stations)}
    stop_names = topo_data.stop_names

    def _name(station_id: str) -> str:
        return stop_names.get(station_id, station_id)

    # --- Load change model ---
    change_model = RouteModel(
        n_stations=ds_change.n_stations,
        n_lines=ds_change.n_lines,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_enc_layers=cfg.n_enc_layers,
        model_type="change",
        max_seq=cfg.max_seq,
        dropout=cfg.dropout,
    ).to(device)

    ls_mask = build_line_station_mask(topo_data, ds_change.stations, ds_change.lines)
    change_model.decoder.set_line_station_mask(ls_mask.to(device))

    change_ckpt = cfg.checkpoint_dir / "model_change_best.pt"
    if not change_ckpt.exists():
        rprint(f"[red]No checkpoint at {change_ckpt}. Train the change model first.")
        return
    change_model.load_state_dict(
        torch.load(change_ckpt, map_location=device, weights_only=True)
    )
    change_model.eval()
    rprint(f"  Loaded change model from {change_ckpt}")

    # --- Load station model ---
    station_cfg = TrainConfig.from_defaults(model_type="station", profile=cfg_profile)
    station_model = RouteModel(
        n_stations=ds_change.n_stations,
        n_lines=ds_change.n_lines,
        d_model=station_cfg.d_model,
        n_heads=station_cfg.n_heads,
        n_enc_layers=station_cfg.n_enc_layers,
        model_type="station",
        max_seq=station_cfg.max_seq,
        dropout=station_cfg.dropout,
    ).to(device)

    adj = build_adj_mask(topo_data, ds_change.stations).to(device)
    station_model.decoder.set_adj_mask(adj)

    station_ckpt = cfg.checkpoint_dir / "model_station_best.pt"
    has_station = station_ckpt.exists()
    if has_station:
        station_model.load_state_dict(
            torch.load(station_ckpt, map_location=device, weights_only=True)
        )
        station_model.eval()
        rprint(f"  Loaded station model from {station_ckpt}")
    else:
        rprint(
            f"  [dim]No station checkpoint at {station_ckpt}, skipping comparison[/]"
        )

    # --- Run on same sample ---
    n_sample = min(200, n_val)
    sample_idx = val_idx[:n_sample]
    _, origins, dests, _labels = ds_change.get_batch(sample_idx)

    rprint(f"\n[bold]Running on {n_sample} examples...[/]")

    # Hierarchical: change model beam → graph walk
    change_beam = beam_decode(
        change_model,
        graph.x,
        graph.edge_index,
        graph.edge_attr,
        origins,
        dests,
        beam_width=5,
    )

    origins_list = origins.tolist()
    dests_list = dests.tolist()

    hier_results = hierarchical_decode(
        topo_data, stations, lines, change_beam, origins_list, dests_list
    )

    all_station_labels = ds_station.get_all_labels_batch(sample_idx)

    # Station model beam
    if has_station:
        station_beam = beam_decode(
            station_model,
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            origins,
            dests,
            beam_width=5,
        )

    # --- Score both ---
    hier_matched = 0
    expand_failed = 0
    wrong_route = 0
    station_matched = 0
    examples_expand_failed: list[dict] = []
    examples_wrong_route: list[dict] = []

    for i, (expanded, rank) in enumerate(hier_results):
        origin_id = stations[origins_list[i]]
        dest_id = stations[dests_list[i]]

        best_beam_tokens = change_beam[i][0][0].tolist() if change_beam[i] else []
        predicted_legs = []
        for j in range(0, len(best_beam_tokens) - 2, 3):
            ln, d, st = (
                best_beam_tokens[j],
                best_beam_tokens[j + 1],
                best_beam_tokens[j + 2],
            )
            if ln < 0 or st < 0 or ln >= len(lines) or st >= len(stations):
                break
            predicted_legs.append(
                (lines[ln], "→" if d == 0 else "←", _name(stations[st]))
            )

        gt_routes = []
        for label in all_station_labels[i]:
            label_len = (label != PAD).sum().item()
            if label_len > 0:
                gt_routes.append(
                    [_name(stations[t]) for t in label[:label_len].tolist()]
                )

        # Hierarchical scoring
        if expanded is None:
            expand_failed += 1
            if len(examples_expand_failed) < 5:
                examples_expand_failed.append(
                    {
                        "journey": f"{_name(origin_id)} → {_name(dest_id)}",
                        "legs": predicted_legs,
                    }
                )
        else:
            expanded_indices = [st2i[s] for s in expanded]
            match = False
            for label in all_station_labels[i]:
                label_len = (label != PAD).sum().item()
                if label_len == 0:
                    continue
                if expanded_indices == label[:label_len].tolist():
                    match = True
                    break

            if match:
                hier_matched += 1
            else:
                wrong_route += 1
                if len(examples_wrong_route) < 5:
                    examples_wrong_route.append(
                        {
                            "journey": f"{_name(origin_id)} → {_name(dest_id)}",
                            "legs": predicted_legs,
                            "expanded": [_name(s) for s in expanded],
                            "gt_sample": gt_routes[0] if gt_routes else [],
                        }
                    )

        # Station model scoring
        if has_station:
            for seq, _score in station_beam[i]:
                seq_list = seq.tolist()
                hit = False
                for label in all_station_labels[i]:
                    label_len = (label != PAD).sum().item()
                    if label_len == 0:
                        continue
                    if seq_list[:label_len] == label[:label_len].tolist():
                        hit = True
                        break
                if hit:
                    station_matched += 1
                    break

    rprint(f"\n[bold green]Results ({n_sample} examples)[/]")
    rprint(f"  Hierarchical:  {hier_matched} ({hier_matched / n_sample:.1%})")
    rprint(f"    expand failed: {expand_failed} ({expand_failed / n_sample:.1%})")
    rprint(f"    wrong route:   {wrong_route} ({wrong_route / n_sample:.1%})")
    if has_station:
        rprint(f"  Station model: {station_matched} ({station_matched / n_sample:.1%})")

    if examples_expand_failed:
        rprint("\n[bold red]Expansion failures:[/]")
        for ex in examples_expand_failed:
            rprint(f"  {ex['journey']}")
            for leg in ex["legs"]:
                rprint(f"    {leg[0]} {leg[1]} {leg[2]}")

    if examples_wrong_route:
        rprint("\n[bold yellow]Wrong route:[/]")
        for ex in examples_wrong_route:
            rprint(f"  {ex['journey']}")
            rprint(f"    predicted legs: {ex['legs']}")
            rprint(
                f"    expanded: {' → '.join(ex['expanded'][:8])}{'...' if len(ex['expanded']) > 8 else ''}"
            )
            rprint(
                f"    gt:       {' → '.join(ex['gt_sample'][:8])}{'...' if len(ex['gt_sample']) > 8 else ''}"
            )


def main():
    p = argparse.ArgumentParser(description="Evaluate hierarchical decode pipeline")
    p.add_argument("--profile", default=None)
    args = p.parse_args()

    from .config import TrainConfig
    from .topology import extract

    cfg = TrainConfig.from_defaults(model_type="change", profile=args.profile)
    topo = extract(cfg.gtfs_path)
    evaluate_hierarchical(topo, cfg_profile=args.profile)
