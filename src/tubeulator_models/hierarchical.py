"""Hierarchical decoding: change model legs → deterministic station fill."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

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
    """
    Convert change-model output into a full station sequence.

    Each leg is (line_id, direction, interchange_station). The final leg's
    interchange_station should be the destination itself.

    Returns the concatenated station sequence (no duplicates at joins),
    or None if any leg can't be walked on the graph.
    """
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
    hypothesis into a full station sequence. Returns (sequence, beam_rank)
    per example — the first successfully expanded hypothesis, or (None, -1).
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
            for i in range(0, len(tokens) - 2, 3):
                ln_idx, dir_idx, st_idx = tokens[i], tokens[i + 1], tokens[i + 2]
                if ln_idx < 0 or st_idx < 0:
                    break
                if ln_idx >= len(lines) or st_idx >= len(stations):
                    break
                legs.append((lines[ln_idx], dir_idx, stations[st_idx]))

            if not legs:
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
    """Load trained change model, run hierarchical decode, compare to station labels."""
    from rich import print as rprint
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

    from .beam import beam_decode
    from .config import TrainConfig
    from .dataset import PAD, GPURouteDataset
    from .graph_enriched import build_enriched_graph
    from .models.combined import RouteModel
    from .topology import build_line_station_mask, extract

    cfg = TrainConfig.from_defaults(model_type="change", profile=cfg_profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    rprint("[bold]Extracting topology...")
    topo_data = extract(cfg.gtfs_path)
    graph = build_enriched_graph(topo_data).to(device)

    # Load change model dataset for beam decode
    ds_change = GPURouteDataset(cfg.routes_path, model="change", device=device)

    # Load station dataset for ground truth comparison
    ds_station = GPURouteDataset(cfg.routes_path, model="station", device=device)

    # Recreate val split (same seed → same split)
    n_total = ds_change.n
    n_val = max(1, int(cfg.val_split * n_total))
    n_train = n_total - n_val
    split_gen = torch.Generator(device=device).manual_seed(cfg.seed)
    perm = torch.randperm(n_total, device=device, generator=split_gen)
    val_idx = perm[n_train:]

    rprint(f"  {n_val:,} val examples")

    # Load model
    model = RouteModel(
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
    model.decoder.set_line_station_mask(ls_mask.to(device))

    ckpt_path = cfg.checkpoint_dir / "model_change_best.pt"
    if not ckpt_path.exists():
        rprint(f"[red]No checkpoint at {ckpt_path}. Train the change model first.")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    rprint(f"  Loaded checkpoint from {ckpt_path}")

    stations = ds_change.stations
    lines = ds_change.lines
    st2i = {s: i for i, s in enumerate(stations)}

    # Run beam decode + hierarchical expand in batches
    expanded_total = 0
    matched_total = 0
    total = 0
    strat_buckets: dict[int, list[bool]] = {}

    with Progress(
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        TextColumn("·"),
        TextColumn("[green]{task.fields[matched]}/{task.fields[total]}"),
        TextColumn("({task.fields[pct]:.1%})"),
        refresh_per_second=4,
    ) as progress:
        n_batches = (n_val + cfg.batch_size - 1) // cfg.batch_size
        task = progress.add_task(
            "Hierarchical eval", total=n_batches, matched=0, total_ex=0, pct=0.0
        )

        for batch_start in range(0, n_val, cfg.batch_size):
            batch_idx = val_idx[batch_start : batch_start + cfg.batch_size]
            _, origins, dests, _labels = ds_change.get_batch(batch_idx)

            beam_results = beam_decode(
                model,
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                origins,
                dests,
                beam_width=cfg.beam_width,
            )

            origins_list = origins.tolist()
            dests_list = dests.tolist()

            hier_results = hierarchical_decode(
                topo_data, stations, lines, beam_results, origins_list, dests_list
            )

            # Get all valid station-sequence labels for this batch
            all_station_labels = ds_station.get_all_labels_batch(batch_idx)

            for i, (expanded, rank) in enumerate(hier_results):
                total += 1
                if expanded is None:
                    continue

                expanded_total += 1
                expanded_indices = [st2i[s] for s in expanded]
                route_len = len(expanded_indices)

                # Check against all valid station sequences
                match = False
                for label in all_station_labels[i]:
                    label_len = (label != PAD).sum().item()
                    if label_len == 0:
                        continue
                    label_tokens = label[:label_len].tolist()
                    if expanded_indices == label_tokens:
                        match = True
                        break

                if match:
                    matched_total += 1

                # Stratify by route length
                if route_len <= 5:
                    bucket = 5
                elif route_len <= 10:
                    bucket = 10
                elif route_len <= 20:
                    bucket = 20
                elif route_len <= 30:
                    bucket = 30
                else:
                    bucket = 50
                strat_buckets.setdefault(bucket, []).append(match)

            progress.update(
                task,
                advance=1,
                matched=matched_total,
                total=total,
                pct=matched_total / max(total, 1),
            )

    rprint(f"\n[bold green]Hierarchical decode results[/]")
    rprint(f"  Total:    {total}")
    rprint(f"  Expanded: {expanded_total} ({expanded_total / max(total, 1):.1%})")
    rprint(f"  Matched:  {matched_total} ({matched_total / max(total, 1):.1%})")

    rprint(f"\n  [bold]Stratified by route length:[/]")
    for bucket in sorted(strat_buckets):
        vals = strat_buckets[bucket]
        acc = sum(vals) / len(vals)
        label = f"≤{bucket}st" if bucket < 50 else "31-50st"
        rprint(f"    {label}: {acc:.1%} ({len(vals)})")


def main():
    p = argparse.ArgumentParser(description="Evaluate hierarchical decode pipeline")
    p.add_argument("--profile", default=None)
    args = p.parse_args()

    from .topology import extract
    from .config import TrainConfig

    cfg = TrainConfig.from_defaults(model_type="change", profile=args.profile)
    topo = extract(cfg.gtfs_path)
    evaluate_hierarchical(topo, cfg_profile=args.profile)