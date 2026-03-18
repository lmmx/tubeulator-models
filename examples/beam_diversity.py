# experiments/beam_diversity.py
"""Check what beam search actually gives us before building the API."""

import torch
from safetensors.torch import load_file

from tubeulator_models.serving.infer import (
    _assign_lines,
    _compute_cumulative_times,
    _display_name,
    _load_from_export,
    resolve_station,
)
from tubeulator_models.training.beam import beam_rollout_nexthop


# ── Load model, but we also need graph tensors ────────────────
# _load_from_export discards them after computing H, so we reload

REPO = "permutans/tube-nexthop-policy"


def load_with_graph(source: str):
    """Load model + keep graph tensors for beam search."""
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    lm = _load_from_export(source)

    # Reload graph tensors (they were discarded after H was computed)
    try:
        graph_path = Path(hf_hub_download(source, "graph.safetensors"))
        saved = load_file(str(graph_path))
        graph_x = saved["x"]
        edge_index = saved["edge_index"]
        edge_attr = saved["edge_attr"]
    except Exception:
        raise RuntimeError("Need graph.safetensors in the repo")

    return lm, graph_x, edge_index, edge_attr


lm, graph_x, edge_index, edge_attr = load_with_graph(REPO)

# ── Test pairs: pick journeys where you'd expect multiple valid routes ──
TEST_PAIRS = [
    ("West Ham Underground", "Shoreditch"),
    ("Camden Town", "Canary Wharf"),
    ("King's Cross", "Waterloo"),
    ("Baker Street", "Liverpool Street"),
    ("Brixton", "Stratford"),
]


def clean(name: str) -> str:
    for suffix in (" Underground Station", " DLR Station", " Station"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


for origin_name, dest_name in TEST_PAIRS:
    orig_idx = resolve_station(origin_name, lm.stations, lm.stop_names)
    dest_idx = resolve_station(dest_name, lm.stations, lm.stop_names)

    origins = torch.tensor([orig_idx])
    dests = torch.tensor([dest_idx])

    beams = beam_rollout_nexthop(
        lm.model,
        graph_x,
        edge_index,
        edge_attr,
        origins,
        dests,
        beam_width=8,
        max_steps=60,
    )[0]  # single OD pair, take first element

    print(f"\n{'=' * 60}")
    print(f"{origin_name} → {dest_name}  ({len(beams)} beams)")
    print(f"{'=' * 60}")

    for rank, (path, score) in enumerate(beams):
        names = [clean(_display_name(i, lm.stations, lm.stop_names)) for i in path]
        reached = path[-1] == dest_idx

        # Show lines used if topo available
        lines_summary = ""
        if lm.topo and len(path) >= 2:
            segments = _assign_lines(path, lm.stations, lm.topo)
            used = []
            seen = set()
            for line_id, _ in segments:
                if line_id and line_id not in seen:
                    used.append(line_id)
                    seen.add(line_id)
            lines_summary = f"  lines: {' → '.join(used)}"

            cum, _, _ = _compute_cumulative_times(
                path, segments, lm.stations, lm.topo, lm.transfer_lookup
            )
            mins = cum[-1] / 60.0
            lines_summary += f"  ({mins:.1f} min)"

        status = "✓" if reached else "✗"
        print(f"\n  [{rank}] {status} score={score:.2f}{lines_summary}")
        print(f"      {' → '.join(names)}")
