# experiments/debug_teleport.py
"""Debug routes that appear to teleport between distant stations."""

from tubeulator_models.serving.infer import (
    _assign_lines,
    _display_name,
    _load_from_export,
    rollout_diverse,
)


REPO = "permutans/tube-nexthop-policy"
lm = _load_from_export(REPO)


def clean(name: str) -> str:
    for suffix in (
        " Underground Station",
        " DLR Station",
        " Rail Station",
        " Station",
        " Rail",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def debug_route(origin: str, dest: str, n: int = 5):
    candidates = rollout_diverse(lm, origin, dest, n_routes=n, beam_width=12)

    for rank, (path, score) in enumerate(candidates):
        print(f"\n  Route {rank + 1} (score={score:.2f})")
        print(f"  Raw path ({len(path)} nodes):")

        segments = _assign_lines(path, lm.stations, lm.topo) if len(path) >= 2 else []

        for i, idx in enumerate(path):
            sid = lm.stations[idx]
            name = clean(_display_name(idx, lm.stations, lm.stop_names))
            line = segments[i][0] if i < len(segments) else None
            alts = segments[i][1] if i < len(segments) else []

            # Check adjacency
            adj_flag = ""
            if i > 0:
                prev_idx = path[i - 1]
                is_adj = lm.adj[prev_idx, idx].item()
                if not is_adj:
                    adj_flag = " ⚠️  NOT ADJACENT"

            # Check what lines serve this edge
            edge_lines = ""
            if i > 0 and lm.topo is not None:
                prev_sid = lm.stations[path[i - 1]]
                lines_on = lm.topo.lines_on_edge(prev_sid, sid)
                edge_lines = (
                    f"  edge_lines={sorted(lines_on)}"
                    if lines_on
                    else "  edge_lines=NONE"
                )

            # Hub membership
            hub = ""
            if lm.topo is not None:
                for h, members in lm.topo.hub_members.items():
                    if sid in members or sid == h:
                        hub = f"  hub={h}"
                        break

            line_tag = f"[{line}]" if line else "[---]"
            print(
                f"    {i:2d}  {line_tag:20s} {name:40s} sid={sid}{hub}{edge_lines}{adj_flag}"
            )


# The two teleportation cases
SUSPECT_PAIRS = [
    ("Notting Hill Gate", "London Bridge"),
    ("Barons Court", "Old Street"),
]

for origin, dest in SUSPECT_PAIRS:
    print(f"\n{'=' * 70}")
    print(f"DEBUG: {origin} → {dest}")
    print(f"{'=' * 70}")
    debug_route(origin, dest)
