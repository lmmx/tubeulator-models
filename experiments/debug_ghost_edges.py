# experiments/debug_ghost_edges.py
"""Find all edges in line_adj but not in edge_time — the phantom edges."""

from tubeulator_models.config import TrainConfig
from tubeulator_models.graph.topology import extract


cfg = TrainConfig.from_defaults(model_type="nexthop", profile="full")
topo = extract(cfg.gtfs_path)

ghosts = []  # (line, from_sid, to_sid)

for line, adj in topo.line_adj.items():
    line_times = topo.edge_time.get(line, {})
    for station, neighbors in adj.items():
        for neighbor in neighbors:
            if (station, neighbor) not in line_times:
                from_name = topo.stop_names.get(station, station)
                to_name = topo.stop_names.get(neighbor, neighbor)
                ghosts.append((line, station, neighbor, from_name, to_name))

print(f"Found {len(ghosts)} ghost edges (in line_adj but not edge_time)\n")

# Group by line
from collections import Counter


by_line = Counter(line for line, *_ in ghosts)
for line, count in by_line.most_common():
    name = topo.route_names.get(line, line)
    print(f"  {name} ({line}): {count} ghost edges")

print("\n--- Ghosts involving key stations ---")
keywords = {"shepherd", "clapham", "highbury", "junction"}
for line, from_sid, to_sid, from_name, to_name in ghosts:
    if any(k in from_name.lower() or k in to_name.lower() for k in keywords):
        name = topo.route_names.get(line, line)
        print(f"  [{name}] {from_name} → {to_name}")
