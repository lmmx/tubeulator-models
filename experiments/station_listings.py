"""Extract station data from topology.json + GTFS."""

import csv
import io
import json
import sys
import zipfile


topo_path = "models/nexthop_policy/topology.json"
gtfs_path = "data/tfl_station_data_gtfs.zip"

with open(topo_path) as f:
    topo = json.load(f)

# Derive stations + lines from line_edges
station_lines = {}  # stop_id → set of line/route names
for line, edges in topo["line_edges"].items():
    # edges is probably list of [s1, s2] pairs or dict of s1 → [neighbors]
    if isinstance(edges, dict):
        for s1, neighbors in edges.items():
            station_lines.setdefault(s1, set()).add(line)
            for s2 in neighbors if isinstance(neighbors, list) else [neighbors]:
                station_lines.setdefault(s2, set()).add(line)
    elif isinstance(edges, list):
        for edge in edges:
            if isinstance(edge, list) and len(edge) >= 2:
                station_lines.setdefault(edge[0], set()).add(line)
                station_lines.setdefault(edge[1], set()).add(line)

print(
    f"Found {len(station_lines)} stations across {len(topo['line_edges'])} lines",
    file=sys.stderr,
)

# Grab names + coords from GTFS
names, coords = {}, {}
with zipfile.ZipFile(gtfs_path) as zf:
    with zf.open("stops.txt") as f:
        for row in csv.DictReader(io.TextIOWrapper(f)):
            sid = row["stop_id"]
            names[sid] = row["stop_name"]
            coords[sid] = (float(row["stop_lat"]), float(row["stop_lon"]))

out = []
missing = 0
for sid, lines in station_lines.items():
    name = names.get(sid)
    lat_lon = coords.get(sid)
    if not name or not lat_lon:
        missing += 1
        continue
    out.append(
        {
            "id": sid,
            "name": name,
            "lat": lat_lon[0],
            "lon": lat_lon[1],
            "lines": sorted(lines),
        }
    )

out.sort(key=lambda x: x["name"])
print(f"Extracted {len(out)} stations ({missing} missing from GTFS)", file=sys.stderr)

with open("stations.json", "w") as f:
    json.dump(out, f, indent=2)
