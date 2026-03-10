import json
import re
from pathlib import Path

from tubeulator_models.topology import extract


topo = extract(Path("data/tfl_station_data_gtfs.zip"))


def _normalize(name):
    s = name.lower().strip()
    if "," in s:
        s = s.split(",", 1)[0].strip()
    s = re.sub(r"\s*\((?:including|inc\.?)\b[^)]*\)", "", s).strip()
    for suffix in (
        "-underground",
        " london underground",
        " underground station",
        " underground",
        " rail station",
        " dlr station",
        " station",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    s = s.replace("'", "").replace("\u2019", "").replace(".", " ")
    return " ".join(s.split())


_ALIASES = {
    "edgware road circle": "edgware road (circle line)",
    "hammersmith district and piccadilly lines": "hammersmith (dist&picc line)",
    "hammersmith hammersmith & city line": "hammersmith (h&c line)",
}

with open("data/interchange_times.json") as f:
    data = json.load(f)

ic_lookup = {}
for item in data:
    norm = _normalize(item["station"])
    norm = _ALIASES.get(norm, norm)
    n_ic = len(
        [ic for ic in item.get("interchanges", []) if ic.get("minutes") is not None]
    )
    if n_ic > 0:
        ic_lookup[norm] = n_ic

for station_id in sorted(topo.interchanges):
    name = topo.stop_names.get(station_id, station_id)
    norm = _normalize(name)
    lines = sorted(topo.station_lines[station_id])
    n_ic = ic_lookup.get(norm, 0)
    if n_ic == 0:
        print(f"  NO DATA: {norm!r} ({len(lines)} lines: {lines})")
    else:
        print(f"  OK: {norm!r} — {n_ic} interchange entries")
