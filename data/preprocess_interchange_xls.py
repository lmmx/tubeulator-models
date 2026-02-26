import json
import re

import polars as pl
import tubeulator as tube
import xlrd


# ============================================================
# STEP 1: Parse the XLS
# ============================================================


def parse_interchange_xls(filepath):
    wb = xlrd.open_workbook(filepath, formatting_info=True)
    ws = wb.sheet_by_index(0)

    merge_map = {}
    for rlo, rhi, clo, chi in ws.merged_cells:
        top_left_val = ws.cell_value(rlo, clo)
        for r in range(rlo, rhi):
            for c in range(clo, chi):
                merge_map[(r, c)] = top_left_val

    def cell_val(row, col):
        if (row, col) in merge_map:
            return merge_map[(row, col)]
        return ws.cell_value(row, col)

    TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*minutes?", re.IGNORECASE)
    SEP_RE = re.compile(r"\s*[:\u2013\u2014]\s*|\s+-\s+")

    def parse_gate_entry(text):
        text = text.strip().rstrip(".")
        if not text:
            return None
        time_match = TIME_RE.search(text)
        if not time_match:
            return {"note": text}
        minutes = float(time_match.group(1))
        if minutes == int(minutes):
            minutes = int(minutes)
        parts = SEP_RE.split(text, maxsplit=1)
        if len(parts) == 2 and not TIME_RE.match(parts[0].strip()):
            return {"line": parts[0].strip(), "minutes": minutes}
        else:
            return {"line": None, "minutes": minutes}

    def parse_interchange_entry(text):
        text = text.strip().rstrip(".")
        if not text:
            return None
        to_match = re.match(
            r"^(.+?)\s+to\s+(.+?)(?:\s*[:\u2013\u2014-]\s*(.*))?$",
            text,
            re.IGNORECASE,
        )
        if to_match:
            from_line = to_match.group(1).strip()
            rest = to_match.group(2).strip()
            after_sep = to_match.group(3)
            note_starters = ("step-free", "interchange", "access", "use ", "via ")
            if from_line.lower().startswith(note_starters):
                return {"note": text}
            if after_sep:
                time_match = TIME_RE.search(after_sep)
                if time_match:
                    minutes = float(time_match.group(1))
                    if minutes == int(minutes):
                        minutes = int(minutes)
                    return {"from_line": from_line, "to_line": rest, "minutes": minutes}
            time_match = TIME_RE.search(rest)
            if time_match:
                to_line = rest[: time_match.start()].rstrip(" :\u2013\u2014-")
                minutes = float(time_match.group(1))
                if minutes == int(minutes):
                    minutes = int(minutes)
                return {"from_line": from_line, "to_line": to_line, "minutes": minutes}
            return {
                "from_line": from_line,
                "to_line": rest,
                "minutes": None,
                "note": "no time specified in source",
            }
        time_match = TIME_RE.search(text)
        if time_match:
            parts = SEP_RE.split(text, maxsplit=1)
            if len(parts) == 2:
                minutes = float(time_match.group(1))
                if minutes == int(minutes):
                    minutes = int(minutes)
                return {"route": parts[0].strip(), "minutes": minutes}
        return {"note": text}

    stations = []
    current_station = None
    current_gate = []
    current_ix = []

    for row_idx in range(1, ws.nrows):
        a = str(cell_val(row_idx, 0)).strip()
        b = str(cell_val(row_idx, 1)).strip()
        c = str(cell_val(row_idx, 2)).strip()

        if a and a != current_station:
            if current_station:
                stations.append(
                    {
                        "station": current_station,
                        "gate_to_platform": [e for e in current_gate if e],
                        "interchanges": [e for e in current_ix if e],
                    }
                )
            current_station = a
            current_gate = []
            current_ix = []

        if b:
            current_gate.append(parse_gate_entry(b))
        if c:
            current_ix.append(parse_interchange_entry(c))

    if current_station:
        stations.append(
            {
                "station": current_station,
                "gate_to_platform": [e for e in current_gate if e],
                "interchanges": [e for e in current_ix if e],
            }
        )

    # --- Deduplicate gate_to_platform ---
    # Merged cells cause identical entries to repeat once per interchange row
    for s in stations:
        seen = set()
        deduped = []
        for g in s["gate_to_platform"]:
            key = (g.get("line"), g.get("minutes"), g.get("note"))
            if key not in seen:
                seen.add(key)
                deduped.append(g)
        s["gate_to_platform"] = deduped

    # Deduplicate interchanges too
    for s in stations:
        seen = set()
        deduped = []
        for ix in s["interchanges"]:
            key = json.dumps(ix, sort_keys=True)
            if key not in seen:
                seen.add(key)
                deduped.append(ix)
        s["interchanges"] = deduped

    return stations


# ============================================================
# STEP 2: Load tubeulator data and build lookups
# ============================================================

platforms = tube.load_platforms_with_stations_and_services()

tb_stations = (
    platforms.select("StationUniqueId", "StationName").unique().sort("StationName")
)
tb_name_to_id = dict(
    zip(tb_stations["StationName"].to_list(), tb_stations["StationUniqueId"].to_list())
)

# For each StationUniqueId, the set of line slugs that serve it
lines_by_station = (
    platforms.select("StationUniqueId", "Line")
    .unique()
    .group_by("StationUniqueId")
    .agg(pl.col("Line").alias("lines"))
)
station_id_to_lines = dict(
    zip(
        lines_by_station["StationUniqueId"].to_list(),
        [set(ls) for ls in lines_by_station["lines"].to_list()],
    )
)


# ============================================================
# STEP 3: Mappings
# ============================================================

SLUG_TO_NAME = {
    "bakerloo": "Bakerloo",
    "central": "Central",
    "circle": "Circle",
    "district": "District",
    "dlr": "DLR",
    "elizabeth": "Elizabeth",
    "hammersmith-city": "Hammersmith & City",
    "jubilee": "Jubilee",
    "liberty": "Liberty",
    "lioness": "Lioness",
    "london-cable-car": "London Cable Car",
    "metropolitan": "Metropolitan",
    "mildmay": "Mildmay",
    "national-rail": "National Rail",
    "northern": "Northern",
    "piccadilly": "Piccadilly",
    "suffragette": "Suffragette",
    "thameslink": "Thameslink",
    "tram": "Tramlink",
    "victoria": "Victoria",
    "waterloo-city": "Waterloo & City",
    "weaver": "Weaver",
    "windrush": "Windrush",
}

STATION_MAP = {
    "Bank, including Monument interchange values": "Bank",
    "Monument, includes interchange values with Bank": "Monument",
    "Bethnal Green, London Underground": "Bethnal Green",
    "Brixton London Underground": "Brixton",
    "Bromley-By-Bow": "Bromley-by-Bow",
    "Caledonian Road London Underground": "Caledonian Road",
    "Canary Wharf DLR": "Canary Wharf",
    "Canary Wharf London Underground": "Canary Wharf",
    "Custom House (for ExCel)": "Custom House for ExCel",
    "Cutty Sark": "Cutty Sark for Maritime Greenwich",
    "Earls Court": "Earl's Court",
    "Edgware Road Bakerloo": "Edgware Road",
    "Edgware Road Circle": "Edgware Road",
    "Elephant & Castle London Underground": "Elephant & Castle",
    "Hammersmith District and Piccadilly lines": "Hammersmith",
    "Hammersmith Hammersmith & City line": "Hammersmith",
    "Harrow-On-The-Hill": "Harrow-on-the-Hill",
    "Heathrow Terminal  5": "Heathrow Terminal 5",
    "Heathrow Terminals 123 London Underground": "Heathrow Terminals 2 & 3",
    "Kensington Olympia": "Kensington (Olympia)",
    "Kings Cross St.Pancras": "King's Cross St Pancras",
    "Moorgate (including First Capital Connect)": "Moorgate",
    "Queens Park": "Queen's Park",
    "Regents Park": "Regent's Park",
    "Shepherds Bush": "Shepherd's Bush",
    "Shepherds Bush Market": "Shepherd's Bush Market",
    "St John\u2019s Wood": "St John's Wood",
    "St.James's Park": "St James's Park",
    "St.Pauls": "St Paul's",
    "Walthamstow Queens Road": "Walthamstow Queen's Road",
}

LINE_MAP = {
    "Bakerloo": "bakerloo",
    "Central": "central",
    "Circle": "circle",
    "District": "district",
    "DLR": "dlr",
    "Hammersmith & City": "hammersmith-city",
    "Jubilee": "jubilee",
    "Metropolitan": "metropolitan",
    "Northern": "northern",
    "Piccadilly": "piccadilly",
    "Victoria": "victoria",
    "Waterloo & City": "waterloo-city",
    "Tramlink": "tram",
    "Jubilee line": "jubilee",
    "Piccadilly line": "piccadilly",
    "Hammersmith & City platform": "hammersmith-city",
    "DLR Canary Wharf": "dlr",
    "DLR Stratford Int": "dlr",
    "London Overground": "london-overground",
    "London Overground (North/West London Lines)": "london-overground",
    "London Overground (Watford)": "london-overground",
    "London Overground, Barking line": "london-overground",
    "London Overground, NLL": "london-overground",
    "London Overground/NR": "london-overground",
    "London Overground/NR (from concourse)": "london-overground",
    "London Overground and National Rail": "london-overground",
    "London Overground and all National Rail": "london-overground",
    "National Rail": "national-rail",
    "National Rail (c2c)": "national-rail",
    "National Rail (concourse)": "national-rail",
    "C2C": "national-rail",
    "c2c": "national-rail",
    "Chiltern": "national-rail",
    "Chiltern Railways": "national-rail",
    "First Capital Connect": "national-rail",
    "First  Capital Connect": "national-rail",
    "First Great Western": "national-rail",
    "Heathrow Connect": "national-rail",
    "Heathrow Express": "national-rail",
    "London Midland": "national-rail",
    "National Express East Anglia": "national-rail",
    "South West Trains": "national-rail",
    "South West Trains (Richmond line)": "national-rail",
    "South West Trains (Richmond)": "national-rail",
    "South West Trains (Wimbledon line)": "national-rail",
    "South West Trains (Wimbledon)": "national-rail",
    "Southeastern": "national-rail",
    "Southern": "national-rail",
    "Virgin Trains": "national-rail",
    "Bakerloo, Jubilee": ["bakerloo", "jubilee"],
    "Circle, District line platform": ["circle", "district"],
    "Circle, Hammersmith & City, Metropolitan": [
        "circle",
        "hammersmith-city",
        "metropolitan",
    ],
}

# Reverse: slug -> set of XLS line names that map to it (for detecting "already named")
SLUG_TO_XLS = {}
for xls_name, slug in LINE_MAP.items():
    if isinstance(slug, str):
        SLUG_TO_XLS.setdefault(slug, set()).add(xls_name)


def resolve_station(xls_name):
    xls_name = xls_name.replace("\u2019", "'")
    tb_name = STATION_MAP.get(xls_name, xls_name)
    uid = tb_name_to_id.get(tb_name)
    return tb_name, uid


def resolve_line(xls_line):
    return LINE_MAP.get(xls_line)


# ============================================================
# STEP 3b: Parse "route" interchange entries into structured form
# ============================================================

SUFFIX_STRIP = re.compile(
    r"\s+(?:line\s+)?(?:trains|services|line|lines)$", re.IGNORECASE
)


def parse_route_interchange(route_text, minutes):
    """
    Parse a 'route' string into a list of structured interchange dicts.
    Returns a list because multi-line groups expand to multiple pairs.
    """
    text = route_text.strip()

    # --- Cross-station interchanges ---
    m = re.match(r"^Interchange with\s+(.+)$", text, re.IGNORECASE)
    if m:
        return [{"cross_station": m.group(1).strip(), "minutes": minutes}]

    # --- Same-line branch interchanges ---
    # "District (between branches)", "District (different branches)"
    m = re.match(r"^(.+?)\s*\((?:between|different)\s+branches\)$", text, re.IGNORECASE)
    if m:
        line = m.group(1).strip().rstrip(",")
        slug = resolve_line(line)
        return [
            {
                "branch_interchange": True,
                "line": line,
                "line_slug": slug,
                "minutes": minutes,
            }
        ]

    # "DLR between branches", "Southern, between branches"
    m = re.match(r"^(.+?)[,\s]+between\s+branches$", text, re.IGNORECASE)
    if m:
        line = m.group(1).strip()
        slug = resolve_line(line)
        return [
            {
                "branch_interchange": True,
                "line": line,
                "line_slug": slug,
                "minutes": minutes,
            }
        ]

    # "Interchange between X branch and Y"
    m = re.match(
        r"^Interchange between\s+(.+?)\s+branch\s+and\s+(.+)$", text, re.IGNORECASE
    )
    if m:
        return [
            {
                "branch_interchange": True,
                "line": "London Overground",
                "line_slug": "london-overground",
                "minutes": minutes,
            }
        ]

    # "Interchange between branches" / "Interchange between different DLR branches"
    m = re.match(
        r"^Interchange between\s+(?:different\s+)?(?:(\w+)\s+)?branches$",
        text,
        re.IGNORECASE,
    )
    if m:
        line = m.group(1) if m.group(1) else None
        slug = resolve_line(line) if line else None
        return [
            {
                "branch_interchange": True,
                "line": line,
                "line_slug": slug,
                "minutes": minutes,
            }
        ]

    # "Connections between different DLR branches" / "Connections between different branches"
    m = re.match(
        r"^Connections between\s+different\s+(?:(\w+)\s+)?branches$",
        text,
        re.IGNORECASE,
    )
    if m:
        line = m.group(1) if m.group(1) else None
        slug = resolve_line(line) if line else None
        return [
            {
                "branch_interchange": True,
                "line": line,
                "line_slug": slug,
                "minutes": minutes,
            }
        ]

    # --- Special case: Northern line branches ---
    if "High Barnet" in text and "Mill Hill East" in text:
        return [
            {
                "branch_interchange": True,
                "line": "Northern",
                "line_slug": "northern",
                "minutes": minutes,
            }
        ]

    # --- "X <> Y" ---
    m = re.match(r"^(.+?)\s*<>\s*(.+)$", text)
    if m:
        a = SUFFIX_STRIP.sub("", m.group(1).strip())
        b = SUFFIX_STRIP.sub("", m.group(2).strip())
        return [
            {
                "from_line": a,
                "to_line": b,
                "minutes": minutes,
                "from_line_slug": resolve_line(a),
                "to_line_slug": resolve_line(b),
            }
        ]

    # --- "London Overground (between X and Y Lines)" ---
    m = re.match(r"^London Overground\s*\(between\s+.+\)$", text, re.IGNORECASE)
    if m:
        return [
            {
                "branch_interchange": True,
                "line": "London Overground",
                "line_slug": "london-overground",
                "minutes": minutes,
            }
        ]

    # --- "Connections between X and Y" / "Connections between X, Y and Z" ---
    m = re.match(r"^Connections between\s+(.+)$", text, re.IGNORECASE)
    if m:
        body = m.group(1).strip()
        # Strip trailing qualifiers: "line trains", "services", "lines", etc.
        body = SUFFIX_STRIP.sub("", body)
        # Also handle mid-text: "and First Capital Connect" at end after main pair
        # Split on " and " — but careful with "Hammersmith & City"
        # Strategy: split on ", " first, then last element on " and "
        parts = [p.strip() for p in body.split(",")]
        # The last part may contain " and X"
        expanded = []
        for i, part in enumerate(parts):
            if " and " in part:
                subparts = part.split(" and ")
                expanded.extend(s.strip() for s in subparts)
            else:
                expanded.append(part)

        # Clean suffixes from each part individually
        cleaned = [SUFFIX_STRIP.sub("", p).strip() for p in expanded]
        # Remove empty strings
        cleaned = [c for c in cleaned if c]

        # Handle special combined text like "District, Hammersmith & City Circle"
        # where "Hammersmith & City" got split wrong — rejoin if we see "Hammersmith" alone
        rejoined = []
        skip_next = False
        for i, c in enumerate(cleaned):
            if skip_next:
                skip_next = False
                continue
            if (
                c == "Hammersmith"
                and i + 1 < len(cleaned)
                and cleaned[i + 1].startswith("City")
            ):
                rest = cleaned[i + 1]
                # "City Circle" -> need to split again
                if rest == "City":
                    rejoined.append("Hammersmith & City")
                elif rest.startswith("City "):
                    rejoined.append("Hammersmith & City")
                    leftover = rest[5:].strip()
                    if leftover:
                        rejoined.append(leftover)
                else:
                    rejoined.append("Hammersmith & City")
                skip_next = True
            else:
                rejoined.append(c)
        cleaned = rejoined

        # Resolve each to slug
        resolved = []
        for name in cleaned:
            slug = resolve_line(name)
            resolved.append((name, slug))

        # Generate all pairs
        if len(resolved) == 1:
            # Shouldn't happen but handle gracefully
            return [{"route": route_text, "minutes": minutes}]

        pairs = []
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                pairs.append(
                    {
                        "from_line": resolved[i][0],
                        "to_line": resolved[j][0],
                        "minutes": minutes,
                        "from_line_slug": resolved[i][1],
                        "to_line_slug": resolved[j][1],
                    }
                )
        return pairs

    # --- Fallback ---
    return [{"route": route_text, "minutes": minutes}]


# ============================================================
# STEP 4: Enrich
# ============================================================

xls_data = parse_interchange_xls("interchange_values.xls")

unmatched_stations = []
unmatched_lines = set()
enriched = []

for station in xls_data:
    tb_name, uid = resolve_station(station["station"])
    if uid is None:
        unmatched_stations.append(station["station"])

    # Collect line slugs that are explicitly named in gate_to_platform
    explicitly_named_slugs = set()
    for g in station["gate_to_platform"]:
        if g.get("line"):
            slug = resolve_line(g["line"])
            if isinstance(slug, list):
                explicitly_named_slugs.update(slug)
            elif slug:
                explicitly_named_slugs.add(slug)

    # Lines at this station according to tubeulator
    tb_lines_here = station_id_to_lines.get(uid, set()) if uid else set()

    # Expand null-line gate entries to cover all remaining lines at station
    expanded_gate = []
    for g in station["gate_to_platform"]:
        if g.get("note"):
            expanded_gate.append(g)
            continue

        if g.get("line"):
            # Explicitly named — resolve and keep
            slug = resolve_line(g["line"])
            if slug is None:
                unmatched_lines.add(g["line"])
            expanded_gate.append(
                {
                    "line": g["line"],
                    "minutes": g["minutes"],
                    "line_slug": slug,
                }
            )
        else:
            # Null line — expand to all lines not already covered
            remaining = sorted(tb_lines_here - explicitly_named_slugs)
            if remaining:
                for line_slug in remaining:
                    expanded_gate.append(
                        {
                            "line": SLUG_TO_NAME.get(line_slug, line_slug),
                            "line_slug": line_slug,
                            "minutes": g["minutes"],
                            "inferred": True,
                        }
                    )
            else:
                # No tubeulator data or everything already named
                expanded_gate.append(
                    {
                        "line": None,
                        "line_slug": None,
                        "minutes": g["minutes"],
                    }
                )

    # Enrich interchanges
    enriched_ix = []
    for ix in station["interchanges"]:
        if ix.get("route"):
            # Parse the route string into structured entries
            parsed = parse_route_interchange(ix["route"], ix["minutes"])
            enriched_ix.extend(parsed)
        else:
            e = dict(ix)
            if ix.get("from_line"):
                slug = resolve_line(ix["from_line"])
                if slug is None:
                    unmatched_lines.add(ix["from_line"])
                e["from_line_slug"] = slug
            if ix.get("to_line"):
                slug = resolve_line(ix["to_line"])
                if slug is None:
                    unmatched_lines.add(ix["to_line"])
                e["to_line_slug"] = slug
            enriched_ix.append(e)

    enriched.append(
        {
            "station": station["station"],
            "station_name_tb": tb_name,
            "station_unique_id": uid,
            "gate_to_platform": expanded_gate,
            "interchanges": enriched_ix,
        }
    )


with open("interchange_times_enriched.json", "w") as f:
    json.dump(enriched, f, indent=2)

# ============================================================
# DIAGNOSTICS
# ============================================================

total_gate = sum(len(s["gate_to_platform"]) for s in enriched)
inferred_gate = sum(
    1 for s in enriched for g in s["gate_to_platform"] if g.get("inferred")
)
note_gate = sum(1 for s in enriched for g in s["gate_to_platform"] if g.get("note"))

print(f"Stations:           {len(enriched)}")
print(f"Matched to TB:      {sum(1 for e in enriched if e['station_unique_id'])}")
print(f"Unmatched stations: {len(unmatched_stations)}")
for s in unmatched_stations:
    print(f"  {s!r}")

print(
    f"\nGate entries:       {total_gate} ({inferred_gate} inferred, {note_gate} notes)"
)
print(f"Unmatched lines:    {len(unmatched_lines)}")
for l in sorted(unmatched_lines):
    print(f"  {l!r}")

remaining_routes = [
    (s["station"], ix) for s in enriched for ix in s["interchanges"] if "route" in ix
]
branch_ix = sum(
    1 for s in enriched for ix in s["interchanges"] if ix.get("branch_interchange")
)
cross_station_ix = sum(
    1 for s in enriched for ix in s["interchanges"] if ix.get("cross_station")
)
print(f"\nInterchange entries: {sum(len(s['interchanges']) for s in enriched)}")
print(f"  Branch interchanges: {branch_ix}")
print(f"  Cross-station:       {cross_station_ix}")
print(f"  Remaining routes:    {len(remaining_routes)}")
for station, ix in remaining_routes:
    print(f"    {station}: {ix}")

# Show Mile End and West Ham for sanity check
for name in ["Mile End", "West Ham", "Plaistow"]:
    for s in enriched:
        if s["station_name_tb"] == name:
            print(f"\n=== {name} ===")
            print(f"Gate entries: {len(s['gate_to_platform'])}")
            for g in s["gate_to_platform"]:
                print(f"  {g}")
            print(f"Interchanges: {len(s['interchanges'])}")
            for ix in s["interchanges"]:
                print(f"  {ix}")
