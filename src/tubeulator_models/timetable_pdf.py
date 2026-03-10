"""Download TfL timetable PDFs and extract to parquet."""

from __future__ import annotations

from pathlib import Path

import httpx
import pdfplumber
import polars as pl
from PIL import ImageDraw

from .defaults import repo_root, resolve_data


__all__ = ["sync_all", "extract_pdf", "download_pdf"]


TFL_PDF_BASE = "https://tfl.gov.uk/cdn/static/cms/documents"

MIN_TIMES_FOR_TIMETABLE = 20

# Words that appear in legend entries, not station names
_LEGEND_WORDS = {"departure", "arrival", "key", "symbol", "interchange"}

# Column header codes that modify service schedules
KNOWN_COLUMN_CODES: dict[str, str] = {
    "SX": "weekdays_only",
    "SO": "saturdays_only",
    "SN": "southern",
    "LN": "london_northwestern",
    "SE": "southeastern",
    "TL": "thameslink",
    "TS": "tue_to_sat_mornings",
    "TWO": "tue_wed_mornings",
    "ThFSO": "thu_fri_sat_mornings",
    "A": "arrival",
    "B": "continues",
    "a": "arrival_time",
    "d": "departure_time",
    "SB": "starts_from",
}


def _timetable_dir() -> Path:
    d = repo_root() / resolve_data()["timetable_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── PDF download ──────────────────────────────────────────────


def download_pdf(filename: str) -> Path:
    """Download a timetable PDF, returning the local path. Skips if already present."""
    dest = _timetable_dir() / filename
    if dest.exists():
        print(f"    {dest.name} already downloaded")
        return dest

    url = f"{TFL_PDF_BASE}/{filename}"
    print(f"    downloading {url}")
    r = httpx.get(url, follow_redirects=True, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


# ── Geometry helpers ──────────────────────────────────────────


def _cluster(values: list[float], tolerance: float) -> list[float]:
    groups: list[list[float]] = []
    for v in sorted(values):
        if groups and v - groups[-1][-1] <= tolerance:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _nearest(val: float, centres: list[float]) -> int:
    return min(range(len(centres)), key=lambda i: abs(centres[i] - val))


def _is_legend_text(name: str) -> bool:
    """True if a left-side label is a legend entry, not a station name."""
    words = set(name.strip().lower().split())
    return bool(words & _LEGEND_WORDS)


def _extract_column_codes(
    labels_right: list,
    row_centres: list[float],
    col_centres: list[float],
    data_rows: set[int],
    metadata_rows: set[int],
    legend_rows: set[int],
    n_cols: int,
    max_gap: float = 30.0,
) -> dict[int, list[str]]:
    """Extract column header codes from metadata rows just above the data region."""
    first_data_y = min(row_centres[r] for r in data_rows)

    col_codes: dict[int, list[str]] = {c: [] for c in range(n_cols)}
    unknown_codes: list[str] = []

    for r in metadata_rows | legend_rows:
        y = row_centres[r]
        # Must be above the data, and close to it — not a distant header
        if y > first_data_y or (first_data_y - y) > max_gap:
            continue
        for lbl in labels_right:
            if _nearest(lbl["top"], row_centres) != r:
                continue
            text = lbl["text"].strip()
            if not text or text.isdigit():
                continue
            c = _nearest(lbl["x0"], col_centres)
            if abs(lbl["x0"] - col_centres[c]) > 15:
                continue
            if text in KNOWN_COLUMN_CODES:
                col_codes[c].append(text)
            elif len(text) <= 5 and text.isalpha():
                unknown_codes.append(text)

    if unknown_codes:
        raise ValueError(
            f"Unknown column codes: {sorted(set(unknown_codes))} — "
            f"add entries to KNOWN_COLUMN_CODES in timetable_pdf.py"
        )

    return col_codes


# ── Page header ───────────────────────────────────────────────


def _page_header_info(page) -> tuple[str, str]:
    words = page.extract_words(x_tolerance=3, y_tolerance=2)
    top_words = sorted(
        [w for w in words if w["top"] < 80],
        key=lambda w: (w["top"], w["x0"]),
    )
    header = " ".join(w["text"] for w in top_words).lower()

    direction = "unknown"
    for marker in ("westbound", "eastbound", "northbound", "southbound"):
        if marker in header:
            direction = marker
            break
    # Overground PDFs may use route descriptions instead
    for marker in ("to stratford", "to richmond", "to clapham"):
        if marker in header:
            direction = marker
            break

    day_type = "unknown"
    if "monday" in header or "friday" in header:
        day_type = "mon-fri"
    elif "saturday" in header:
        day_type = "saturday"
    elif "sunday" in header:
        day_type = "sunday"

    return direction, day_type


# ── Single page extraction ────────────────────────────────────


def _strip_station_codes(name: str) -> tuple[str, list[str]]:
    """Remove known operator/schedule codes from a station name.

    Codes adjacent to '&' are kept (e.g. 'Plts A & B' stays intact).
    """
    words = name.split()
    clean = []
    codes = []
    for i, w in enumerate(words):
        prev = words[i - 1] if i > 0 else ""
        nxt = words[i + 1] if i < len(words) - 1 else ""
        if w in KNOWN_COLUMN_CODES and prev != "&" and nxt != "&":
            codes.append(w)
        else:
            clean.append(w)
    return " ".join(clean), codes


def _absorb_code_legends(
    legend_rows: set[int],
    data_rows: set[int],
    row_centres: list[float],
    row_labels_left: dict[int, list],
    station_codes: dict[int, list[str]],
) -> set[int]:
    """Legend rows that are purely known codes get absorbed into the nearest data row.

    Returns the remaining (non-absorbed) legend rows.
    """
    remaining = set()

    for r in legend_rows:
        words = [lbl["text"].strip() for lbl in row_labels_left.get(r, [])]
        if not words:
            remaining.add(r)
            continue

        if all(w in KNOWN_COLUMN_CODES for w in words):
            # Find nearest data row
            best = min(data_rows, key=lambda dr: abs(row_centres[dr] - row_centres[r]))
            station_codes.setdefault(best, []).extend(words)
        else:
            remaining.add(r)

    return remaining


def _extract_page(page) -> dict | None:
    """Extract timetable grid from a single page."""
    words = page.extract_words(x_tolerance=3, y_tolerance=2, keep_blank_chars=False)

    times = []
    labels = []
    for w in words:
        text = w["text"].strip()
        if len(text) == 4 and text.isdigit():
            times.append(w)
        elif text:
            labels.append(w)

    if len(times) < MIN_TIMES_FOR_TIMETABLE:
        return None

    # Columns from times only
    time_x = sorted(set(round(t["x0"], 1) for t in times))
    col_centres = _cluster(time_x, tolerance=8)
    left_boundary = col_centres[0] - 20

    labels_left = [lbl for lbl in labels if lbl["x0"] < left_boundary]
    labels_right = [lbl for lbl in labels if lbl["x0"] >= left_boundary]

    # Row clustering from times + left labels only
    grid_words = times + labels_left
    all_y = sorted(set(round(w["top"], 1) for w in grid_words))
    row_centres = _cluster(all_y, tolerance=4)

    # Bucket left labels into rows
    row_labels_left: dict[int, list] = {r: [] for r in range(len(row_centres))}
    for lbl in labels_left:
        r = _nearest(lbl["top"], row_centres)
        row_labels_left[r].append(lbl)

    # Build grid for ALL rows that have left labels
    n_cols = len(col_centres)
    candidate_rows = {r for r in range(len(row_centres)) if row_labels_left[r]}

    grid: dict[int, list[str]] = {}
    station_names: dict[int, str] = {}
    station_codes: dict[int, list[str]] = {}

    for r in sorted(candidate_rows):
        grid[r] = [""] * n_cols
        for t in times:
            if _nearest(t["top"], row_centres) == r:
                c = _nearest(t["x0"], col_centres)
                grid[r][c] = t["text"]

        name_parts = sorted(row_labels_left[r], key=lambda w: w["x0"])
        raw_name = " ".join(w["text"] for w in name_parts)
        station_names[r], station_codes[r] = _strip_station_codes(raw_name)

    # Key heuristic: a real station row has at least one time in the grid.
    # Footer legends have station-name-like text but zero times.
    data_rows = {
        r
        for r in candidate_rows
        if any(grid[r]) and not _is_legend_text(station_names[r])
    }
    legend_rows = candidate_rows - data_rows
    legend_rows = _absorb_code_legends(
        legend_rows,
        data_rows,
        row_centres,
        row_labels_left,
        station_codes,
    )

    # Everything else is metadata
    metadata_rows = set(range(len(row_centres))) - data_rows

    # Extract column codes from metadata/legend rows:
    # short non-numeric text in the grid area that snaps to a column centre
    col_codes = _extract_column_codes(
        labels_right,
        row_centres,
        col_centres,
        data_rows,
        metadata_rows,
        legend_rows,
        n_cols,
    )

    # Annotations on data rows
    annotations: dict[int, str] = {}
    for r in sorted(data_rows):
        right = [lbl for lbl in labels_right if _nearest(lbl["top"], row_centres) == r]
        if right:
            annotations[r] = " ".join(
                w["text"] for w in sorted(right, key=lambda w: w["x0"])
            )

    # Collect metadata text for debug
    metadata = []
    for r in sorted(metadata_rows | legend_rows):
        row_words = [t for t in times if _nearest(t["top"], row_centres) == r]
        row_words += [
            lbl
            for lbl in labels_left + labels_right
            if _nearest(lbl["top"], row_centres) == r
        ]
        row_words.sort(key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in row_words)
        metadata.append(
            {"row": r, "y": row_centres[r], "text": text, "words": row_words}
        )

    return {
        "station_names": station_names,
        "grid": {r: grid[r] for r in data_rows},
        "n_cols": n_cols,
        "col_codes": col_codes,
        "row_centres": row_centres,
        "col_centres": col_centres,
        "data_rows": data_rows,
        "legend_rows": legend_rows,
        "metadata_rows": metadata_rows,
        "metadata": metadata,
        "annotations": annotations,
        "times": times,
        "labels_left": labels_left,
        "labels_right": labels_right,
        "row_labels_left": row_labels_left,
        "station_codes": station_codes,
    }


# ── Debug image rendering ─────────────────────────────────────


def _render_debug_image(page, result: dict, out_path: Path) -> None:
    """Draw detected grid and classified words onto the page image."""
    im = page.to_image(resolution=150)
    scale = 150 / 72
    draw = ImageDraw.Draw(im.annotated)

    for y in result["row_centres"]:
        sy = y * scale
        draw.line([(0, sy), (im.annotated.width, sy)], fill="lime", width=1)

    for x in result["col_centres"]:
        sx = x * scale
        draw.line([(sx, 0), (sx, im.annotated.height)], fill="cyan", width=1)

    # Data row times — red
    for r in result["data_rows"]:
        for t in result["times"]:
            if _nearest(t["top"], result["row_centres"]) == r:
                box = (
                    t["x0"] * scale,
                    t["top"] * scale,
                    t["x1"] * scale,
                    t["bottom"] * scale,
                )
                draw.rectangle(box, outline="red", width=1)

    # Station labels — yellow for names, green for stripped codes
    for r in result["data_rows"]:
        stripped = set(result.get("station_codes", {}).get(r, []))
        for lbl in result["row_labels_left"].get(r, []):
            is_code = lbl["text"].strip() in stripped
            box = (
                lbl["x0"] * scale,
                lbl["top"] * scale,
                lbl["x1"] * scale,
                lbl["bottom"] * scale,
            )
            draw.rectangle(
                box, outline="lime" if is_code else "yellow", width=2 if is_code else 1
            )

    # Annotations on data rows — magenta
    for r in result["data_rows"]:
        for lbl in result["labels_right"]:
            if _nearest(lbl["top"], result["row_centres"]) == r:
                box = (
                    lbl["x0"] * scale,
                    lbl["top"] * scale,
                    lbl["x1"] * scale,
                    lbl["bottom"] * scale,
                )
                draw.rectangle(box, outline="magenta", width=2)

    # Metadata rows — orange
    for m in result["metadata"]:
        for w in m["words"]:
            box = (
                w["x0"] * scale,
                w["top"] * scale,
                w["x1"] * scale,
                w["bottom"] * scale,
            )
            draw.rectangle(box, outline="orange", width=2)

    # Legend rows (had station-like text but no times) — white
    for r in result.get("legend_rows", set()):
        for lbl in result["row_labels_left"].get(r, []):
            box = (
                lbl["x0"] * scale,
                lbl["top"] * scale,
                lbl["x1"] * scale,
                lbl["bottom"] * scale,
            )
            draw.rectangle(box, outline="deeppink", width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.annotated.save(out_path)


# ── Full PDF extraction ───────────────────────────────────────


def extract_pdf(pdf_path: Path, line_id: str, debug: bool = True) -> pl.DataFrame:
    """Extract all timetable pages from a PDF into a long-format DataFrame."""
    debug_dir = _timetable_dir() / "pages" / line_id if debug else None

    records = []
    service_counter = 0

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            result = _extract_page(page)

            if result is None:
                print(f"    page {i}: skipped")
                if debug_dir:
                    im = page.to_image(resolution=150)
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    im.save(debug_dir / f"page_{i:02d}_skipped.png")
                continue

            direction, day_type = _page_header_info(page)
            grid = result["grid"]
            station_names = result["station_names"]
            n_cols = result["n_cols"]
            col_codes = result["col_codes"]
            rows_sorted = sorted(grid.keys())

            if debug_dir:
                _render_debug_image(page, result, debug_dir / f"page_{i:02d}.png")

            for col_idx in range(n_cols):
                col_times = [
                    (r, grid[r][col_idx]) for r in rows_sorted if grid[r][col_idx]
                ]
                if not col_times:
                    continue

                service_counter += 1
                codes = ",".join(col_codes.get(col_idx, []))

                for r, time_str in col_times:
                    s_codes = result["station_codes"].get(r, [])
                    records.append(
                        {
                            "direction": direction,
                            "day_type": day_type,
                            "page": i,
                            "service": service_counter,
                            "station": station_names[r],
                            "time": time_str,
                            "col_codes": codes,
                            "station_codes": ",".join(s_codes),
                        }
                    )

    df = pl.DataFrame(records)
    print(f"    {df.height} stop-times, {df['service'].n_unique()} services")
    return df


# ── Top-level sync ────────────────────────────────────────────


def sync_all() -> None:
    """Download and extract all configured PDF timetable sources."""
    cfg = resolve_data()
    sources = cfg.get("pdf_timetables", {})

    if not sources:
        print("No PDF timetable sources configured.")
        return

    out_dir = _timetable_dir()

    for line_id, spec in sources.items():
        filename = spec["filename"]
        parquet_path = out_dir / f"{line_id}_timetable.parquet"

        print(f"  {line_id}:")

        if parquet_path.exists():
            print(f"    {parquet_path.name} already exists, skipping")
            continue

        pdf_path = download_pdf(filename)
        df = extract_pdf(pdf_path, line_id=line_id)
        df.write_parquet(parquet_path)
        print(f"    wrote {parquet_path.name}")
