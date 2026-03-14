from pathlib import Path

import pdfplumber
import polars as pl


pdf_path = next(Path().glob("elizabeth-line-*.pdf"))


def cluster(values, tolerance):
    groups = []
    for v in sorted(values):
        if groups and v - groups[-1][-1] <= tolerance:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


MIN_TIMES_FOR_TIMETABLE = 20


def extract_timetable(page):
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
        return None, {}

    time_x = sorted(set(round(t["x0"], 1) for t in times))
    col_centres = cluster(time_x, tolerance=8)
    left_boundary = col_centres[0] - 20

    labels_left = [lbl for lbl in labels if lbl["x0"] < left_boundary]
    labels_right = [lbl for lbl in labels if lbl["x0"] >= left_boundary]

    grid_words = times + labels_left
    all_y = sorted(set(round(w["top"], 1) for w in grid_words))
    row_centres = cluster(all_y, tolerance=4)

    def nearest(val, centres):
        return min(range(len(centres)), key=lambda i: abs(centres[i] - val))

    row_labels_left = {r: [] for r in range(len(row_centres))}
    for lbl in labels_left:
        r = nearest(lbl["top"], row_centres)
        row_labels_left[r].append(lbl)

    data_rows = set()
    for r in range(len(row_centres)):
        if len(row_labels_left[r]) > 0:
            data_rows.add(r)

    n_cols = len(col_centres)
    grid = {}
    station_names = {}

    for r in sorted(data_rows):
        grid[r] = [""] * n_cols
        for t in times:
            tr = nearest(t["top"], row_centres)
            if tr == r:
                c = nearest(t["x0"], col_centres)
                grid[r][c] = t["text"]

        name_parts = sorted(row_labels_left[r], key=lambda w: w["x0"])
        station_names[r] = " ".join(w["text"] for w in name_parts)

    # collect annotations for reporting
    annotations = {}
    for r in sorted(data_rows):
        right = [lbl for lbl in labels_right if nearest(lbl["top"], row_centres) == r]
        if right:
            annotations[r] = " ".join(
                w["text"] for w in sorted(right, key=lambda w: w["x0"])
            )

    return (station_names, grid, n_cols), annotations


def page_header_info(page):
    """Extract direction and day type from page header text."""
    words = page.extract_words(x_tolerance=3, y_tolerance=2)
    top_words = sorted(
        [w for w in words if w["top"] < 80],
        key=lambda w: (w["top"], w["x0"]),
    )
    header_text = " ".join(w["text"] for w in top_words)

    direction = "unknown"
    if "Westbound" in header_text:
        direction = "westbound"
    elif "Eastbound" in header_text:
        direction = "eastbound"

    day_type = "unknown"
    header_lower = header_text.lower()
    if "monday" in header_lower or "friday" in header_lower:
        day_type = "mon-fri"
    elif "saturday" in header_lower:
        day_type = "saturday"
    elif "sunday" in header_lower:
        day_type = "sunday"

    return direction, day_type


records = []
service_counter = 0

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        result, annotations = extract_timetable(page)
        if result is None:
            print(f"Page {i}: skipped")
            continue

        station_names, grid, n_cols = result
        direction, day_type = page_header_info(page)
        rows_sorted = sorted(grid.keys())

        print(
            f"Page {i}: {direction} {day_type} — {len(rows_sorted)} stations × {n_cols} services"
        )
        for r, ann in sorted(annotations.items()):
            print(f"  annotation: {ann}")

        for col_idx in range(n_cols):
            # check if this column has any times at all
            col_times = [(r, grid[r][col_idx]) for r in rows_sorted if grid[r][col_idx]]
            if not col_times:
                continue

            service_counter += 1
            for r, time_str in col_times:
                records.append(
                    {
                        "direction": direction,
                        "day_type": day_type,
                        "page": i,
                        "service": service_counter,
                        "station": station_names[r],
                        "time": time_str,
                    }
                )

df = pl.DataFrame(records)
print(f"\\n{df.shape[0]} stop-time records, {df['service'].n_unique()} services")
print(df)

df.write_parquet("elizabeth_timetable.parquet")
print("Wrote elizabeth_timetable.parquet")
