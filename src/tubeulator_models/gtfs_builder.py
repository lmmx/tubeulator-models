"""Build a GTFS zip from TfL open data via the tubeulator library."""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import tubeulator as tube

from .defaults import repo_root, resolve_data


SCHEDULE_MAP = {
    "monday - friday": "MTF",
    "monday to friday": "MTF",
    "saturday": "SAT",
    "sunday": "SUN",
}

GTFS_ROUTE_TYPE = {
    "tube": 1,
    "dlr": 0,
    "elizabeth-line": 2,
    "overground": 2,
    "tram": 0,
    "national-rail": 2,
}


PARQUET_SERVICE_MAP = {
    "mon-fri": "MTF",
    "saturday": "SAT",
    "sunday": "SUN",
}

PARQUET_DIRECTION_MAP = {
    "westbound": 0,
    "eastbound": 1,
}


# Aliases for PDF station names that don't match API CommonName
PARQUET_NAME_ALIASES: dict[str, str] = {
    "liverpool st plts a & b": "liverpool street",
    "liverpool st plts 15–17": "liverpool street",
    "paddington plts 11 & 12": "paddington",
    "paddington plts a & b": "paddington",
    "burnham": "burnham (berks)",
    "langley": "langley (berks)",
    "custom house": "custom house",
}


@dataclass
class Stop:
    """Normalized stop record that works for both API object types."""

    Id: str
    Name: str
    Lat: float
    Lon: float


def _normalize_stop_name(name: str) -> str:
    s = name.strip().lower()
    # Normalize apostrophes
    s = s.replace("\u2019", "'")  # right single quote
    s = s.replace("\u2018", "'")  # left single quote
    # Strip trailing operator codes (LN = London Northwestern, SN = Southern)
    for suffix in (" ln sn", " ln", " sn"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    # Strip station type suffixes
    for suffix in (" rail station", " underground station", " station"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def _make_stop(obj) -> Stop:
    """Normalize a StopPoint or MatchedStop into a Stop."""
    sid = getattr(obj, "NaptanId", None) or getattr(obj, "Id", None) or ""
    name = getattr(obj, "CommonName", None) or getattr(obj, "Name", None) or ""
    lat = getattr(obj, "Lat", None) or 0.0
    lon = getattr(obj, "Lon", None) or 0.0
    return Stop(Id=sid, Name=name, Lat=lat, Lon=lon)


def _minutes_to_gtfs_time(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}:00"


def _load_from_parquet(
    line_id: str,
    parquet_path: Path,
    trip_rows: list,
    stop_time_rows: list,
    stops: dict,
) -> None:
    """Load timetable data from a pre-extracted parquet file."""
    stop_points = tube.fetch.line.stop_points_by_id(id=line_id)
    for sp in stop_points:
        s = _make_stop(sp)
        if s.Id and s.Id not in stops:
            stops[s.Id] = s

    # Stops missing from the line's stop_points API response
    MISSING_STOPS = {"HUBCUS": "Custom House"}
    for sid, name in MISSING_STOPS.items():
        if sid not in stops:
            try:
                sp = tube.fetch.stop_point.stop_by_id(id=sid)
                stops[sid] = _make_stop(sp)
            except Exception:
                stops[sid] = Stop(Id=sid, Name=name, Lat=51.5095, Lon=0.0255)

    df = pl.read_parquet(parquet_path)

    # Name → stop_id lookup, normalized
    name_to_sid: dict[str, str] = {}
    for sid, st in stops.items():
        norm = _normalize_stop_name(st.Name)
        if norm:
            name_to_sid[norm] = sid

    def _resolve(station_name: str) -> str | None:
        norm = _normalize_stop_name(station_name)
        alias = PARQUET_NAME_ALIASES.get(norm)
        if alias:
            norm = alias
        return name_to_sid.get(norm)

    unmatched: set[str] = set()
    n_trips = 0

    for (direction, day_type, service), group in df.group_by(
        ["direction", "day_type", "service"], maintain_order=True
    ):
        service_id = PARQUET_SERVICE_MAP.get(day_type, "MTF")
        dir_id = PARQUET_DIRECTION_MAP.get(direction, 0)

        rows = group.sort("time")
        first_time = rows["time"][0]

        trip_id = f"{line_id}_{direction}_{service_id}_{first_time}_s{service}"
        trip_rows.append([line_id, service_id, trip_id, "", dir_id])

        for seq, row in enumerate(rows.iter_rows(named=True)):
            stop_id = _resolve(row["station"])

            if stop_id is None:
                unmatched.add(row["station"])
                continue

            hhmm = row["time"]
            h, m = int(hhmm[:2]), int(hhmm[2:])
            gtfs_time = f"{h:02d}:{m:02d}:00"

            stop_time_rows.append([trip_id, gtfs_time, gtfs_time, stop_id, seq])

        n_trips += 1

    print(f"    {n_trips} trips from {parquet_path.name}")
    if unmatched:
        print(f"    UNMATCHED ({len(unmatched)}):")
        for name in sorted(unmatched):
            norm = _normalize_stop_name(name)
            print(f"      '{norm}' not in stop names")
        print(f"    Available names: {sorted(name_to_sid.keys())}")
        raise ValueError(
            f"{len(unmatched)} unmatched Elizabeth line stations — "
            f"add entries to PARQUET_NAME_ALIASES"
        )


def _load_from_api(
    line_id: str,
    line_obj,
    trip_rows: list,
    stop_time_rows: list,
    stops: dict,
) -> None:
    """Load timetable data from the TfL API."""
    try:
        route_data = tube.fetch.line.route_by_ids(ids=line_id)
    except Exception as e:
        print(f"    skipping {line_id} (route fetch failed: {e})")
        return

    route_data = route_data if isinstance(route_data, list) else [route_data]
    termini: set[tuple[str, str]] = set()
    for lo in route_data:
        for section in lo.RouteSections or []:
            if section.Originator and section.Destination:
                termini.add((section.Originator, section.Destination))

    for orig, dest in termini:
        data = None
        try:
            data = tube.fetch.line.timetable_by_id_from_to_stop(
                id=line_id,
                fromStopPointId=orig,
                toStopPointId=dest,
            )
        except Exception:
            pass

        if data is None or data.Timetable is None:
            try:
                data = tube.fetch.line.timetable_by_id_from_stop(
                    id=line_id,
                    fromStopPointId=orig,
                )
            except Exception as e:
                print(f"    skipping {orig}→{dest}: {e}")
                continue

        if data is None or data.Timetable is None:
            print(f"    skipping {orig}→{dest}: no timetable")
            continue

        for st in (data.Stations or []) + (data.Stops or []):
            s = _make_stop(st)
            if s.Id and s.Id not in stops:
                stops[s.Id] = s

        timetable = data.Timetable
        departure_stop_id = timetable.DepartureStopId or orig

        for route_idx, route in enumerate(timetable.Routes or []):
            interval_map: dict[str, list[tuple[str, float]]] = {
                str(si.Id): [
                    (iv.StopId, iv.TimeToArrival) for iv in (si.Intervals or [])
                ]
                for si in (route.StationIntervals or [])
            }

            for schedule in route.Schedules or []:
                svc_name = (schedule.Name or "").lower()
                service_id = next(
                    (v for k, v in SCHEDULE_MAP.items() if k in svc_name),
                    "MTF",
                )

                for journey in schedule.KnownJourneys or []:
                    hour = int(journey.Hour or 0)
                    minute = int(journey.Minute or 0)
                    interval_id = str(journey.IntervalId or 0)
                    dep_minutes = hour * 60 + minute

                    trip_id = (
                        f"{line_id}_{orig}_{dest}_{service_id}"
                        f"_{hour:02d}{minute:02d}_r{route_idx}"
                    )
                    trip_rows.append([line_id, service_id, trip_id, dest, 0])

                    dep_time = _minutes_to_gtfs_time(dep_minutes)
                    stop_time_rows.append(
                        [trip_id, dep_time, dep_time, departure_stop_id, 0]
                    )

                    for seq, (stop_id, tta) in enumerate(
                        interval_map.get(interval_id, []), start=1
                    ):
                        t = _minutes_to_gtfs_time(dep_minutes + int(round(tta)))
                        stop_time_rows.append([trip_id, t, t, stop_id, seq])


def _parquet_sources() -> dict[str, Path]:
    cfg = resolve_data()
    timetable_dir = repo_root() / cfg.get("timetable_dir", "data/timetables")
    return {
        line_id: timetable_dir / f"{line_id}_timetable.parquet"
        for line_id in cfg.get("pdf_timetables", {})
    }


def build_gtfs(output_path: Path) -> None:
    today = date.today()
    cal_start = today.strftime("%Y%m%d")
    cal_end = (today + timedelta(days=365)).strftime("%Y%m%d")

    agency_rows = [
        ["TFL", "Transport for London", "https://tfl.gov.uk", "Europe/London", "EN"]
    ]
    route_rows = []
    calendar_rows = [
        ["MTF", 1, 1, 1, 1, 1, 0, 0, cal_start, cal_end],
        ["SAT", 0, 0, 0, 0, 0, 1, 0, cal_start, cal_end],
        ["SUN", 0, 0, 0, 0, 0, 0, 1, cal_start, cal_end],
    ]
    trip_rows = []
    stop_time_rows = []
    stops: dict[str, object] = {}

    cfg = resolve_data()
    lines = []
    for mode in cfg["modes"]:
        print(f"Fetching lines for mode: {mode}")
        lines.extend(tube.fetch.line.lines_by_modes(modes=mode))

    for line in lines:
        line_id = line.Id
        print(f"  {line_id}...")
        try:
            route_type = GTFS_ROUTE_TYPE[line.ModeName]
        except KeyError:
            raise ValueError(
                f"No entry for {line.ModeName} in GTFS_ROUTE_TYPE "
                f"(options: {list(GTFS_ROUTE_TYPE)})"
            )
        route_rows.append([line_id, "TFL", line_id, line.Name, route_type])

        parquet_sources = _parquet_sources()
        if line_id in parquet_sources:
            parquet_path = parquet_sources[line_id]
            if not parquet_path.exists():
                raise FileNotFoundError(
                    f"Parquet source for {line_id} not found: {parquet_path}"
                )
            _load_from_parquet(line_id, parquet_path, trip_rows, stop_time_rows, stops)
        else:
            _load_from_api(line_id, line, trip_rows, stop_time_rows, stops)

    print("Writing GTFS zip...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        files: dict[str, tuple[list, list]] = {
            "agency.txt": (
                [
                    "agency_id",
                    "agency_name",
                    "agency_url",
                    "agency_timezone",
                    "agency_lang",
                ],
                agency_rows,
            ),
            "routes.txt": (
                [
                    "route_id",
                    "agency_id",
                    "route_short_name",
                    "route_long_name",
                    "route_type",
                ],
                route_rows,
            ),
            "calendar.txt": (
                [
                    "service_id",
                    "monday",
                    "tuesday",
                    "wednesday",
                    "thursday",
                    "friday",
                    "saturday",
                    "sunday",
                    "start_date",
                    "end_date",
                ],
                calendar_rows,
            ),
            "trips.txt": (
                ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id"],
                trip_rows,
            ),
            "stops.txt": (
                ["stop_id", "stop_name", "stop_lat", "stop_lon"],
                [[s.Id, s.Name, s.Lat, s.Lon] for s in stops.values()],
            ),
            "stop_times.txt": (
                [
                    "trip_id",
                    "arrival_time",
                    "departure_time",
                    "stop_id",
                    "stop_sequence",
                ],
                stop_time_rows,
            ),
        }
        for filename, (header, rows) in files.items():
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(header)
            w.writerows(rows)
            zf.writestr(filename, buf.getvalue())

    print(f"Done: {output_path}")
    print(
        f"  {len(stops):,} stops, {len(trip_rows):,} trips, "
        f"{len(stop_time_rows):,} stop_times"
    )
