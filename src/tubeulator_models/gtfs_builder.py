"""Build a GTFS zip from TfL open data via the tubeulator library."""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import date, timedelta
from pathlib import Path

import tubeulator as tube


SCHEDULE_MAP = {
    "monday - friday": "MTF",
    "monday to friday": "MTF",
    "saturday": "SAT",
    "sunday": "SUN",
}


def _minutes_to_gtfs_time(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}:00"


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
    stops: dict[str, object] = {}  # stop_id → MatchedStop

    lines = tube.fetch.line.lines_by_modes(modes="tube")

    for line in lines:
        line_id = line.Id
        print(f"  {line_id}...")
        route_rows.append([line_id, "TFL", line_id, line.Name, 1])

        try:
            route_data = tube.fetch.line.route_by_ids(ids=line_id)
        except Exception as e:
            print(f"  skipping {line_id} (route fetch failed: {e})")
            continue

        # route_by_ids returns a list of Line models
        route_data = route_data if isinstance(route_data, list) else [route_data]
        termini: set[tuple[str, str]] = set()
        for line_obj in route_data:
            for section in line_obj.RouteSections or []:
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
                if st.Id and st.Id not in stops:
                    stops[st.Id] = st

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
        f"  {len(stops):,} stops, {len(trip_rows):,} trips, {len(stop_time_rows):,} stop_times"
    )
