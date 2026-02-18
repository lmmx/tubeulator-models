# Tubeulator Models

Models of the TfL transit network

## Usage

First build a GTFS zip file from the API timetable data:

```bash
tm-build-gtfs
```

```
Building GTFS from TfL API → /home/louis/dev/tubeulator-models/data/tfl_station_data_gtfs.zip
  bakerloo...
  central...
  circle...
  district...
  hammersmith-city...
  jubilee...
  metropolitan...
  northern...
  piccadilly...
  victoria...
  waterloo-city...
Writing GTFS zip...
Done: /home/louis/dev/tubeulator-models/data/tfl_station_data_gtfs.zip
  270 stops, 44,212 trips, 1,349,594 stop_times
```

To plot a nice visualisation of the network, run `tm-plot`

Then convert the GTFS into PyG graph objects (node and edges parquet files too!) either step by step
or all in one with `tm-gtfs2pyg` (recommended)

```bash
# Full pipeline
uv run --group prep --group pyg tm-gtfs2pyg

# or staged — e.g. to re-run pyg conversion only
uv run --group prep    tm-gtfs2graph
uv run --group pyg     tm-graph2pyg
```
