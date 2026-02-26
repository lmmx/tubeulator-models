"""Upload interchange times dataset to HuggingFace."""

import json
from huggingface_hub import HfApi, metadata_update

REPO_ID = "permutans/tube-interchange-times"

with open("interchange_times_enriched.json") as f:
    data = json.load(f)

# Write the dataset card
README = """\
---
license: cc-by-4.0
language:
- en
tags:
- transport
- london
- tfl
- tube
- underground
- interchange
- walking-times
- open-data
pretty_name: London Tube Interchange Times
size_categories:
- n<1K
---

# London Tube Interchange Times

Structured interchange and gate-to-platform time estimates for 355 London Underground, DLR, and London Overground stations.

## Description

This dataset contains two types of time estimates for each station:

- **Gate-to-platform times** (column B in the source): the maximum walking time from ticket gates to platform, per line. These are pure walking times with no waiting component.
- **Interchange times** (column C in the source): the maximum time to transfer between two lines at a station. These are total interchange penalties including both walking and waiting for the next service, not pure walking time. This distinction is important for routing applications.

The data also includes branch interchange times (changing between branches of the same line) and a small number of cross-station interchange references.

## Source

Derived from Transport for London's "LU LO DLR interchange values" spreadsheet, obtained via Freedom of Information request ([FOI-0986-1516, September 2015](https://www.whatdotheyknow.com/request/interchange_time_at_london_under)). TfL noted at the time of release that the document had not been updated and that changes were likely minimal as there had been few station adjustments.

Station and line identifiers were matched against TfL's station topology data using the [tubeulator](https://github.com/lmmx/tubeulator) library.

> Contains public sector information licensed under the Open Government Licence v3.0.

## Limitations

- The source data is from 2015 and has not been updated by TfL. Stations that have been significantly reconfigured since then (e.g. Bank, Battersea Power Station, Nine Elms) may have inaccurate or missing values.
- Interchange times include waiting time, which varies by time of day and service frequency. Using these values directly as transfer costs in a routing model will overestimate if the model already accounts for headway-based waiting.
- One interchange (West Ham, Hammersmith & City ↔ Jubilee) has no time value in the source data.
- Four stations have notes instead of times: Blackfriars (closed at time of data collection), Vauxhall and Victoria (National Rail operators listed without times).
- The Elizabeth line did not exist when this data was collected and is not represented in interchange values, though it appears in inferred gate-to-platform entries via current station topology.

## Schema

Each record represents a station with the following structure:

### Top level

| Field | Type | Description |
|---|---|---|
| `station` | string | Station name as it appeared in the TfL spreadsheet |
| `station_name_tb` | string | Normalised station name matching TfL topology data |
| `station_unique_id` | string | TfL StationUniqueId (e.g. `HUBWEH`, `940GZZLUMED`) |
| `gate_to_platform` | array | Gate-to-platform walking times per line |
| `interchanges` | array | Line-to-line interchange times |

### Gate-to-platform entries

| Field | Type | Description |
|---|---|---|
| `line` | string | Line name (human-readable) |
| `line_slug` | string or array | Line identifier slug(s) matching TfL API conventions |
| `minutes` | number | Maximum walking time from gate to platform |
| `inferred` | boolean | `true` if the line was not named in the source and was inferred from station topology |
| `note` | string | Present instead of `minutes` when the source contained a note rather than a time |

### Interchange entries

Standard interchange:

| Field | Type | Description |
|---|---|---|
| `from_line` | string | Origin line name |
| `to_line` | string | Destination line name |
| `from_line_slug` | string | Origin line slug |
| `to_line_slug` | string | Destination line slug |
| `minutes` | number | Maximum interchange time (walking + waiting) |

Branch interchange:

| Field | Type | Description |
|---|---|---|
| `branch_interchange` | boolean | `true` |
| `line` | string | Line name |
| `line_slug` | string | Line slug |
| `minutes` | number | Time to change between branches |

Cross-station interchange:

| Field | Type | Description |
|---|---|---|
| `cross_station` | string | Name of the other station |
| `minutes` | number | Interchange time |

## Example
```json
{
  "station": "Mile End",
  "station_name_tb": "Mile End",
  "station_unique_id": "940GZZLUMED",
  "gate_to_platform": [
    {"line": "Central", "line_slug": "central", "minutes": 1.5, "inferred": true},
    {"line": "District", "line_slug": "district", "minutes": 1.5, "inferred": true},
    {"line": "Hammersmith & City", "line_slug": "hammersmith-city", "minutes": 1.5, "inferred": true}
  ],
  "interchanges": [
    {"from_line": "Central", "to_line": "District", "minutes": 3, "from_line_slug": "central", "to_line_slug": "district"},
    {"from_line": "Central", "to_line": "Hammersmith & City", "minutes": 3, "from_line_slug": "central", "to_line_slug": "hammersmith-city"},
    {"from_line": "District", "to_line": "Hammersmith & City", "minutes": 3, "from_line_slug": "district", "to_line_slug": "hammersmith-city"}
  ]
}
```

## Citation

If you use this dataset, please attribute both this processed version and the original TfL source:
```bibtex
@dataset{tube_interchange_times_2025,
  title={London Tube Interchange Times},
  author={permutans},
  year={2025},
  url={https://huggingface.co/datasets/permutans/tube-interchange-times},
  note={Derived from TfL FOI-0986-1516. Contains public sector information licensed under the Open Government Licence v3.0.}
}
```
"""

api = HfApi()

# Create the repo
api.create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    exist_ok=True,
)

# Upload README
api.upload_file(
    path_or_fileobj=README.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)

# Upload the dataset
api.upload_file(
    path_or_fileobj="interchange_times_enriched.json",
    path_in_repo="data/interchange_times.json",
    repo_id=REPO_ID,
    repo_type="dataset",
)

print(f"Published to https://huggingface.co/datasets/{REPO_ID}")