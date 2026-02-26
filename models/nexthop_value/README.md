---
library_name: pytorch
license: mit
tags:
  - graph-neural-network
  - gatv2
  - distance-estimation
  - shortest-path
  - london-underground
  - transport
  - pytorch-geometric
metrics:
  - mae
model-index:
  - name: tube-distance-field
    results:
      - task:
          type: graph-distance-estimation
          name: Travel Time Prediction
        dataset:
          type: custom
          name: TfL GTFS Timetable (London Underground)
        metrics:
          - type: mae
            name: Mean Absolute Error (minutes)
            value: 0.49
---

# Tube Distance Field

A GATv2 encoder + MLP value head trained to predict shortest travel time between
any pair of London Underground stations.

Given an origin and destination, the model outputs estimated travel time in
minutes.  **MAE: 0.49 min** — 90% of predictions within 30 seconds of the
Floyd–Warshall ground truth.

## Intended Use

This model is a research artifact demonstrating learned distance estimation on
a transit graph. It predicts the shortest travel time in minutes between any
two London Underground stations using the graph topology and scheduled edge
weights from TfL's GTFS timetable feed.

## Architecture

| Component | Details |
|---|---|
| Encoder | 16-layer GATv2, d=512, 8 heads |
| Value head | 4-layer MLP with LayerNorm |
| Graph | 272 stations |
| Parameters | 11,174,673 |
| Training signal | Huber loss (δ=2 min) vs. Floyd–Warshall all-pairs shortest times |

## Evaluation Results

| Metric | Value |
|---|---|
| Mean Absolute Error | 0.49 min |

## Limitations

While the value head is accurate as a distance oracle, Bellman rollout using
these predictions achieves only 43% routing success due to error compounding
across hops. For routing, use the companion policy model.

## Usage
```python
import json, torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

repo = "permutans/tube-distance-field"
config = json.loads(open(hf_hub_download(repo, "config.json")).read())
weights = load_file(hf_hub_download(repo, "model.safetensors"))
metadata = json.loads(open(hf_hub_download(repo, "metadata.json")).read())
# See the project repository for full inference code.
```

## Links

- **Code:** [tubeulator-models](https://github.com/lmmx/tubeulator-models)
- **Companion model:** Next-hop policy for optimal routing

Part of the [Model Trains](https://github.com/lmmx/tubeulator-models) series.
Trained on GTFS timetable data from Transport for London.
