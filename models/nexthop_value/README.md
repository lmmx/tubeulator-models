---
library_name: pytorch
tags:
  - graph-neural-network
  - distance-estimation
  - london-underground
  - gatv2
license: mit
---

# Tube Distance Field

A GATv2 encoder + MLP value head trained to predict shortest travel time between
any pair of London Underground stations.

Given an origin and destination, the model outputs estimated travel time in
minutes.  **MAE: 0.25 min** — 90% of predictions within 30 seconds of the
Floyd–Warshall ground truth.

## Architecture

- **Encoder:** 15-layer GATv2, d=512, 8 heads
- **Value head:** 4-layer MLP with LayerNorm
- **Training signal:** Huber loss (δ=2 min) against Floyd–Warshall all-pairs
  shortest times
- **Parameters:** 10,641,169

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

## Limitations

While the value head is accurate as a distance oracle, Bellman rollout using
these predictions achieves only 43% routing success due to error compounding
across hops.  For routing, use the companion policy model.

## Training

Part of the [Model Trains](https://github.com/lmmx/tubeulator-models) series.
Trained on GTFS timetable data from Transport for London.
