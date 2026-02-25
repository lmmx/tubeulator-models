---
library_name: pytorch
tags:
  - graph-neural-network
  - routing
  - london-underground
  - gatv2
license: mit
---

# Tube Next-Hop Policy

A GATv2 encoder + MLP decoder trained to route on the London Underground graph.

Given a current station and a destination, the model outputs a distribution over
adjacent stations.  Greedy rollout achieves **1.00 Dijkstra ratio** (optimal
shortest travel time) with **100% success rate** across all origin–destination
pairs.

## Architecture

- **Encoder:** 15-layer GATv2, d=512, 8 heads
- **Decoder:** 3-layer MLP with adjacency masking (272 stations)
- **Training signal:** KL divergence against Q-soft targets derived from
  Floyd–Warshall shortest paths
- **Parameters:** 9,457,425

## Usage
```python
import json, torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

repo = "permutans/tube-nexthop-policy"
config = json.loads(open(hf_hub_download(repo, "config.json")).read())
weights = load_file(hf_hub_download(repo, "model.safetensors"))
metadata = json.loads(open(hf_hub_download(repo, "metadata.json")).read())
# See the project repository for full inference code.
```

## Training

Part of the [Model Trains](https://github.com/lmmx/tubeulator-models) series.
Trained on GTFS timetable data from Transport for London.
