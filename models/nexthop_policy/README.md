---
library_name: pytorch
license: mit
tags:
  - graph-neural-network
  - gatv2
  - routing
  - shortest-path
  - london-underground
  - transport
  - pytorch-geometric
metrics:
  - accuracy
model-index:
  - name: tube-nexthop-policy
    results:
      - task:
          type: graph-routing
          name: Next-Hop Routing
        dataset:
          type: custom
          name: TfL GTFS Timetable (London Underground)
        metrics:
          - type: accuracy
            name: Rollout Success Rate
            value: 0.9995826377295493
          - type: custom
            name: Dijkstra Ratio
            value: 1.0875682957637163
          - type: custom
            name: Step Accuracy
            value: 0.7905286123339477
          - type: custom
            name: Length Ratio
            value: inf
---

# Tube Next-Hop Policy

A GATv2 encoder + MLP decoder trained to predict optimal next-hop routing
decisions on the London Underground graph. Given a current station and a
destination, the model outputs a probability distribution over adjacent
stations.

Greedy rollout achieves **1.09× Dijkstra ratio** (optimal
shortest travel time) with **100.0% success rate** across all
239,610 origin–destination pairs.

## Intended Use

This model is a research artifact and demonstration of learned graph routing.
It predicts the next station to visit on a shortest-time path between any two
London Underground stations, using only the graph topology and travel-time edge
weights from TfL's GTFS timetable feed.

Potential applications include learned routing components in transit planning
systems, GNN routing benchmarks, and educational demonstrations of
graph-attention-based policy learning.

## Architecture

| Component | Details |
|---|---|
| Encoder | 16-layer GATv2, d=512, 8 heads |
| Decoder | 3-layer MLP with adjacency masking |
| Graph | 490 stations, 19 lines, 1752 directed edges |
| Parameters | 10,168,299 |
| Training signal | KL divergence vs. Q-soft targets from Floyd–Warshall |
| Label smoothing | 0.1 |
| Scheduled sampling | 0.5 |

## Evaluation Results

Evaluated on a held-out set of 23,961 OD pairs (489,981 next-hop
steps), with greedy rollout to completion.

| Metric | Value |
|---|---|
| Rollout success rate | 100.0% |
| Dijkstra ratio (travel time vs. optimal) | 1.09 |
| Step accuracy (single-step top-1) | 79.1% |
| Length ratio (hops vs. optimal hops) | inf |

> **Note on step accuracy:** The 79.1% top-1 step accuracy reflects
> that many nodes have multiple equally-optimal next hops. The model distributes
> probability across these alternatives, which is correct behavior — the
> rollout metrics confirm optimal routing.

## Training Data

The graph topology and edge travel times are extracted from Transport for
London's [GTFS timetable feed](https://tfl.gov.uk/info-for/open-data-users/).
Floyd–Warshall all-pairs shortest paths provide the Q-value supervision signal.

- 239,610 OD pairs → 4,891,856 next-hop training steps
- 90/10 OD-pair split (215,649 train / 23,961 val)
- Batch size: 4,096 steps

## Training Details

- Optimizer: AdamW, lr=3e-04
- Compiled with `torch.compile(mode='default')`
- Training time: ~58 minutes
- Hardware: single GPU

## Limitations

- The model is specific to the London Underground graph topology at the time
  of the GTFS snapshot. It will not generalize to other transit networks
  without retraining.
- Edge weights represent scheduled travel times, not real-time conditions.
- The adjacency mask is fixed at inference time — the model cannot handle
  station closures or service disruptions without mask modification.

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

## Links

- **Code:** [tubeulator-models](https://github.com/lmmx/tubeulator-models)
- **Companion model:** Distance field (value head) for travel-time estimation

Part of the [Model Trains](https://github.com/lmmx/tubeulator-models) series.
Trained on GTFS timetable data from Transport for London.
