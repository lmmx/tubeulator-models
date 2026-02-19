# Default recipe
default:
    @just --list

# ── Data pipeline ────────────────────────────────────────────

# Fetch timetables from TfL API
fetch:
    uv run --group pull tm-build-gtfs

# Build GTFS → graph → PyG pipeline
graph:
    uv run --group prep --group pyg tm-gtfs2pyg

# Build route training data
routes:
    uv run --group prep tm-build-routes

# Full data pipeline
data: fetch graph routes

# Plot the network
plot:
    uv run --group prep --group plot tm-plot

# ── Training ─────────────────────────────────────────────────

train-hi profile="" *args="":
    uv run --group pyg tm-hierarchical {{ if profile != "" { "--profile " + profile } else { "" } }} {{args}}

# Train a single model (e.g. just train line)
train model="change" profile="" *args="":
    uv run --group pyg tm-train --model {{model}} {{ if profile != "" { "--profile " + profile } else { "" } }} {{args}}

# Train all three models sequentially
train-all profile="":
    #!/usr/bin/env bash
    set -euo pipefail
    for m in line change station; do
        echo "═══ Training model: $m ═══"
        just train "$m" "{{profile}}"
    done

# Train all with full profile
train-full:
    just train-all full

# ── Shortcuts ────────────────────────────────────────────────

# Build everything from scratch and train all models
all: data train-all

# Quick dev cycle: rebuild routes + train change model
dev:
    just routes
    just train change dev

# ── Housekeeping ─────────────────────────────────────────────

# Lint
lint:
    ruff check src/

# Clean generated data
clean:
    rm -rf data/ checkpoints/
