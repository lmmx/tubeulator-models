# Default recipe
default:
    @just --list

# ── Data pipeline ────────────────────────────────────────────

# Fetch timetables from TfL API
fetch:
    uv run --group pull tm-sync-tts
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

# Train a single model (e.g. just train line)
train model="change" profile="" *args="":
    uv run --group pyg tm-train --model {{model}} {{ if profile != "" { "--profile " + profile } else { "" } }} {{args}}

# Train all models sequentially
train-all profile="full":
    #!/usr/bin/env bash
    set -euo pipefail
    just train nexthop "{{profile}}"
    just train nexthop "{{profile}}" --value-primary --batch-size 1024 --epochs 200
    done

# Train all with full profile
train-full:
    just train-all full

# ── Model Export & Upload ────────────────────────────────────

# Export model to HF-ready directory (no upload)
export variant profile="full":
    uv run --group pyg tm-export {{variant}} --profile {{profile}}

# Export and upload to Hugging Face
upload variant profile="full":
    uv run --group pyg tm-export {{variant}} --profile {{profile}} --upload

# Export both models
export-all: (export "policy") (export "value")

# Upload both models
upload-all: (upload "policy") (upload "value")

# ── Inference ────────────────────────────────────────────────

# Policy rollout
route origin dest via="":
    uv run --group pyg tm-infer policy -o "{{origin}}" -d "{{dest}}" {{ if via == "" { "" } else { "-v \"" + via + "\"" } }}

# Travel time prediction
time origin dest via="":
    uv run --group pyg tm-infer value -o "{{origin}}" -d "{{dest}}" {{ if via == "" { "" } else { "-v \"" + via + "\"" } }}

# Both: route + time
plan origin dest via="":
    uv run --group pyg tm-infer route -o "{{origin}}" -d "{{dest}}" {{ if via == "" { "" } else { "-v \"" + via + "\"" } }}

# ── Shortcuts ────────────────────────────────────────────────

# Build everything from scratch and train all models
all: data train-all

# ── Housekeeping ─────────────────────────────────────────────

# Lint
lint:
    ruff check src/

# Clean generated data
clean:
    rm -rf data/ checkpoints/

# Publish Python package
publish:
    uv build
    uv publish -u __token__ -p $(keyring get PYPIRC_TOKEN "")
