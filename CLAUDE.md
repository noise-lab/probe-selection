# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**probe-selection** optimizes the selection of network measurement probes by identifying which subset of probes captures the most significant shared latency anomalies. The `src/` package implements the full pipeline: ingesting Netrics (FLOTO) RTT data, detecting congestion events via the Jitterbug algorithm, finding pairwise spike overlaps across devices, and running greedy/baseline probe selection algorithms.

## Setup

```bash
uv sync --dev   # create .venv and install all dependencies
```

## Running the Pipeline

The pipeline is orchestrated from `src/main.py`. Steps are commented/uncommented manually to run selectively:

```bash
uv run python -m src.main
```

Individual pipeline steps can also be invoked directly:

```python
from src.scripts import detect_jumps, compute_event_overlaps, run_sampling_algo
detect_jumps.run()               # Step 1: run Jitterbug per device
compute_event_overlaps.run()     # Step 2: compute pairwise spike overlaps
run_sampling_algo.run()          # Step 3: run probe selection experiments
```

The `detect-jumps` script is also available as a CLI:

```bash
uv run detect-jumps --output /data/changepoints.parquet
```

## Architecture

### Core Detection Pipeline (`src/apis/` + `src/jitterbug/`)

The congestion detection chain for a single device:

1. **`JitterbugAnalyzer.load_rtts()`** — preprocesses raw RTT measurements into three DataFrames: full-resolution RTTs, 15-min aggregated minimums (`mins_df`), and 15-min jitter (mdev).

2. **`JitterbugAnalyzer.analyze()`** — runs changepoint detection on `mins_df` using one of three algorithms:
   - `pelt` (default): PELT via `ruptures`
   - `cusum`: cumulative sum via `detecta`
   - `bcp`: Bayesian changepoint detection (`src/jitterbug/bcp.py`)

3. **`LatencyJumps.fit()`** (`src/jitterbug/_latency_jump.py`) — classifies each changepoint as a positive latency jump (amplitude > `latency_jump_threshold`) or not.

4. **Congestion inference** — one of three methods controlled by `inference_method`:
   - `jd` (default): jitter dispersion (IQR of RTT diffs within segment) via `src/jitterbug/signal_energy.py`
   - `ks`: KS test comparing segment distributions via `src/jitterbug/kstest.py`
   - `lj_only`: latency jump amplitude alone (skips jitter analysis)

5. **`CongestionInference.fit()`** — combines latency jump and jitter signals into spike labels: `"Congestion"`, `"Path Change"`, `"Normal"`, or `"Congestion + Path Change"`.

**`DeviceProcessor`** wraps this for Netrics data (includes ISP/zipcode cleaning).

**`CongestionDetector`** (`src/apis/congestion_detection.py`) applies sliding-window detection across a device's full deployment period, then deduplicates and merges adjacent changepoints.

### Probe Selection (`src/optimization/baselines.py` + `src/scripts/run_sampling_algo.py`)

Probe selection works in two phases:

1. **Spike ID assignment** (`assign_spike_ids`): spikes from different probes that overlap in time (IoU ≥ threshold) with similar amplitude are merged to the same `spike_id`. Impact is computed as `amplitude × duration_hours`; `log_impact` is used for the objective.

2. **Selection algorithms**:
   - `greedy_probe_selection`: iteratively picks the probe with highest marginal log-impact until a coverage fraction is met.
   - `sort_by_impact_probe_selection`: sorts probes by total unique impact, selects greedily.
   - `random_probe_selection`: Monte Carlo baseline over N trials.

`produce_reduced_dataset()` applies greedy selection per latency target and returns the union of selected probes.

### Evaluation (`src/evaluation/`)

`dataset_builders.py` provides utilities for building labeled train/test datasets from detected spikes: `label_anomalous_rows`, `build_augmented_timeseries` (rebalances anomaly fraction), and `build_anomaly_only_timeseries`.


## Data Sources

- **Netrics (FLOTO)**: Residential measurement devices in Chicago. Provided as a pre-built `netrics.db` DuckDB file placed at the project root (see README for download link). The path is resolved via `src/config.py:DB_PATH`. Contains per-target latency tables (`google`, `atlanta`, `chicago`, etc.) with columns `time`, `device_id`, `latency_min/avg/mdev/max`, `isp_whois`, `zipcode`, etc.

Processed intermediate files are cached as Parquet under `/data/device_placement/`.
