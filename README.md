# synthsensor

[![R-CMD-check](https://github.com/AstridMarie2/synthsensor/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/AstridMarie2/synthsensor/actions/workflows/R-CMD-check.yaml)

**This repo is modified from the original synthsensor package by Astrid Marie Skålvik (@AstridMarie2) to generate anomalies for multivariate time series.**

Original repository: [synthsensor GitHub](https://github.com/AstridMarie2/synthsensor.git)

Demo website: [synthsensor Interactive App](https://sensordiagnostics.shinyapps.io/app_synth/)

The modified code lives in `Python/`. The original R implementation is in `R/`.

---

## Overview

**synthsensor** generates **labeled, multi-sensor synthetic time series** for benchmarking anomaly detection and uncertainty methods. It supports configurable background signals (AR(1), random walk, Poisson moving average, sine wave), along with **spikes** (correlated/uncorrelated), **drifts** (correlated/uncorrelated), and outputs per-timestep **anomaly flags**.

---

## Python — Synthetic Data Generation

All generation logic is in the `Python/` folder. The entry point is `Python/main.py`, which uses [Hydra](https://hydra.cc/) for configuration management.

### 1. Install dependencies

From the `Python/` directory:

```bash
pip install hydra-core omegaconf numpy pandas matplotlib scipy
```

### 2. Create your config file

Copy the example config to create your own:

```bash
cp Python/conf/config.yaml.example Python/conf/config.yaml
```

Edit `Python/conf/config.yaml` to adjust generation settings. Key options:

| Key | Description |
|-----|-------------|
| `generation.used_settings` | Which settings file to load from `generation/config/` (e.g. `settings_five`) |
| `generation.num_iterations` | Number of generation iterations |
| `generation.plot_data` | Whether to save plots alongside the data |
| `generation.mix_anomalies` | Whether to mix anomaly types |
| `generation.enrich` | Whether to run class-imbalance enrichment instead of standard generation |
| `random.seed` | Random seed for reproducibility |

### 3. Run generation

From the `Python/` directory:

```bash
cd Python
python main.py
```

Hydra will read `conf/config.yaml` automatically. To override a setting without editing the file:

```bash
python main.py generation.used_settings=settings_six generation.plot_data=false
```

### Output

Generated data is saved under `Python/generation/`:

```
generation/
  data/<used_settings>/       # CSV files per batch + training set
  zip/<used_settings>/        # Compressed versions
  history/<used_settings>/    # Generation history logs
  config/                     # Per-settings YAML configs
```

Each output CSV contains sensor columns (`Sensor0`, `Sensor1`, ...) and anomaly flag columns (`AnomalyFlag0`, `AnomalyFlag1`, ...) with labels: `Normal`, `Spike`, `SpikeCorr`, `Drift`, `DriftCorr`, `Both`.

Hydra run logs are saved to `Python/hydra_outputs/`.

---