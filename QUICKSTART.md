# Quick Start Guide

## Installation

### 1. Create conda environment

```bash
conda env create -f environment.yaml
conda activate financial-ml
```

### 2. Install package

```bash
pip install -e .
```

## Basic Usage

### 1. Prepare data

Place your Dukascopy tick data in `data/raw/`:

```
data/raw/
  └── EURUSD.parquet
```

Or use the download script (placeholder):

```bash
python scripts/download_dukascopy.py --symbol EURUSD --start 2023-01-01 --end 2023-12-31
```

### 2. Validate configuration

```bash
python scripts/validate_config.py experiment=eurusd_scalping
```

### 3. Run experiment

```bash
python run_experiment.py experiment=eurusd_scalping
```

### 4. Monitor with MLflow

```bash
mlflow ui
```

Open http://localhost:5000 to view experiments.

## Configuration Override Examples

### Change asset

```bash
python run_experiment.py experiment=eurusd_scalping assets.symbol=GBPUSD
```

### Change bar type

```bash
python run_experiment.py experiment=eurusd_scalping data.bars.type=volume data.bars.threshold=5000
```

### Adjust labeling parameters

```bash
python run_experiment.py experiment=eurusd_scalping \
    labeling.triple_barrier.tp_ticks=150 \
    labeling.triple_barrier.sl_ticks=75
```

### Change session times

```bash
python run_experiment.py experiment=eurusd_scalping \
    session.session_end="20:00" \
    session.friday_end="18:00"
```

## Docker Usage

### Build image

```bash
docker build -t financial-ml-pipeline .
```

### Run experiment

```bash
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/outputs:/app/outputs \
           -v $(pwd)/mlruns:/app/mlruns \
           financial-ml-pipeline \
           python run_experiment.py experiment=eurusd_scalping
```

## Useful Scripts

### Inspect data

```bash
python scripts/inspect_data.py data/raw/EURUSD.parquet
```

### View configuration

```bash
python scripts/validate_config.py experiment=eurusd_scalping
```

## Project Structure

```
FinancialMLPipeline/
├── src/                    # Source code
│   ├── data/              # Data processing
│   ├── features/          # Feature engineering
│   ├── labeling/          # Labeling (triple barrier, session calendar)
│   ├── models/            # ML models (HMM, RF)
│   ├── validation/        # Time-series CV
│   ├── backtest/          # Backtrader strategy
│   ├── risk/              # Monte Carlo risk analysis
│   ├── reporting/         # HTML report generation
│   └── pipeline/          # Main orchestration
├── configs/               # Hydra configuration
├── scripts/               # Utility scripts
├── templates/             # Jinja2 report templates
├── run_experiment.py      # Main entry point
└── docs/                  # Documentation
```

## Next Steps

1. Read the [documentation](docs/INDEX.md)
2. Review the [architecture](docs/ARCHITECTURE.md)
3. Check [coding standards](docs/CODING_STANDARDS.md)
4. Explore the [configuration reference](docs/CONFIG_REFERENCE.md)

## Troubleshooting

### Import errors

Make sure you've installed the package:

```bash
pip install -e .
```

### MLflow not found

Install MLflow:

```bash
pip install mlflow
```

### Data not found

Check that your data is in `data/raw/` with the correct filename format (e.g., `EURUSD.parquet`).

### Session times

Ensure session times are in 24-hour format (HH:MM) and in UTC.

