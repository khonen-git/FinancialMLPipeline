# MLflow Integration Guide

This document explains how to use MLflow for experiment tracking, monitoring, and artifact management in the trading ML pipeline.

MLflow is used to:
- track experiment parameters and metrics
- log artifacts (reports, plots, trade logs, equity curves)
- compare runs across different configurations
- reproduce past experiments
- monitor model performance over time

---

## 1. Overview

The pipeline uses MLflow to track:
- **Parameters**: Configuration values (TP/SL, bar types, model hyperparameters)
- **Metrics**: Performance metrics (accuracy, precision, Sharpe ratio, drawdown, PnL)
- **Artifacts**: Files (configs, reports, plots, trade logs, equity curves)
- **Tags**: Metadata (git commit, experiment type, asset symbol)

All experiments are logged automatically when you run the pipeline.

---

## 2. Configuration

### 2.1 Local Tracking (Default)

The default configuration uses local file-based tracking:

**File**: `configs/mlflow/local.yaml`

```yaml
mlflow:
  enabled: true
  tracking_uri: "file:./mlruns"
  experiment_name: "financial_ml"
  autolog: false
  
  log_artifacts:
    config_dump: true
    equity_curves: true
    trade_logs: true
    reports: true
    models: true
  
  tags:
    project: "financial-ml-pipeline"
```

**Key settings**:
- `tracking_uri`: Path to MLflow tracking store (relative to project root)
- `experiment_name`: Default experiment name (can be overridden per run)
- `enabled`: Enable/disable MLflow logging

### 2.2 Remote Tracking (Optional)

For team collaboration or cloud deployment, you can use a remote tracking server:

**File**: `configs/mlflow/remote.yaml`

```yaml
mlflow:
  enabled: true
  tracking_uri: "http://your-mlflow-server:5000"
  experiment_name: "financial_ml"
  autolog: false
```

Or use MLflow's managed tracking (Databricks, etc.):

```yaml
mlflow:
  tracking_uri: "databricks://your-workspace"
```

---

## 3. Starting the MLflow UI

### 3.1 Local Tracking

To view experiments locally, start the MLflow UI:

```bash
# From project root directory
mlflow ui --backend-store-uri file:./mlruns
```

**Important**: 
- Run this command from the **project root** (where `mlruns/` directory is located)
- The `--backend-store-uri` must match the `tracking_uri` in your config
- If `mlruns/` doesn't exist yet, it will be created on first experiment run

### 3.2 Accessing the UI

Once started, open your browser to:

```
http://127.0.0.1:5000
```

The UI shows:
- **Experiments**: List of all experiments
- **Runs**: Individual experiment runs
- **Metrics**: Performance metrics over time
- **Parameters**: Configuration values
- **Artifacts**: Files logged with each run

### 3.3 Troubleshooting

**Problem**: `mlflow ui` command not found
- **Solution**: Install MLflow: `pip install mlflow`

**Problem**: UI shows "No experiments found"
- **Solution**: 
  - Check that you've run at least one experiment
  - Verify `mlruns/` directory exists and contains data
  - Ensure `--backend-store-uri` matches your config

**Problem**: UI shows empty runs
- **Solution**: 
  - Check that MLflow is enabled in config (`mlflow.enabled: true`)
  - Verify the experiment ran successfully (check logs)
  - Check that `tracking_uri` in config matches the UI command

**Problem**: Permission errors
- **Solution**: Ensure you have read/write permissions to `mlruns/` directory

---

## 4. What Gets Logged

### 4.1 Parameters

The pipeline logs key configuration parameters:

- `asset`: Asset symbol (e.g., "EURUSD")
- `bars_type`: Bar construction type
- `tp_ticks`: Take profit in ticks
- `sl_ticks`: Stop loss in ticks
- `n_labels`: Number of labels created
- `n_features`: Number of features
- `cv_type`: Cross-validation type
- `model_type`: Model used (e.g., "random_forest")

### 4.2 Metrics

Performance metrics logged per fold and aggregated:

**Primary Model Metrics**:
- `fold_{n}_primary_accuracy`: Accuracy per fold
- `fold_{n}_primary_precision`: Precision per fold
- `fold_{n}_primary_recall`: Recall per fold
- `fold_{n}_primary_f1`: F1 score per fold
- `mean_primary_cv_accuracy`: Mean accuracy across folds

**Meta Model Metrics**:
- `fold_{n}_meta_accuracy`: Meta model accuracy per fold
- `fold_{n}_meta_precision`: Meta model precision per fold
- `fold_{n}_meta_recall`: Meta model recall per fold
- `fold_{n}_meta_f1`: Meta model F1 score per fold
- `mean_meta_cv_accuracy`: Mean meta model accuracy

**Backtest Metrics**:
- `backtest_total_trades`: Total number of trades
- `backtest_win_rate`: Win rate (0-1)
- `backtest_sharpe_ratio`: Sharpe ratio
- `backtest_max_drawdown`: Maximum drawdown (0-1)
- `backtest_total_pnl`: Total profit/loss
- `backtest_total_return`: Total return (fraction)

**Risk Metrics**:
- `prob_ruin`: Probability of ruin (Monte Carlo)
- `prob_profit_target`: Probability of reaching profit target

### 4.3 Artifacts

Files saved with each run:

**Configuration**:
- `config/config.yaml`: Full Hydra configuration dump

**Reports**:
- `reports/{experiment_name}_report.html`: HTML experiment report

**Backtest Results**:
- `backtest_trade_log.csv`: Detailed trade log
- `backtest_equity_curve.csv`: Equity curve over time

**Plots** (if generated):
- `plots/equity.png`: Equity curve plot
- `plots/drawdown.png`: Drawdown curve plot
- `plots/trade_histogram.png`: Trade distribution

---

## 5. Viewing Experiments

### 5.1 Experiment List

In the MLflow UI, the **Experiments** page shows:
- Experiment names
- Number of runs per experiment
- Last modified time

### 5.2 Run Details

Click on a run to see:
- **Overview**: Parameters, metrics, tags
- **Artifacts**: All logged files
- **Metrics**: Time series of metrics (if logged over time)

### 5.3 Comparing Runs

To compare multiple runs:
1. Select runs using checkboxes
2. Click "Compare"
3. View side-by-side comparison of:
   - Parameters (highlighting differences)
   - Metrics (with min/max/mean)
   - Artifacts

---

## 6. Searching and Filtering

### 6.1 Search Runs

Use the search bar to filter runs by:
- Parameter values: `params.tp_ticks = "10"`
- Metrics: `metrics.backtest_sharpe_ratio > 1.0`
- Tags: `tags.asset = "EURUSD"`

**Examples**:
```
params.tp_ticks = "10" AND metrics.backtest_win_rate > 0.5
tags.asset = "EURUSD" AND metrics.backtest_sharpe_ratio > 1.0
```

### 6.2 Sorting

Sort runs by:
- Any metric (ascending/descending)
- Start time
- Duration

---

## 7. Downloading Artifacts

### 7.1 From UI

1. Navigate to a run
2. Go to **Artifacts** tab
3. Click on files to download

### 7.2 From Command Line

```bash
# Download all artifacts from a run
mlflow artifacts download -u file:./mlruns -r <run_id>

# Download specific artifact
mlflow artifacts download -u file:./mlruns -r <run_id> -a "reports/experiment_report.html"
```

---

## 8. Reproducing Experiments

### 8.1 From MLflow UI

1. Find the run you want to reproduce
2. Download `config/config.yaml` artifact
3. Note the `git_commit` tag
4. Checkout the git commit:
   ```bash
   git checkout <git_commit_hash>
   ```
5. Use the downloaded config or recreate the same parameters

### 8.2 From Run ID

```bash
# Get run details
mlflow runs describe -u file:./mlruns -r <run_id>

# Get parameters
mlflow runs get-param -u file:./mlruns -r <run_id> <param_name>
```

---

## 9. Best Practices

### 9.1 Experiment Naming

Use descriptive experiment names:
- Include asset: `eurusd_tick1000_rf`
- Include key parameters: `eurusd_tp10_sl10_cpcv`
- Include date range: `eurusd_2023_2024`

### 9.2 Tagging

Add meaningful tags:
- `asset`: Asset symbol
- `model_type`: Model used
- `cv_type`: Cross-validation method
- `git_commit`: Git commit hash (auto-logged)

### 9.3 Metric Logging

- Log metrics at consistent intervals (per fold, per segment)
- Use descriptive metric names
- Include both raw and normalized metrics

### 9.4 Artifact Management

- Keep artifacts small (compress large files)
- Use consistent naming conventions
- Log only essential artifacts to avoid clutter

---

## 10. Advanced Usage

### 10.1 Custom Metrics

You can log custom metrics in the pipeline:

```python
import mlflow

mlflow.log_metric('custom_metric', value)
mlflow.log_metrics({'metric1': 1.0, 'metric2': 2.0})
```

### 10.2 Custom Artifacts

Log custom files:

```python
mlflow.log_artifact('path/to/file.csv')
mlflow.log_artifacts('path/to/directory/')
```

### 10.3 Tags

Add custom tags:

```python
mlflow.set_tag('custom_tag', 'value')
mlflow.set_tags({'tag1': 'value1', 'tag2': 'value2'})
```

### 10.4 Model Registry (Future)

For production deployments, use MLflow Model Registry:
- Register models with versioning
- Stage models (Staging, Production)
- Track model lineage

---

## 11. Troubleshooting

### 11.1 Common Issues

**Issue**: MLflow UI shows "No experiments"
- **Check**: Have you run at least one experiment?
- **Check**: Is `mlflow.enabled: true` in config?
- **Check**: Does `mlruns/` directory exist?

**Issue**: Metrics not appearing
- **Check**: Experiment ran successfully (check logs)
- **Check**: MLflow logging code executed (no exceptions)
- **Check**: Metric names are valid (no special characters)

**Issue**: Artifacts not saving
- **Check**: File paths are correct
- **Check**: Permissions to write to `mlruns/`
- **Check**: Files exist before logging

**Issue**: UI not starting
- **Check**: MLflow is installed: `pip install mlflow`
- **Check**: Port 5000 is available
- **Check**: You're in the correct directory

### 11.2 Debugging

Enable verbose logging:

```bash
mlflow ui --backend-store-uri file:./mlruns --verbose
```

Check MLflow tracking store:

```bash
# List experiments
mlflow experiments list -u file:./mlruns

# List runs in experiment
mlflow runs list -u file:./mlruns -e <experiment_name>
```

---

## 12. Integration with Pipeline

The pipeline automatically:
1. Sets tracking URI from config
2. Creates/uses experiment (from `experiment.name` in config)
3. Starts a run for each experiment execution
4. Logs parameters, metrics, and artifacts
5. Ends the run when pipeline completes

**No manual MLflow calls needed** - everything is handled by the pipeline.

---

## 13. Quick Reference

### Start UI
```bash
mlflow ui --backend-store-uri file:./mlruns
```

### Access UI
```
http://127.0.0.1:5000
```

### List Experiments
```bash
mlflow experiments list -u file:./mlruns
```

### Get Run Info
```bash
mlflow runs describe -u file:./mlruns -r <run_id>
```

### Download Artifacts
```bash
mlflow artifacts download -u file:./mlruns -r <run_id>
```

---

## 14. AI Usage Guidelines

When modifying MLflow integration:
- Always use the config-based tracking URI (don't hardcode)
- Log consistent metrics across all runs
- Use descriptive artifact names
- Follow the artifact structure documented here
- Don't change the default `mlruns/` location without updating docs

**This file is the source of truth for MLflow usage in the project.**

