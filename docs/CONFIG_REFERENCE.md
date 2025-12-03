# Configuration Reference

This document describes the configuration structure used by the trading ML pipeline, based on Hydra.

It is meant for:

- **Developers** (you + AI assistants) to know which config keys exist and what they mean.
- **Experiment users** to know where to change instruments, capital, risk, and experiment settings.

> **Important:**  
> The AI must not silently invent new configuration keys or rename existing ones.  
> Any new key must be documented here and added in a dedicated config file.

---

## 1. Hydra Layout Overview

All configs live under the `configs/` directory.

Recommended structure:

```text
configs/
  config.yaml          # main entrypoint (defaults and overrides)
  experiment/
    base.yaml
    walkforward.yaml
  data/
    dukascopy.yaml
    bars.yaml
  assets/
    EURUSD.yaml
    USDJPY.yaml
  session/
    default_fx.yaml
  labeling/
    triple_barrier.yaml
  features/
    base_features.yaml
  models/
    rf_cpu.yaml
    rf_gpu.yaml
    meta.yaml
    hmm.yaml
  risk/
    basic.yaml
    monte_carlo.yaml
  backtest/
    base_bt.yaml
  reporting/
    report.yaml
  mlflow/
    local.yaml
    remote.yaml
  runtime/
    hardware.yaml
```

`config.yaml` typically composes these pieces, e.g.:

```yaml
defaults:
  - experiment: base
  - data: dukascopy
  - data/bars: bars
  - assets: EURUSD
  - session: default_fx
  - labeling: triple_barrier
  - features: base_features
  - models: rf_cpu
  - models/hmm: hmm
  - risk: basic
  - risk/monte_carlo: monte_carlo
  - backtest: base_bt
  - reporting: report
  - mlflow: local
  - runtime: hardware
```

---

## 2. Experiment Configuration

**File**: `configs/experiment/base.yaml` (or similar)

Controls the high-level experiment metadata.

```yaml
experiment:
  name: "eurusd_tick1000_rf"
  description: "Baseline RF on EURUSD with 1000-tick bars"
  seed: 42
  mode: "train"       # train | backtest-only | analysis
  tags:
    - "eurusd"
    - "rf"
    - "tick1000"
```

- `name`: human-readable experiment name (also used in MLflow).
- `description`: short description of what is being tested.
- `seed`: random seed for reproducibility.
- `mode`: can control behavior (e.g. only backtest, or full train+backtest).
- `tags`: optional list of strings to tag MLflow runs.

---

## 3. Asset Configuration

**File**: `configs/assets/<SYMBOL>.yaml`

Each asset has its own config file. Example for EURUSD:

```yaml
asset:
  symbol: "EURUSD"
  tick_size: 0.00001        # minimum price increment
  contract_size: 100000     # nominal size of 1 lot
  quote_currency: "USD"
  category: "fx_major"
```

For USDJPY:

```yaml
asset:
  symbol: "USDJPY"
  tick_size: 0.001
  contract_size: 100000
  quote_currency: "JPY"
  category: "fx_major"
```

**Notes**:

- `tick_size` is the smallest price movement for that symbol (no `pip_size` key here).
- `contract_size` can be used to compute notional exposure if needed.
- Do not hardcode tick size in the code; always read from asset config.

---

## 4. Capital & Risk Model

**File**: could be in `configs/risk/basic.yaml` or a dedicated `capital.yaml`.

Example:

```yaml
capital:
  starting_capital: 10000.0
  currency: "USD"
  compounding: true          # if true, position size is recalculated as capital changes

risk_model:
  type: "fixed_fractional"   # fixed_fractional | fixed_amount
  risk_per_trade: 0.01       # 1% per trade if fixed_fractional
  fixed_amount: 100.0        # used if type == fixed_amount
  max_daily_loss: 0.03       # 3% of starting_capital or equity (define policy)
  max_total_drawdown: 0.2    # 20% overall drawdown
  max_open_positions: 1
```

- `compounding`:
  - `true` → size computed as fraction of current equity.
  - `false` → computed from starting_capital (no compounding).
- `risk_per_trade`:
  - For `fixed_fractional`, fraction of equity (or starting capital) risked per trade.
- `max_daily_loss`, `max_total_drawdown`:
  - can be used for risk checks / stopping criteria in backtests or risk analysis.

---

## 5. Data Configuration

**File**: `configs/data/dukascopy.yaml`

Controls raw data source and storage layout.

```yaml
data:
  source: "dukascopy"
  format: "auto"                # auto | csv | parquet
  raw_path: "data/raw/eurusd/"
  clean_path: "data/clean/eurusd/"
  timezone: "UTC"

  date_range:
    start: "2019-01-01"
    end: "2020-12-31"

  schema:
    timestamp: "timestamp"      # epoch ms
    ask: "askPrice"
    bid: "bidPrice"
    ask_volume: "askVolume"
    bid_volume: "bidVolume"

  volume_policy:
    require_volume: false       # if true, raise error if volume is missing
    allow_synthetic: true       # some FX volumes are synthetic
```

---

## 6. Bar Construction Configuration

**File**: `configs/data/bars.yaml`

Defines how to build bars from ticks.

```yaml
bars:
  main_type: "tick1000"
  available:
    - "tick100"
    - "tick1000"
    - "volume_bar"
  tick:
    sizes:
      - 100
      - 1000
  volume:
    enabled: true
    target_volume: 1000000.0    # units of volume per bar (if volume makes sense)
  dollar:
    enabled: false              # future extension
  heikin_ashi:
    enabled: false              # applied as a transform on top of OHLC
```

**Notes**:

- `main_type` is the bar series used for evaluation / labels.
- Additional bars can be built and used as feature sources.

---

## 7. Session Configuration (No Overnight Constraint)

**File**: `configs/session/default_fx.yaml`

Controls trading sessions and "no overnight" behavior.

```yaml
session:
  timezone: "UTC"
  session_start: "00:00"
  session_end: "21:55"      # time to be flat before rollover
  friday_end: "20:00"       # early close on Friday
  weekend_trading: false    # no Sat/Sun trading

  labeling:
    max_horizon_bars: 50
    min_horizon_bars: 10    # no labels if < 10 bars left before session_end
```

- Used both in labeling (triple barrier) and in the backtest (Backtrader).
- `max_horizon_bars` and `min_horizon_bars` tie to the triple barrier logic.

---

## 8. Labeling Configuration (Triple Barrier)

**File**: `configs/labeling/triple_barrier.yaml`

Defines how triple-barrier labeling is applied.

```yaml
labeling:
  type: "triple_barrier"

  # distance specification: choose exactly one mode
  distance_mode: "mfe_mae"          # ticks | mfe_mae

  # if distance_mode == "ticks"
  tp_ticks: 100
  sl_ticks: 100
  
  # if distance_mode == "mfe_mae"
  # TP/SL are computed from MFE/MAE quantiles
  # MFE quantile → TP, MAE quantile → SL
  mfe_mae:
    horizon_bars: 8  # Horizon in bars (e.g., 8 bars of 100 ticks = 800 ticks)
    tp_quantile: 0.5  # Quantile for MFE (take profit)
    sl_quantile: 0.5  # Quantile for MAE (stop loss)

  max_horizon_bars: 50          # overridden by session-aware logic
  min_horizon_bars: 10

  long_only: true               # base labels are long-only
  meta_label:
    enabled: true
    base_signal_source: "model" # or "rule"
```

**Notes**:

- **Distance modes**:
  - `ticks`: TP/SL specified in ticks, converted to price using `convert_ticks_to_price()` and `tick_size`
  - `mfe_mae`: TP/SL computed automatically from MFE/MAE quantiles
    - Uses MFE quantile for TP (Maximum Favorable Excursion)
    - Uses MAE quantile for SL (Maximum Adverse Excursion)
    - Requires `mfe_mae` configuration block with `horizon_bars`, `tp_quantile`, and `sl_quantile`
    - The computed TP/SL values override `tp_ticks` and `sl_ticks` in the config
  
- Internally, TP/SL distances are always translated into price space.
- For `distance_mode="ticks"`, conversion uses `convert_ticks_to_price()` function.
- For `distance_mode="mfe_mae"`, TP/SL are computed from price excursions and converted to ticks.
- Session-aware logic uses `max_horizon_bars` and `min_horizon_bars` with the session config.

---

## 9. Feature Configuration

**File**: `configs/features/base_features.yaml`

Activates feature groups and their parameters.

```yaml
features:
  price:
    enabled: true
    returns_lookbacks: [1, 5, 20]
    volatility_lookbacks: [20, 50]
    range_lookbacks: [20]

  microstructure:
    enabled: true
    order_flow_lookbacks: [10, 20]
    spread_stats_lookbacks: [10]

  bars:
    enabled: true
    include_volume: true
    include_tick_count: true

  fracdiff:
    enabled: false
    d: 0.4
    columns: ["close"]

  scaling:
    enabled: false            # if we decide to scale features
    method: "robust"          # robust | standard | none
```

**Notes**:

- The code should not hardcode which features exist; it should read from this config where possible.
- Some features (e.g. regime) will be added after HMMs are trained.

---

## 10. Model Configuration

### 10.1 Base Model (RF / GBM)

**File**: `configs/models/rf_cpu.yaml` (or `rf_gpu.yaml` for cuML)

```yaml
model:
  name: "rf_cpu"
  type: "random_forest"
  backend: "sklearn"          # sklearn | cuml

  params:
    n_estimators: 200
    max_depth: 10
    min_samples_leaf: 5
    n_jobs: -1
    class_weight: "balanced"

  calibration:
    enabled: false
    method: "isotonic"        # isotonic | sigmoid
```

### 10.2 Meta-Label Model

**File**: `configs/models/meta.yaml`

```yaml
meta_model:
  enabled: true
  type: "random_forest"
  backend: "sklearn"
  params:
    n_estimators: 200
    max_depth: 8
    min_samples_leaf: 5
```

### 10.3 HMM Configuration

**File**: `configs/models/hmm.yaml`

```yaml
hmm:
  macro:
    enabled: true
    n_states: 3
    covariance_type: "full"
    n_init: 5
    max_iter: 500
  micro:
    enabled: true
    n_states: 3
    covariance_type: "full"
    n_init: 5
    max_iter: 500
```

---

## 11. Validation & Cross-Validation Configuration

**Files**: `configs/validation/tscv.yaml` (default) or `configs/validation/cpcv.yaml`

Two types of cross-validation are supported:

### 11.1 Time-Series Cross-Validation (Baseline)

**File**: `configs/validation/tscv.yaml`

```yaml
validation:
  cv_type: "time_series"  # Use TimeSeriesCV (baseline sklearn-based)
  n_splits: 3  # Number of CV folds
  test_size: 2000  # Number of samples in each test set (optional)
  gap: 0  # Number of samples to skip between train and test sets
```

**Notes**:
- Baseline CV method: wrapper around `sklearn.model_selection.TimeSeriesSplit` with optional gap
- Serves as a benchmark to compare against CPCV (Combinatorial Purged CV)
- Unlike CPCV, this method:
  - Does not perform purging based on label intervals
  - Does not apply embargo
  - Simply splits data temporally with an optional gap between train and test

### 11.2 Combinatorial Purged Cross-Validation (CPCV)

**File**: `configs/validation/cpcv.yaml`

```yaml
validation:
  cv_type: "cpcv"  # Use CombinatorialPurgedCV
  n_groups: 10  # Number of temporal groups (blocks)
  n_test_groups: 2  # Number of groups used as test per fold
  embargo_duration: 20  # Size of embargo (in bars) after each test block
  max_combinations: null  # Optional: limit number of folds if C(n_groups, n_test_groups) is too large
```

**Notes**:
- Partitions data into N temporal groups
- Generates folds by selecting k groups as test (combinatorial)
- Allows training on data before and after test (unlike walk-forward)
- Creates richer distribution of out-of-sample performance estimates
- If `max_combinations` is None, uses all C(n_groups, n_test_groups) combinations

**Usage in experiments**:
- Override in experiment config: `- override /validation: cpcv`
- Or specify directly in experiment YAML (will override defaults)

---

## 12. Risk Module Configuration (Monte Carlo)

**File**: `configs/risk/monte_carlo.yaml`

```yaml
risk:
  monte_carlo:
    enabled: true
    n_sims: 1000
    block_bootstrap: false
    block_size: 20       # used if block_bootstrap == true

    prop_firm:
      enabled: true
      max_daily_loss: 0.03
      max_total_loss: 0.10
      target_profit: 0.10
```

**Notes**:

- Monte Carlo operates on trade sequences produced by the session-aware backtest.
- `prop_firm` block is tied to evaluation in prop firm style (e.g. FTMO).

---

## 13. Backtest Configuration

**File**: `configs/backtest/base_bt.yaml`

```yaml
backtest:
  engine: "backtrader"

  commission:
    type: "per_lot"
    value: 0.0            # raw spread account, commission included in spread

  slippage:
    enabled: false        # can be turned on later
    model: "none"         # none | fixed | proportional
    fixed_slippage_ticks: 1.0

  position_sizing:
    use_risk_model: true  # read from capital/risk_model config

  session:
    use_session_calendar: true
```

---

## 14. Reporting Configuration

**File**: `configs/reporting/report.yaml`

```yaml
reporting:
  enabled: true
  html:
    enabled: true
    output_path: "reports/html"
  pdf:
    enabled: false        # can be turned on later (LaTeX / wkhtmltopdf etc.)
  include_sections:
    data_overview: true
    modelling: true
    walk_forward: true
    backtest: true
    risk: true
    feature_importance: true
    regimes: true
```

---

## 15. MLflow Configuration

**File**: `configs/mlflow/local.yaml`

```yaml
mlflow:
  enabled: true
  tracking_uri: "file:./mlruns"
  experiment_name: "trading_ml"
  autolog: false

  log_artifacts:
    config_dump: true
    equity_curves: true
    trade_logs: true
    reports: true
    models: true

  tags:
    project: "trading-ml-pipeline"
```

**Notes**:

The code should log:

- git commit hash,
- full Hydra config dump,
- data version / date range.

---

## 16. Runtime / Hardware Configuration

**File**: `configs/runtime/hardware.yaml`

```yaml
runtime:
  use_gpu: false           # true if RAPIDS / cuML etc. available
  n_jobs: -1               # for CPU-bound models
  memory_limit_gb: 16      # soft limit for certain operations
  deterministic: true      # enforce deterministic behavior where possible
```

---

## 17. AI Guardrails for Configuration

To avoid configuration chaos when using AI assistants:

- **Do not rename existing top-level keys** (`experiment`, `data`, `asset`, `session`, `labeling`, `features`, `models`, `risk`, `backtest`, `reporting`, `mlflow`, `runtime`).
- **Do not move config sections across files** without updating this reference and the docs.
- **Any new key must**:
  - be added to this `CONFIG_REFERENCE.md`,
  - be documented with meaning and type,
  - be used consistently in code (no half-implemented flags).
- **Do not hardcode parameters in code** if a config key already exists for that purpose.

**This document is the single source of truth for configuration semantics.**

If the code and this document disagree, the document should be updated first, then the code or config changed accordingly.

