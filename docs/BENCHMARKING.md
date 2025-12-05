# Benchmarking Framework & Experimentation Protocol

This document describes how the FinancialMLPipeline is benchmarked across modeling choices, data-engineering backends, validation strategies, feature configurations, and system performance.

It defines the experimental structure, configuration system, measurement protocol, MLflow integration, and methodology for scaling to thousands of runs without combinatorial explosion.

**The objective is to compare results (Sharpe, accuracy, P&L...) and runtime performance (latency, CPU/GPU usage...) across diverse pipeline variants.**

---

## Table of Contents

1. [Experiment Scope](#1-experiment-scope)
2. [Experiment Configuration Structure](#2-experiment-configuration-structure)
3. [Baseline Strategies](#3-baseline-strategies)
4. [Benchmark Strategy (Three-Stage Method)](#4-benchmark-strategy-three-stage-method)
5. [Benchmark Framework](#5-benchmark-framework)
6. [Statistical Significance Testing](#6-statistical-significance-testing)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [MLflow Experiment Organization](#8-mlflow-experiment-organization)
9. [Result Analysis & Reporting](#9-result-analysis--reporting)
10. [Reproducibility Rules](#10-reproducibility-rules)
11. [Best Practices](#11-best-practices)
12. [Example Workflow](#12-example-workflow)
13. [Final Output](#13-final-output)
14. [Integration with MLflow](#14-integration-with-mlflow)

---

## 1. Experiment Scope

The benchmarking system evaluates both **modeling performance** and **computational efficiency** under multiple configurations.

### 1.1 Dimensions Under Study

Benchmarked components include:

**Dataframe backends**:
- pandas, polars, cuDF

**ML libraries**:
- scikit-learn, cuML

**Model variants**:
- base model, HMM regime detection on/off, meta-model stacking on/off

**Cross-validation schemes**:
- TimeSeriesSplit, CPCV, walk-forward (future)

**Feature extraction**:
- different MFE/MAE parameter sets (lags, windows, volatility lookback)

**Labeling/target variations**:
- triple barrier configs, event sources, aggregation methods

Each experiment is fully reproducible using a structured config object logged to MLflow.

### 1.2 Overview

A benchmark is a reference strategy used to evaluate whether a machine learning model provides genuine value over simple approaches.

**Why benchmark?**:

- A model with 60% accuracy might be worse than a simple moving average crossover
- High Sharpe ratio might be due to luck, not skill
- Need statistical tests to confirm improvements are real

**Objective**: Compare results (Sharpe, accuracy, P&L...) and runtime performance (latency, CPU/GPU usage...) across diverse pipeline variants.

---

## 2. Experiment Configuration Structure

Every run is defined by a structured configuration.

### 2.1 Config Schema

```yaml
benchmark:
  scenario_id: "eurusd_2020Q1"
  df_backend: "pandas | polars | cudf"
  ml_backend: "sklearn | cuml"
  use_hmm: true/false
  use_meta_model: true/false

cv:
  mode: "TimeSeriesSplit | CPCV"
  n_splits: 5
  purge: 30   # bars
  embargo: 30 # bars

features:
  mfe_params: 
    lookback: 50
    vol_window: 30
  mae_params: 
    fracdiff: false
    diff_window: 20

compute:
  seed: 42
  track_memory: true
  gpu_monitoring: optional
```

**Logged in MLflow as**:

- `params` (scalar values)
- `config JSON` artifact (full configuration)
- `tags` (benchmark class, scenario, backend type)

### 2.2 Orchestrated Run Execution

**Benchmark Runner Logic**:

```
Load Config → Init Backend & Pipeline → Build Features (MFE/MAE) → 
Cross-Validation & Fit → Standard Backtest → Write Metrics to MLflow
```

### 2.3 Runtime Metrics Recorded

| Category | Metrics |
|----------|---------|
| **Feature Engineering** | `time_feature_engineering`, `memory_peak` |
| **Training** | `time_fit`, `CPU/GPU usage` |
| **Prediction/Backtest** | `time_inference`, `time_backtest` |
| **Total Run Cost** | `time_total`, `ram_max`, `gpu_utilization` |

All metrics are logged automatically per run.

---

## 3. Baseline Strategies

### 3.1 Buy-and-Hold

The simplest strategy: buy at the start, hold until the end.

**Implementation**:

```python
# src/benchmarks/baselines.py
class BuyAndHold:
    """Buy-and-hold baseline strategy."""
    
    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate buy-and-hold signals."""
        signals = pd.DataFrame(index=bars.index)
        signals['prediction'] = 1  # Always long
        signals['probability'] = 1.0
        signals['meta_decision'] = 1
        return signals
```

**When to use**: Baseline for directional strategies.

### 3.2 Random Strategy

Random entry signals (50/50 long/short or long-only).

**Implementation**:

```python
class RandomStrategy:
    """Random baseline strategy."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate random signals."""
        signals = pd.DataFrame(index=bars.index)
        # Random long signals (50% probability)
        signals['prediction'] = self.rng.choice([0, 1], size=len(bars), p=[0.5, 0.5])
        signals['probability'] = 0.5
        signals['meta_decision'] = signals['prediction']
        return signals
```

**When to use**: Null hypothesis test (model should beat random).

### 3.3 Moving Average Crossover

Simple technical indicator: buy when short MA crosses above long MA.

**Implementation**:

```python
class MovingAverageCrossover:
    """MA crossover baseline strategy."""
    
    def __init__(self, short_period: int = 10, long_period: int = 50):
        self.short_period = short_period
        self.long_period = long_period
    
    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate MA crossover signals."""
        close = bars['bid_close']  # Use bid for consistency
        
        short_ma = close.rolling(self.short_period).mean()
        long_ma = close.rolling(self.long_period).mean()
        
        # Signal: 1 when short MA > long MA, 0 otherwise
        signals = pd.DataFrame(index=bars.index)
        signals['prediction'] = (short_ma > long_ma).astype(int)
        signals['probability'] = 0.5  # Not probabilistic
        signals['meta_decision'] = signals['prediction']
        
        return signals
```

**When to use**: Common technical analysis baseline.

### 3.4 RSI Strategy

Relative Strength Index-based strategy.

**Implementation**:

```python
class RSIStrategy:
    """RSI-based baseline strategy."""
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI-based signals."""
        close = bars['bid_close']
        rsi = self.calculate_rsi(close)
        
        signals = pd.DataFrame(index=bars.index)
        # Buy when RSI < oversold, sell when RSI > overbought
        signals['prediction'] = (rsi < self.oversold).astype(int)
        signals['probability'] = 0.5
        signals['meta_decision'] = signals['prediction']
        
        return signals
```

**When to use**: Mean-reversion baseline.

---

## 4. Benchmark Strategy (Three-Stage Method)

The parameter space is infinite — benchmarks must be layered to avoid brute-force combinatorics.

### 4.1 Stage 1 — Backend/Stack Comparison

**Purpose**: choose foundation of the pipeline.

**Benchmark grid**:

| df_backend | ml_backend | HMM | Meta | CV |
|------------|------------|-----|------|-----|
| pandas | sklearn | off | off | TimeSeriesSplit |
| polars | sklearn | off | off | TimeSeriesSplit |
| cudf | cuml | off | off | TimeSeriesSplit |
| ... | ... | ... | ... | ... |

**Outputs**:

- runtime scaling curves (small/medium/large dataset)
- best stack candidate for further experiments

### 4.2 Stage 2 — Ablation Experiments

Evaluate HMM, meta-model, CPCV individually and combined.

**Baseline**:

```yaml
df_backend: <best_from_stage1>
ml_backend: <best_from_stage1>
HMM: false
Meta: false
CV: TimeSeriesSplit
```

**Ablations**:

- +HMM
- +Meta
- +CPCV
- +HMM + Meta
- +Meta + CPCV
- +HMM + Meta + CPCV

**Recorded metrics**:

- ΔSharpe vs baseline
- Δruntime vs baseline
- stability across splits

### 4.3 Stage 3 — Fine Parameter Search (MFE/MAE + Hyperparams)

Only executed on best architecture(s) from stage 1-2.

**Examples**:

- MAE windows = [20, 50, 100]
- MFE volatility lookback = [30, 60]
- Meta model = XGBoost, RF, LGBM
- HMM states = [2, 3, 4]

**Search methods**:

- grid search
- randomized search
- Optuna-based optimizer

---

## 5. Benchmark Framework

### 5.1 Benchmark Runner

```python
# src/benchmarks/runner.py
from typing import Dict, List
import pandas as pd
from src.backtest.runner import run_backtest

class BenchmarkRunner:
    """Run and compare multiple strategies."""
    
    def __init__(self, bars: pd.DataFrame, session_calendar, config: dict):
        self.bars = bars
        self.session_calendar = session_calendar
        self.config = config
    
    def run_benchmark(self, strategy, name: str) -> Dict:
        """Run a single benchmark strategy."""
        signals = strategy.generate_signals(self.bars)
        
        results = run_backtest(
            bars=self.bars,
            predictions=signals,
            session_calendar=self.session_calendar,
            config=self.config,
            labeling_config=self.config.get('labeling', {}),
            assets_config=self.config.get('assets', {})
        )
        
        results['strategy_name'] = name
        return results
    
    def compare_strategies(self, strategies: Dict[str, object]) -> pd.DataFrame:
        """Compare multiple strategies."""
        all_results = []
        
        for name, strategy in strategies.items():
            results = self.run_benchmark(strategy, name)
            all_results.append(results)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'strategy': [r['strategy_name'] for r in all_results],
            'sharpe_ratio': [r['sharpe_ratio'] for r in all_results],
            'total_return': [r['total_return'] for r in all_results],
            'max_drawdown': [r['max_drawdown'] for r in all_results],
            'win_rate': [r['win_rate'] for r in all_results],
            'total_trades': [r['total_trades'] for r in all_results],
        })
        
        return comparison
```

### 5.2 Integration with Pipeline

```python
# src/pipeline/main_pipeline.py
if cfg.benchmarks.get('enabled', False):
    from src.benchmarks.runner import BenchmarkRunner
    from src.benchmarks.baselines import BuyAndHold, RandomStrategy, MovingAverageCrossover
    
    logger.info("Running benchmark strategies")
    
    benchmark_runner = BenchmarkRunner(
        bars=bars,
        session_calendar=calendar,
        config=dict(cfg)
    )
    
    strategies = {
        'buy_and_hold': BuyAndHold(),
        'random': RandomStrategy(seed=42),
        'ma_crossover': MovingAverageCrossover(short_period=10, long_period=50),
    }
    
    benchmark_comparison = benchmark_runner.compare_strategies(strategies)
    
    # Log to MLflow
    for _, row in benchmark_comparison.iterrows():
        mlflow.log_metric(f"benchmark_{row['strategy']}_sharpe", row['sharpe_ratio'])
        mlflow.log_metric(f"benchmark_{row['strategy']}_return", row['total_return'])
    
    # Save comparison
    comparison_path = Path('benchmark_comparison.csv')
    benchmark_comparison.to_csv(comparison_path, index=False)
    mlflow.log_artifact(str(comparison_path))
```

---

## 6. Statistical Significance Testing

### 6.1 T-Test for Returns

Compare model returns to baseline returns:

```python
# src/benchmarks/statistical_tests.py
from scipy import stats
import numpy as np

def t_test_returns(model_returns: np.ndarray, baseline_returns: np.ndarray) -> Dict:
    """Perform t-test to compare returns.
    
    Args:
        model_returns: Array of model returns
        baseline_returns: Array of baseline returns
    
    Returns:
        Dictionary with test results
    """
    t_stat, p_value = stats.ttest_ind(model_returns, baseline_returns)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'model_mean': np.mean(model_returns),
        'baseline_mean': np.mean(baseline_returns),
    }
```

### 6.2 Mann-Whitney U Test

Non-parametric test (doesn't assume normal distribution):

```python
def mann_whitney_test(model_returns: np.ndarray, baseline_returns: np.ndarray) -> Dict:
    """Perform Mann-Whitney U test (non-parametric)."""
    u_stat, p_value = stats.mannwhitneyu(model_returns, baseline_returns, alternative='two-sided')
    
    return {
        'u_statistic': u_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }
```

### 6.3 Bootstrap Confidence Intervals

Estimate confidence intervals for Sharpe ratio:

```python
def bootstrap_sharpe_ratio(returns: np.ndarray, n_bootstrap: int = 1000) -> Dict:
    """Bootstrap confidence interval for Sharpe ratio."""
    sharpe_ratios = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = np.random.choice(returns, size=len(returns), replace=True)
        
        if np.std(resampled) > 0:
            sharpe = np.mean(resampled) / np.std(resampled) * np.sqrt(252)
            sharpe_ratios.append(sharpe)
    
    sharpe_ratios = np.array(sharpe_ratios)
    
    return {
        'mean_sharpe': np.mean(sharpe_ratios),
        'ci_lower': np.percentile(sharpe_ratios, 2.5),
        'ci_upper': np.percentile(sharpe_ratios, 97.5),
    }
```

---

## 7. Evaluation Metrics

### 7.1 Model Performance

Metrics logged per run:

**Classification**:
- accuracy, f1, recall, precision, logloss, auc

**Trading**:
- P&L, Sharpe, Sortino, maxDD, hitrate, turnover

**Stability**:
- cross-validation mean/variance

### 7.2 Efficiency & Scaling

- `time_total`
- `time_feature_eng` / `time_training` / `time_backtest`
- scaling tests vs dataset size
- `gpu_speedup_ratio` (if GPU enabled)

### 7.3 Efficiency Frontier

Comparisons performed visually:

**Sharpe ↑ vs Runtime ↓**

Plot performance vs computational cost to identify optimal trade-offs.

### 7.4 Extended Performance Metrics

Beyond Sharpe ratio, calculate:

```python
# src/backtest/metrics.py (extend existing)

def calculate_sortino_ratio(trade_log: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Sortino ratio (Sharpe with downside deviation)."""
    returns = trade_log['pnl_pct'].values
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    downside_std = np.std(downside_returns)
    mean_return = np.mean(returns)
    
    return (mean_return - risk_free_rate / 252) / downside_std * np.sqrt(252)

def calculate_profit_factor(trade_log: pd.DataFrame) -> float:
    """Profit factor = gross profit / gross loss."""
    profits = trade_log[trade_log['pnl'] > 0]['pnl'].sum()
    losses = abs(trade_log[trade_log['pnl'] < 0]['pnl'].sum())
    
    return profits / losses if losses > 0 else np.inf

def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """Calmar ratio = annual return / max drawdown."""
    if max_drawdown == 0:
        return np.inf
    return total_return / abs(max_drawdown)

def calculate_expectancy(trade_log: pd.DataFrame) -> float:
    """Average profit per trade."""
    return trade_log['pnl'].mean()
```

### 7.5 Comprehensive Comparison

```python
def comprehensive_comparison(model_results: Dict, baseline_results: Dict) -> pd.DataFrame:
    """Comprehensive strategy comparison."""
    comparison = pd.DataFrame({
        'metric': [
            'Sharpe Ratio',
            'Sortino Ratio',
            'Total Return',
            'Max Drawdown',
            'Profit Factor',
            'Calmar Ratio',
            'Win Rate',
            'Expectancy',
        ],
        'model': [
            model_results['sharpe_ratio'],
            calculate_sortino_ratio(model_results['trade_log']),
            model_results['total_return'],
            model_results['max_drawdown'],
            calculate_profit_factor(model_results['trade_log']),
            calculate_calmar_ratio(
                model_results['total_return'],
                model_results['max_drawdown']
            ),
            model_results['win_rate'],
            calculate_expectancy(model_results['trade_log']),
        ],
        'baseline': [
            baseline_results['sharpe_ratio'],
            calculate_sortino_ratio(baseline_results['trade_log']),
            baseline_results['total_return'],
            baseline_results['max_drawdown'],
            calculate_profit_factor(baseline_results['trade_log']),
            calculate_calmar_ratio(
                baseline_results['total_return'],
                baseline_results['max_drawdown']
            ),
            baseline_results['win_rate'],
            calculate_expectancy(baseline_results['trade_log']),
        ],
    })
    
    comparison['improvement'] = comparison['model'] - comparison['baseline']
    comparison['improvement_pct'] = (
        (comparison['model'] - comparison['baseline']) / 
        comparison['baseline'].abs() * 100
    )
    
    return comparison
```

---

## 8. MLflow Experiment Organization

### 8.1 Run Structure

```
mlruns/
  ├── stack_benchmarks/
  ├── ablation_hmm_meta/
  ├── mfe_mae_search/
  ├── regression_tests/
```

### 8.2 Tagging Convention

| Tag | Meaning |
|-----|---------|
| `benchmark:stack` | backend performance comparison |
| `benchmark:ablation` | HMM/meta/CPCV toggles |
| `benchmark:params` | MFE/MAE parametrization |
| `scenario:<id>` | dataset period or symbol |
| `baseline:true` | reference configuration |

**Example**:

```python
mlflow.set_tags({
    'benchmark:stack': 'pandas_sklearn',
    'scenario': 'eurusd_2023',
    'baseline': 'true'
})
```

### 8.3 Benchmark Configuration

#### Configuration File

```yaml
# configs/benchmarks/baselines.yaml
benchmarks:
  enabled: true
  
  strategies:
    buy_and_hold:
      enabled: true
    
    random:
      enabled: true
      seed: 42
    
    ma_crossover:
      enabled: true
      short_period: 10
      long_period: 50
    
    rsi:
      enabled: false
      period: 14
      oversold: 30
      overbought: 70
  
  statistical_tests:
    enabled: true
    tests:
      - t_test
      - mann_whitney
    significance_level: 0.05
  
  metrics:
    - sharpe_ratio
    - sortino_ratio
    - profit_factor
    - calmar_ratio
    - expectancy
```

### 8.4 Integration

```python
# src/pipeline/main_pipeline.py
if cfg.benchmarks.get('enabled', False):
    benchmark_config = cfg.benchmarks
    
    # Run enabled benchmarks
    strategies = {}
    
    if benchmark_config.strategies.buy_and_hold.enabled:
        strategies['buy_and_hold'] = BuyAndHold()
    
    if benchmark_config.strategies.random.enabled:
        strategies['random'] = RandomStrategy(
            seed=benchmark_config.strategies.random.seed
        )
    
    # ... run comparisons ...
```

---

## 9. Result Analysis & Reporting

### 9.1 Performance Tables & Rankings

Analysis performed through MLflow queries or a dedicated notebook:

- generate performance tables & rankings
- compute deltas vs baseline
- plot tradeoffs runtime/performance
- build leaderboards

**Example: Stack Comparison Pivot**

| backend | Sharpe | time_total | ΔSharpe | speedup |
|---------|--------|------------|---------|---------|
| pandas+sklearn | 1.05 | 18min | baseline | 1x |
| polars+sklearn | 1.07 | 11min | +1.9% | 1.6x faster |
| cudf+cuml | 1.01 | 4min | -3.8% | 4.5x faster |

### 9.2 Visualization

### 7.1 Comparison Charts

```python
# src/reporting/visualizations.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_benchmark_comparison(comparison_df: pd.DataFrame, output_path: Path = None):
    """Plot benchmark strategy comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sharpe ratio comparison
    axes[0, 0].bar(comparison_df['strategy'], comparison_df['sharpe_ratio'])
    axes[0, 0].set_title('Sharpe Ratio Comparison')
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Total return comparison
    axes[0, 1].bar(comparison_df['strategy'], comparison_df['total_return'])
    axes[0, 1].set_title('Total Return Comparison')
    axes[0, 1].set_ylabel('Total Return')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Max drawdown comparison
    axes[1, 0].bar(comparison_df['strategy'], comparison_df['max_drawdown'])
    axes[1, 0].set_title('Max Drawdown Comparison')
    axes[1, 0].set_ylabel('Max Drawdown')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Win rate comparison
    axes[1, 1].bar(comparison_df['strategy'], comparison_df['win_rate'])
    axes[1, 1].set_title('Win Rate Comparison')
    axes[1, 1].set_ylabel('Win Rate')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 10. Reproducibility Rules

All experiments must store full config JSON.

**Requirements**:

- Seeds fixed for deterministic comparison
- Dataset version frozen via hashing/artifact tracking
- Any experiment must be replayable using `config.yaml`
- No silent fallback or auto-altered settings allowed

**Example**:

```python
# Log full config to MLflow
import json
from omegaconf import OmegaConf

config_dict = OmegaConf.to_container(cfg, resolve=True)
mlflow.log_dict(config_dict, 'config.json')

# Log dataset hash
dataset_hash = hash_dataset(dataset_path)
mlflow.log_param('dataset_hash', dataset_hash)

# Log git commit
import subprocess
git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
mlflow.set_tag('git_commit', git_commit)
```

---

## 11. Best Practices

### 11.1 Benchmark Selection

- **Always compare to buy-and-hold**: Minimum baseline
- **Include random strategy**: Null hypothesis test
- **Use domain-appropriate baselines**: MA crossover for trend-following, RSI for mean-reversion
- **Multiple baselines**: More robust comparison

### 11.2 Statistical Rigor

- **Test significance**: Don't just compare numbers, test if difference is real
- **Multiple time periods**: Test on different market conditions
- **Out-of-sample**: Benchmarks must use same out-of-sample data as model
- **Bootstrap confidence**: Account for uncertainty in metrics

### 11.3 Interpretation

- **Economic vs statistical significance**: Small improvements might be statistically significant but not economically meaningful
- **Risk-adjusted metrics**: Don't just look at returns, consider risk (Sharpe, Sortino)
- **Consistency**: Model should outperform across multiple metrics, not just one

---

## 12. Example Workflow

### 9.1 Running Benchmarks

```bash
# Enable benchmarks in config
python scripts/run_experiment.py \
    experiment=test \
    benchmarks.enabled=true \
    benchmarks.strategies.buy_and_hold.enabled=true \
    benchmarks.strategies.random.enabled=true
```

### 9.2 Analyzing Results

```python
# Load benchmark comparison
comparison = pd.read_csv('benchmark_comparison.csv')

# Check if model beats baselines
model_sharpe = comparison[comparison['strategy'] == 'model']['sharpe_ratio'].values[0]
baseline_sharpe = comparison[comparison['strategy'] == 'buy_and_hold']['sharpe_ratio'].values[0]

if model_sharpe > baseline_sharpe:
    print(f"Model outperforms buy-and-hold: {model_sharpe:.2f} vs {baseline_sharpe:.2f}")
else:
    print(f"Model underperforms buy-and-hold: {model_sharpe:.2f} vs {baseline_sharpe:.2f}")

# Statistical test
model_returns = load_model_returns()
baseline_returns = load_baseline_returns()

test_result = t_test_returns(model_returns, baseline_returns)
if test_result['significant']:
    print(f"Difference is statistically significant (p={test_result['p_value']:.4f})")
```

---

## 13. Final Output

Each benchmark produces:

- MLflow logged runs (with full config and metrics)
- comparison tables (performance vs baseline)
- performance vs cost plots (efficiency frontier)
- recommended system configuration
- best hyperparameter sets for production

**The benchmarking framework becomes the foundation for continuous ML improvements, regression testing, and model selection.**

---

## 14. Integration with MLflow

### 14.1 Logging Benchmark Results

```python
# Log benchmark metrics
for strategy_name in ['buy_and_hold', 'random', 'ma_crossover']:
    results = benchmark_results[strategy_name]
    
    mlflow.log_metric(f'benchmark_{strategy_name}_sharpe', results['sharpe_ratio'])
    mlflow.log_metric(f'benchmark_{strategy_name}_return', results['total_return'])
    mlflow.log_metric(f'benchmark_{strategy_name}_max_dd', results['max_drawdown'])

# Log comparison
comparison_df.to_csv('benchmark_comparison.csv', index=False)
mlflow.log_artifact('benchmark_comparison.csv')

# Log visualization
plot_benchmark_comparison(comparison_df, 'benchmark_comparison.png')
mlflow.log_artifact('benchmark_comparison.png')
```

---

## References

- [BACKTESTING.md](BACKTESTING.md) - Backtesting system details
- [TESTING.md](TESTING.md) - Regression test guidelines
- [REPORTING.md](REPORTING.md) - Report generation with benchmarks
- [DETAILED_REVIEW.md](../DETAILED_REVIEW.md) - Benchmarking requirements

