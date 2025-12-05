# Profiling & Optimization Strategy

This document defines how profiling is used to measure runtime efficiency across the FinancialMLPipeline, identify bottlenecks, and guide performance optimization.

It describes recommended tools, workflow, interpretation rules, and implementation patterns for profiling Python code (tick → bars → labeling → features → ML).

**Profiling is mandatory before any optimisation work** — optimisation decisions must be evidence-based, not intuitive.

---

## Table of Contents

1. [Profiling Fundamentals](#1-profiling-fundamentals)
2. [Why Profiling is Required](#2-why-profiling-is-required)
3. [Profiling Tools](#3-profiling-tools)
4. [Global Profiling (Full Pipeline)](#4-global-profiling-full-pipeline)
5. [Visual Profiling](#5-visual-profiling)
6. [Targeted Profiling (Single Component)](#6-targeted-profiling-single-component)
7. [Profiling the Pipeline](#7-profiling-the-pipeline)
8. [Analyzing Profile Results](#8-analyzing-profile-results)
9. [Common Bottlenecks](#9-common-bottlenecks)
10. [Optimization Strategy](#10-optimization-strategy)
11. [Profiling Specific Components](#11-profiling-specific-components)
12. [Performance Benchmarks](#12-performance-benchmarks)
13. [Profiling Workflow for FinancialMLPipeline](#13-profiling-workflow-for-financialmlpipeline)
14. [Output Storage & Conventions](#14-output-storage--conventions)
15. [When Profiling Must Be Performed](#15-when-profiling-must-be-performed)
16. [Memory Profiling](#16-memory-profiling)
17. [Continuous Performance Monitoring](#17-continuous-performance-monitoring)
18. [Goal](#18-goal)
19. [Best Practices](#19-best-practices)
20. [Tools and Resources](#20-tools-and-resources)

---

## 1. Profiling Fundamentals

### 1.1 What Profiling Does

Profiling executes the code normally (no simulation) and measures execution time per function.

It tracks:
- number of calls (`ncalls`)
- time spent inside a function (`tottime`)
- cumulative time including subcalls (`cumtime`)

**Profiling is a performance measurement system, not a predictor.**

### 1.2 Key Metrics

| Metric | Meaning |
|--------|---------|
| `ncalls` | Times function was called |
| `tottime` | Time spent only inside function (excluding subcalls) |
| `cumtime` | Function + children functions execution time |
| `percall` | Average time per call |
| `filename:lineno(function)` | Origin of code |

### 1.3 Output Example

```
ncalls  tottime  cumtime  function
500     2.100    4.400    triple_barrier_labeling
10000   0.250    0.350    rolling_features
1       0.010    5.000    run_pipeline
```

`triple_barrier_labeling` is the bottleneck here.

### 1.4 Overview

The pipeline processes large amounts of tick data and performs computationally intensive operations:

- Bar construction from millions of ticks
- Feature engineering with rolling windows
- Cross-validation with purging and embargo
- Backtesting with session-aware logic
- Monte Carlo simulations

Profiling helps identify which operations consume the most time and resources.

---

## 2. Why Profiling is Required

**Profiling must be run before optimizing code.**

**Reasons**:

- intuition about slow code is often wrong
- most runtime is concentrated in a few functions (Pareto 80/20 rule)
- premature optimisation wastes development time
- avoids micro-optimising components that represent < 5% of runtime

**Optimization Roadmap**:

```
Profile → Identify bottlenecks → Fix → Re-profile → Repeat
```

**Never optimize below the threshold of significance** (components < 5% of total runtime).

---

## 3. Profiling Tools

### 3.1 cProfile (Standard Library)

Python's built-in profiler, suitable for most use cases.

**Basic usage**:

```python
import cProfile
import pstats
from io import StringIO

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Code to profile
    run_pipeline(cfg)
    
    profiler.disable()
    
    # Generate report
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(50)  # Top 50 functions
    
    print(s.getvalue())
```

**Command-line usage**:

```bash
# Profile entire script
python -m cProfile -o pipeline.prof scripts/run_experiment.py experiment=test

# View results
python -m pstats pipeline.prof
```

### 3.2 line_profiler

Profiles line-by-line execution time.

**Installation**:

```bash
pip install line_profiler
```

**Usage**:

```python
# Decorate function to profile
@profile
def slow_function():
    # Code to profile
    pass

# Run with kernprof
kernprof -l -v script.py
```

### 3.3 memory_profiler

Profiles memory usage.

**Installation**:

```bash
pip install memory_profiler
```

**Usage**:

```python
@profile
def memory_intensive_function():
    # Code to profile
    pass

# Run with mprof
mprof run script.py
mprof plot
```

### 3.4 py-spy (Sampling Profiler)

Low-overhead sampling profiler, good for production code.

**Installation**:

```bash
pip install py-spy
```

**Usage**:

```bash
# Profile running process
py-spy record -o profile.svg -- python scripts/run_experiment.py

# Top output
py-spy top --pid <PID>
```

---

## 4. Global Profiling (Full Pipeline)

Used to detect the heaviest blocks before optimization.

### 4.1 Running Global Profiling

**Command line**:

```bash
python -m cProfile -o profile.out scripts/run_experiment.py experiment=test
```

**Inside Python**:

```python
import cProfile

if __name__ == "__main__":
    cProfile.run("run_pipeline()", "profile.out")
```

### 4.2 Reading Output

```python
import pstats

stats = pstats.Stats("profile.out")
stats.strip_dirs().sort_stats("cumtime").print_stats(40)
```

**Sorting keys**:

| Mode | Use Case |
|------|----------|
| `tottime` | identify heavy internal logic |
| `cumtime` | identify system-level impact |
| `calls` | check over-calling inefficiencies |

---

## 5. Visual Profiling

### 5.1 Snakeviz (Recommended)

```bash
pip install snakeviz
snakeviz profile.out
```

Uses graphical flame/folded view to inspect call tree.

### 5.2 When to Use

| Scenario | Tool |
|----------|------|
| Pipeline overview | cProfile + Snakeviz |
| Single function bottleneck | cProfile targeted |
| Line-by-line timing | line_profiler |
| Memory usage | memory_profiler |
| Asynchronous profiling | pyinstrument |

---

## 6. Targeted Profiling (Single Component)

When a function is identified as a bottleneck, run focused profiling:

```python
import cProfile

cProfile.run("triple_barrier_labeling(data)", sort="tottime")
```

Useful for:

- bar construction (tick bars, rolling windows)
- triple-barrier labeling
- feature generation loops
- purged K-fold cross-validation
- Monte Carlo backtesting scenarios

---

## 7. Profiling the Pipeline

### 7.1 Script-Based Profiling

Create a profiling script:

```python
# scripts/profile_pipeline.py
#!/usr/bin/env python3
"""Profile the ML pipeline execution."""

import cProfile
import pstats
from pathlib import Path
import sys
import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.main_pipeline import run_pipeline

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def profile_pipeline(cfg: DictConfig):
    """Run pipeline with profiling enabled."""
    
    # Enable profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        # Run pipeline
        run_pipeline(cfg)
    finally:
        profiler.disable()
        
        # Save profile data
        profile_file = Path('pipeline.prof')
        profiler.dump_stats(str(profile_file))
        print(f"Profile saved to {profile_file}")
        
        # Generate text report
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(50)
        
        report_file = Path('profile_report.txt')
        with open(report_file, 'w') as f:
            f.write(s.getvalue())
        print(f"Report saved to {report_file}")

if __name__ == '__main__':
    profile_pipeline()
```

**Usage**:

```bash
python scripts/profile_pipeline.py experiment=test
```

### 7.2 Integrated Profiling

Add profiling support directly in the pipeline:

```python
# src/pipeline/main_pipeline.py
def run_pipeline(cfg: DictConfig) -> None:
    """Run full ML pipeline."""
    
    # Enable profiling if configured
    if cfg.runtime.get('profile', False):
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    
    try:
        # ... pipeline code ...
        pass
    finally:
        if cfg.runtime.get('profile', False):
            profiler.disable()
            profiler.dump_stats('pipeline.prof')
            logger.info("Profile saved to pipeline.prof")
```

**Configuration**:

```yaml
# configs/runtime/hardware.yaml
runtime:
  profile: true  # Enable profiling
  profile_output: "pipeline.prof"  # Output file
```

---

## 8. Analyzing Profile Results

### 8.1 Using pstats

```python
import pstats

# Load profile
stats = pstats.Stats('pipeline.prof')

# Sort by cumulative time
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions

# Sort by total time
stats.sort_stats('tottime')
stats.print_stats(20)

# Filter by module
stats.print_stats('src/data')  # Only src/data functions
```

### 8.2 Using snakeviz (Visualization)

**Installation**:

```bash
pip install snakeviz
```

**Usage**:

```bash
# Generate interactive HTML visualization
snakeviz pipeline.prof

# Opens browser with interactive call graph
```

### 8.3 Interpreting Results

**Key metrics**:

- **ncalls**: Number of calls
- **tottime**: Total time in function (excluding subcalls)
- **cumtime**: Cumulative time (including subcalls)
- **percall**: Time per call

**What to look for**:

1. **High cumulative time**: Functions called many times or with expensive subcalls
2. **High total time**: Functions doing expensive work themselves
3. **Many calls**: Functions that could be optimized or cached

**Example output**:

```
         1234567 function calls (1234567 primitive calls) in 45.678 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000   45.678   45.678 main_pipeline.py:49(run_pipeline)
        1    0.123    0.123   30.456   30.456 bars_pandas.py:31(build_tick_bars)
   100000    5.234    0.000   25.123    0.000 bars_pandas.py:64(_build_bid_ask_ohlc)
   100000   15.789    0.000   15.789    0.000 bars_pandas.py:110(iterrows)  # SLOW!
```

---

## 9. Common Bottlenecks

### 9.1 DataFrame Iteration

**Problem**: Using `iterrows()` is very slow.

```python
# SLOW
for idx, row in df.iterrows():
    process(row)

# FASTER
for row in df.itertuples():
    process(row)

# FASTEST (vectorized)
df.apply(process, axis=1)
# or fully vectorized operations
```

**Profile detection**:

```
ncalls  tottime  cumtime  filename:lineno(function)
100000  15.789  15.789   pandas/core/frame.py:iterrows
```

### 9.2 Repeated Computations

**Problem**: Computing the same thing multiple times.

**Solution**: Cache results or compute once.

```python
# BAD
for i in range(n):
    result = expensive_computation(data)

# GOOD
cached_result = expensive_computation(data)
for i in range(n):
    result = use_cached(cached_result)
```

### 9.3 Large Data Copies

**Problem**: Unnecessary DataFrame copies.

```python
# BAD (creates copy)
df_new = df.copy()
df_new['new_col'] = values

# GOOD (in-place when possible)
df['new_col'] = values
```

### 9.4 Non-Vectorized Operations

**Problem**: Python loops instead of NumPy/pandas vectorization.

```python
# SLOW
result = []
for val in series:
    result.append(transform(val))

# FAST
result = series.apply(transform)
# or
result = transform_vectorized(series.values)
```

---

## 10. Optimization Strategy

### 10.1 Priority-Driven Approach

**Do not optimise everything.**

**Optimise only what profiling shows as expensive.**

| Priority | Component | Optimisation Direction |
|----------|-----------|------------------------|
| **P1** | Triple Barrier | numba JIT, vectorisation, parallelisation |
| **P2** | Rolling features | pandas rolling optimised, strides, numba |
| **P3** | Purged CV + Embargo | joblib parallel, caching |
| **P4** | Data joins / merge | indexed joins, avoid repeated merges |
| **P5** | ML Training loops | reduce parameter search, early stopping |
| **LOW** | Minor overhead (<5%) | ignore until global gains completed |

### 10.2 Rules

- **Optimise measured, not guessed**
- **Always re-profile after every optimisation step**
- **Prefer algorithmic improvement over micro-speed hacks**

### 10.3 Indicators for "Obvious Optimisation"

May be done before profiling:

- ✔ python loops over millions of ticks/bars
- ✔ repeated recalculation inside loops
- ✔ absence of vectorisation
- ✔ computation repeated across folds
- ✔ unnecessary dataframe copies

**Still recommended to measure impact afterward.**

### 10.4 Numba JIT Compilation

For numerical loops, use Numba:

```python
from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def compute_mfe_mae_numba(entry_price, future_high, future_low, horizon):
    """Numba-optimized MFE/MAE computation."""
    n = len(entry_price)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    
    for i in range(n - horizon):
        start_price = entry_price[i]
        start_idx = i + 1
        end_idx = min(start_idx + horizon, n)
        
        max_high = np.max(future_high[start_idx:end_idx])
        min_low = np.min(future_low[start_idx:end_idx])
        
        mfe[i] = max_high - start_price
        mae[i] = start_price - min_low
    
    return mfe, mae
```

**When to use**:

- Numerical loops over arrays
- Functions called many times
- Pure NumPy operations

### 10.5 Vectorization

Replace loops with vectorized operations:

```python
# SLOW
result = []
for i in range(len(data)):
    result.append(data[i] * 2)

# FAST
result = data * 2
```

### 10.6 Parallelization

For independent operations:

```python
from joblib import Parallel, delayed

# Parallel Monte Carlo simulations
results = Parallel(n_jobs=-1)(
    delayed(run_simulation)(config)
    for _ in range(n_simulations)
)
```

### 10.7 Caching

Cache expensive computations:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(param1, param2):
    # Expensive operation
    return result
```

---

## 11. Profiling Specific Components

### 11.1 Bar Construction

```python
# Profile bar construction
import cProfile

profiler = cProfile.Profile()
profiler.enable()

bars = build_tick_bars(ticks, tick_size=100)

profiler.disable()
profiler.print_stats()
```

### 11.2 Feature Engineering

```python
# Profile feature engineering
profiler = cProfile.Profile()
profiler.enable()

features = engineer_features(bars, cfg)

profiler.disable()
profiler.print_stats()
```

### 11.3 Cross-Validation

```python
# Profile CV splits
profiler = cProfile.Profile()
profiler.enable()

for train_idx, test_idx in cv.split(X, label_indices=label_indices):
    # Training and evaluation
    pass

profiler.disable()
profiler.print_stats()
```

### 11.4 Backtesting

```python
# Profile backtest execution
profiler = cProfile.Profile()
profiler.enable()

bt_results = run_backtest(bars, predictions, ...)

profiler.disable()
profiler.print_stats()
```

---

## 12. Performance Benchmarks

### 12.1 Baseline Measurements

Establish baseline performance:

```python
# scripts/benchmark_baseline.py
import time
from src.data.bars import BarBuilder

def benchmark_bar_construction():
    """Benchmark bar construction performance."""
    ticks = load_large_dataset()  # 1M ticks
    
    start = time.time()
    bars = BarBuilder(cfg).build_tick_bars(ticks, tick_size=100)
    elapsed = time.time() - start
    
    print(f"Bar construction: {elapsed:.2f}s for {len(ticks)} ticks")
    print(f"Rate: {len(ticks)/elapsed:.0f} ticks/s")
    
    return elapsed
```

### 12.2 Regression Detection

Compare against baseline:

```python
def test_performance_regression():
    """Ensure performance doesn't degrade."""
    baseline_time = 10.0  # seconds
    current_time = benchmark_bar_construction()
    
    # Allow 20% degradation
    threshold = baseline_time * 1.2
    
    assert current_time < threshold, \
        f"Performance regression: {current_time:.2f}s > {threshold:.2f}s"
```

---

## 13. Profiling Workflow for FinancialMLPipeline

This procedure must be used whenever performance tuning is considered.

**Workflow**:

```
Start → Run full pipeline → cProfile global profiling → 
Identify top bottlenecks → Run focused profiling on heavy functions → 
Implement optimisation → Re-profile to validate performance gain → 
More bottlenecks? → (Yes) → Identify top bottlenecks
                    (No) → Stop optimization
```

**Steps**:

1. **Run full pipeline** with profiling enabled
2. **Global profiling** → Identify top time consumers (cumulative time)
3. **Targeted profiling** → Focus on heavy functions (total time)
4. **Optimize** → Apply optimization (Numba, vectorization, parallelization)
5. **Re-profile** → Validate performance gain
6. **Test** → Ensure correctness not broken
7. **Repeat** → Next bottleneck

**Never optimise below the threshold of significance** (components < 5% of total runtime).

### 13.1 Before/After Comparison

```python
# Compare implementations
def compare_implementations():
    """Compare old vs new implementation."""
    
    # Old implementation
    profiler_old = cProfile.Profile()
    profiler_old.enable()
    result_old = old_implementation(data)
    profiler_old.disable()
    
    # New implementation
    profiler_new = cProfile.Profile()
    profiler_new.enable()
    result_new = new_implementation(data)
    profiler_new.disable()
    
    # Compare
    stats_old = pstats.Stats(profiler_old)
    stats_new = pstats.Stats(profiler_new)
    
    print("Old implementation:")
    stats_old.print_stats(10)
    
    print("\nNew implementation:")
    stats_new.print_stats(10)
    
    # Verify correctness
    assert np.allclose(result_old, result_new)
```

---

## 14. Output Storage & Conventions

Profiling results are saved inside:

```
profiling/
  full_run_<date>.prof
  triple_barrier_<version>.prof
  features_roll_<version>.prof
  notes.md  # Optional: manual insights, changes, improvements
```

**Profiling reports must be versioned to track evolution.**

**Example structure**:

```bash
profiling/
├── 2024-01-15_full_pipeline.prof
├── 2024-01-15_triple_barrier_v1.prof
├── 2024-01-20_triple_barrier_v2_numba.prof
└── notes.md
```

**Configuration**:

```yaml
# configs/runtime/hardware.yaml
runtime:
  profile: true
  profile_output_dir: "profiling"
  profile_versioning: true  # Append date/version to filenames
```

---

## 15. When Profiling Must Be Performed

Profiling is mandatory:

- **after adding a new feature block**
- **after modifying bar/label logic**
- **before hyperparameter scaling**
- **before Monte-Carlo expansion**
- **whenever runtime increases unexpectedly**

**Backtests and ML pipelines scale exponentially → performance must be monitored continuously.**

---

## 16. Memory Profiling

### 10.1 Using memory_profiler

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Code that uses a lot of memory
    large_array = np.zeros((1000000, 100))
    result = process(large_array)
    return result
```

**Run**:

```bash
python -m memory_profiler script.py
```

### 10.2 Memory Optimization

**Common issues**:

1. **Large DataFrame copies**: Use views when possible
2. **Unnecessary data loading**: Load only what's needed
3. **Memory leaks**: Ensure proper cleanup

**Solutions**:

```python
# Use views instead of copies
df_view = df.loc[start:end]  # View, not copy

# Delete large objects when done
del large_array
import gc
gc.collect()

# Use chunked processing
for chunk in pd.read_csv(file, chunksize=10000):
    process(chunk)
```

---

## 17. Continuous Performance Monitoring

### 11.1 Logging Performance Metrics

```python
# src/pipeline/main_pipeline.py
import time

def run_pipeline(cfg: DictConfig) -> None:
    timings = {}
    
    # Time each step
    start = time.time()
    bars = build_bars(ticks, cfg)
    timings['bar_construction'] = time.time() - start
    
    start = time.time()
    features = engineer_features(bars, cfg)
    timings['feature_engineering'] = time.time() - start
    
    # Log to MLflow
    for step, duration in timings.items():
        mlflow.log_metric(f'timing_{step}', duration)
        logger.info(f"{step}: {duration:.2f}s")
```

### 11.2 Performance Regression Tests

```python
# tests/performance/test_performance_regression.py
def test_bar_construction_performance():
    """Ensure bar construction meets performance target."""
    ticks = load_fixture('large_tick_data')
    
    start = time.time()
    bars = build_tick_bars(ticks, tick_size=100)
    elapsed = time.time() - start
    
    # Target: < 10s for 1M ticks
    assert elapsed < 10.0, f"Bar construction too slow: {elapsed:.2f}s"
```

---

## 18. Goal

The objective is not just speed — but **efficient iteration cycles**, enabling:

- faster feature experimentation
- faster research iteration
- scalable ML backtesting
- reduced compute waste
- production-grade ML workflows

**Performance is an asset. Profiling is how it is measured.**

---

## 19. Best Practices

### 12.1 When to Profile

- Before major optimizations (know your baseline)
- After adding new features (check for regressions)
- When performance is a concern
- Periodically to track performance over time

### 12.2 What to Profile

- **Hot paths**: Code executed many times
- **Bottlenecks**: Operations taking significant time
- **User-facing**: Operations affecting user experience
- **Scalability**: Code that must handle large datasets

### 12.3 Profiling Guidelines

- Profile realistic workloads (not tiny test data)
- Profile multiple times (account for variance)
- Profile in production-like environment
- Don't optimize prematurely (measure first)

---

## 20. Tools and Resources

### 13.1 Python Profilers

- **cProfile**: Standard library, good for most cases
- **line_profiler**: Line-by-line profiling
- **memory_profiler**: Memory usage profiling
- **py-spy**: Sampling profiler, low overhead

### 13.2 Visualization Tools

- **snakeviz**: Interactive HTML visualization
- **gprof2dot**: Generate call graph images
- **pycallgraph**: Call graph visualization

### 13.3 Optimization Libraries

- **Numba**: JIT compilation for numerical code
- **Cython**: Compile Python to C
- **NumPy**: Vectorized operations
- **joblib**: Parallel processing

---

## References

- [TESTING.md](TESTING.md) - Performance test guidelines
- [CODING_STANDARDS.md](CODING_STANDARDS.md) - Code optimization practices
- [DETAILED_REVIEW.md](../DETAILED_REVIEW.md) - Performance bottlenecks identified

