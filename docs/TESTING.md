# Testing & Validation Strategy

This document describes the testing logic used in the FinancialMLPipeline project.

It defines test types, code organization, methodological rules, and how to integrate test experiments (complete configurations) to validate the pipeline end-to-end.

The goal is to ensure:

- reliability of internal modules (bar construction, labeling, feature engineering...)
- stability of the financial engine (no silent regressions after modifications)
- ability to quickly test a new strategy/config
- explicit performance measurement over time
- scientific integrity of the pipeline (no information leakage)

---

## 1. Test Types

The project uses five complementary test categories:

| Type | Objective | Granularity | Location |
|------|-----------|-------------|----------|
| **Unit Tests** | Verify an isolated function | Very fine | `tests/unit/` |
| **Integration Tests** | Test 2+ modules together | Medium | `tests/integration/` |
| **Regression Tests** | Prevent degradation of results | Global/numerical | `tests/regression/` |
| **End-to-End / Experiment Tests** | Verify complete workflow via config | Global operation | `tests/e2e/` |
| **Performance Tests** | Control computation time & scalability | Machine-dependent | `tests/performance/` |

Each module must have at least one unit test.

Complete pipelines must be validated by an e2e test and, when relevant, a regression test.

---

## 2. Unit Tests

Unit tests validate the precise behavior of a function, with small synthetic data.

**Examples**:

- `ticks_to_bars_100t()` returns a correct DataFrame
- `apply_triple_barrier()` generates labels `{-1, 0, 1}` correctly
- `fracdiff()` respects index and does not introduce future leakage

**Rules**:

- One test → one idea → one clear assertion
- No network access, no CPU-heavy operations
- No dependency on test execution order
- Failure = bug or explicit logical change

**Recommended structure**:

```
tests/unit/
  data/
    test_resampling.py
    test_bars.py
  labeling/
    test_triple_barrier.py
    test_session_calendar.py
  features/
    test_fracdiff.py
    test_price_features.py
    test_microstructure_features.py
  models/
    test_training.py
    test_hmm.py
  validation/
    test_cpcv.py
    test_tscv.py
```

**Example**:

```python
# tests/unit/labeling/test_triple_barrier.py
import pytest
import pandas as pd
from src.labeling.triple_barrier import compute_triple_barrier

def test_tp_hit_before_sl():
    """TP hit before SL → label = 1, barrier_hit = 'tp'."""
    # Arrange
    dates = pd.date_range('2024-01-09 10:00', periods=10, freq='5min', tz='UTC')
    bars = create_sample_bars(dates, ...)
    events = create_sample_events(dates[0])
    
    # Act
    labels = compute_triple_barrier(events, bars, ...)
    
    # Assert
    assert labels.iloc[0]['label'] == 1
    assert labels.iloc[0]['barrier_hit'] == 'tp'
```

---

## 3. Integration Tests

Integration tests validate consistency between modules:

**Examples**:

- Tick → Bars → Features works without breaking data shape
- Labeling + session calendar gives a coherent result
- Assembling features + labels produces a complete dataset

They focus on:

- ✔ chronological consistency
- ✔ absence of information leakage
- ✔ abusive config validation

These tests use small datasets (internal fixtures) to stay fast.

**Example**:

```python
# tests/integration/test_data_pipeline.py
def test_tick_to_features_pipeline():
    """Test complete data pipeline: ticks → bars → features."""
    # Load sample ticks
    ticks = load_fixture('small_tick_data')
    
    # Build bars
    bars = build_tick_bars(ticks, tick_size=100)
    assert len(bars) > 0
    assert 'bid_close' in bars.columns
    
    # Build features
    features = engineer_features(bars, cfg)
    assert len(features) == len(bars)
    assert not features.empty
    
    # Verify no future leakage
    assert_no_forward_looking_operations(features)
```

---

## 4. Regression Tests

Regression tests detect when a modification degrades a previously validated result.

They are based on a **Golden Benchmark**, for example:

| Metric | Threshold |
|--------|-----------|
| Sharpe Ratio min | > 1.2 |
| Max Drawdown | > -7% |
| % label=1 vs -1 | Stable ratio (±5%) |
| Resampling speed | Stable ±20% |

**Objective**:

If the result drops significantly → alert → revision necessary

This is not about checking binary exactness, but an acceptable range.

**Examples**:

- If adding a new feature changes Sharpe → OK if +, KO if - too strong
- If code modification divides backtest time by 3 → good to keep
- If system becomes 10× slower → degradations blocked

Regressions are critical in a financial system, as a small leak or logic change can invalidate an entire strategy.

**Example**:

```python
# tests/regression/test_performance_metrics.py
def test_sharpe_ratio_regression():
    """Ensure Sharpe ratio does not degrade below threshold."""
    # Run experiment with golden config
    results = run_experiment('configs/experiment/golden_benchmark.yaml')
    
    sharpe = results['backtest']['sharpe_ratio']
    assert sharpe > 1.2, f"Sharpe ratio {sharpe} below threshold 1.2"
    
def test_label_distribution_stability():
    """Ensure label distribution remains stable."""
    results = run_experiment('configs/experiment/golden_benchmark.yaml')
    
    label_counts = results['labeling']['label_counts']
    ratio = label_counts[1] / label_counts[-1]
    
    # Should be within ±5% of expected ratio
    expected_ratio = 0.5
    assert abs(ratio - expected_ratio) < 0.05, \
        f"Label ratio {ratio} deviates from expected {expected_ratio}"
```

---

## 5. End-to-End / Experiment Tests

They use YAML config files to reconstruct a mini complete experiment, from raw ticks to final report.

These tests serve to validate the production workflow:

```
ticks → cleaning → session filtering → bars → features → 
labeling → ML → backtest → metrics export
```

Each E2E experiment must be:

- reproducible
- isolated
- autonomous
- linked to a versioned config

**Recommended structure**:

```
tests/e2e/
  configs/
    exp_sanity.yaml
    exp_small_eurusd.yaml
    exp_feature_ablation.yaml
  test_e2e_sanity.py
  test_e2e_small_eurusd.py
  test_e2e_feature_ablation.py
```

**Example**:

```python
# tests/e2e/test_e2e_small_eurusd.py
import pytest
from pathlib import Path
from src.pipeline.main_pipeline import run_pipeline
import hydra
from omegaconf import DictConfig

@pytest.mark.e2e
def test_e2e_small_eurusd():
    """Test complete pipeline with small EURUSD dataset."""
    # Load test config
    config_path = Path(__file__).parent / 'configs' / 'exp_small_eurusd.yaml'
    
    # Run pipeline
    with hydra.initialize(config_path=str(config_path.parent), version_base=None):
        cfg = hydra.compose(config_name='exp_small_eurusd')
        run_pipeline(cfg)
    
    # Verify outputs
    assert Path('mlruns').exists()
    assert Path('outputs/reports').exists()
    
    # Verify metrics logged
    import mlflow
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=[...])
    assert len(runs) > 0
    assert runs[0].data.metrics.get('backtest_sharpe_ratio') is not None
```

---

## 6. Performance Tests

Performance tests evaluate:

- bar resampling speed
- feature engineering time
- backtesting cost
- scaling with volume (100k → 1M ticks)

They should not block development.

They are separated because often long or machine-dependent.

```
tests/performance/
  test_perf_resampling.py
  test_perf_backtest.py
  test_perf_feature_engineering.py
```

**Guideline**:

| Objective | Method |
|-----------|--------|
| Decrease latency | timer + threshold |
| Track perf over time | log versioned metrics |
| Compare implementations | bench side-by-side |

If integrated to CI → opt-in mode via marker:

```bash
pytest -m "perf"
```

**Example**:

```python
# tests/performance/test_perf_resampling.py
import pytest
import time
from src.data.bars import BarBuilder

@pytest.mark.perf
def test_bar_resampling_performance():
    """Ensure bar resampling completes within time threshold."""
    ticks = load_fixture('large_tick_data')  # 1M ticks
    
    start = time.time()
    bars = BarBuilder(cfg).build_tick_bars(ticks, tick_size=100)
    elapsed = time.time() - start
    
    # Should complete in < 10 seconds for 1M ticks
    assert elapsed < 10.0, f"Bar resampling took {elapsed:.2f}s, threshold: 10s"
    assert len(bars) > 0
```

---

## 7. Fixtures & Synthetic Data

A `conftest.py` file centralizes fixtures:

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def small_tick_data():
    """Small synthetic tick dataset for unit tests."""
    dates = pd.date_range('2024-01-01 00:00', periods=1000, freq='1s', tz='UTC')
    return pd.DataFrame({
        'timestamp': dates,
        'bidPrice': 1.1000 + np.random.randn(1000) * 0.0001,
        'askPrice': 1.1005 + np.random.randn(1000) * 0.0001,
        'bidVolume': np.random.uniform(0.5, 1.5, 1000),
        'askVolume': np.random.uniform(0.5, 1.5, 1000),
    })

@pytest.fixture
def sample_bar_data(small_tick_data):
    """Sample bars built from small_tick_data."""
    from src.data.bars import BarBuilder
    builder = BarBuilder({'type': 'tick', 'threshold': 100})
    return builder.build_tick_bars(small_tick_data, tick_size=100)

@pytest.fixture
def large_tick_data():
    """Large dataset for performance tests."""
    # Generate 1M ticks
    pass

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for test isolation."""
    return tmp_path / 'outputs'
```

**Benefits**:

- 🔁 reusability, reproducibility, clear config

Real data should not be versioned except minimal samples.

---

## 8. Quality Gates & CI Recommendations

To secure the pipeline:

| Level | Rule | CI Action |
|-------|------|-----------|
| **Lint & Typecheck** | Clean code | `ruff`, `mypy` |
| **Unit Tests** | 100% of critical modules | Block merge |
| **Integration** | 1 scenario min | Block merge |
| **Regression** | Thresholds validated | Warning or block if failure |
| **Perf** | Non-regressive thresholds | Manual or periodic execution |

**Recommended pipeline**:

```
Commit → Lint → Unit → Integration → Regression → Perf → Build
```

**Example CI configuration** (`.github/workflows/tests.yml`):

```yaml
name: Tests

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install ruff mypy
      - run: ruff check src/
      - run: mypy src/

  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -e .
      - run: pytest tests/unit/ -v

  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -e .
      - run: pytest tests/integration/ -v

  regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -e .
      - run: pytest tests/regression/ -v
      # Allow failure but report

  performance:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'  # Run periodically
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -e .
      - run: pytest tests/performance/ -m perf -v
```

---

## 9. Best Practices

**Fast tests** → short feedback (<2s for unit)

**Explicit tests** → clear failure message

**Deterministic tests** → fixed seed

**Financial tests** → never ignore future leaks

**E2E tests** → versioned configs

**Priority**: consistency + reproducibility + non-regression

---

## 10. Running Tests

### 10.1 All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

### 10.2 With Markers

```bash
# Run only unit tests
pytest -m "not e2e and not perf"

# Run only performance tests
pytest -m "perf"

# Run only e2e tests
pytest -m "e2e"
```

### 10.3 Verbose Output

```bash
# Verbose output
pytest -v

# Very verbose (show print statements)
pytest -vv -s

# Show local variables on failure
pytest -l
```

---

## 11. Writing New Tests

### 11.1 Unit Test Template

```python
# tests/unit/<module>/test_<function>.py
import pytest
from src.<module>.<function> import function_to_test

def test_function_behavior():
    """Test description."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result is not None
    assert len(result) == expected_length
    assert result['column'].dtype == expected_dtype
```

### 11.2 Integration Test Template

```python
# tests/integration/test_<pipeline>.py
import pytest

def test_pipeline_integration(fixture1, fixture2):
    """Test integration between modules."""
    # Test module A
    result_a = module_a.process(fixture1)
    
    # Test module B with A's output
    result_b = module_b.process(result_a)
    
    # Verify consistency
    assert result_b is not None
    assert_no_data_leakage(result_b)
```

### 11.3 E2E Test Template

```python
# tests/e2e/test_<experiment>.py
import pytest
from pathlib import Path
import hydra

@pytest.mark.e2e
def test_experiment_name():
    """Test complete experiment workflow."""
    config_path = Path(__file__).parent / 'configs' / 'exp_name.yaml'
    
    with hydra.initialize(config_path=str(config_path.parent), version_base=None):
        cfg = hydra.compose(config_name='exp_name')
        run_pipeline(cfg)
    
    # Verify outputs
    verify_mlflow_artifacts()
    verify_report_generated()
    verify_metrics_logged()
```

---

## 12. Data Leakage Prevention in Tests

🧠 **Note anti-leakage for AI & future contributors**:

Any modification of pipeline behavior must be justified, documented, and validated against regression tests.

No optimization should introduce lookahead bias, data leakage, or silent modification of trading logic.

**Test helpers for leakage detection**:

```python
# tests/utils/leakage_helpers.py
def assert_no_forward_looking_operations(features):
    """Assert no forward-looking operations in features."""
    # Check for negative shifts
    # Check for future data access
    # Check for improper rolling windows
    pass

def assert_temporal_consistency(features, labels):
    """Assert features and labels are temporally consistent."""
    # Check indices match
    # Check no future features used for past labels
    pass
```

---

## 13. Test Data Management

### 13.1 Synthetic Data

- Use `pytest.fixture` for reusable synthetic data
- Keep fixtures small for fast tests
- Use deterministic random seeds

### 13.2 Real Data Samples

- Store minimal samples in `tests/fixtures/data/`
- Never commit large datasets
- Document data source and date range

### 13.3 Data Cleanup

- Use `tmp_path` fixture for temporary files
- Clean up after tests
- Isolate test outputs

---

## 14. Continuous Integration

See [ARCH_INFRA.md §5](ARCH_INFRA.md#5-continuous-integration) for CI/CD setup details.

**Key points**:

- All tests must pass before merge
- Regression tests can warn but not block (for now)
- Performance tests run periodically, not on every commit
- Coverage reports generated and tracked

---

## 15. Troubleshooting

### 15.1 Tests Failing Locally

- Check environment matches `environment.yaml`
- Verify test data fixtures exist
- Check for file path issues (use `Path` objects)

### 15.2 Flaky Tests

- Ensure deterministic random seeds
- Avoid time-dependent logic
- Isolate test state

### 15.3 Slow Tests

- Use markers to skip slow tests: `pytest -m "not slow"`
- Profile slow tests to identify bottlenecks
- Consider moving to performance test suite

---

## References

- [CODING_STANDARDS.md](CODING_STANDARDS.md) - Code style and structure
- [DATA_HANDLING.md](DATA_HANDLING.md) - Data leakage prevention
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) - Reproducibility guidelines
- [ARCH_INFRA.md](ARCH_INFRA.md) - CI/CD infrastructure

