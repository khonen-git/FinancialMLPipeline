# DEVBOOK - FinancialMLPipeline Implementation

**Started**: 2024-11-28  
**Goal**: Implement complete ML trading pipeline following documentation  
**Language**: English (code + commits)  
**Strategy**: Code everything, pause for validation on critical modules

---

## 📋 Implementation Roadmap

### ✅ Phase 0: Documentation (DONE)
- [x] All markdown docs created and reviewed
- [x] 15 mermaid diagrams
- [x] Config reference complete
- [x] Glossary complete

### 🔄 Phase 1: Project Foundation (IN PROGRESS)
- [ ] Project structure (src/, configs/, scripts/, tests/)
- [ ] setup.py + requirements.txt + environment.yml
- [ ] .gitignore
- [ ] Hydra configs (all YAML files)
- [ ] src/utils/ (logging, config helpers)

### ⏳ Phase 2: Data Pipeline
- [ ] src/data/schema_detection.py
- [ ] src/data/cleaning.py
- [ ] src/data/bars_pandas.py (tick bars, volume bars)
- [ ] src/data/bars_polars.py (optional, faster)
- [ ] scripts/prepare_data.py

### ⏸️ Phase 3: Labeling (CRITICAL - REQUIRES VALIDATION)
- [ ] src/labeling/session_calendar.py
- [ ] ⚠️ src/labeling/triple_barrier.py (session-aware logic)
- [ ] src/labeling/meta_labeling.py
- **STOP HERE FOR VALIDATION**

### ⏳ Phase 4: Features
- [ ] src/features/price.py (returns, volatility, ranges)
- [ ] src/features/microstructure.py (spread, order flow, tick direction)
- [ ] src/features/bars.py (bar statistics)
- [ ] src/features/fracdiff.py (optional)

### ⏸️ Phase 5: Models (REQUIRES VALIDATION FOR FEATURE SELECTION)
- [ ] ⚠️ src/models/hmm_macro.py (need feature list)
- [ ] ⚠️ src/models/hmm_micro.py (need feature list)
- [ ] src/models/rf_cpu.py
- [ ] src/models/rf_gpu_cuml.py (optional)
- [ ] src/models/calibration.py (optional)
- [ ] src/models/registry_mlflow.py
- **ASK ABOUT HMM FEATURES**

### ⏳ Phase 6: Validation
- [ ] src/validation/time_split.py
- [ ] src/validation/purging_embargo.py
- [ ] src/validation/walk_forward.py

### ⏸️ Phase 7: Backtesting (REQUIRES VALIDATION)
- [ ] ⚠️ src/backtest/bt_adapter.py (custom bid/ask feed)
- [ ] ⚠️ src/backtest/bt_strategies.py (session-aware strategy)
- [ ] src/backtest/bt_to_metrics.py
- **VALIDATE CUSTOM FEED STRUCTURE**

### ⏳ Phase 8: Risk
- [ ] src/risk/monte_carlo.py
- [ ] src/risk/drawdown.py

### ⏳ Phase 9: Reporting
- [ ] src/reporting/templates/ (Jinja2 HTML templates)
- [ ] src/reporting/build_report.py
- [ ] src/reporting/export_pdf.py (optional)

### ⏳ Phase 10: Main Entrypoint
- [ ] scripts/run_experiment.py (orchestrates everything)
- [ ] scripts/run_backtest.py
- [ ] scripts/run_risk.py
- [ ] scripts/predict_cli.py

### ⏳ Phase 11: Tests
- [ ] tests/test_schema_detection.py
- [ ] tests/test_bars.py
- [ ] tests/test_triple_barrier.py
- [ ] tests/test_session_calendar.py
- [ ] Integration tests

---

## 📝 Implementation Notes

### Key Design Decisions

#### Data Types
- Use `float64` for prices
- Use `int64` for timestamps (epoch ms)
- Use `pd.Timestamp` for datetime operations

#### Naming Conventions
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

#### Logging
- Use standard Python `logging` module
- Logger per module: `logger = logging.getLogger(__name__)`
- Levels: DEBUG, INFO, WARNING, ERROR

#### Configuration
- All config via Hydra YAML
- Never hardcode parameters
- Document new keys in CONFIG_REFERENCE.md

---

## ❓ Questions / Validation Points

### 1. Triple Barrier Session-Aware ✅ VALIDATED
**File**: `src/labeling/triple_barrier.py`

**Decisions**:
- [x] ✅ Logic: `effective_horizon_bars = min(max_horizon_bars, bars_until_session_end)`
- [x] ✅ Edge case: If `effective_horizon_bars < min_horizon_bars` → **SKIP event (no label)**
- [x] ✅ Friday: Same logic, managed by `SessionCalendar` (uses `friday_end` internally)
- [x] ✅ Horizon in **bars** (not seconds)

**Implementation**:
```python
def compute_triple_barrier(
    events: pd.DataFrame,
    prices: pd.DataFrame,  # bid/ask
    tp_ticks: int,
    sl_ticks: int,
    max_horizon_bars: int,
    session_calendar: SessionCalendar,
    min_horizon_bars: int = 10
) -> pd.DataFrame:
    # For each event at t0:
    # 1. bars_until_session_end = session_calendar.bars_until_session_end(t0)
    # 2. effective_horizon_bars = min(max_horizon_bars, bars_until_session_end)
    # 3. If effective_horizon_bars < min_horizon_bars:
    #       skip event (no label, or NaN)
    # 4. Scan forward up to effective_horizon_bars for TP/SL/time barrier
    # 5. Exit always at BID
    pass
```

**Key principle**: Triple barrier must be **perfectly aligned** with Backtrader strategy (no overnight, no-trade zone).

---

### 2. HMM Feature Selection ✅ VALIDATED
**Files**: `src/models/hmm_macro.py`, `src/models/hmm_micro.py`

**Macro HMM features** (slow market regimes):
- [x] ✅ `ret_long` - log-return on longer window (e.g., 50 bars)
- [x] ✅ `vol_long` - rolling std of returns (e.g., 50 bars)
- [x] ✅ `trend_slope` - linear regression slope of price
- [x] ✅ `trend_strength` - R² or |slope|/vol
- [x] Optional: `vol_of_vol`, `drawdown_recent`

**Micro HMM features** (order flow regimes):
- [x] ✅ `of_imbalance` - order flow imbalance (buy_vol - sell_vol) / total
- [x] ✅ `spread` - ask - bid
- [x] ✅ `spread_change` - spread - spread.shift(1)
- [x] ✅ `tick_direction` - signed (+1 up, -1 down, 0 flat)
- [x] Optional: `quote_update_freq`, `signed_volume`

**Config structure**:
```yaml
hmm:
  macro:
    enabled: true
    features:
      - "ret_long"
      - "vol_long"
      - "trend_slope"
      - "trend_strength"
  micro:
    enabled: true
    features:
      - "of_imbalance"
      - "spread"
      - "spread_change"
      - "tick_direction"
```

**Implementation**: HMM code takes `df[feature_list]`, applies StandardScaler, fits HMM(n_states), returns regime.

---

### 3. Backtrader Custom Feed ✅ VALIDATED
**File**: `src/backtest/bt_adapter.py`

**Decision**: 
- [x] ✅ Standard OHLC = **BID** prices
- [x] ✅ Extra lines for **ASK** (ask_open, ask_high, ask_low, ask_close)
- [x] ❌ NO redundant bid_* lines (OHLC already = bid)

**Implementation**:
```python
class PandasDataBidAsk(bt.feeds.PandasData):
    """
    Custom PandasData feed with:
    - standard OHLC = BID prices (for exits, SL/TP)
    - extra ASK OHLC lines (for entries)
    """
    lines = ('ask_open', 'ask_high', 'ask_low', 'ask_close',)
    
    params = (
        ('datetime', None),
        ('open', 'bid_open'),      # standard = BID
        ('high', 'bid_high'),
        ('low', 'bid_low'),
        ('close', 'bid_close'),
        ('volume', 'volume'),
        ('openinterest', -1),
        
        # Extra ASK fields
        ('ask_open', 'ask_open'),
        ('ask_high', 'ask_high'),
        ('ask_low', 'ask_low'),
        ('ask_close', 'ask_close'),
    )
```

**Strategy usage**:
- Entry: `self.data.ask_close` (buy at ask)
- Exit: `self.data.close` (sell at bid)
- SL/TP checks: use `self.data.high/low` (bid prices)

---

## 🐛 Known Issues / TODOs

- [ ] None yet (clean start)

---

## 📊 Progress Tracker

| Phase | Status | Commits | Notes |
|-------|--------|---------|-------|
| 0. Documentation | ✅ DONE | - | 13 docs, 15 diagrams |
| 1. Foundation | 🔄 IN PROGRESS | 0 | Starting now |
| 2. Data Pipeline | ⏳ TODO | 0 | |
| 3. Labeling | ⏸️ BLOCKED | 0 | Need validation |
| 4. Features | ⏳ TODO | 0 | |
| 5. Models | ⏸️ BLOCKED | 0 | Need feature lists |
| 6. Validation | ⏳ TODO | 0 | |
| 7. Backtesting | ⏸️ BLOCKED | 0 | Need validation |
| 8. Risk | ⏳ TODO | 0 | |
| 9. Reporting | ⏳ TODO | 0 | |
| 10. Main Script | ⏳ TODO | 0 | |
| 11. Tests | ⏳ TODO | 0 | |

---

## 🎯 Current Focus

**NOW**: Phase 1 - Project Foundation

**NEXT VALIDATION POINT**: Triple Barrier implementation

---

## 💡 Quick References

- Doc root: `docs/INDEX.md`
- Config reference: `docs/CONFIG_REFERENCE.md`
- Glossary: `docs/GLOSSARY.md`
- Coding standards: `docs/CODING_STANDARDS.md`

---

*This DEVBOOK is updated continuously during development.*

