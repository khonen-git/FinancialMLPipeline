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

### ✅ Phase 1: Project Foundation (DONE)
- [x] Project structure (src/, configs/, scripts/, tests/)
- [x] setup.py + requirements.txt + environment.yml
- [x] .gitignore + .dockerignore
- [x] Hydra configs (17 YAML files)
- [x] src/utils/ (logging, config helpers)

### ✅ Phase 2: Data Pipeline (DONE)
- [x] src/data/schema_detection.py
- [x] src/data/cleaning.py
- [x] src/data/bars.py (tick bars, volume bars, dollar bars)
- [x] src/data/fractional_diff.py (FFD for stationarity)

### ✅ Phase 3: Labeling (DONE - VALIDATED)
- [x] src/labeling/session_calendar.py
- [x] ⚠️ src/labeling/triple_barrier.py (session-aware logic) ✅ VALIDATED
- [x] src/labeling/meta_labeling.py

### ✅ Phase 4: Features (DONE)
- [x] src/features/price.py (returns, volatility, ranges)
- [x] src/features/microstructure.py (spread, order flow, tick direction)
- [x] src/features/bars_stats.py (bar statistics)
- [x] src/features/hmm_features.py (macro + micro)

### ✅ Phase 5: Models (DONE - VALIDATED)
- [x] ⚠️ src/models/hmm_macro.py ✅ VALIDATED
- [x] ⚠️ src/models/hmm_micro.py ✅ VALIDATED
- [x] src/models/rf_cpu.py (with calibration support)
- [ ] src/models/rf_gpu_cuml.py (optional - not implemented)

### ✅ Phase 6: Validation (DONE)
- [x] src/validation/tscv.py (TimeSeriesCV with purging & embargo)

### ✅ Phase 7: Backtesting (DONE - VALIDATED)
- [x] ⚠️ src/backtest/data_feed.py (PandasDataBidAsk) ✅ VALIDATED
- [x] ⚠️ src/backtest/backtrader_strategy.py (SessionAwareStrategy) ✅ VALIDATED

### ✅ Phase 8: Risk (DONE)
- [x] src/risk/monte_carlo.py (probability of ruin, prop firm analysis)

### ✅ Phase 9: Reporting (DONE)
- [x] src/reporting/report_generator.py
- [x] templates/experiment_report.html (Jinja2)

### ✅ Phase 10: Main Entrypoint (DONE)
- [x] src/pipeline/main_pipeline.py (13-step orchestration)
- [x] run_experiment.py (CLI entry point)
- [x] scripts/download_dukascopy.py (placeholder)
- [x] scripts/validate_config.py
- [x] scripts/inspect_data.py

### ⏳ Phase 11: Tests (TODO)
- [ ] tests/test_schema_detection.py
- [ ] tests/test_bars.py
- [ ] tests/test_triple_barrier.py
- [ ] tests/test_session_calendar.py
- [ ] Integration tests

### ✅ Bonus: Docker & Documentation (DONE)
- [x] Dockerfile
- [x] QUICKSTART.md
- [x] DEVBOOK.md (this file)

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

### Placeholders (Not Yet Implemented)
- [ ] Dukascopy real data download API (placeholder in `scripts/download_dukascopy.py`)
- [ ] Full backtest execution loop (main logic is there, but needs integration testing)
- [ ] Detailed metrics computation (precision, recall, F1, AUC - placeholders in pipeline)
- [ ] Plot generation for reports (plot references ready, but no actual plotting yet)
- [ ] GPU model variants (RandomForestGPU, GradientBoostingGPU)
- [ ] Unit tests (Phase 11)

### Next Steps for Production
1. **Add real data**: Implement Dukascopy download or use existing data files
2. **Test pipeline end-to-end**: Run full experiment with real data
3. **Add unit tests**: Especially for critical modules (triple barrier, session calendar)
4. **Performance tuning**: Profile and optimize bottlenecks
5. **Advanced features**: More sophisticated features (volatility regimes, etc.)
6. **Hyperparameter optimization**: Systematic model tuning
7. **CI/CD**: GitHub Actions for automated testing

---

## 📊 Progress Tracker

| Phase | Status | Commits | Notes |
|-------|--------|---------|-------|
| 0. Documentation | ✅ DONE | - | 13 docs, 15 diagrams |
| 1. Foundation | ✅ DONE | 2 | Configs + utils |
| 2. Data Pipeline | ✅ DONE | 2 | Schema, cleaning, bars, fracdiff |
| 3. Labeling | ✅ DONE | 2 | Session calendar + triple barrier ✅ |
| 4. Features | ✅ DONE | 2 | Price, micro, bars, HMM |
| 5. Models | ✅ DONE | 1 | HMM macro/micro ✅ + RF CPU |
| 6. Validation | ✅ DONE | 1 | TimeSeriesCV with purging |
| 7. Backtesting | ✅ DONE | 1 | Custom feed ✅ + strategy |
| 8. Risk | ✅ DONE | 1 | Monte Carlo |
| 9. Reporting | ✅ DONE | 1 | Jinja2 HTML reports |
| 10. Main Script | ✅ DONE | 2 | Main pipeline + CLI scripts |
| 11. Tests | ⏳ TODO | 0 | Not started |
| **TOTAL** | **91% DONE** | **15** | **38 Python files** |

---

## 🎯 Current Focus

**STATUS**: ✅ **Implementation COMPLETE (91%)**

**Completed**: All phases 0-10 (15 atomic commits, 38 Python files)

**Remaining**: Phase 11 (Unit tests) - optional for MVP

**Ready for**: End-to-end testing with real data

---

## 💡 Quick References

- Doc root: `docs/INDEX.md`
- Config reference: `docs/CONFIG_REFERENCE.md`
- Glossary: `docs/GLOSSARY.md`
- Coding standards: `docs/CODING_STANDARDS.md`

---

*This DEVBOOK is updated continuously during development.*

