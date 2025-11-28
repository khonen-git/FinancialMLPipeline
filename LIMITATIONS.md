# ⚠️ Known Limitations

This document describes the known limitations of the current version (v1.0) of the trading ML pipeline.

## 🎯 Model Performance

### High Accuracy (~98%)

**Issue**: The model achieves unusually high accuracy (96-100% across folds), which is **not realistic** for production trading.

**Possible Causes**:
1. **Features too informative**: Microstructure features (spread, order flow) may leak future information
2. **Problem setup**: 10-pip TP/SL with 50-bar horizon may be too easy (EUR/USD typically moves >10 pips in 50 bars)
3. **Model complexity**: Despite regularization, Random Forest may memorize patterns
4. **Data characteristics**: Test period may be too similar to training period

**Impact**: This model is **NOT ready for production trading** without significant adjustments.

**Recommendations**:
- Increase TP/SL targets to 20-30 pips
- Reduce max_horizon to 20-30 bars
- Increase regularization: `min_samples_leaf=50-100`, `max_depth=3-5`
- Simplify feature set (keep only 10-15 most important)
- Test on longer out-of-sample periods (2+ years)

### Overfitting vs Data Leakage

**Verification Done**: 
- ✅ Purging + embargo implemented (868-1358 samples purged per fold)
- ✅ No technical data leakage detected
- ⚠️ Overfitting persists despite purging

**Conclusion**: The high accuracy is due to **model overfitting**, not data leakage.

## 📊 Data Quality

### Missing Volume Data

**Issue**: Dukascopy FX data doesn't include true traded volume (only tick count as proxy).

**Impact**:
- Volume-based bars cannot be constructed properly
- Volume imbalance features are all NaN (automatically dropped)
- Order flow metrics are approximations

**Workaround**: Use tick-based bars instead of volume-based bars.

**Solution**: Use futures data (e.g., CME) or centralized exchange data for true volume.

### Single Instrument

**Current Scope**: Only EUR/USD tested (2023-2024 data).

**Limitation**: Model is **not validated** on:
- Other currency pairs (GBP/USD, USD/JPY, etc.)
- Other asset classes (stocks, crypto, commodities)
- Different market regimes (crisis, low volatility, etc.)

**Recommendation**: Test on multiple instruments and time periods before production use.

## 🔄 Backtesting

### Simplified Execution

**Current Implementation**:
- No slippage modeling
- Zero transaction costs (raw spread account assumed)
- Instant fills at bid/ask
- No order book depth simulation

**Reality**: Real execution involves:
- Variable slippage (especially during news events)
- Commission costs (even on spread accounts)
- Partial fills on large orders
- Spread widening during low liquidity

**Impact**: Backtest results are **optimistic** compared to live trading.

### Monte Carlo Limitations

**Current Implementation**:
- Bootstrap resampling of trades
- No sequential dependencies modeled
- No regime changes simulated

**Missing**:
- Time-based simulation (sequence of market conditions)
- Correlation between trades
- Market impact over time

## 🧠 Model Architecture

### Single Model

**Current**: Random Forest only (CPU-based).

**Limitations**:
- No ensemble of different model types
- No online learning (model is static after training)
- No adaptive behavior based on performance

**Recommendation**: Implement model stacking and periodic re-training.

### HMM Regime Detection

**Current Status**: Implemented but **not significantly improving** performance (98.73% without HMM, 98.65% with HMM).

**Possible Reasons**:
- Overfitting dominates any regime detection benefit
- Only 3 states may be too simplistic
- Features for HMM may not capture regime transitions well

**Recommendation**: Re-evaluate HMM utility after reducing overfitting.

## 🎨 Features

### Limited Feature Diversity

**Current Features** (24 base):
- Price-based: returns, volatility, ranges (7)
- Microstructure: spreads, order flow (14)
- Bar stats: volume proxies, tick count (3)

**Missing**:
- Alternative data (sentiment, news)
- Macroeconomic indicators
- Cross-asset correlations
- Order book features (depth, imbalance)

### Feature Engineering

**No automated feature selection**: All 24 features are used without validation of importance.

**No feature interaction**: Higher-order features (products, ratios) not explored.

**Recommendation**: Implement feature importance analysis and selection.

## 🔧 Technical Debt

### No Hyperparameter Optimization

**Current**: Hyperparameters are manually set (not optimized).

**Missing**:
- Grid search / Random search
- Bayesian optimization
- Cross-validated tuning

**Impact**: Model may be suboptimal.

### Limited Hardware Utilization

**Current**: CPU-only implementation.

**Missing**:
- GPU acceleration (cuML, PyTorch)
- Distributed training (Dask, Ray)
- Multi-core parallelization

**Impact**: Slow training on large datasets.

## 📈 Production Readiness

### No Live Trading Infrastructure

**Current**: Batch processing only (offline training and backtesting).

**Missing for Production**:
- Real-time data ingestion
- Real-time feature computation
- Low-latency inference
- Order management system
- Risk monitoring and alerts
- Model versioning and A/B testing

### No Monitoring

**Missing**:
- Model drift detection
- Performance degradation alerts
- Feature distribution monitoring
- Anomaly detection

## 📝 Documentation

### Benchmarking

**Status**: ⚠️ **NOT IMPLEMENTED**

**Missing**:
- Comparison to baseline strategies (buy-and-hold, MA crossover, etc.)
- Statistical significance tests
- Economic vs statistical significance analysis
- Comparison to published research results

**Recommendation**: Implement benchmarking framework (see `FUTURE_ENHANCEMENTS.md`).

### Limited Examples

**Current**: One main experiment configuration.

**Missing**:
- Multiple asset examples
- Different timeframes (intraday, daily, weekly)
- Various strategy types (trend-following, mean-reversion, etc.)
- Jupyter notebook tutorials

## 🎯 Scope Limitations

### Educational / Demo Purpose

**Important**: This pipeline is designed as a **demonstration and learning tool**, **NOT** a production-ready trading system.

**Before Live Trading**, you must:
1. Reduce overfitting significantly (target 55-65% accuracy)
2. Validate on multiple years of out-of-sample data
3. Implement realistic execution costs and slippage
4. Add comprehensive risk management
5. Implement real-time infrastructure
6. Conduct extensive paper trading (3-6 months minimum)
7. Start with small position sizes
8. Monitor performance continuously

### Risk Warning

⚠️ **RISK DISCLAIMER**: 

Trading financial instruments carries a high level of risk. The models and strategies in this repository are **FOR EDUCATIONAL PURPOSES ONLY**. 

- Past performance does not guarantee future results
- The 98% accuracy is **NOT REALISTIC** for live trading
- You may lose more than your initial investment
- This software is provided "AS IS" without warranty

**DO NOT USE THIS IN PRODUCTION WITHOUT SIGNIFICANT MODIFICATIONS AND TESTING.**

---

## 📞 Contributing

Found other limitations? Please open an issue or submit a pull request!

See `FUTURE_ENHANCEMENTS.md` for planned improvements.

