# Glossary

Technical terms and concepts used throughout the FinancialMLPipeline documentation.

---

## A

### Ask Price
The price at which you **buy** (go long). Always higher than bid price.  
Example: If ask = 1.12001, you pay 1.12001 to enter a long position.

### Asset
A tradeable instrument (e.g., EURUSD, USDJPY). Each asset has its own config file with `tick_size` and `contract_size`.

---

## B

### Bar
An aggregated price candle built from multiple ticks. Types include:
- **Tick bars**: aggregated every N ticks
- **Volume bars**: aggregated every N units of volume
- **Dollar bars**: aggregated every N notional units

### Bid Price
The price at which you **sell** (exit long or enter short). Always lower than ask price.  
Example: If bid = 1.11998, you receive 1.11998 when closing a long position.

### Backtest
Simulation of a trading strategy on historical data. In this project, handled by **Backtrader** with session-aware logic.

---

## C

### Calibration
Post-processing of model probabilities to ensure they match empirical frequencies. Optional for Random Forest models.

### Commission
Trading cost per lot. In raw spread accounts, commission = 0 because the spread already covers costs.

### Contract Size
Nominal size of 1 lot for an asset (e.g., 100,000 for EURUSD). Used to compute notional exposure.

### cuML / cuDF
GPU-accelerated libraries from RAPIDS for machine learning (cuML) and dataframes (cuDF). Used for GPU-based Random Forest.

---

## D

### Dollar Bars
Bars aggregated when cumulative notional value (price × volume) reaches a threshold. Optional in this project.

### Drawdown (DD)
Loss from peak equity. **Max drawdown** = largest peak-to-trough loss. Critical for prop firm evaluation.

### Dukascopy
FX tick data provider. Data is ingested via `dukascopy-node`.

---

## E

### Embargo
Time gap between train and test windows in cross-validation. Prevents leakage from overlapping label horizons.

### Equity Curve
Plot of account balance over time. Used to visualize backtest performance.

### Event
A point in time where a trade signal is generated and a triple-barrier label is computed.

---

## F

### Fractional Differencing
Technique to make a time series stationary while preserving memory. Optional feature transformation.

### Feature Engineering
Creating predictive variables from raw data (e.g., log returns, rolling volatility, spread changes).

### FTMO
Prop firm (proprietary trading firm) that provides funded accounts under strict rules (daily loss limits, max drawdown).

---

## G

### Gap Risk
Risk of price jumping overnight or over weekends. Eliminated in this project via **no overnight positions**.

---

## H

### Heikin-Ashi
Smoothed candlestick type. Computed as transformation of OHLC bars. Used only for features, never for labeling.

### HMM (Hidden Markov Model)
Probabilistic model used to detect latent market regimes:
- **Macro HMM**: slow-moving regimes (trending, volatile, range-bound)
- **Micro HMM**: order flow regimes (buyer aggression, spread regime)

### Hydra
Configuration framework by Facebook. Used to manage YAML configs with composition and overrides.

---

## I

### Inference
Applying a trained model to new data to generate predictions. Handled via `scripts/predict_cli.py`.

---

## L

### Label
Target variable for supervised learning. In this project:
- `+1` = take profit hit first
- `-1` = stop loss hit first
- `0` = time barrier hit (neutral)

### Leakage (Data Leakage)
Using future information during training. Strictly prevented via purging, embargo, and session-aware logic.

### Lopez de Prado
Author of *Advances in Financial Machine Learning*. Source of triple-barrier labeling and meta-labeling methods.

### Long Position
Buying an asset expecting price to rise. Entry at **ask**, exit at **bid**.

---

## M

### Meta-Labeling
Secondary model that decides whether to take a trade suggested by the base model:
- `meta = 1` → take trade
- `meta = 0` → skip trade

Improves risk-adjusted performance.

### Mid-Price
Average of bid and ask: `(bid + ask) / 2`.  
**Not used for execution** in this project (realistic bid/ask execution only).

### MLflow
Experiment tracking platform. Logs parameters, metrics, artifacts, and models for each run.

### Monte Carlo
Simulation technique. In this project: bootstrap trade sequences to estimate distribution of equity curves, drawdowns, and probability of ruin.

---

## N

### No Overnight Constraint
Hard rule: all positions must be **flat before session end**. Avoids swap fees, gap risk, and weekend exposure.

---

## O

### OHLC
Open, High, Low, Close prices for a bar.

### Order Flow
Direction and intensity of market orders. Used in microstructure features.

---

## P

### Parquet
Columnar storage format (Apache Arrow). Used for fast, efficient data storage after cleaning.

### Pip
Common term in FX for smallest price movement (e.g., 0.0001 for EURUSD).  
**In this project, we use "ticks"** to avoid confusion with tick_size.

### Prop Firm
Proprietary trading firm that funds traders under strict risk rules (e.g., FTMO).

### Purging
Removing overlapping labels from train/test sets to prevent leakage. Essential for time-series CV.

---

## R

### Regime
Latent market state detected by HMM (e.g., high volatility, trending up, buyer aggression).

### Rollover
Broker's daily cutoff time (typically 22:00 or 23:00 UTC). Positions held past rollover incur swap fees.

### Raw Spread Account
Account where commission = 0 but spread is wide. All execution costs are in the bid/ask spread.

---

## S

### Session
Trading period between `session_start` and `session_end`. Positions must be flat before `session_end`.

### Session Calendar
Configuration defining session hours, Friday early close, and weekend trading rules.

### SHAP
Explainability method for machine learning models. Optional in this project due to computational cost.

### Short Position
Selling an asset expecting price to fall. Entry at **bid**, exit at **ask**.  
**Not implemented in v1** (long-only).

### Slippage
Difference between expected and actual execution price. Set to 0 by default (deterministic backtests).

### Spread
Difference between ask and bid: `spread = ask - bid`. Trading cost in raw spread accounts.

### Stop Loss (SL)
Price level at which a losing position is closed to limit loss.  
In triple barrier: lower barrier for longs.

---

## T

### Take Profit (TP)
Price level at which a winning position is closed to lock in profit.  
In triple barrier: upper barrier for longs.

### Tick
Single price update from the market. Contains timestamp, bid, ask, and optionally volumes.

### Tick Bars
Bars aggregated every N ticks (e.g., 100-tick bars, 1000-tick bars). Default bar type.

### Tick Size
Minimum price increment for an asset (e.g., 0.00001 for EURUSD, 0.001 for USDJPY).  
**Do not confuse with "pip".**

### Time Barrier
Maximum holding period for a trade. In this project: capped by `time_until_session_end` (session-aware).

### Triple Barrier
Labeling method with three exit conditions:
1. **Upper barrier** (take profit)
2. **Lower barrier** (stop loss)
3. **Time barrier** (max holding period)

First barrier hit determines the label.

---

## V

### Volume Bars
Bars aggregated when cumulative volume reaches a threshold. Only used if volume data is meaningful (not synthetic).

---

## W

### Walk-Forward Validation
Sequential train/test procedure simulating real-world deployment:
- Train on window 1 → Test on window 2
- Train on window 2 → Test on window 3
- etc.

### Weekend Trading
In FX, markets are closed Saturday/Sunday. This project enforces `weekend_trading: false`.

---

## Related Documents

- **[ARCH_DATA_PIPELINE.md](ARCH_DATA_PIPELINE.md)** - Data flow and labeling
- **[BACKTESTING.md](BACKTESTING.md)** - Execution and session logic
- **[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)** - Configuration parameters
- **[ARCH_ML_PIPELINE.md](ARCH_ML_PIPELINE.md)** - Models and regimes

