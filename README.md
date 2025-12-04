# FinancialMLPipeline

A professional-grade machine learning pipeline for FX trading with session-aware backtesting, regime detection, and comprehensive risk analysis.

Built for raw spread accounts with strict **no overnight positions** constraint.

---

## 🎯 Key Features

- ✅ **Session-Aware Trading**: No overnight positions, Friday early close, weekend gap prevention
- 🎲 **Triple-Barrier Labeling**: Lopez de Prado methodology with realistic bid/ask execution
- 🧠 **Regime Detection**: Dual HMM system (macro market regimes + microstructure states)
- 🌲 **Random Forest Models**: CPU (sklearn) and GPU (RAPIDS cuML) implementations
- 🎯 **Meta-Labeling**: Secondary model for trade filtering and risk management
- 📊 **Walk-Forward Validation**: Time-series CV with purging & embargo
- 📈 **Backtrader Integration**: Session-aware backtesting with realistic execution
- 🎰 **Monte Carlo Risk Analysis**: Bootstrap simulations for prop firm evaluation (FTMO)
- 📋 **MLflow Tracking**: Complete experiment versioning and reproducibility
- 📄 **Automated Reporting**: HTML/PDF reports via Jinja2

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate financial-ml

# Install project
pip install -e .
```

### 2. Prepare Data

```bash
# Download and clean tick data (Dukascopy)
python scripts/prepare_data.py \
  assets=EURUSD \
  data=dukascopy
```

### 3. Run First Experiment

```bash
# Baseline experiment: EURUSD with 1000-tick bars
python scripts/run_experiment.py \
  experiment=base \
  assets=EURUSD \
  models=rf_cpu
```

### 4. View Results

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# Open http://127.0.0.1:5000
```

**📖 Full guide**: [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md)

---

## 📚 Documentation

Complete documentation is available in [`docs/`](docs/):

- **[docs/INDEX.md](docs/INDEX.md)** 📋 - Documentation index and reading order
- **[docs/GLOSSARY.md](docs/GLOSSARY.md)** 📖 - Technical terms and concepts

### Quick Links

| Document | Description |
|----------|-------------|
| [HOW_TO_RUN.md](docs/HOW_TO_RUN.md) | Setup and execution guide |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System overview |
| [CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) | Hydra configuration reference |
| [CODING_STANDARDS.md](docs/CODING_STANDARDS.md) | Code style and AI guardrails |
| [BACKTESTING.md](docs/BACKTESTING.md) | Backtesting architecture |

**Architecture Deep Dive**:
- [ARCH_DATA_PIPELINE.md](docs/ARCH_DATA_PIPELINE.md) - Data flow and labeling
- [ARCH_ML_PIPELINE.md](docs/ARCH_ML_PIPELINE.md) - Model training and regimes
- [ARCH_INFRA.md](docs/ARCH_INFRA.md) - Infrastructure and MLflow
- [ARCH_RISK.md](docs/ARCH_RISK.md) - Risk analysis and Monte Carlo

---

## 🏗️ Project Structure

```
FinancialMLPipeline/
├── configs/              # Hydra YAML configurations
│   ├── config.yaml       # Main entrypoint
│   ├── assets/           # Per-asset configs (EURUSD, USDJPY)
│   ├── session/          # Session calendars
│   ├── labeling/         # Triple barrier configs
│   ├── models/           # Model configs (RF, HMM, meta)
│   └── ...
│
├── src/                  # Source code
│   ├── data/             # Ingestion, schema detection, bar builders
│   ├── features/         # Feature engineering
│   ├── labeling/         # Triple barrier, meta-labeling
│   ├── models/           # RF, HMM, MLflow registry
│   ├── validation/       # CV, purging, embargo, walk-forward
│   ├── backtest/         # Backtrader strategies
│   ├── risk/             # Monte Carlo, drawdown metrics
│   ├── interpretability/ # Feature importance, SHAP
│   ├── reporting/        # Jinja2 reports
│   └── utils/            # Logging, helpers
│
├── scripts/              # CLI entrypoints
│   ├── prepare_data.py   # Data preparation
│   ├── run_experiment.py # Main experiment runner
│   └── ...
│
├── data/                 # Data storage
│   ├── raw/              # Raw tick data
│   ├── clean/            # Cleaned Parquet files
│   └── processed/        # Final datasets with labels
│
├── docs/                 # Documentation
├── tests/                # Unit and integration tests
├── notebooks/            # Exploratory analysis
└── mlruns/               # MLflow tracking data
```

---

## 🎯 Core Design Principles

### 1. No Overnight Positions

All positions are **flat before session end** to avoid:
- Swap fees and rollover costs
- Weekend gap risk
- Overnight volatility exposure

Session logic is enforced in:
- Triple-barrier labeling
- Backtrader strategy
- Risk analysis

### 2. Realistic Execution

- **Long entry**: `askPrice` (you buy at the ask)
- **Long exit**: `bidPrice` (you sell at the bid)
- **Spread always included** (raw spread account)
- No mid-price assumptions

### 3. Data Integrity

- Purging & embargo in cross-validation
- No forward-looking bias
- Session-aware labeling
- Strict time-series validation

---

## 🔬 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Data Processing** | Pandas, Polars, PyArrow (Parquet) |
| **ML Framework** | scikit-learn, RAPIDS cuML (GPU) |
| **Regime Detection** | hmmlearn / pomegranate |
| **Backtesting** | Backtrader |
| **Configuration** | Hydra |
| **Experiment Tracking** | MLflow |
| **Reporting** | Jinja2 (HTML/PDF) |
| **Environment** | Python 3.12+, Conda |

---

## 🤖 AI Assistant Integration

This project is designed to work seamlessly with AI coding assistants.

**Key guardrails**:
- Configuration keys must be documented in [CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md)
- No overnight logic must be preserved
- Bid/ask execution rules are non-negotiable
- See [CODING_STANDARDS.md](docs/CODING_STANDARDS.md) for complete AI guidelines

---

## 📊 Example Results

Each experiment produces:

- ✅ **Metrics**: Sharpe ratio, max drawdown, PnL, win rate
- 📈 **Plots**: Equity curves, drawdown charts, regime diagrams
- 📄 **HTML Report**: Complete experiment summary
- 🎲 **Monte Carlo**: Risk distribution and probability of ruin
- 💾 **Artifacts**: All logged in MLflow for reproducibility

---

## 🔄 Reproducibility

Every experiment logs:

- Git commit hash
- Full Hydra config
- Data version and date range
- Python environment (`requirements.txt`)

**See**: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)

---

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format
black src/ scripts/

# Lint
ruff check src/ scripts/
```

### Adding New Features

1. Read [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md)
2. Update relevant architecture docs
3. Add configuration to [CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md)
4. Write tests
5. Update [GLOSSARY.md](docs/GLOSSARY.md) if needed

---

## 📝 Citation

If you use this pipeline in your research or trading, please reference:

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

---

## 📄 License

[Add your license here]

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Read the [documentation](docs/)
2. Follow [CODING_STANDARDS.md](docs/CODING_STANDARDS.md)
3. Ensure tests pass
4. Update docs if needed

---

## 📧 Contact

[Add contact information]

---

**⚠️ Disclaimer**: This is educational software. Trading involves risk of loss. Past performance does not guarantee future results.

