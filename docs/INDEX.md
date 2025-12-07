# Documentation Index

This directory contains the complete documentation for the trading machine learning pipeline.

The project implements a session-aware, raw-spread FX trading system with:
- Triple-barrier labeling (Lopez de Prado)
- HMM regime detection (macro + microstructure)
- Random Forest models (CPU/GPU)
- Meta-labeling for trade filtering
- Walk-forward validation with purging & embargo
- Backtrader backtesting
- Monte Carlo risk analysis
- MLflow experiment tracking

---

## 🚀 Quick Start

**New to the project?** Read in this order:

1. 📖 **[HOW_TO_RUN.md](HOW_TO_RUN.md)** - Setup and execution guide
2. 🏗️ **[ARCHITECTURE.md](ARCHITECTURE.md)** - High-level system overview
3. 📊 **[ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)** - Visual architecture diagrams (Mermaid)
4. ⚙️ **[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)** - Hydra configuration reference
5. 📚 **[GLOSSARY.md](GLOSSARY.md)** - Key terms and concepts

---

## 📚 Architecture Documentation (Read in Order)

### Core Pipeline

1. **[ARCH_DATA_PIPELINE.md](ARCH_DATA_PIPELINE.md)**
   - Raw tick data ingestion (Dukascopy)
   - Bar construction (tick, volume, dollar bars)
   - Feature engineering
   - Triple-barrier labeling with session-aware logic

2. **[ARCH_ML_PIPELINE.md](ARCH_ML_PIPELINE.md)**
   - HMM regime detection (macro + microstructure)
   - Base model training (Random Forest / GBM)
   - Meta-labeling
   - Walk-forward validation

3. **[BACKTESTING.md](BACKTESTING.md)**
   - Backtrader integration
   - Session-aware strategy (no overnight positions)
   - Realistic bid/ask execution
   - Trade logging

4. **[ARCH_RISK.md](ARCH_RISK.md)**
   - Monte Carlo on trade sequences
   - Drawdown metrics
   - Prop firm evaluation (FTMO)

5. **[ARCH_INFRA.md](ARCH_INFRA.md)**
   - Repository structure
   - MLflow tracking
   - Environment setup
   - Deployment strategy

---

## 🔧 Implementation Guidelines

### For Developers

- **[CODING_STANDARDS.md](CODING_STANDARDS.md)**
  - Code style and structure
  - Logging conventions
  - AI-specific guardrails
  - Git workflow

- **[DATA_HANDLING.md](DATA_HANDLING.md)**
  - Data preprocessing rules
  - Schema validation
  - Data leakage prevention
  - Output formats

- **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**
  - Git commit tracking
  - Configuration versioning
  - Environment versioning
  - Reproduction checklist

- **[REPORTING.md](REPORTING.md)**
  - Jinja2 template system
  - HTML/PDF report generation
  - MLflow artifact integration
  - Report sections

- **[TESTING.md](TESTING.md)**
  - Test types and organization
  - Unit, integration, and E2E tests
  - Regression testing strategy
  - Performance testing guidelines

- **[PROFILING.md](PROFILING.md)**
  - Performance profiling with cProfile
  - Identifying bottlenecks
  - Optimization strategies (Numba, vectorization)
  - Memory profiling

- **[BENCHMARKING.md](BENCHMARKING.md)**
  - Baseline strategy implementations
  - Statistical significance testing
  - Performance comparison framework
  - Extended metrics (Sortino, Calmar, etc.)

- **[DEPLOYMENT_API.md](DEPLOYMENT_API.md)**
  - Guide de déploiement de l'API d'inférence
  - Endpoints et exemples d'utilisation
  - Configuration Docker et docker-compose
  - Dépannage et bonnes pratiques

- **[GPU_SETUP.md](GPU_SETUP.md)**
  - GPU setup guide for RAPIDS cuML and cuDF
  - CUDA version detection and compatibility
  - Installation instructions for all NVIDIA GPUs
  - Troubleshooting GPU issues

---

## 📖 Reference

- **[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)** - Complete Hydra config documentation
- **[GLOSSARY.md](GLOSSARY.md)** - Technical terms and definitions

---

## 🎯 Core System Constraints

All components respect these **hard constraints**:

### 1. No Overnight Positions
- All positions flat before session end
- No weekend positions
- Friday early close
- Documented in: [ARCH_DATA_PIPELINE.md §6.7](ARCH_DATA_PIPELINE.md#67-session-aware-time-barrier-no-overnight-constraint)

### 2. Raw Spread Account
- Entry: `askPrice`
- Exit: `bidPrice`
- No mid-price assumption
- Spread cost always included

### 3. Time-Series Integrity
- Purging overlapping labels
- Embargo between train/test
- No forward-looking bias
- Walk-forward validation

---

## 🤖 For AI Assistants

**Important files for AI understanding**:

1. **[CODING_STANDARDS.md](CODING_STANDARDS.md)** - Code rules and AI guardrails
2. **[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)** - Never invent config keys
3. **[DATA_HANDLING.md](DATA_HANDLING.md)** - Data leakage prevention rules
4. **[BACKTESTING.md](BACKTESTING.md)** - Backtest must match labeling logic

**AI must respect**:
- No overnight positions (session calendar)
- Bid/ask execution (no mid-price)
- Config integrity (document new keys)
- API stability (update tests when changing signatures)
- Test coverage (see [TESTING.md](TESTING.md))
- Performance considerations (see [PROFILING.md](PROFILING.md))

---

## 📂 File Organization

```
docs/
├── INDEX.md                     # This file (documentation index)
├── GLOSSARY.md                  # Technical terms
│
├── HOW_TO_RUN.md                # Quick start guide
├── ARCHITECTURE.md              # System overview
├── ARCHITECTURE_DIAGRAMS.md     # Visual diagrams (Mermaid)
│
├── ARCH_DATA_PIPELINE.md        # Data flow
├── ARCH_ML_PIPELINE.md          # Model training
├── ARCH_INFRA.md                # Infrastructure
├── ARCH_RISK.md                 # Risk analysis
├── BACKTESTING.md               # Backtesting system
│
├── CODING_STANDARDS.md          # Code style
├── CONFIG_REFERENCE.md          # Config reference
├── DATA_HANDLING.md             # Data processing
├── REPRODUCIBILITY.md           # Reproducibility
├── REPORTING.md                 # Report generation
├── TESTING.md                   # Testing strategy
├── PROFILING.md                 # Performance profiling
├── BENCHMARKING.md              # Benchmarking guide
├── GPU_SETUP.md                 # GPU setup and RAPIDS installation
└── DEPLOYMENT_API.md            # API deployment guide
```

---

## 📝 Contributing to Documentation

When updating documentation:

1. Keep terminology consistent (use [GLOSSARY.md](GLOSSARY.md))
2. Update cross-references when moving sections
3. Follow markdown formatting (see [CODING_STANDARDS.md](CODING_STANDARDS.md))
4. Test all code examples
5. Update this INDEX if adding new files

---

## 🔗 Quick Links

- **Setup**: [HOW_TO_RUN.md §1](HOW_TO_RUN.md#1-prerequisites)
- **First Run**: [HOW_TO_RUN.md §3.2](HOW_TO_RUN.md#32-typical-first-run)
- **Config Example**: [CONFIG_REFERENCE.md §2](CONFIG_REFERENCE.md#2-experiment-configuration)
- **Session Logic**: [ARCH_DATA_PIPELINE.md §6.7](ARCH_DATA_PIPELINE.md#67-session-aware-time-barrier-no-overnight-constraint)
- **Backtest Strategy**: [BACKTESTING.md §4](BACKTESTING.md#4-backtrader-strategy-implementation)
- **MLflow Tracking**: [ARCH_INFRA.md §4](ARCH_INFRA.md#4-mlflow-tracking-system)

