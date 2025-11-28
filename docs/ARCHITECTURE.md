# Architecture Overview

This document provides a high-level overview of the entire trading machine-learning pipeline, including data ingestion, bar construction, feature engineering, regime detection, model training, walk-forward validation, risk analysis, reporting, and inference.

The goal is to expose the global system architecture to both human readers and AI coding assistants, ensuring clarity, consistency, and maintainability.

---

## Core System Constraints

- **No overnight positions**: all positions flat before session end (see ARCH_DATA_PIPELINE.md §6.7)
- **Raw spread account**: realistic bid/ask execution, no mid-price assumption
- **Time-series integrity**: purging, embargo, no forward-looking bias

---

# 1. Global End-to-End Pipeline

The diagram below describes the full lifecycle of a model run:

- Download and ingest tick data (Dukascopy)
- Detect format (CSV vs Parquet)
- Construct multiple bar types
- Compute features and labels (Lopez de Prado)
- Train micro and macro HMMs
- Train the base ML model + meta-label model
- Perform walk-forward validation with purging/embargo
- Run Monte Carlo on the resulting trades
- Generate MLflow logs + experiment report

```mermaid
flowchart LR
    subgraph RAW[Data sources]
        TICKS["Tick data (Dukascopy)<br/>timestamp, askPrice, bidPrice,<br/>askVolume, bidVolume"]
        MARKET["Other market data<br/>volatility, calendar, news (optional)"]
    end

    RAW -->|download via dukascopy-node<br/>and local ingest| D1["Raw data ingestion<br/>CSV or Parquet detection"]

    subgraph PREP[Data preparation]
        FORMAT["Format & schema check<br/>CSV vs Parquet, volume availability"]
        BARS["Bar construction<br/>100 / 1000 ticks, volume / dollar bars"]
        FRACDIFF["Fractional differencing<br/>optional stationarization"]
        FEATS["Feature engineering<br/>returns, volatility, microstructure"]
        LABELS["Labelling (Lopez de Prado)<br/>triple barrier, meta-labels"]
    end

    D1 --> FORMAT --> BARS --> FRACDIFF --> FEATS --> LABELS

    subgraph ML[Modelling and regimes]
        CVNODE["Time-series CV<br/>purging, embargo, walk-forward"]
        HMM_MACRO["Macro HMM<br/>slow market regimes"]
        HMM_MICRO["Microstructure HMM<br/>order flow regimes"]
        RF_BASE["Base models<br/>Random Forest / Gradient Boosting<br/>CPU (sklearn) / GPU (cuML)"]
        META["Meta-label model<br/>trade filtering and sizing"]
    end

    FEATS --> HMM_MACRO
    FEATS --> HMM_MICRO

    LABELS --> CVNODE
    CVNODE --> RF_BASE
    HMM_MACRO --> RF_BASE
    HMM_MICRO --> RF_BASE

    RF_BASE --> META
    HMM_MACRO --> META
    HMM_MICRO --> META

    subgraph RISK[Risk analysis]
        WF["Walk-forward analysis<br/>performance stability over time"]
        MC_TRADES["Monte Carlo on trades<br/>bootstrap equity, risk of ruin"]
    end

    META --> WF
    META --> MC_TRADES

    subgraph TRACK[Tracking and reporting]
        MLFLOW["MLflow<br/>params, metrics, artifacts, models"]
        REPORTS["Jinja2 → HTML / PDF<br/>experiment reports"]
    end

    RF_BASE --> MLFLOW
    META --> MLFLOW
    WF --> MLFLOW
    MC_TRADES --> MLFLOW
    MLFLOW --> REPORTS

    subgraph BT[Backtest and execution]
        BTDR["Backtrader<br/>backtests, PnL, drawdown, risk"]
        LIVE["Inference script (CLI)<br/>future MT5 integration"]
    end

    META --> BTDR
    BTDR --> MLFLOW
    META --> LIVE
