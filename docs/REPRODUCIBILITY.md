# Reproducibility & Traceability

This project guarantees reproducibility at three levels:

1. **Code versioning** (Git)
2. **Experiment versioning** (MLflow)
3. **Configuration versioning** (Hydra config dumps)

---

## 1. Git Commit Tracking

Each MLflow run logs:

- the current Git commit hash,
- whether the repository is "clean" or has uncommitted changes.

This ensures full traceability between:

- code version,
- experiments,
- results.

**No experiment should run without logging the Git commit.**

---

## 2. Hydra Configuration Dump

For every experiment, the full Hydra configuration (after resolving defaults) is stored in MLflow:

```
<run>/artifacts/config/config.yaml
```

This includes:

- asset parameters,
- bar construction parameters,
- triple barrier configuration,
- model hyperparameters,
- walk-forward settings,
- risk model parameters.

**This guarantees that any experiment can be fully reconstructed.**

---

## 3. Data Versioning

Each experiment logs:

- the date range used,
- the dataset path or version identifier,
- a hash of the clean Parquet file (optional but recommended).

This ensures traceability of the input data.

---

## 4. Environment Versioning

The following environment metadata is logged:

- Python version
- Major library versions (numpy, pandas, sklearn, cudf, cuml, hmmlearn, etc.)
- Optionally: `pip freeze` as artifact (`requirements.txt`)

This avoids inconsistencies over time.

---

## 5. Reproduction Checklist

To reproduce any MLflow run:

1. Checkout the Git commit hash.
2. Restore the dataset version.
3. Load the logged Hydra `config.yaml`.
4. Recreate the environment using the logged requirements.
5. Run the same experiment entrypoint.

**This guarantees faithful reproduction of training, backtest, and risk analysis results.**

---

## 6. Reproduction Workflow

```mermaid
flowchart TB
    subgraph MLflow["MLflow Experiment Run"]
        RUN[Run ID: abc123]
        GIT[Git Commit: 293fbd2]
        CFG[Config YAML]
        DATA[Dataset: 2019-2020]
        ENV[requirements.txt]
    end
    
    subgraph Reproduce["Reproduction Steps"]
        CHECKOUT[git checkout 293fbd2]
        LOADCFG[Load config.yaml]
        GETDATA[Restore Dataset]
        CREATEENV[conda env create]
    end
    
    subgraph Rerun["Re-run"]
        EXEC[python scripts/run_experiment.py]
        VERIFY[Compare Metrics]
        MATCH{Match?}
    end
    
    RUN --> GIT
    RUN --> CFG
    RUN --> DATA
    RUN --> ENV
    
    GIT --> CHECKOUT
    CFG --> LOADCFG
    DATA --> GETDATA
    ENV --> CREATEENV
    
    CHECKOUT --> EXEC
    LOADCFG --> EXEC
    GETDATA --> EXEC
    CREATEENV --> EXEC
    
    EXEC --> VERIFY
    VERIFY --> MATCH
    MATCH -->|Yes ✅| SUCCESS([Reproduced!])
    MATCH -->|No ❌| DEBUG[Check Differences]
    
    style SUCCESS fill:#90EE90
    style DEBUG fill:#FFB6C1
```

This workflow ensures that any experiment can be faithfully reproduced by following the logged metadata.

