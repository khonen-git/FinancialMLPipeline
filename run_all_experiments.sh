#!/bin/bash
# Script to run all MFE/MAE experiments
# Usage: ./run_all_experiments.sh

set -e  # Exit on error

echo "=========================================="
echo "Running MFE/MAE Experiments"
echo "=========================================="
echo ""

# Check if micromamba or conda is available
if command -v micromamba &> /dev/null; then
    echo "Activating micromamba environment: financial-ml"
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate financial-ml
    echo "✓ Environment activated (micromamba)"
elif command -v conda &> /dev/null; then
    echo "Activating conda environment: financial-ml"
    eval "$(conda shell.bash hook)"
    conda activate financial-ml
    echo "✓ Environment activated (conda)"
else
    echo "⚠ Warning: neither micromamba nor conda found. Make sure you have activated the environment manually:"
    echo "  micromamba activate financial-ml  # or conda activate financial-ml"
    echo ""
fi

# Check Python and dependencies
echo "Checking Python environment..."
python --version
python -c "import hydra, mlflow, backtrader; print('✓ Dependencies OK')" || {
    echo "✗ Missing dependencies. Please install:"
    echo "  conda env create -f environment.yaml"
    exit 1
}

echo ""
echo "=========================================="
echo "Experiment 1/4: EURUSD CPCV"
echo "=========================================="
python run_experiment.py experiment=eurusd_2023_100ticks_32bars_mfe_mae_cpcv
echo ""

echo "=========================================="
echo "Experiment 2/4: EURUSD TSCV"
echo "=========================================="
python run_experiment.py experiment=eurusd_2023_100ticks_32bars_mfe_mae_tscv
echo ""

echo "=========================================="
echo "Experiment 3/4: USDJPY CPCV"
echo "=========================================="
if [ -f "data/raw/USDJPY_2023.csv" ] && [ -f "data/raw/USDJPY_2024.csv" ]; then
    python run_experiment.py experiment=usdjpy_2023_100ticks_32bars_mfe_mae_cpcv
else
    echo "⚠ Skipping: USDJPY data files not found"
    echo "  Missing: data/raw/USDJPY_2023.csv or data/raw/USDJPY_2024.csv"
fi
echo ""

echo "=========================================="
echo "Experiment 4/4: USDJPY TSCV"
echo "=========================================="
if [ -f "data/raw/USDJPY_2023.csv" ] && [ -f "data/raw/USDJPY_2024.csv" ]; then
    python run_experiment.py experiment=usdjpy_2023_100ticks_32bars_mfe_mae_tscv
else
    echo "⚠ Skipping: USDJPY data files not found"
    echo "  Missing: data/raw/USDJPY_2023.csv or data/raw/USDJPY_2024.csv"
fi
echo ""

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "View results in MLflow:"
echo "  mlflow ui"
echo ""

