#!/usr/bin/env python3
"""CLI script to run experiments.

Usage:
    python run_experiment.py experiment=eurusd_scalping
    python run_experiment.py experiment=gbpusd_trend assets.symbol=GBPUSD
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.main_pipeline import run_pipeline

if __name__ == '__main__':
    run_pipeline()

