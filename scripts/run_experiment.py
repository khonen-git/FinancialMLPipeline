#!/usr/bin/env python3
"""CLI script to run experiments.

Usage:
    python scripts/run_experiment.py experiment=eurusd_scalping
    python scripts/run_experiment.py experiment=gbpusd_trend assets.symbol=GBPUSD
"""

import sys
from pathlib import Path
from typing import NoReturn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.main_pipeline import run_pipeline


def main() -> NoReturn:
    """Main entry point for experiment runner."""
    run_pipeline()


if __name__ == '__main__':
    main()

