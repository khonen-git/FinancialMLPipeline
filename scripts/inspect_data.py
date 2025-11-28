#!/usr/bin/env python3
"""Inspect tick data.

Usage:
    python scripts/inspect_data.py data/raw/EURUSD.parquet
"""

import argparse
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def inspect_data(data_path: Path):
    """Inspect tick data file.
    
    Args:
        data_path: Path to parquet file
    """
    logger.info(f"Inspecting {data_path}")
    
    if not data_path.exists():
        logger.error(f"File not found: {data_path}")
        return
    
    # Load data
    df = pd.read_parquet(data_path)
    
    print("=" * 80)
    print(f"Data inspection: {data_path.name}")
    print("=" * 80)
    
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index: {df.index.name}")
    
    print("\nFirst rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Spread analysis
    if 'askPrice' in df.columns and 'bidPrice' in df.columns:
        spread = df['askPrice'] - df['bidPrice']
        print(f"\nSpread statistics:")
        print(f"  Mean: {spread.mean():.5f}")
        print(f"  Std: {spread.std():.5f}")
        print(f"  Min: {spread.min():.5f}")
        print(f"  Max: {spread.max():.5f}")
        print(f"  Zero spreads: {(spread <= 0).sum()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Inspect tick data")
    parser.add_argument('data_path', type=str, help="Path to parquet file")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    inspect_data(Path(args.data_path))


if __name__ == '__main__':
    main()

