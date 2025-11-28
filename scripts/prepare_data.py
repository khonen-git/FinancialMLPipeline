#!/usr/bin/env python3
"""Prepare tick data for training and testing.

Converts CSV to Parquet and organizes data for experiments.

Usage:
    # Convert all CSV files
    python scripts/prepare_data.py --convert-all
    
    # Convert specific file
    python scripts/prepare_data.py --convert data/eurusd-tick-2023-01-01-2024-01-01.csv
    
    # Create sample for quick testing
    python scripts/prepare_data.py --create-sample --asset EURUSD --year 2023 --n-rows 100000
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_csv_to_parquet(csv_path: Path, output_dir: Path = None):
    """Convert CSV tick data to Parquet format.
    
    Args:
        csv_path: Path to CSV file
        output_dir: Output directory (default: data/raw/)
    """
    if output_dir is None:
        output_dir = Path("data/raw")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting {csv_path.name} to Parquet...")
    
    # Extract asset and year from filename (e.g., eurusd-tick-2023-01-01-2024-01-01.csv)
    filename = csv_path.stem
    parts = filename.split('-')
    asset = parts[0].upper()
    year = parts[3]  # Year from start date
    
    # Read CSV in chunks for memory efficiency
    logger.info(f"Reading CSV file...")
    df = pd.read_csv(csv_path)
    
    logger.info(f"Loaded {len(df):,} rows")
    
    # Basic validation
    expected_cols = ['timestamp', 'askPrice', 'bidPrice', 'askVolume', 'bidVolume']
    if not all(col in df.columns for col in expected_cols):
        logger.error(f"Missing columns. Expected: {expected_cols}, Got: {list(df.columns)}")
        return
    
    # Convert timestamp to datetime for inspection
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Spread: min={df['askPrice'].min() - df['bidPrice'].max():.5f}, "
                f"max={df['askPrice'].max() - df['bidPrice'].min():.5f}")
    
    # Save to Parquet
    output_path = output_dir / f"{asset}_{year}.parquet"
    df.to_parquet(output_path, engine='pyarrow', compression='snappy')
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Saved to {output_path} ({file_size_mb:.1f} MB)")
    
    return output_path


def create_sample(asset: str, year: str, n_rows: int = 100000):
    """Create a small sample for quick testing.
    
    Args:
        asset: Asset symbol (e.g., 'EURUSD')
        year: Year (e.g., '2023')
        n_rows: Number of rows to sample
    """
    logger.info(f"Creating sample: {asset} {year} ({n_rows:,} rows)")
    
    # Find source CSV
    csv_pattern = f"data/{asset.lower()}-tick-{year}-*.csv"
    import glob
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        logger.error(f"No CSV file found matching {csv_pattern}")
        return
    
    csv_path = Path(csv_files[0])
    logger.info(f"Reading from {csv_path.name}...")
    
    # Read sample
    df = pd.read_csv(csv_path, nrows=n_rows)
    
    logger.info(f"Loaded {len(df):,} rows")
    
    # Save as Parquet
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{asset}_{year}_sample.parquet"
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.to_parquet(output_path, engine='pyarrow', compression='snappy')
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Sample saved to {output_path} ({file_size_mb:.1f} MB)")
    
    return output_path


def convert_all_csv():
    """Convert all CSV files in data/ directory."""
    data_dir = Path("data")
    csv_files = list(data_dir.glob("*-tick-*.csv"))
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    for csv_path in csv_files:
        # Skip empty files
        if csv_path.stat().st_size == 0:
            logger.warning(f"Skipping empty file: {csv_path.name}")
            continue
        
        try:
            convert_csv_to_parquet(csv_path)
        except Exception as e:
            logger.error(f"Error converting {csv_path.name}: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare tick data")
    parser.add_argument('--convert', type=str, help="Convert specific CSV file")
    parser.add_argument('--convert-all', action='store_true', help="Convert all CSV files")
    parser.add_argument('--create-sample', action='store_true', help="Create sample dataset")
    parser.add_argument('--asset', type=str, default='EURUSD', help="Asset for sample")
    parser.add_argument('--year', type=str, default='2023', help="Year for sample")
    parser.add_argument('--n-rows', type=int, default=100000, help="Number of rows for sample")
    
    args = parser.parse_args()
    
    if args.convert:
        csv_path = Path(args.convert)
        if not csv_path.exists():
            logger.error(f"File not found: {csv_path}")
            return
        convert_csv_to_parquet(csv_path)
    
    elif args.convert_all:
        convert_all_csv()
    
    elif args.create_sample:
        create_sample(args.asset, args.year, args.n_rows)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

