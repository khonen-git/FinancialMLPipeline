"""Schema detection and validation for raw tick data.

According to DATA_HANDLING.md, validates:
- CSV vs Parquet format
- Required columns presence
- Data types
- Volume usability
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


@dataclass
class SchemaValidationResult:
    """Result of schema validation."""
    
    is_valid: bool
    format: str  # 'csv' | 'parquet'
    has_volume: bool
    volume_usable: bool
    missing_columns: List[str]
    extra_columns: List[str]
    dtype_issues: Dict[str, str]
    error_message: Optional[str] = None


class SchemaDetector:
    """Detect and validate data schema for tick data."""
    
    REQUIRED_COLUMNS = ['timestamp', 'askPrice', 'bidPrice']
    OPTIONAL_COLUMNS = ['askVolume', 'bidVolume']
    
    EXPECTED_DTYPES = {
        'timestamp': 'int64',
        'askPrice': 'float64',
        'bidPrice': 'float64',
        'askVolume': 'float64',
        'bidVolume': 'float64',
    }
    
    def __init__(self, config: Dict):
        """Initialize schema detector.
        
        Args:
            config: Data configuration from Hydra
        """
        self.config = config
        self.schema_mapping = config.get('schema', {})
    
    def detect_format(self, file_path: Path) -> str:
        """Detect file format (CSV or Parquet).
        
        Args:
            file_path: Path to data file
            
        Returns:
            Format string: 'csv' | 'parquet'
            
        Raises:
            ValueError: If format cannot be determined
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return 'csv'
        elif suffix in ['.parquet', '.pq']:
            return 'parquet'
        else:
            # Try to detect from content
            try:
                pq.read_table(file_path, use_threads=False)
                return 'parquet'
            except:
                try:
                    pd.read_csv(file_path, nrows=1)
                    return 'csv'
                except:
                    raise ValueError(f"Cannot determine format of {file_path}")
    
    def validate_schema(
        self,
        df: pd.DataFrame,
        file_format: str
    ) -> SchemaValidationResult:
        """Validate DataFrame schema against expected structure.
        
        Args:
            df: DataFrame to validate
            file_format: Detected file format
            
        Returns:
            SchemaValidationResult with validation details
        """
        # Check required columns
        missing_cols = [
            col for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]
        
        if missing_cols:
            return SchemaValidationResult(
                is_valid=False,
                format=file_format,
                has_volume=False,
                volume_usable=False,
                missing_columns=missing_cols,
                extra_columns=[],
                dtype_issues={},
                error_message=f"Missing required columns: {missing_cols}"
            )
        
        # Check optional columns (volume)
        has_volume = all(
            col in df.columns for col in self.OPTIONAL_COLUMNS
        )
        
        # Check data types
        dtype_issues = {}
        for col, expected_dtype in self.EXPECTED_DTYPES.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not actual_dtype.startswith(expected_dtype.split('64')[0]):
                    dtype_issues[col] = f"Expected {expected_dtype}, got {actual_dtype}"
        
        # Check volume usability
        volume_usable = False
        if has_volume:
            # Volume is usable if:
            # 1. Not all zeros
            # 2. Not all NaN
            # 3. Has reasonable variance
            ask_vol = df['askVolume'].dropna()
            bid_vol = df['bidVolume'].dropna()
            
            if len(ask_vol) > 0 and len(bid_vol) > 0:
                if (ask_vol != 0).any() and (bid_vol != 0).any():
                    # Check variance (not constant)
                    if ask_vol.std() > 0 and bid_vol.std() > 0:
                        volume_usable = True
        
        # Extra columns
        all_expected = self.REQUIRED_COLUMNS + self.OPTIONAL_COLUMNS
        extra_cols = [col for col in df.columns if col not in all_expected]
        
        is_valid = len(missing_cols) == 0 and len(dtype_issues) == 0
        
        return SchemaValidationResult(
            is_valid=is_valid,
            format=file_format,
            has_volume=has_volume,
            volume_usable=volume_usable,
            missing_columns=missing_cols,
            extra_columns=extra_cols,
            dtype_issues=dtype_issues,
            error_message=None if is_valid else "Schema validation failed"
        )
    
    def validate_file(self, file_path: Path) -> SchemaValidationResult:
        """Validate a data file (end-to-end).
        
        Args:
            file_path: Path to data file
            
        Returns:
            SchemaValidationResult
        """
        logger.info(f"Validating schema for {file_path}")
        
        try:
            file_format = self.detect_format(file_path)
            logger.debug(f"Detected format: {file_format}")
            
            # Load sample
            if file_format == 'csv':
                df = pd.read_csv(file_path, nrows=1000)
            else:
                df = pq.read_table(file_path).to_pandas()[:1000]
            
            result = self.validate_schema(df, file_format)
            
            if result.is_valid:
                logger.info(f"Schema validation passed. Volume usable: {result.volume_usable}")
            else:
                logger.warning(f"Schema validation failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return SchemaValidationResult(
                is_valid=False,
                format='unknown',
                has_volume=False,
                volume_usable=False,
                missing_columns=[],
                extra_columns=[],
                dtype_issues={},
                error_message=str(e)
            )

