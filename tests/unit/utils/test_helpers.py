"""Unit tests for utility helpers."""

import pytest
from pathlib import Path
from src.utils.helpers import convert_ticks_to_price, ensure_dir


@pytest.mark.unit
class TestHelpers:
    """Test utility helper functions."""
    
    def test_convert_ticks_to_price(self):
        """Test tick to price conversion."""
        tick_size = 0.00001
        ticks = 50
        
        price = convert_ticks_to_price(ticks, tick_size)
        
        assert price == 50 * 0.00001
        assert price == 0.0005
    
    def test_convert_ticks_to_price_zero(self):
        """Test conversion with zero ticks."""
        price = convert_ticks_to_price(0, 0.00001)
        
        assert price == 0.0
    
    def test_ensure_dir(self, tmp_path):
        """Test directory creation."""
        new_dir = tmp_path / "new_directory"
        
        result = ensure_dir(new_dir)
        
        assert result.exists()
        assert result.is_dir()
    
    def test_ensure_dir_existing(self, tmp_path):
        """Test ensure_dir with existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        
        result = ensure_dir(existing_dir)
        
        assert result.exists()
        assert result.is_dir()

