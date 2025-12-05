"""Unit tests for report generator."""

import pytest
from pathlib import Path
from src.reporting.report_generator import ReportGenerator


@pytest.mark.unit
class TestReportGenerator:
    """Test report generator functionality."""
    
    def test_report_generator_init(self, tmp_path):
        """Test ReportGenerator initialization."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        
        generator = ReportGenerator(template_dir)
        
        assert generator is not None
        assert generator.template_dir == template_dir
    
    def test_report_generator_init_missing_template_dir(self, tmp_path):
        """Test error when template directory doesn't exist."""
        template_dir = tmp_path / "nonexistent"
        
        # Should still initialize (Jinja2 will handle missing templates)
        generator = ReportGenerator(template_dir)
        
        assert generator is not None

