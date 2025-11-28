"""HTML report generation using Jinja2 templates.

Generates comprehensive experiment reports with:
- Summary statistics
- Metrics tables
- Performance plots
- Risk analysis
- Backtest results
"""

import logging
import pandas as pd
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate HTML reports from experiment results."""
    
    def __init__(self, template_dir: Path):
        """Initialize report generator.
        
        Args:
            template_dir: Directory containing Jinja2 templates
        """
        self.template_dir = template_dir
        self.env = Environment(loader=FileSystemLoader(template_dir))
    
    def generate_report(
        self,
        results: dict,
        output_path: Path,
        config: dict
    ) -> Path:
        """Generate full experiment report.
        
        Args:
            results: Dictionary with all experiment results
            output_path: Path to save HTML report
            config: Experiment configuration
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating report: {output_path}")
        
        # Load template
        template = self.env.get_template('experiment_report.html')
        
        # Prepare context
        context = {
            'experiment_name': config.get('experiment', {}).get('name', 'Unknown'),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': config,
            'summary': self._prepare_summary(results),
            'metrics': self._prepare_metrics(results),
            'backtest': self._prepare_backtest_results(results),
            'risk': self._prepare_risk_results(results),
            'plots': self._prepare_plots(results)
        }
        
        # Render template
        html = template.render(**context)
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding='utf-8')
        
        logger.info(f"Report saved: {output_path}")
        
        return output_path
    
    def _prepare_summary(self, results: dict) -> dict:
        """Prepare summary section.
        
        Args:
            results: Experiment results
            
        Returns:
            Dictionary with summary data
        """
        return {
            'n_samples': results.get('n_samples', 0),
            'n_features': results.get('n_features', 0),
            'n_labels': results.get('n_labels', 0),
            'train_duration': results.get('train_duration', 'N/A'),
            'test_duration': results.get('test_duration', 'N/A')
        }
    
    def _prepare_metrics(self, results: dict) -> dict:
        """Prepare metrics section.
        
        Args:
            results: Experiment results
            
        Returns:
            Dictionary with metrics
        """
        metrics = results.get('metrics', {})
        
        return {
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0),
            'auc': metrics.get('auc', 0)
        }
    
    def _prepare_backtest_results(self, results: dict) -> dict:
        """Prepare backtest section.
        
        Args:
            results: Experiment results
            
        Returns:
            Dictionary with backtest results
        """
        bt = results.get('backtest', {})
        
        return {
            'total_trades': bt.get('total_trades', 0),
            'win_rate': bt.get('win_rate', 0),
            'avg_pnl': bt.get('avg_pnl', 0),
            'sharpe_ratio': bt.get('sharpe_ratio', 0),
            'max_drawdown': bt.get('max_drawdown', 0),
            'final_value': bt.get('final_value', 0)
        }
    
    def _prepare_risk_results(self, results: dict) -> dict:
        """Prepare risk analysis section.
        
        Args:
            results: Experiment results
            
        Returns:
            Dictionary with risk results
        """
        risk = results.get('risk', {})
        
        return {
            'prob_ruin': risk.get('prob_ruin', 0),
            'prob_profit_target': risk.get('prob_profit_target', 0),
            'mean_max_dd': risk.get('mean_max_dd', 0),
            'is_viable': risk.get('is_viable', False)
        }
    
    def _prepare_plots(self, results: dict) -> list:
        """Prepare plot references.
        
        Args:
            results: Experiment results
            
        Returns:
            List of plot file paths
        """
        plots = results.get('plots', [])
        return plots

