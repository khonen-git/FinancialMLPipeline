#!/usr/bin/env python3
"""Profile the ML pipeline execution.

This script runs the pipeline with cProfile enabled and saves the results
to profiling/ directory with timestamped filenames.
"""

import cProfile
import pstats
from pathlib import Path
import sys
from datetime import datetime
from io import StringIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig

from src.pipeline.main_pipeline import run_pipeline


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def profile_pipeline(cfg: DictConfig) -> None:
    """Run pipeline with profiling enabled."""
    
    # Create profiling directory
    profiling_dir = Path('profiling')
    profiling_dir.mkdir(exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = cfg.experiment.name.replace('/', '_').replace(' ', '_')
    profile_file = profiling_dir / f"full_run_{experiment_name}_{timestamp}.prof"
    report_file = profiling_dir / f"full_run_{experiment_name}_{timestamp}.txt"
    
    print(f"🔍 Starting profiling for: {cfg.experiment.name}")
    print(f"📁 Profile will be saved to: {profile_file}")
    
    # Enable profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        # Run pipeline
        run_pipeline(cfg)
    finally:
        profiler.disable()
        
        # Save profile data
        profiler.dump_stats(str(profile_file))
        print(f"✅ Profile saved to {profile_file}")
        
        # Generate text report
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        
        # Sort by cumulative time (most impactful functions first)
        ps.sort_stats('cumulative')
        ps.print_stats(50)  # Top 50 functions
        
        # Also sort by total time (functions doing heavy work themselves)
        s.write("\n" + "=" * 80 + "\n")
        s.write("TOP FUNCTIONS BY TOTAL TIME (excluding subcalls):\n")
        s.write("=" * 80 + "\n")
        ps.sort_stats('tottime')
        ps.print_stats(30)  # Top 30 functions
        
        # Save report
        with open(report_file, 'w') as f:
            f.write(s.getvalue())
        
        print(f"✅ Report saved to {report_file}")
        print(f"\n📊 Quick summary (top 10 by cumulative time):")
        print("-" * 80)
        
        # Print quick summary to console
        print("\n" + "=" * 80)
        print("TOP 10 FUNCTIONS BY CUMULATIVE TIME:")
        print("=" * 80)
        ps.sort_stats('cumulative')
        ps.print_stats(10)
        
        print(f"\n💡 To view full report: cat {report_file}")
        print(f"💡 To visualize with snakeviz: snakeviz {profile_file}")


if __name__ == '__main__':
    profile_pipeline()

