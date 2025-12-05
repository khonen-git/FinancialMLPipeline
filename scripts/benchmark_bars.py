#!/usr/bin/env python3
"""Benchmark different bar construction backends.

Compares:
- pandas (baseline)
- optimized (Numba)
- polars
- cudf (GPU)
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bar_factory import create_bar_builder, get_available_backends


def load_sample_data(n_ticks: int = 100000) -> pd.DataFrame:
    """Generate sample tick data for benchmarking."""
    print(f"Generating {n_ticks:,} sample ticks...")
    
    dates = pd.date_range('2023-01-01 00:00', periods=n_ticks, freq='1s', tz='UTC')
    
    # Simulate realistic price movements
    np.random.seed(42)
    base_price = 1.1000
    price_changes = np.random.randn(n_ticks) * 0.0001
    bid_prices = base_price + np.cumsum(price_changes)
    ask_prices = bid_prices + np.random.uniform(0.00001, 0.00005, n_ticks)
    
    ticks = pd.DataFrame({
        'bidPrice': bid_prices,
        'askPrice': ask_prices,
        'bidVolume': np.random.uniform(100, 1000, n_ticks),
        'askVolume': np.random.uniform(100, 1000, n_ticks),
    }, index=dates)
    
    return ticks


def benchmark_backend(backend: str, ticks: pd.DataFrame, config: dict, n_runs: int = 3) -> dict:
    """Benchmark a single backend."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {backend.upper()}")
    print(f"{'='*60}")
    
    try:
        builder = create_bar_builder(config, backend=backend)
    except (ImportError, ValueError) as e:
        print(f"❌ Backend '{backend}' not available: {e}")
        return {
            'backend': backend,
            'available': False,
            'error': str(e)
        }
    
    times = []
    n_bars = None
    
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}...", end=' ', flush=True)
        
        # Warmup run (not counted)
        if run == 0:
            _ = builder.build_bars(ticks)
            continue
        
        start = time.time()
        bars = builder.build_bars(ticks)
        elapsed = time.time() - start
        
        times.append(elapsed)
        n_bars = len(bars)
        
        print(f"{elapsed:.3f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\n  Results:")
    print(f"    Average: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"    Min:     {min_time:.3f}s")
    print(f"    Max:     {max_time:.3f}s")
    print(f"    Bars:    {n_bars:,}")
    print(f"    Throughput: {len(ticks) / avg_time:,.0f} ticks/s")
    
    return {
        'backend': backend,
        'available': True,
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'n_bars': n_bars,
        'n_ticks': len(ticks),
        'throughput': len(ticks) / avg_time
    }


def main():
    """Run benchmarks."""
    print("="*60)
    print("BAR CONSTRUCTION BENCHMARK")
    print("="*60)
    
    # Configuration
    config = {
        'type': 'tick',
        'threshold': 100
    }
    
    # Test with different data sizes
    data_sizes = [10000, 100000, 1000000]
    
    all_results = []
    
    for n_ticks in data_sizes:
        print(f"\n{'#'*60}")
        print(f"# Testing with {n_ticks:,} ticks")
        print(f"{'#'*60}")
        
        ticks = load_sample_data(n_ticks)
        
        # Get available backends
        available_backends = get_available_backends()
        print(f"\nAvailable backends: {', '.join(available_backends)}")
        
        # Benchmark each backend
        results = {}
        for backend in available_backends:
            result = benchmark_backend(backend, ticks, config, n_runs=3)
            results[backend] = result
            all_results.append(result)
        
        # Compare results
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        
        available_results = {k: v for k, v in results.items() if v.get('available', False)}
        
        if len(available_results) > 1:
            # Find fastest
            fastest = min(available_results.items(), key=lambda x: x[1]['avg_time'])
            fastest_backend, fastest_result = fastest
            
            print(f"\n🏆 Fastest: {fastest_backend.upper()} ({fastest_result['avg_time']:.3f}s)")
            print(f"\nSpeedup vs fastest:")
            
            for backend, result in available_results.items():
                if backend == fastest_backend:
                    continue
                speedup = result['avg_time'] / fastest_result['avg_time']
                print(f"  {backend:12s}: {speedup:.2f}x {'(slower)' if speedup > 1 else '(faster)'}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    # Group by backend
    backend_summary = {}
    for result in all_results:
        if not result.get('available', False):
            continue
        
        backend = result['backend']
        if backend not in backend_summary:
            backend_summary[backend] = []
        backend_summary[backend].append(result)
    
    for backend, results in backend_summary.items():
        avg_times = [r['avg_time'] for r in results]
        avg_throughput = [r['throughput'] for r in results]
        
        print(f"\n{backend.upper()}:")
        print(f"  Average time: {np.mean(avg_times):.3f}s")
        print(f"  Average throughput: {np.mean(avg_throughput):,.0f} ticks/s")


if __name__ == '__main__':
    main()

