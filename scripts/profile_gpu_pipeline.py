#!/usr/bin/env python3
"""Profile the ML pipeline execution with GPU support.

This script runs the main pipeline with cProfile enabled and saves profiling results.
It also monitors GPU usage if cuML is available.
"""

import cProfile
import pstats
import io
from pathlib import Path
import sys
import hydra
from omegaconf import DictConfig
import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.main_pipeline import run_pipeline

# Try to import GPU monitoring tools
try:
    import pynvml
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    print("⚠️  pynvml not available. GPU monitoring disabled. Install with: pip install nvidia-ml-py")


def get_gpu_info():
    """Get GPU information if available."""
    if not GPU_MONITORING_AVAILABLE:
        return {}
    
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        gpu_info = {
            'device_count': device_count,
            'devices': []
        }
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name_bytes = pynvml.nvmlDeviceGetName(handle)
            name = name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else name_bytes
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            gpu_info['devices'].append({
                'index': i,
                'name': name,
                'total_memory_mb': mem_info.total / 1024**2,
                'free_memory_mb': mem_info.free / 1024**2,
                'used_memory_mb': mem_info.used / 1024**2
            })
        
        return gpu_info
    except Exception as e:
        print(f"⚠️  Error getting GPU info: {e}")
        return {}


def monitor_gpu_usage(duration_seconds: float = 1.0):
    """Monitor GPU usage during execution.
    
    Args:
        duration_seconds: How long to monitor
        
    Returns:
        Dictionary with GPU usage statistics
    """
    if not GPU_MONITORING_AVAILABLE:
        return {}
    
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        if device_count == 0:
            return {}
        
        # Monitor first GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Get initial state
        initial_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # Sample utilization
        utilizations = []
        import time
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilizations.append(util.gpu)
            except:
                pass
            time.sleep(0.1)
        
        # Get final state
        final_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        return {
            'avg_utilization': sum(utilizations) / len(utilizations) if utilizations else 0,
            'max_utilization': max(utilizations) if utilizations else 0,
            'initial_memory_mb': initial_mem.used / 1024**2,
            'final_memory_mb': final_mem.used / 1024**2,
            'peak_memory_mb': final_mem.used / 1024**2  # Approximate
        }
    except Exception as e:
        print(f"⚠️  Error monitoring GPU: {e}")
        return {}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def profile_pipeline(cfg: DictConfig):
    """Run pipeline with profiling enabled."""
    
    # Check if GPU backend is requested
    model_backend = cfg.models.model.get('backend', 'cpu').lower()
    use_gpu = model_backend in ['gpu', 'cuml']
    
    if use_gpu:
        print("🚀 GPU profiling mode enabled")
        gpu_info = get_gpu_info()
        if gpu_info:
            print(f"📊 GPU Info: {gpu_info.get('device_count', 0)} device(s)")
            for device in gpu_info.get('devices', []):
                print(f"  - Device {device['index']}: {device['name']}")
                print(f"    Memory: {device['used_memory_mb']:.0f}MB / {device['total_memory_mb']:.0f}MB")
        else:
            print("⚠️  No GPU detected or GPU monitoring unavailable")
    else:
        print("💻 CPU profiling mode")
    
    # Ensure profiling is enabled in the config for integrated profiling
    # Note: We don't modify cfg directly as it may be in struct mode
    # Profiling is handled by cProfile in this script
    
    # Generate a unique filename for the profile output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = cfg.experiment.name if 'experiment' in cfg and 'name' in cfg.experiment else "default_experiment"
    backend_suffix = "gpu" if use_gpu else "cpu"
    profile_dir = Path('profiling')
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_file = profile_dir / f"full_run_{experiment_name}_{backend_suffix}_{timestamp}.prof"
    report_file = profile_dir / f"full_run_{experiment_name}_{backend_suffix}_{timestamp}.txt"
    
    print(f"📊 Starting profiling. Profile will be saved to {profile_file}")
    
    # Enable profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Monitor GPU if available and GPU backend is used
    gpu_stats = {}
    if use_gpu and GPU_MONITORING_AVAILABLE:
        print("📈 Starting GPU monitoring...")
        import threading
        import time
        
        gpu_monitoring_active = True
        gpu_samples = []
        
        def monitor_gpu_thread():
            while gpu_monitoring_active:
                try:
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_samples.append({
                        'time': time.time(),
                        'utilization': util.gpu,
                        'memory_mb': mem.used / 1024**2
                    })
                except:
                    pass
                time.sleep(0.5)
        
        monitor_thread = threading.Thread(target=monitor_gpu_thread, daemon=True)
        monitor_thread.start()
    
    try:
        # Run pipeline
        run_pipeline(cfg)
    except Exception as e:
        print(f"❌ An error occurred during pipeline execution: {e}")
        raise
    finally:
        # Stop GPU monitoring
        if use_gpu and GPU_MONITORING_AVAILABLE:
            gpu_monitoring_active = False
            monitor_thread.join(timeout=2)
            
            if gpu_samples:
                gpu_stats = {
                    'avg_utilization': sum(s['utilization'] for s in gpu_samples) / len(gpu_samples),
                    'max_utilization': max(s['utilization'] for s in gpu_samples),
                    'avg_memory_mb': sum(s['memory_mb'] for s in gpu_samples) / len(gpu_samples),
                    'peak_memory_mb': max(s['memory_mb'] for s in gpu_samples),
                    'samples': len(gpu_samples)
                }
                print(f"📊 GPU Stats: Avg Util={gpu_stats['avg_utilization']:.1f}%, "
                      f"Peak Memory={gpu_stats['peak_memory_mb']:.0f}MB")
        
        profiler.disable()
        
        # Save profile data
        profiler.dump_stats(str(profile_file))
        print(f"✅ Profile saved to {profile_file}")
        
        # Generate text report
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CPU PROFILING RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            s = pstats.Stats(profiler, stream=f)
            s.sort_stats('cumulative').print_stats(50)  # Top 50 functions by cumulative time
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("TOP FUNCTIONS BY TOTAL TIME (excluding subcalls):\n")
            f.write("=" * 80 + "\n")
            s.sort_stats('tottime').print_stats(50)  # Top 50 functions by total time
            
            # Add GPU stats if available
            if gpu_stats:
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("GPU MONITORING RESULTS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Average GPU Utilization: {gpu_stats['avg_utilization']:.1f}%\n")
                f.write(f"Peak GPU Utilization: {gpu_stats['max_utilization']:.1f}%\n")
                f.write(f"Average GPU Memory: {gpu_stats['avg_memory_mb']:.0f} MB\n")
                f.write(f"Peak GPU Memory: {gpu_stats['peak_memory_mb']:.0f} MB\n")
                f.write(f"Monitoring Samples: {gpu_stats['samples']}\n")
        
        print(f"✅ Report saved to {report_file}")
        
        # Quick summary for terminal
        print("\n📊 Quick summary (top 10 by cumulative time):")
        summary_stream = io.StringIO()
        pstats.Stats(profiler, stream=summary_stream).sort_stats('cumulative').print_stats(10)
        print(summary_stream.getvalue())
        
        if gpu_stats:
            print(f"\n🎮 GPU Summary:")
            print(f"  Average Utilization: {gpu_stats['avg_utilization']:.1f}%")
            print(f"  Peak Utilization: {gpu_stats['max_utilization']:.1f}%")
            print(f"  Peak Memory: {gpu_stats['peak_memory_mb']:.0f} MB")


if __name__ == '__main__':
    profile_pipeline()

