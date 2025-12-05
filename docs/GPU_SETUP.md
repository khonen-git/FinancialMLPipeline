# GPU Setup Guide (RTX 4070)

This guide explains how to set up GPU support with RAPIDS cuML for the FinancialMLPipeline on a system with NVIDIA RTX 4070.

---

## Prerequisites

- **GPU**: NVIDIA RTX 4070 (or compatible CUDA-capable GPU)
- **CUDA**: CUDA 12.x (RTX 4070 supports CUDA 12.6+)
- **Environment Manager**: micromamba (or conda/mamba)
- **Python**: 3.12

---

## 1. Verify GPU and CUDA

First, verify your GPU is detected and check CUDA version:

```bash
nvidia-smi
```

You should see your RTX 4070 listed with CUDA version 12.x.

---

## 2. Install cuML via micromamba

### Option A: Using the installation script (Recommended)

```bash
cd /home/khonen/Dev/FinancialMLPipeline
./scripts/install_gpu_support.sh
```

### Option B: Manual installation

```bash
# Activate your environment
micromamba activate financial-ml

# Install RAPIDS cuML with CUDA 12.x support
micromamba install -y -c rapidsai -c conda-forge -c nvidia \
    cuml=24.02 \
    cudatoolkit=12.4 \
    python=3.12

# Install GPU monitoring tool (optional but recommended)
pip install nvidia-ml-py
```

### Option C: Add to environment.yaml

Uncomment and modify the GPU section in `environment.yaml`:

```yaml
dependencies:
  # ... existing dependencies ...
  - cudatoolkit=12.4
  - cuml=24.02
```

Then recreate the environment:

```bash
micromamba env update -f environment.yaml
```

---

## 3. Verify Installation

Test that cuML is working:

```bash
python -c "
import cuml
print(f'cuML version: {cuml.__version__}')

from cuml.ensemble import RandomForestClassifier
print('✅ RandomForestClassifier imported successfully')

# Check GPU availability
from cuml.common import device_type
print(f'Device type: {device_type.get_device_type()}')
"
```

Expected output:
```
cuML version: 24.02.xx
✅ RandomForestClassifier imported successfully
Device type: gpu
```

---

## 4. Run Experiments with GPU

### Quick Test: CPU vs GPU Comparison

1. **Run with CPU** (baseline):
```bash
python scripts/run_experiment.py \
    +experiment=test_cpu_vs_gpu \
    +models=rf_cpu
```

2. **Run with GPU**:
```bash
python scripts/run_experiment.py \
    +experiment=test_cpu_vs_gpu \
    +models=rf_gpu
```

3. **Compare results in MLflow**:
```bash
mlflow ui
```

Look for metrics:
- `model_backend`: "cpu" vs "gpu"
- `fold_X_fit_time`: Training time per fold
- `available_backends`: List of available backends

### Full Experiment with GPU

```bash
python scripts/run_experiment.py \
    +experiment=eurusd_2023_100ticks_32bars_mfe_mae_cpcv \
    +models=rf_gpu
```

---

## 5. GPU Profiling

Profile GPU performance to identify bottlenecks:

```bash
python scripts/profile_gpu_pipeline.py \
    +experiment=test_cpu_vs_gpu \
    +models=rf_gpu
```

This will:
- Monitor GPU utilization and memory usage
- Profile CPU execution with cProfile
- Generate combined CPU + GPU profiling report
- Save results to `profiling/` directory

---

## 6. Troubleshooting

### Issue: cuML not found

**Error**: `ImportError: cuML is not available`

**Solution**:
```bash
# Verify installation
micromamba list | grep cuml

# Reinstall if needed
micromamba install -y -c rapidsai -c conda-forge -c nvidia cuml=24.02
```

### Issue: CUDA version mismatch

**Error**: CUDA version incompatibility

**Solution**: Check CUDA version:
```bash
nvidia-smi  # Check CUDA version
nvcc --version  # Check installed CUDA toolkit
```

Install matching cudatoolkit version:
```bash
micromamba install -y -c nvidia cudatoolkit=12.4  # Match your CUDA version
```

### Issue: GPU not detected

**Error**: cuML falls back to CPU

**Solution**:
1. Verify GPU is detected: `nvidia-smi`
2. Check CUDA drivers are installed
3. Verify cuML installation: `python -c "import cuml; print(cuml.__version__)"`

### Issue: Out of memory

**Error**: GPU memory errors

**Solution**:
- Reduce `n_estimators` in model config
- Reduce batch size or dataset size
- Monitor GPU memory: `watch -n 1 nvidia-smi`

---

## 7. Performance Comparison

After running both CPU and GPU experiments, compare:

| Metric | CPU (sklearn) | GPU (cuML) | Speedup |
|--------|---------------|------------|---------|
| Training time (fold 0) | X seconds | Y seconds | X/Y |
| Total pipeline time | X seconds | Y seconds | X/Y |
| GPU utilization | N/A | Z% | - |
| GPU memory peak | N/A | W MB | - |

Check MLflow for detailed metrics.

---

## 8. Best Practices

1. **Start with small experiments** to verify GPU setup
2. **Monitor GPU utilization** during training
3. **Compare CPU vs GPU** on same dataset for fair comparison
4. **Profile before optimizing** to identify bottlenecks
5. **Use GPU for large datasets** where speedup is significant

---

## 9. Additional Resources

- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [CUDA Compatibility Guide](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [RTX 4070 Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4070/)

---

## Notes

- RTX 4070 supports CUDA 12.x (recommended: 12.4+)
- cuML 24.02 is compatible with CUDA 12.x
- GPU acceleration is most beneficial for:
  - Large datasets (>100k samples)
  - Many features (>100 features)
  - Large models (many trees, deep trees)
- CPU may still be faster for small datasets due to overhead

