# GPU Setup Guide

This guide explains how to set up GPU support with RAPIDS cuML and cuDF for the FinancialMLPipeline on systems with NVIDIA CUDA-capable GPUs.

---

## Prerequisites

- **GPU**: NVIDIA CUDA-capable GPU (Compute Capability 6.0+)
  - GeForce series (GTX 10xx, RTX 20xx, RTX 30xx, RTX 40xx, etc.)
  - Quadro series
  - Tesla/Data Center GPUs (V100, A100, etc.)
- **CUDA**: CUDA 11.0+ (CUDA 11.8 or 12.x recommended)
- **Environment Manager**: micromamba (or conda/mamba)
- **Python**: 3.12

**Note**: Check [RAPIDS compatibility](https://docs.rapids.ai/install) for the latest CUDA version requirements.

---

## 1. Verify GPU and CUDA

First, verify your GPU is detected and check CUDA version:

```bash
nvidia-smi
```

You should see your GPU listed with its CUDA version. Note the CUDA version shown (e.g., 11.8, 12.0, 12.4).

**Check GPU Compute Capability**:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

RAPIDS requires Compute Capability 6.0 or higher (most modern GPUs meet this requirement).

---

## 2. Determine CUDA Version

RAPIDS cuML/cuDF versions are tied to specific CUDA versions. Check which CUDA version your system supports:

```bash
# Check driver CUDA version
nvidia-smi | grep "CUDA Version"

# Check installed CUDA toolkit (if available)
nvcc --version 2>/dev/null || echo "CUDA toolkit not found in PATH"
```

**Common CUDA versions and RAPIDS compatibility**:
- **CUDA 11.8**: RAPIDS 23.12, 24.02 (cuML/cuDF)
- **CUDA 12.0/12.1**: RAPIDS 24.02+ (cuML/cuDF)
- **CUDA 12.4+**: RAPIDS 24.02+ (cuML/cuDF)

---

## 3. Install cuML and cuDF via micromamba

### Option A: Using the installation script (Recommended)

The installation script automatically detects your CUDA version and installs the appropriate RAPIDS packages:

```bash
./scripts/install_gpu_support.sh
```

### Option B: Manual installation

#### Step 1: Determine CUDA version

Check your CUDA version from `nvidia-smi` output. For example:
- If `nvidia-smi` shows "CUDA Version: 12.4", use `cudatoolkit=12.4`
- If it shows "CUDA Version: 11.8", use `cudatoolkit=11.8`

#### Step 2: Install RAPIDS packages

**For CUDA 12.x**:
```bash
# Activate your environment
micromamba activate financial-ml

# Install RAPIDS cuML and cuDF with CUDA 12.x support
micromamba install -y -c rapidsai -c conda-forge -c nvidia \
    cuml=24.02 \
    cudf=24.02 \
    cudatoolkit=12.4 \
    python=3.12
```

**For CUDA 11.8**:
```bash
micromamba activate financial-ml

# Install RAPIDS cuML and cuDF with CUDA 11.8 support
micromamba install -y -c rapidsai -c conda-forge -c nvidia \
    cuml=24.02 \
    cudf=24.02 \
    cudatoolkit=11.8 \
    python=3.12
```

**Note**: Replace `cudatoolkit=12.4` or `cudatoolkit=11.8` with the version matching your system's CUDA version.

#### Step 3: Install GPU monitoring tool (optional but recommended)

```bash
pip install nvidia-ml-py
```

### Option C: Add to environment.yaml

Uncomment and modify the GPU section in `environment.yaml` to match your CUDA version:

```yaml
dependencies:
  # ... existing dependencies ...
  - cudatoolkit=12.4  # Match your CUDA version (11.8 or 12.x)
  - cuml=24.02
  - cudf=24.02
```

Then recreate the environment:

```bash
micromamba env update -f environment.yaml
```

---

## 4. Verify Installation

Test that cuML and cuDF are working:

```bash
python -c "
import cuml
import cudf
print(f'cuML version: {cuml.__version__}')
print(f'cuDF version: {cudf.__version__}')

from cuml.ensemble import RandomForestClassifier
print('✅ RandomForestClassifier imported successfully')

# Check GPU availability
from cuml.common import device_type
print(f'✅ cuML device type: {device_type.get_device_type()}')

# Test cuDF
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
gdf = cudf.from_pandas(df)
print(f'✅ cuDF working: {len(gdf)} rows')
"
```

Expected output:
```
cuML version: 24.02.xx
cuDF version: 24.02.xx
✅ RandomForestClassifier imported successfully
✅ cuML device type: gpu
✅ cuDF working: 3 rows
```

---

## 5. Run Experiments with GPU

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

## 6. GPU Profiling

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

## 7. Troubleshooting

### Issue: cuML not found

**Error**: `ImportError: cuML is not available`

**Solution**:
```bash
# Verify installation
micromamba list | grep cuml

# Reinstall if needed (adjust CUDA version)
micromamba install -y -c rapidsai -c conda-forge -c nvidia cuml=24.02 cudatoolkit=12.4
```

### Issue: CUDA version mismatch

**Error**: CUDA version incompatibility

**Solution**: Check CUDA version:
```bash
nvidia-smi  # Check driver CUDA version
nvcc --version  # Check installed CUDA toolkit (if available)
```

Install matching cudatoolkit version:
```bash
# For CUDA 12.x
micromamba install -y -c nvidia cudatoolkit=12.4

# For CUDA 11.8
micromamba install -y -c nvidia cudatoolkit=11.8
```

**Important**: The `cudatoolkit` version should match or be compatible with your driver's CUDA version shown in `nvidia-smi`.

### Issue: GPU not detected

**Error**: cuML falls back to CPU

**Solution**:
1. Verify GPU is detected: `nvidia-smi`
2. Check CUDA drivers are installed and up to date
3. Verify cuML installation: `python -c "import cuml; print(cuml.__version__)"`
4. Check GPU Compute Capability: `nvidia-smi --query-gpu=compute_cap --format=csv`
   - Must be 6.0 or higher

### Issue: Out of memory

**Error**: GPU memory errors

**Solution**:
- Reduce `n_estimators` in model config
- Reduce batch size or dataset size
- Monitor GPU memory: `watch -n 1 nvidia-smi`
- Use smaller models or process data in chunks

### Issue: cuDF not working

**Error**: cuDF import or conversion errors

**Solution**:
```bash
# Verify cuDF installation
micromamba list | grep cudf

# Reinstall if needed
micromamba install -y -c rapidsai -c conda-forge -c nvidia cudf=24.02 cudatoolkit=12.4
```

---

## 8. Performance Comparison

After running both CPU and GPU experiments, compare:

| Metric | CPU (sklearn) | GPU (cuML) | Speedup |
|--------|---------------|------------|---------|
| Training time (fold 0) | X seconds | Y seconds | X/Y |
| Total pipeline time | X seconds | Y seconds | X/Y |
| GPU utilization | N/A | Z% | - |
| GPU memory peak | N/A | W MB | - |

Check MLflow for detailed metrics.

---

## 9. Best Practices

1. **Start with small experiments** to verify GPU setup
2. **Monitor GPU utilization** during training
3. **Compare CPU vs GPU** on same dataset for fair comparison
4. **Profile before optimizing** to identify bottlenecks
5. **Use GPU for large datasets** where speedup is significant (>100k samples)
6. **Match CUDA versions** - ensure `cudatoolkit` matches your driver's CUDA version
7. **Check GPU memory** - ensure sufficient VRAM for your dataset size

---

## 10. Supported GPU Models

RAPIDS cuML/cuDF supports most modern NVIDIA GPUs with Compute Capability 6.0+:

**GeForce Series**:
- GTX 10xx series (Pascal)
- RTX 20xx series (Turing)
- RTX 30xx series (Ampere)
- RTX 40xx series (Ada Lovelace)

**Quadro/Professional Series**:
- P100, P4000, P5000, P6000
- RTX 4000, RTX 5000, RTX 6000, RTX 8000

**Data Center GPUs**:
- V100 (Volta)
- A100, A40, A10 (Ampere)
- H100 (Hopper)

**Check your GPU**: Visit [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) to verify compatibility.

---

## 11. Additional Resources

- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [RAPIDS cuDF Documentation](https://docs.rapids.ai/api/cudf/stable/)
- [CUDA Compatibility Guide](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [RAPIDS Installation Guide](https://docs.rapids.ai/install)
- [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)

---

## Notes

- **CUDA Version Compatibility**: 
  - RAPIDS 24.02 supports CUDA 11.8 and CUDA 12.x
  - Always match `cudatoolkit` version with your driver's CUDA version
  - Check [RAPIDS release notes](https://docs.rapids.ai/release_notes) for latest compatibility

- **GPU acceleration is most beneficial for**:
  - Large datasets (>100k samples)
  - Many features (>100 features)
  - Large models (many trees, deep trees)
  - Data processing with cuDF (bar construction, feature engineering)

- **CPU may still be faster for**:
  - Small datasets (<10k samples) due to GPU overhead
  - Simple models with few features
  - When GPU memory is insufficient

- **Memory Considerations**:
  - GPU memory (VRAM) is typically smaller than system RAM
  - Monitor GPU memory usage: `nvidia-smi`
  - Process large datasets in batches if needed
