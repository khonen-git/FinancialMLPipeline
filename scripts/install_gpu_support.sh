#!/bin/bash
# Install GPU support (cuML and cuDF) for FinancialMLPipeline
# Automatically detects CUDA version and installs appropriate RAPIDS packages

set -e

echo "🚀 Installing GPU support (RAPIDS cuML and cuDF)"
echo ""

# Check if micromamba is available
if ! command -v micromamba &> /dev/null; then
    echo "❌ Error: micromamba not found. Please install micromamba first."
    exit 1
fi

# Check if environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No conda/micromamba environment activated."
    echo "   Activating 'financial-ml' environment..."
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate financial-ml
fi

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. NVIDIA GPU drivers may not be installed."
    echo "   Please install NVIDIA drivers first."
    exit 1
fi

echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
echo ""

# Detect CUDA version from nvidia-smi
CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -n1 | cut -d'.' -f1,2)
echo "🔍 Detected CUDA version: ${CUDA_VERSION}"

# Determine cudatoolkit version
if [[ "$CUDA_VERSION" == "12."* ]]; then
    CUDATOOLKIT="12.4"
    echo "   Using cudatoolkit=${CUDATOOLKIT} for CUDA 12.x"
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    CUDATOOLKIT="11.8"
    echo "   Using cudatoolkit=${CUDATOOLKIT} for CUDA 11.x"
else
    echo "⚠️  Warning: Unsupported CUDA version: ${CUDA_VERSION}"
    echo "   RAPIDS supports CUDA 11.8 and 12.x"
    echo "   Attempting with cudatoolkit=12.4 (you may need to adjust manually)"
    CUDATOOLKIT="12.4"
fi

# Check GPU Compute Capability
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
echo "   GPU Compute Capability: ${COMPUTE_CAP}"
if [[ $(echo "$COMPUTE_CAP < 6.0" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
    echo "⚠️  Warning: GPU Compute Capability ${COMPUTE_CAP} may not be fully supported"
    echo "   RAPIDS requires Compute Capability 6.0 or higher"
fi

# Install cuML and cuDF with detected CUDA version
echo ""
echo "📦 Installing RAPIDS cuML and cuDF (CUDA ${CUDATOOLKIT}) via micromamba..."
echo "   This may take several minutes..."
echo ""

micromamba install -y -c rapidsai -c conda-forge -c nvidia \
    cuml=24.02 \
    cudf=24.02 \
    cudatoolkit=${CUDATOOLKIT} \
    python=3.12

# Install nvidia-ml-py for GPU monitoring (optional but recommended)
echo ""
echo "📦 Installing nvidia-ml-py for GPU monitoring..."
micromamba run -n financial-ml pip install nvidia-ml-py

# Verify installation
echo ""
echo "✅ Installation complete! Verifying..."
python -c "
try:
    import cuml
    print(f'✅ cuML version: {cuml.__version__}')
    
    import cudf
    print(f'✅ cuDF version: {cudf.__version__}')
    
    from cuml.ensemble import RandomForestClassifier
    print('✅ RandomForestClassifier imported successfully')
    
    # Check GPU availability
    from cuml.common import device_type
    device = device_type.get_device_type()
    print(f'✅ cuML device type: {device}')
    
    if device != 'gpu':
        print('⚠️  Warning: cuML is not using GPU. Check CUDA installation.')
    
    print('')
    print('🎉 GPU support is ready!')
except ImportError as e:
    print(f'❌ Error importing RAPIDS packages: {e}')
    exit(1)
"

echo ""
echo "📝 Next steps:"
echo "   1. Update your experiment config: models.model.backend='gpu'"
echo "   2. Run experiment: python scripts/run_experiment.py +models=rf_gpu +experiment=<your_experiment>"
echo "   3. Profile GPU: python scripts/profile_gpu_pipeline.py +models=rf_gpu +experiment=<your_experiment>"
