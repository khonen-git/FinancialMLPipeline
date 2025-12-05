#!/bin/bash
# Install GPU support (cuML) for FinancialMLPipeline
# For RTX 4070 with CUDA 12.x

set -e

echo "🚀 Installing GPU support (RAPIDS cuML) for RTX 4070"
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
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
else
    echo "📊 GPU Information:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo ""
fi

# Install cuML with CUDA 12.x support
echo "📦 Installing RAPIDS cuML (CUDA 12.x) via micromamba..."
echo "   This may take several minutes..."
echo ""

micromamba install -y -c rapidsai -c conda-forge -c nvidia \
    cuml=24.02 \
    cudatoolkit=12.4 \
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
    
    from cuml.ensemble import RandomForestClassifier
    print('✅ RandomForestClassifier imported successfully')
    
    # Check GPU availability
    from cuml.common import device_type
    print(f'✅ cuML device type: {device_type.get_device_type()}')
    
    print('')
    print('🎉 GPU support is ready!')
except ImportError as e:
    print(f'❌ Error importing cuML: {e}')
    exit(1)
"

echo ""
echo "📝 Next steps:"
echo "   1. Update your experiment config: models.model.backend='gpu'"
echo "   2. Run experiment: python scripts/run_experiment.py +models=rf_gpu +experiment=<your_experiment>"
echo "   3. Profile GPU: python scripts/profile_gpu_pipeline.py +models=rf_gpu +experiment=<your_experiment>"

