#!/bin/bash
# Run Experiment 1 on A100 GPU

set -e  # Exit on error

echo "=========================================="
echo "VeriLLM Experiment 1: Homogeneous Baseline"
echo "Hardware: NVIDIA A100"
echo "=========================================="

# Environment check
echo "[1/5] Checking environment..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on the A100 server?"
    exit 1
fi

echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# Activate virtual environment
echo ""
echo "[2/5] Activating virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
else
    echo "ERROR: venv not found. Run setup first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check Python dependencies
echo ""
echo "[3/5] Checking dependencies..."
python -c "import torch; import transformers; print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Create output directories
echo ""
echo "[4/5] Creating output directories..."
mkdir -p data/raw/exp1
mkdir -p logs

# Run experiment
echo ""
echo "[5/5] Running Experiment 1..."
echo "This will take approximately 15-30 minutes per trial (3 trials total)"
echo "Progress will be logged to logs/exp1_homogeneous_*.log"
echo ""

cd experiments
python exp1_homogeneous.py

echo ""
echo "=========================================="
echo "Experiment 1 Complete!"
echo "Results saved in: data/raw/exp1/"
echo "=========================================="
echo ""
echo "Quick summary:"
ls -lh ../data/raw/exp1/*.json
