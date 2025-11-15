#!/bin/bash

# VeriLLM Experiments - Parallel Execution Script for 3 GPUs
# Optimized for NVIDIA Pro 6000 x3

set -e

echo "======================================================================"
echo "VeriLLM Experiments - Parallel Execution"
echo "Hardware: 3x NVIDIA Pro 6000"
echo "======================================================================"

# Create logs directory
mkdir -p logs

# Check GPUs
echo ""
echo "Checking GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 3 ]; then
    echo "Warning: Expected 3 GPUs, found $GPU_COUNT"
    echo "Continuing anyway..."
fi

echo ""
echo "======================================================================"
echo "Phase 1: Running Experiments 1 and 3 in parallel"
echo "======================================================================"
echo "  - Exp 1 (Homogeneous FP16) on GPU 0"
echo "  - Exp 3 (Quantized Homogeneous) on GPU 2"
echo ""

cd experiments

# Start Experiment 1 on GPU 0
echo "[$(date '+%H:%M:%S')] Starting Experiment 1 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python exp1_homogeneous.py > ../logs/exp1_$(date '+%Y%m%d_%H%M%S').log 2>&1 &
EXP1_PID=$!
echo "  PID: $EXP1_PID"

# Wait a bit to avoid simultaneous model loading
sleep 10

# Start Experiment 3 on GPU 2
echo "[$(date '+%H:%M:%S')] Starting Experiment 3 on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup python exp3_quantized_inference_homogeneous.py > ../logs/exp3_$(date '+%Y%m%d_%H%M%S').log 2>&1 &
EXP3_PID=$!
echo "  PID: $EXP3_PID"

echo ""
echo "Waiting for Experiments 1 and 3 to complete..."
echo "You can monitor progress with:"
echo "  watch -n 1 nvidia-smi"
echo "  tail -f logs/exp1_*.log"
echo "  tail -f logs/exp3_*.log"
echo ""

wait $EXP1_PID
echo "[$(date '+%H:%M:%S')] Experiment 1 completed"

wait $EXP3_PID
echo "[$(date '+%H:%M:%S')] Experiment 3 completed"

echo ""
echo "======================================================================"
echo "Phase 2: Running Experiments 2, 4, and 5"
echo "======================================================================"
echo "  - Exp 2 (Heterogeneous FP16) on GPU 0 & 1"
echo "  - Exp 4 (Quantized Heterogeneous) - sequential after Exp 2"
echo "  - Exp 5 (FP16 to Quantized) on GPU 2"
echo ""

# Start Experiment 5 on GPU 2
echo "[$(date '+%H:%M:%S')] Starting Experiment 5 on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup python exp5_full_inference_quantized_verification.py > ../logs/exp5_$(date '+%Y%m%d_%H%M%S').log 2>&1 &
EXP5_PID=$!
echo "  PID: $EXP5_PID"

sleep 5

# Start Experiment 2 on GPU 0 and 1
echo "[$(date '+%H:%M:%S')] Starting Experiment 2 on GPU 0 & 1..."
python exp2_heterogeneous.py > ../logs/exp2_$(date '+%Y%m%d_%H%M%S').log 2>&1
echo "[$(date '+%H:%M:%S')] Experiment 2 completed"

# Wait for Experiment 5
wait $EXP5_PID
echo "[$(date '+%H:%M:%S')] Experiment 5 completed"

# Start Experiment 4
echo "[$(date '+%H:%M:%S')] Starting Experiment 4 on GPU 0 & 1..."
python exp4_quantized_inference_heterogeneous.py > ../logs/exp4_$(date '+%Y%m%d_%H%M%S').log 2>&1
echo "[$(date '+%H:%M:%S')] Experiment 4 completed"

cd ..

echo ""
echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "======================================================================"
echo ""
echo "Results location:"
echo "  - data/raw/exp1/"
echo "  - data/raw/exp2/"
echo "  - data/raw/exp3/"
echo "  - data/raw/exp4/"
echo "  - data/raw/exp5/"
echo ""
echo "Logs location: logs/"
echo ""
echo "View summary results:"
echo "  for exp in exp1 exp2 exp3 exp4 exp5; do"
echo "    echo \"=== \$exp ===\";"
echo "    cat data/raw/\$exp/*_summary.json | jq '.aggregate.avg_accept_rate';"
echo "  done"
echo ""
echo "======================================================================"
