# Multi-GPU Parallel Execution Guide

This guide explains how to utilize all 3 RTX PRO 6000 GPUs for parallel experiment execution.

## Hardware Configuration

Your server has:
- 3x NVIDIA RTX PRO 6000 Ada Generation
- 95GB VRAM per GPU (285GB total)
- Blackwell architecture

## Execution Strategies

### Strategy 1: Run All Experiments in Parallel (Recommended)

Run all 5 experiments simultaneously across the 3 GPUs:

```bash
cd /home/user/verillm-experiments
source venv/bin/activate
python scripts/run_parallel_experiments.py --mode all --gpus 0 1 2
```

**GPU Assignment:**
- GPU 0: Exp 1 (Homogeneous FP16)
- GPU 1: Exp 2 (Heterogeneous FP16)
- GPU 2: Exp 3 (Quantized + Homogeneous)
- GPU 0: Exp 4 (Quantized + Heterogeneous) - reuses after Exp 1 completes
- GPU 1: Exp 5 (FP16 Inference + Quantized Verification) - reuses after Exp 2 completes

**Expected Runtime:** ~15-25 minutes (compared to 50-70 minutes sequential)

---

### Strategy 2: Run Specific Experiments

Run only specific experiments:

```bash
# Run experiments 1, 3, and 5 (homogeneous experiments)
python scripts/run_parallel_experiments.py \
    --mode experiments \
    --experiments exp1 exp3 exp5 \
    --gpus 0 1 2

# Run experiments 1 and 2
python scripts/run_parallel_experiments.py \
    --mode experiments \
    --experiments exp1 exp2 \
    --gpus 0 1
```

---

### Strategy 3: Run Multiple Trials for Threshold Optimization

Run many trials of a single experiment in parallel for better threshold statistics:

```bash
# Run 30 trials of Experiment 1 across 3 GPUs
python scripts/run_parallel_experiments.py \
    --mode trials \
    --experiment exp1 \
    --trials 30 \
    --gpus 0 1 2

# This will distribute trials:
# GPU 0: trials 1, 4, 7, 10, 13, 16, 19, 22, 25, 28
# GPU 1: trials 2, 5, 8, 11, 14, 17, 20, 23, 26, 29
# GPU 2: trials 3, 6, 9, 12, 15, 18, 21, 24, 27, 30
```

**Use cases:**
- Threshold optimization (need 20-50 trials for good statistics)
- Reproducibility testing
- Statistical significance validation

---

## Performance Comparison

| Method | Experiments | Runtime | GPU Utilization |
|--------|-------------|---------|-----------------|
| Sequential | All 5 | ~70 min | ~33% (1 GPU) |
| Parallel (3 GPUs) | All 5 | ~25 min | ~100% (all 3 GPUs) |
| Parallel trials | 30 trials | ~15 min | ~100% (all 3 GPUs) |

**Speedup:** 2.8-3x faster with parallel execution

---

## Memory Requirements

Per-experiment VRAM usage:

| Experiment | Model | VRAM Usage | Fits on PRO 6000? |
|-----------|-------|------------|-------------------|
| Exp 1 | FP16 (inference + verification) | ~28GB | Yes (95GB available) |
| Exp 2 | FP16 (cross-device) | ~14GB per GPU | Yes |
| Exp 3 | INT4 + FP16 | ~22GB | Yes |
| Exp 4 | INT4 + FP16 (cross-device) | ~18GB | Yes |
| Exp 5 | FP16 + INT4 | ~22GB | Yes |

All experiments fit comfortably on a single PRO 6000 with 95GB VRAM.

---

## Advanced Usage

### Custom GPU Assignment

Specify which GPUs to use:

```bash
# Use only GPU 0 and GPU 2
python scripts/run_parallel_experiments.py \
    --mode all \
    --gpus 0 2

# Use all 3 GPUs
python scripts/run_parallel_experiments.py \
    --mode all \
    --gpus 0 1 2
```

### Different Model

```bash
python scripts/run_parallel_experiments.py \
    --mode all \
    --gpus 0 1 2 \
    --model llama-3.1-8b
```

### Monitoring Execution

Watch GPU utilization in real-time:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

---

## Output and Results

Results are saved to:
- Individual trials: `data/raw/exp{N}/`
- Parallel execution summary: `data/parallel_execution/parallel_execution_{mode}_{timestamp}.json`

The summary includes:
- Per-task execution time
- GPU assignment
- Success/failure status
- Accept rates (for trials mode)

---

## Troubleshooting

### Out of Memory

If you encounter OOM errors:

1. Reduce max_tokens in `configs/prompts.yaml`
2. Run fewer experiments in parallel
3. Use 2 GPUs instead of 3

### Process Hanging

If a process hangs:

```bash
# Kill all Python processes
pkill -9 python

# Check GPU processes
nvidia-smi

# Kill specific GPU process
sudo fuser -k /dev/nvidia0
```

### Uneven Load

The script uses round-robin GPU assignment. For uneven experiment runtimes:
- Faster experiments will allow GPU reuse
- Monitor with `nvidia-smi` to see real-time load

---

## Recommended Workflow

For initial testing:

```bash
# 1. Run all experiments once to verify setup
python scripts/run_parallel_experiments.py --mode all --gpus 0 1 2

# 2. If successful, run 30 trials for threshold optimization
python scripts/run_parallel_experiments.py \
    --mode trials \
    --experiment exp1 \
    --trials 30 \
    --gpus 0 1 2
```

For production runs:

```bash
# Run 50 trials of each experiment for robust statistics
for exp in exp1 exp3 exp5; do
    python scripts/run_parallel_experiments.py \
        --mode trials \
        --experiment $exp \
        --trials 50 \
        --gpus 0 1 2
done
```

---

## Integration with Threshold Optimization

After running parallel trials, use the collected data for threshold optimization:

```bash
# The parallel trials save results in the same format
# You can analyze them with the threshold optimization tools

python scripts/optimize_thresholds.py \
    --experiment exp1 \
    --trials 30
```

---

## Summary

With 3x RTX PRO 6000 GPUs, you can:
- Run all 5 experiments in ~25 minutes (vs 70 minutes sequential)
- Collect 30 trials for threshold optimization in ~15 minutes
- Fully utilize your hardware investment
- Get statistically significant results faster

The parallel execution script handles GPU assignment, error recovery, and result aggregation automatically.
