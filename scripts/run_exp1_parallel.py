#!/usr/bin/env python3
"""
Quick script to run Experiment 1 with Parallel Verification on 3 GPUs

This demonstrates the full utilization of all 3 RTX PRO 6000 GPUs for a single experiment:
- GPU 0: Inference
- GPU 0, 1, 2: Parallel verification (3x speedup)

Usage:
    python scripts/run_exp1_parallel.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments"))

from exp1_parallel_verification import Exp1ParallelVerification


def main():
    print("=" * 80)
    print("Running Experiment 1 with Parallel Verification")
    print("This will use all 3 GPUs simultaneously:")
    print("  - GPU 0: Inference phase")
    print("  - GPUs 0, 1, 2: Parallel verification phase (3x speedup)")
    print("=" * 80)
    print()

    # Configuration
    MODEL_NAME = "qwen2.5-7b"
    GPU_IDS = [0, 1, 2]
    NUM_VERIFIERS = 3

    # Create and run experiment
    exp = Exp1ParallelVerification(
        model_name=MODEL_NAME,
        gpu_ids=GPU_IDS,
        num_verifiers=NUM_VERIFIERS
    )

    print(f"Starting experiment with model: {MODEL_NAME}")
    print(f"Using GPUs: {GPU_IDS}")
    print(f"Running {NUM_VERIFIERS} verifiers in parallel")
    print()

    exp.run()

    print()
    print("=" * 80)
    print("Experiment complete!")
    print("Check data/raw/exp1_parallel/ for results")
    print("=" * 80)


if __name__ == "__main__":
    main()
