# VeriLLM Experiments Implementation Guide

## Overview

This guide helps you complete all 5 experiments for the VeriLLM paper replication.

## Experiment Framework

### Completed Components

1. **Base Experiment Class** ([experiments/base_experiment.py](experiments/base_experiment.py))
   - Handles common functionality: prompt loading, hidden state comparison, overhead calculation
   - Provides consistent result saving and logging

2. **Enhanced Statistics Module** ([src/analysis/statistics.py](src/analysis/statistics.py))
   - Implements VeriLLM paper Table 3 & 4 metrics
   - Bit-level float comparison (Pe, Pm, Pw)
   - Delta magnitude categorization

3. **Experiment 1 - Homogeneous** ([experiments/exp1_homogeneous.py](experiments/exp1_homogeneous.py))
   - Baseline: same hardware for inference and verification
   - 3 diverse prompts × 3 verifiers per trial
   - Complete implementation ready to run

## Experiment Specifications

### Experiment 1: Homogeneous Hardware Baseline ✅ DONE

**File**: `experiments/exp1_homogeneous.py`

**Setup**:
- Inference device: NVIDIA GPU (or Mac M4)
- Verification device: Same as inference
- Model: Full precision (FP16/FP32)
- Verifiers: 3 per trial

**Run**:
```bash
cd experiments
python exp1_homogeneous.py
```

**Expected Output**:
- `data/raw/exp1/{model}_{device}_trial{1-3}.json` - Individual trials
- `data/raw/exp1/{model}_{device}_summary.json` - Aggregated results

---

### Experiment 2: Heterogeneous Hardware

**File**: `experiments/exp2_heterogeneous.py` (TO CREATE)

**Setup**:
- Inference device: NVIDIA GPU
- Verification device: Mac M4 CPU/MPS (or vice versa)
- Model: Full precision on both sides
- Expected: Slightly higher Pe due to vendor-specific floating-point handling

**Implementation Template**:

```python
"""
Experiment 2: Heterogeneous Hardware Cross-Verification
"""
from base_experiment import BaseExperiment
from src.models.model_loader import ModelLoader
from src.inference.inferencer import Inferencer
from src.verification.verifier import Verifier


class Exp2Heterogeneous(BaseExperiment):
    def __init__(self, model_name: str, inference_device: str, verification_device: str, num_verifiers: int = 3):
        super().__init__(
            exp_name=f"exp2_heterogeneous_{model_name}_{inference_device}_to_{verification_device}",
            output_dir="data/raw/exp2"
        )
        self.model_name = model_name
        self.inference_device = inference_device
        self.verification_device = verification_device
        self.num_verifiers = num_verifiers

    def run_single_trial(self, prompt: str, trial_id: int) -> dict:
        # Load model on inference device
        model_loader_inf = ModelLoader()
        model_inf, tokenizer = model_loader_inf.load_model(self.model_name, device=self.inference_device)

        # Run inference
        inferencer = Inferencer(model_inf, tokenizer, self.inference_device, self.logger)
        inference_result = inferencer.generate_with_hidden_states(
            prompt=prompt, max_new_tokens=1000, temperature=0.7, top_p=0.9, sample_layers_every=8
        )

        # Load model on verification device
        model_loader_ver = ModelLoader()
        model_ver, _ = model_loader_ver.load_model(self.model_name, device=self.verification_device)

        verification_results = []
        for ver_id in range(self.num_verifiers):
            verifier = Verifier(model_ver, tokenizer, self.verification_device, self.logger)
            ver_result = verifier.verify_with_prefill(
                prompt=inference_result['prompt'],
                generated_tokens=inference_result['generated_tokens'],
                sample_layers_every=8
            )
            verification_results.append(ver_result)

        # Compare and aggregate (similar to Exp1)
        # ...

    def run(self):
        # Similar to Exp1
        pass


def main():
    exp = Exp2Heterogeneous(
        model_name="qwen2.5-7b",
        inference_device="cuda",
        verification_device="mps",  # or "cpu"
        num_verifiers=3
    )
    exp.run()


if __name__ == "__main__":
    main()
```

---

### Experiment 3: Quantized Inference + Homogeneous Verification

**File**: `experiments/exp3_quantized_homo.py` (TO CREATE)

**Setup**:
- Inference: Quantized model (AWQ/GPTQ) on NVIDIA
- Verification: Full precision model on same NVIDIA GPU
- Expected: Higher Pe, detectable drift in statistics

**Key Changes**:
```python
# In ModelLoader, support quantized models:
def load_model(self, model_name: str, device: str, quantized: bool = False):
    if quantized:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            f"{model_name}-AWQ",  # e.g., "Qwen/Qwen2.5-7B-Instruct-AWQ"
            device_map=device
        )
    else:
        # Load normal model
        ...
```

**Run**:
```python
exp = Exp3QuantizedHomo(
    model_name="qwen2.5-7b",
    device="cuda",
    inference_quantized=True,   # ← KEY
    verification_quantized=False,
    num_verifiers=3
)
```

---

### Experiment 4: Quantized Inference + Heterogeneous Verification

**File**: `experiments/exp4_quantized_hetero.py` (TO CREATE)

**Setup**:
- Inference: Quantized model on NVIDIA
- Verification: Full precision on Mac M4
- Expected: Highest Pe/Pm, combined effect of quantization + cross-vendor drift

**Implementation**: Combine Exp2 and Exp3 logic

---

### Experiment 5: Normal Inference + Quantized Verification

**File**: `experiments/exp5_normal_inf_quantized_ver.py` (TO CREATE)

**Setup**:
- Inference: Full precision on NVIDIA
- Verification: Quantized model on same/different hardware
- Expected: Should FAIL verification (verifier shortcuts computation)

**Purpose**: Demonstrate detection of lazy/malicious verifier using lower-precision model

---

## Batch Execution Script

**File**: `run_all_experiments.sh` (TO CREATE)

```bash
#!/bin/bash

echo "=== VeriLLM Experiment Suite ==="

# Experiment 1
echo "[1/5] Running Experiment 1: Homogeneous..."
python experiments/exp1_homogeneous.py

# Experiment 2
echo "[2/5] Running Experiment 2: Heterogeneous..."
python experiments/exp2_heterogeneous.py

# Experiment 3
echo "[3/5] Running Experiment 3: Quantized Inference + Homo Verification..."
python experiments/exp3_quantized_homo.py

# Experiment 4
echo "[4/5] Running Experiment 4: Quantized Inference + Hetero Verification..."
python experiments/exp4_quantized_hetero.py

# Experiment 5
echo "[5/5] Running Experiment 5: Normal Inference + Quantized Verification..."
python experiments/exp5_normal_inf_quantized_ver.py

echo "=== All Experiments Complete ==="
echo "Results saved in data/raw/exp{1-5}/"
```

Make executable:
```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

---

## Expected Results Format

Each experiment generates JSON files with this structure:

```json
{
  "experiment": "exp1_homogeneous",
  "model": "qwen2.5-7b",
  "device": "cuda",
  "trials": [
    {
      "trial_id": 1,
      "inference": {
        "generated_length": 856,
        "timing": { "total_time": 127.3, "tokens_per_second": 6.7 }
      },
      "verifiers": [
        {
          "verifier_id": 0,
          "statistics": {
            "summary": {
              "accept_rate": 0.985,
              "avg_mean_error": 0.009,
              "avg_Pe": 0.048
            }
          },
          "overhead": { "percentage": 0.78 },
          "verdict": "PASS"
        }
      ]
    }
  ],
  "aggregate": {
    "avg_accept_rate": 0.982,
    "avg_overhead_percentage": 0.85
  }
}
```

## Generating Paper Tables

After running all experiments, create analysis script:

**File**: `scripts/generate_paper_tables.py`

```python
"""Generate Tables 3 & 4 from VeriLLM paper"""
import json
from pathlib import Path

def load_exp_results(exp_num):
    """Load all trial results for an experiment"""
    results_dir = Path(f"data/raw/exp{exp_num}")
    summary_files = list(results_dir.glob("*_summary.json"))
    return [json.load(open(f)) for f in summary_files]

def generate_table_3():
    """Table 3: Homogeneous Hardware (M4, RTX 5090)"""
    exp1_results = load_exp_results(1)
    # Extract: Exact, Exp Match, Exp Mismatch, Mean ε
    # Format as markdown table
    ...

def generate_table_4():
    """Table 4: Heterogeneous + Quantization"""
    exp2_results = load_exp_results(2)
    exp3_results = load_exp_results(3)
    exp4_results = load_exp_results(4)
    # Compare cross-device and quantization effects
    ...

if __name__ == "__main__":
    generate_table_3()
    generate_table_4()
```

---

## Next Steps

1. **Create Exp2-5 scripts** following the templates above
2. **Update [model_loader.py](src/models/model_loader.py)** to support quantized models
3. **Run experiments** on your hardware
4. **Analyze results** and generate paper tables
5. **Commit to GitHub** with updated README showing results

## Troubleshooting

### CUDA Out of Memory
- Reduce `max_new_tokens` from 1000 to 500
- Use smaller model (Qwen2.5-3B)

### Model Loading Issues
- Ensure model names match Hugging Face repository names
- For quantized models, install AutoAWQ or AutoGPTQ:
  ```bash
  pip install auto-gptq autoawq
  ```

### Cross-device Timing Issues
- Ensure fair comparison by warming up GPU
- Run multiple trials and average

---

**Questions?** Check `QUICKSTART.md` or open an issue.
