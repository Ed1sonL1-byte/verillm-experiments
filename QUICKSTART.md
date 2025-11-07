# Quick Start Guide

## 1. Environment Setup (5 minutes)

```bash
# Extract or clone the project
cd verillm-experiments

# Setup environment (for Linux)
bash scripts/setup_environment.sh

# Activate virtual environment
source venv/bin/activate
```

## 2. Test Environment (1 minute)

```bash
python test_setup.py
```

Expected output:
```
✅ CUDA available - RTX 5090
✅ Model loaded successfully
✅ Inference successful
✅ All tests passed!
```

## 3. Download Models (30 minutes - 1 hour)

```bash
# Method 1: Using script (recommended)
bash scripts/download_models.sh

# Method 2: Manual download
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
"
```

## 4. Run First Experiment (10-20 minutes)

```bash
# Run experiment 1
python experiments/exp1_homogeneous.py
```

**Expected output:**
```
[2024-11-06 10:00:00] Starting inference, max tokens: 1000
[2024-11-06 10:05:00] Inference complete, generated 800 tokens
[2024-11-06 10:05:05] Verification complete, time: 0.5s
[2024-11-06 10:05:05] Verification overhead ratio: 0.85%
[2024-11-06 10:05:06] Verification pass rate: 98.5%
[2024-11-06 10:05:06] Final verdict: PASS
```

**Result files:**
- `data/raw/exp1/qwen2.5-7b_cuda_run1.json` - Experiment data
- `logs/exp1_homogeneous_YYYYMMDD_HHMMSS.log` - Detailed logs

## 5. View Results

```python
import json

# Load results
with open('data/raw/exp1/qwen2.5-7b_cuda_run1.json', 'r') as f:
    result = json.load(f)

# Key metrics
print(f"Verification overhead: {result['overhead']['percentage']:.2f}%")
print(f"Verification pass rate: {result['statistics']['summary']['accept_rate']*100:.2f}%")
print(f"Verdict: {result['verdict']}")
```

## Common Issues

### Q1: CUDA out of memory
**Solution**: Reduce max_new_tokens
```python
# Modify in experiments/exp1_homogeneous.py
inference_result = inferencer.generate_with_hidden_states(
    prompt=prompt,
    max_new_tokens=500,  # Change from 1000 to 500
    ...
)
```

### Q2: Model download too slow
**Solution**: Use domestic mirror (for China)
```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_models.sh
```

### Q3: GPU not detected
**Check**: CUDA installation
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

### Q4: Permission denied on scripts
**Solution**: Make scripts executable
```bash
chmod +x scripts/*.sh
```

## Next Steps

- Modify configuration files: `configs/experiments.yaml`
- Add custom prompts: `configs/prompts.yaml`
- Run other experiments: `experiments/exp2_heterogeneous.py`

## Get Help

View full documentation: `README.md`
