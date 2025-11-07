# VeriLLM Experiments

Experimental reproduction project based on the paper "VeriLLM: Verifiable Decentralized Inference for LLMs".

## Project Overview

This project aims to validate the LLM verification mechanism proposed in the VeriLLM paper by detecting inference honesty through comparing hidden states between the inference side and verification side.

### Core Experiments

1. **Experiment 1**: Homogeneous Hardware Baseline Verification
2. **Experiment 2**: Heterogeneous Hardware Compatibility Testing
3. **Experiment 3**: Quantization Attack Detection (Homogeneous)
4. **Experiment 4**: Quantization Attack Detection (Heterogeneous)
5. **Experiment 5**: Lazy Verifier Detection
6. **Experiment 6**: Multi-Verifier Parallel Verification

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration Check

```bash
# Run environment test
python test_setup.py
```

### 3. Download Models

```bash
# Use provided script to download models
bash scripts/download_models.sh
```

### 4. Run First Experiment

```bash
# Run experiment 1
python experiments/exp1_homogeneous.py
```

## Project Structure

```
verillm-experiments/
├── configs/           # Configuration files
├── src/              # Source code
│   ├── models/       # Model management
│   ├── inference/    # Inference side
│   ├── verification/ # Verification side
│   ├── analysis/     # Data analysis
│   └── utils/        # Utility functions
├── experiments/      # Experiment scripts
├── data/            # Experimental data
├── models/          # Model files
└── logs/            # Log files
```

## Hardware Requirements

### Minimum Configuration
- GPU: NVIDIA RTX 5090 (24GB) or Apple M3 Max (64GB+)
- RAM: 64GB
- Storage: 200GB

### Recommended Configuration
- GPU: NVIDIA RTX 5090 + Apple M3 Max
- RAM: 128GB
- Storage: 500GB

## Experiment Workflow

### Experiment 1 Example
```python
from src.models.model_loader import ModelLoader
from src.inference.inferencer import Inferencer
from src.verification.verifier import Verifier

# Load model
loader = ModelLoader()
model, tokenizer = loader.load_model("qwen2.5-7b", device="cuda")

# Inference
inferencer = Inferencer(model, tokenizer, "cuda", logger)
result = inferencer.generate_with_hidden_states(prompt)

# Verification
verifier = Verifier(model, tokenizer, "cuda", logger)
verification = verifier.verify_with_prefill(prompt, result['generated_tokens'])
```

## Data Output

Each experiment generates:
- JSON result file (statistical data)
- Log file (detailed process)
- Hidden States (optional)

## Common Issues

### 1. CUDA out of memory
- Reduce batch size or sequence length
- Use quantized models

### 2. Model download failure
- Check network connection
- Use domestic mirror sources

### 3. MPS device unavailable
- Ensure macOS version >= 14.0
- Ensure Xcode Command Line Tools are installed

## Citation

```bibtex
@article{verillm2024,
  title={VeriLLM: Verifiable Decentralized Inference for LLMs},
  author={...},
  year={2024}
}
```

## Contact

For questions, please submit an Issue or contact the project maintainer.

## License

MIT License
