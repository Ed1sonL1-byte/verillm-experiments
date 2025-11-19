"""模型加载器"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import yaml
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent

class ModelLoader:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = project_root / "configs/models.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_model(self, model_name: str, device: str = "cuda", quantized: bool = False):
        """加载模型和tokenizer"""
        if model_name not in self.config['models']:
            raise ValueError(f"Model '{model_name}' not found")
        model_config = self.config['models'][model_name]

        # Determine model path: use HuggingFace name first, fallback to local if exists
        if quantized and 'quantized_version' in model_config:
            model_path = model_config['quantized_version']['name']
            local_path = Path(model_config['quantized_version'].get('local_path', ''))
        else:
            model_path = model_config['name']
            local_path = Path(model_config.get('local_path', ''))

        # If local path exists, use it; otherwise use HuggingFace
        if local_path.exists():
            model_path = str(local_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, 
            device_map=device if device != "mps" else None,
            trust_remote_code=True, low_cpu_mem_usage=True
        )
        if device == "mps":
            model = model.to(device)
        model.eval()
        return model, tokenizer
