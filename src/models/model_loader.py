"""模型加载器"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import yaml
from pathlib import Path

class ModelLoader:
    def __init__(self, config_path: str = "configs/models.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_model(self, model_name: str, device: str = "cuda", quantized: bool = False):
        """加载模型和tokenizer"""
        if model_name not in self.config['models']:
            raise ValueError(f"Model '{model_name}' not found")
        model_config = self.config['models'][model_name]
        if quantized and 'quantized_version' in model_config:
            model_path = model_config['quantized_version'].get('local_path', model_config['quantized_version']['name'])
        else:
            model_path = model_config.get('local_path', model_config['name'])
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
