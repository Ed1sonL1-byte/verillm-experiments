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
        """加载模型和tokenizer

        Args:
            model_name: Model name from configs/models.yaml
            device: Target device (cuda:0, cuda:1, mps, cpu)
            quantized: If True, load quantized version (AWQ/GPTQ)
        """
        if model_name not in self.config['models']:
            raise ValueError(f"Model '{model_name}' not found")
        model_config = self.config['models'][model_name]

        # Determine model path and quantization method
        quant_method = None
        if quantized and 'quantized_version' in model_config:
            model_path = model_config['quantized_version']['name']
            local_path = Path(model_config['quantized_version'].get('local_path', ''))
            quant_method = model_config['quantized_version'].get('method', 'awq')
        else:
            model_path = model_config['name']
            local_path = Path(model_config.get('local_path', ''))

        # If local path exists, use it; otherwise use HuggingFace
        if local_path.exists():
            model_path = str(local_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model based on quantization method
        if quantized and quant_method:
            model = self._load_quantized_model(model_path, device, quant_method)
        else:
            model = self._load_full_precision_model(model_path, device)

        model.eval()
        return model, tokenizer

    def _load_full_precision_model(self, model_path: str, device: str):
        """Load full-precision (FP16/FP32) model"""
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device if device != "mps" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        if device == "mps":
            model = model.to(device)
        return model

    def _load_quantized_model(self, model_path: str, device: str, quant_method: str):
        """Load quantized model (AWQ or GPTQ)"""
        if quant_method.lower() == 'awq':
            return self._load_awq_model(model_path, device)
        elif quant_method.lower() == 'gptq':
            return self._load_gptq_model(model_path, device)
        else:
            raise ValueError(f"Unsupported quantization method: {quant_method}")

    def _load_awq_model(self, model_path: str, device: str):
        """Load AWQ quantized model"""
        try:
            from awq import AutoAWQForCausalLM
            model = AutoAWQForCausalLM.from_quantized(
                model_path,
                fuse_layers=True,
                trust_remote_code=True,
                device_map=device
            )
            return model
        except ImportError:
            # Fallback: try loading with transformers (some AWQ models support this)
            print("Warning: autoawq not installed, trying transformers AutoModelForCausalLM...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            return model

    def _load_gptq_model(self, model_path: str, device: str):
        """Load GPTQ quantized model"""
        try:
            from transformers import GPTQConfig
            quantization_config = GPTQConfig(bits=4, disable_exllama=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            return model
        except ImportError:
            # Fallback: try loading without explicit GPTQ config
            print("Warning: GPTQ support issue, trying basic loading...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            return model
