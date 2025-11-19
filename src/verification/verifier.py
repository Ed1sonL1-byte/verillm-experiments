"""验证器核心模块"""
import torch
from typing import Dict, List
import numpy as np

class Verifier:
    def __init__(self, model, tokenizer, device, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger
    
    def verify_with_prefill(self, prompt: str, generated_tokens: List[int],
                           sample_layers_every: int = 8) -> Dict:
        """通过Prefill验证生成结果"""
        self.logger.info(f"开始验证，序列长度: {len(generated_tokens)}")
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        full_sequence_ids = prompt_ids + generated_tokens
        input_ids = torch.tensor([full_sequence_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        prompt_length = len(prompt_ids)
        import time
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True, return_dict=True)
        end_time = time.perf_counter()
        verification_time = end_time - start_time
        hidden_states_record = {}
        for token_idx in range(len(generated_tokens)):
            actual_pos = prompt_length + token_idx
            hidden_states_record[token_idx] = {}
            for layer_idx in range(0, len(outputs.hidden_states), sample_layers_every):
                hidden = outputs.hidden_states[layer_idx][:, actual_pos, :]
                hidden_states_record[token_idx][layer_idx] = hidden.cpu().numpy()
        return {
            'hidden_states': hidden_states_record,
            'timing': {'total_time': verification_time,
                      'tokens_per_second': len(generated_tokens) / verification_time}
        }
