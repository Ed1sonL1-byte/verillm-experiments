"""推理器核心模块"""
import torch
from typing import Dict, List
import numpy as np
from tqdm import tqdm

class Inferencer:
    def __init__(self, model, tokenizer, device, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger
    
    def generate_with_hidden_states(self, prompt: str, max_new_tokens: int = 1000,
                                    temperature: float = 0.7, top_p: float = 0.9,
                                    sample_layers_every: int = 8) -> Dict:
        """自回归生成并记录Hidden States"""
        self.logger.info(f"开始推理，最大tokens: {max_new_tokens}")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        prompt_length = input_ids.shape[1]
        generated_tokens = []
        hidden_states_record = {}
        import time
        start_time = time.perf_counter()
        with torch.no_grad():
            for step in tqdm(range(max_new_tokens), desc="生成中"):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                   output_hidden_states=True, return_dict=True)
                logits = outputs.logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens.append(next_token.item())
                hidden_states_record[step] = {}
                for layer_idx in range(0, len(outputs.hidden_states), sample_layers_every):
                    hidden = outputs.hidden_states[layer_idx][:, -1, :]
                    hidden_states_record[step][layer_idx] = hidden.cpu().numpy()
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return {
            'prompt': prompt, 'generated_tokens': generated_tokens,
            'generated_text': generated_text, 'hidden_states': hidden_states_record,
            'timing': {'total_time': inference_time, 
                      'tokens_per_second': len(generated_tokens) / inference_time}
        }
