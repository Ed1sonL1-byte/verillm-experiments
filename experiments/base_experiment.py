"""Base Experiment Runner for VeriLLM Experiments"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import numpy as np
from typing import Dict, List, Tuple
from src.utils.logger import setup_logger
from src.utils.data_saver import DataSaver
from src.models.model_loader import ModelLoader
from src.inference.inferencer import Inferencer
from src.verification.verifier import Verifier
from src.analysis.statistics import HiddenStateStatistics


class BaseExperiment:
    """Base class for all VeriLLM experiments"""

    def __init__(self, exp_name: str, output_dir: str):
        self.exp_name = exp_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(exp_name)

        # Load experiment configuration
        with open(project_root / "configs/experiments.yaml", 'r') as f:
            self.exp_config = yaml.safe_load(f)

        # Load prompts
        with open(project_root / "configs/prompts.yaml", 'r') as f:
            self.prompts_config = yaml.safe_load(f)

    def get_prompts(self, num_prompts: int = 3) -> List[str]:
        """Get diverse prompts for testing"""
        prompts = []
        for prompt_type in ['type_a', 'type_b', 'type_c']:
            templates = self.prompts_config['prompts'][prompt_type]['templates']
            prompts.extend(templates[:num_prompts // 3 + 1])
        return prompts[:num_prompts]

    def get_prompts_with_config(self, num_prompts: int = 3) -> List[Tuple[str, int, int]]:
        """
        Get diverse prompts with their generation configs.

        Returns:
            List of tuples: (prompt_text, max_tokens, min_tokens)
        """
        prompts_with_config = []
        prompt_types = ['type_a', 'type_b', 'type_c']

        # Distribute prompts across types
        prompts_per_type = num_prompts // len(prompt_types)
        remainder = num_prompts % len(prompt_types)

        for idx, prompt_type in enumerate(prompt_types):
            type_config = self.prompts_config['prompts'][prompt_type]
            templates = type_config['templates']
            max_tokens = type_config.get('max_tokens', 3000)
            min_tokens = type_config.get('min_tokens', 500)

            # Take more prompts from first types if there's a remainder
            num_to_take = prompts_per_type + (1 if idx < remainder else 0)

            for template in templates[:num_to_take]:
                prompts_with_config.append((template, max_tokens, min_tokens))

        return prompts_with_config[:num_prompts]

    def compare_hidden_states(self, inf_hs: Dict, ver_hs: Dict) -> Dict:
        """
        Compare hidden states from inference and verification.

        Args:
            inf_hs: {token_idx: {layer_idx: np.ndarray}}
            ver_hs: {token_idx: {layer_idx: np.ndarray}}

        Returns:
            Statistics dict with per-layer and summary results
        """
        all_layer_stats = {}

        for token_idx in inf_hs.keys():
            if token_idx not in ver_hs:
                continue

            for layer_idx in inf_hs[token_idx].keys():
                if layer_idx not in ver_hs[token_idx]:
                    continue

                inf_hidden = inf_hs[token_idx][layer_idx]
                ver_hidden = ver_hs[token_idx][layer_idx]

                stats = HiddenStateStatistics.compute_float_diff_statistics(
                    inf_hidden, ver_hidden
                )

                thresholds = self.exp_config['experiment']['thresholds']
                verdict = HiddenStateStatistics.check_thresholds(stats, thresholds)

                key = f"token_{token_idx}_layer_{layer_idx}"
                all_layer_stats[key] = {
                    **stats,
                    'verdict': 'ACCEPT' if verdict else 'REJECT'
                }

        # Compute summary statistics
        accept_count = sum(1 for s in all_layer_stats.values() if s['verdict'] == 'ACCEPT')
        total_count = len(all_layer_stats)
        accept_rate = accept_count / total_count if total_count > 0 else 0.0

        # Aggregate metrics
        avg_mean_error = np.mean([s['mean_error'] for s in all_layer_stats.values()])
        avg_Pe = np.mean([s['Pe'] for s in all_layer_stats.values()])

        return {
            'per_layer': all_layer_stats,
            'summary': {
                'accept_rate': accept_rate,
                'accept_count': accept_count,
                'total_count': total_count,
                'avg_mean_error': float(avg_mean_error),
                'avg_Pe': float(avg_Pe)
            }
        }

    def compute_verification_overhead(self, inf_time: float, ver_time: float) -> Dict:
        """Compute verification overhead metrics"""
        ratio = ver_time / inf_time if inf_time > 0 else 0.0
        percentage = ratio * 100

        return {
            'inference_time': inf_time,
            'verification_time': ver_time,
            'ratio': ratio,
            'percentage': percentage
        }

    def save_result(self, result: Dict, filename: str):
        """Save experiment result to JSON file"""
        output_file = self.output_dir / filename
        DataSaver.save_json(result, str(output_file))
        self.logger.info(f"Result saved to: {output_file}")

    def run(self):
        """Override this method in subclasses"""
        raise NotImplementedError("Subclasses must implement run() method")
