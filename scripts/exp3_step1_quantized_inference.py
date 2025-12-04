"""
Experiment 3 - Step 1: Quantized Inference Only

This script performs inference using a QUANTIZED model and saves hidden states.
The hidden states can then be used for verification with a full-precision model.

Goal: Test if quantization causes detectable deviation in hidden states.

Usage:
    python scripts/exp3_step1_quantized_inference.py --trials 10 --device cuda:0

    # For faster inference with reduced tokens:
    python scripts/exp3_step1_quantized_inference.py --trials 10 --device cuda:0 --max-tokens 1000
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments"))

import argparse
import json
import pickle
import time
import numpy as np
import torch
from datetime import datetime

from src.models.model_loader import ModelLoader
from src.inference.inferencer import Inferencer
from src.utils.logger import setup_logger


class Exp3QuantizedInferenceOnly:
    """Run inference with QUANTIZED model and save hidden states for later verification"""

    def __init__(self, model_name: str, device: str, output_dir: Path, max_tokens_override: int = None):
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(f"exp3_quantized_inference_{model_name}_{device}")
        self.max_tokens_override = max_tokens_override

        # Model will be loaded once and reused
        self.model = None
        self.tokenizer = None

        # Load prompts config
        import yaml
        with open(project_root / "configs/prompts.yaml", 'r') as f:
            self.prompts_config = yaml.safe_load(f)

    def load_model(self):
        """Load QUANTIZED model once for all trials"""
        if self.model is None:
            self.logger.info(f"Loading QUANTIZED model {self.model_name} to {self.device}...")
            model_loader = ModelLoader()
            self.model, self.tokenizer = model_loader.load_model(
                self.model_name,
                device=self.device,
                quantized=True  # Load quantized version
            )
            self.logger.info("Quantized model loaded successfully")

    def unload_model(self):
        """Free GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            self.logger.info("Model unloaded, GPU memory freed")

    def get_prompts_with_config(self, num_prompts: int = 10):
        """Get diverse prompts with their generation configs"""
        prompts_with_config = []
        prompt_types = ['short', 'medium', 'long']

        prompts_per_type = num_prompts // len(prompt_types)
        remainder = num_prompts % len(prompt_types)

        for idx, prompt_type in enumerate(prompt_types):
            type_config = self.prompts_config['prompts'][prompt_type]
            templates = type_config['templates']
            max_tokens = type_config.get('max_tokens', 4000)
            min_tokens = type_config.get('min_tokens', 500)

            num_to_take = prompts_per_type + (1 if idx < remainder else 0)

            for template in templates[:num_to_take]:
                prompts_with_config.append((template, max_tokens, min_tokens, prompt_type))

        return prompts_with_config[:num_prompts]

    def run_inference(self, prompt: str, trial_id: int, max_tokens: int,
                      min_tokens: int, input_category: str) -> dict:
        """Run inference with QUANTIZED model and return results with hidden states"""
        # Apply max_tokens override if set
        if self.max_tokens_override:
            max_tokens = self.max_tokens_override

        self.logger.info("=" * 80)
        self.logger.info(f"Trial {trial_id}: {self.model_name} (QUANTIZED) on {self.device}")
        self.logger.info(f"Input category: {input_category}")
        self.logger.info(f"Prompt length: {len(prompt)} chars")
        self.logger.info(f"Max tokens: {max_tokens}")
        self.logger.info("=" * 80)

        # Model should already be loaded
        if self.model is None:
            self.load_model()

        # Run inference (reuse loaded model)
        inferencer = Inferencer(self.model, self.tokenizer, self.device, self.logger)
        self.logger.info("Starting QUANTIZED inference (Prefill + Decode)...")

        inference_result = inferencer.generate_with_hidden_states(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            sample_layers_every=8
        )

        self.logger.info(f"Inference complete: {len(inference_result['generated_tokens'])} tokens")
        self.logger.info(f"Inference time: {inference_result['timing']['total_time']:.2f}s")

        return inference_result

    def save_inference_result(self, result: dict, trial_id: int, input_category: str):
        """Save inference result with hidden states to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"exp3_quantized_inference_trial{trial_id}_{input_category}_{timestamp}"

        # Save hidden states separately (large file, use pickle for efficiency)
        hidden_states_file = self.output_dir / f"{base_filename}_hidden_states.pkl"
        with open(hidden_states_file, 'wb') as f:
            pickle.dump(result['hidden_states'], f)
        self.logger.info(f"Hidden states saved to: {hidden_states_file}")

        # Save metadata and generated tokens (JSON for readability)
        metadata = {
            'experiment': 'exp3_quantized_inference_homogeneous',
            'step': 'quantized_inference',
            'trial_id': trial_id,
            'input_category': input_category,
            'model': self.model_name,
            'model_precision': 'quantized',
            'inference_device': self.device,
            'timestamp': timestamp,
            'prompt': result['prompt'],
            'generated_tokens': result['generated_tokens'],
            'generated_text': result['generated_text'],
            'timing': result['timing'],
            'hidden_states_file': str(hidden_states_file.name),
            'num_tokens_generated': len(result['generated_tokens']),
            'num_layers_sampled': len(result['hidden_states'].get(0, {})) if result['hidden_states'] else 0
        }

        metadata_file = self.output_dir / f"{base_filename}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Metadata saved to: {metadata_file}")

        return {
            'hidden_states_file': str(hidden_states_file),
            'metadata_file': str(metadata_file)
        }

    def run(self, num_trials: int = 10):
        """Run quantized inference for multiple trials"""
        self.logger.info("=" * 80)
        self.logger.info("EXPERIMENT 3 - STEP 1: QUANTIZED INFERENCE ONLY")
        self.logger.info(f"Model: {self.model_name} (QUANTIZED)")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Number of trials: {num_trials}")
        self.logger.info(f"Output directory: {self.output_dir}")
        if self.max_tokens_override:
            self.logger.info(f"Max tokens override: {self.max_tokens_override}")
        self.logger.info("=" * 80)

        # Load model once before all trials
        self.load_model()

        prompts_config = self.get_prompts_with_config(num_prompts=num_trials)
        all_files = []

        for trial_id, (prompt, max_tokens, min_tokens, category) in enumerate(prompts_config, 1):
            self.logger.info(f"\n===== Trial {trial_id}/{num_trials} ({category}) =====")

            try:
                # Run inference (model already loaded)
                result = self.run_inference(prompt, trial_id, max_tokens, min_tokens, category)

                # Save results
                files = self.save_inference_result(result, trial_id, category)
                all_files.append(files)

                self.logger.info(f"Trial {trial_id} completed successfully")

            except Exception as e:
                self.logger.error(f"Trial {trial_id} failed: {e}")
                import traceback
                traceback.print_exc()

        # Unload model after all trials
        self.unload_model()

        # Save summary
        summary = {
            'experiment': 'exp3_quantized_inference_homogeneous',
            'step': 'quantized_inference',
            'model': self.model_name,
            'model_precision': 'quantized',
            'device': self.device,
            'num_trials': num_trials,
            'completed_trials': len(all_files),
            'output_files': all_files,
            'timestamp': datetime.now().isoformat()
        }

        summary_file = self.output_dir / "exp3_quantized_inference_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("QUANTIZED INFERENCE COMPLETE")
        self.logger.info(f"Completed {len(all_files)}/{num_trials} trials")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Summary saved to: {summary_file}")
        self.logger.info("=" * 80)
        self.logger.info("\nNext step: Run exp3_step2_full_precision_verification.py")


def main():
    parser = argparse.ArgumentParser(description='Exp3 Step 1: Quantized Inference Only')
    parser.add_argument('--model', type=str, default='qwen2.5-7b',
                        help='Model name (default: qwen2.5-7b)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference (default: cuda:0)')
    parser.add_argument('--trials', type=int, default=10,
                        help='Number of trials (default: 10)')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Override max tokens for faster testing (default: use config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: data/raw/exp3/quantized_inference)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data/raw/exp3/quantized_inference"

    runner = Exp3QuantizedInferenceOnly(
        model_name=args.model,
        device=args.device,
        output_dir=output_dir,
        max_tokens_override=args.max_tokens
    )

    runner.run(num_trials=args.trials)


if __name__ == "__main__":
    main()
