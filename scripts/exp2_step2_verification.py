"""
Experiment 2 - Step 2: Verification Only (Run on Mac)

This script loads hidden states from inference (done on NVIDIA) and performs
verification using prefill on Mac hardware.

Usage:
    python scripts/exp2_step2_verification.py --input-dir data/raw/exp2/inference --device mps
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments"))

import argparse
import json
import pickle
import numpy as np
import torch
from datetime import datetime

from src.models.model_loader import ModelLoader
from src.verification.verifier import Verifier
from src.analysis.statistics import HiddenStateStatistics
from src.utils.logger import setup_logger

import yaml


class Exp2VerificationOnly:
    """Run verification only using saved hidden states from inference"""

    def __init__(self, model_name: str, device: str, input_dir: Path, output_dir: Path):
        self.model_name = model_name
        self.device = device
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(f"exp2_verification_{model_name}_{device}")

        # Load experiment config for thresholds
        with open(project_root / "configs/experiments.yaml", 'r') as f:
            self.exp_config = yaml.safe_load(f)

    def load_inference_data(self, metadata_file: Path) -> dict:
        """Load inference metadata and hidden states"""
        self.logger.info(f"Loading inference data from {metadata_file}")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Load hidden states
        hidden_states_file = metadata_file.parent / metadata['hidden_states_file']
        with open(hidden_states_file, 'rb') as f:
            hidden_states = pickle.load(f)

        metadata['hidden_states'] = hidden_states
        return metadata

    def compare_hidden_states(self, inf_hs: dict, ver_hs: dict) -> dict:
        """Compare hidden states from inference and verification"""
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

    def run_verification(self, inference_data: dict, num_verifiers: int = 3) -> dict:
        """Run verification on loaded inference data"""
        self.logger.info("=" * 80)
        self.logger.info(f"Verification for Trial {inference_data['trial_id']}")
        self.logger.info(f"Original inference device: {inference_data['inference_device']}")
        self.logger.info(f"Verification device: {self.device}")
        self.logger.info(f"Generated tokens: {inference_data['num_tokens_generated']}")
        self.logger.info("=" * 80)

        # Load model for verification
        self.logger.info(f"Loading model {self.model_name} to {self.device}...")
        model_loader = ModelLoader()
        model, tokenizer = model_loader.load_model(self.model_name, device=self.device)

        # Run multiple verifications
        verification_results = []

        for ver_id in range(num_verifiers):
            self.logger.info(f"Starting verification {ver_id + 1}/{num_verifiers}...")

            verifier = Verifier(model, tokenizer, self.device, self.logger)

            ver_result = verifier.verify_with_prefill(
                prompt=inference_data['prompt'],
                generated_tokens=inference_data['generated_tokens'],
                sample_layers_every=8
            )

            self.logger.info(f"Verification {ver_id + 1} complete: {ver_result['timing']['total_time']:.2f}s")

            # Compare hidden states
            stats = self.compare_hidden_states(
                inference_data['hidden_states'],
                ver_result['hidden_states']
            )

            verification_results.append({
                'verifier_id': ver_id,
                'timing': ver_result['timing'],
                'statistics': stats,
                'verdict': 'PASS' if stats['summary']['accept_rate'] > 0.90 else 'FAIL'
            })

            self.logger.info(f"Verifier {ver_id + 1}: Accept rate = {stats['summary']['accept_rate']*100:.2f}%")
            self.logger.info(f"Verifier {ver_id + 1}: Verdict = {verification_results[-1]['verdict']}")

        # Free model memory
        del model
        if self.device == 'mps':
            torch.mps.empty_cache()
        elif self.device.startswith('cuda'):
            torch.cuda.empty_cache()

        # Aggregate results
        avg_accept_rate = np.mean([v['statistics']['summary']['accept_rate'] for v in verification_results])
        avg_verification_time = np.mean([v['timing']['total_time'] for v in verification_results])

        # Compute overhead (verification time vs inference time)
        inference_time = inference_data['timing']['total_time']
        overhead_percentage = (avg_verification_time / inference_time) * 100

        return {
            'experiment': 'exp2_heterogeneous',
            'step': 'verification',
            'trial_id': inference_data['trial_id'],
            'input_category': inference_data['input_category'],
            'model': self.model_name,
            'inference_device': inference_data['inference_device'],
            'verification_device': self.device,
            'inference_timing': inference_data['timing'],
            'verifiers': verification_results,
            'summary': {
                'avg_accept_rate': float(avg_accept_rate),
                'avg_verification_time': float(avg_verification_time),
                'inference_time': float(inference_time),
                'overhead_percentage': float(overhead_percentage),
                'final_verdict': 'PASS' if avg_accept_rate > 0.90 else 'FAIL'
            }
        }

    def run(self, num_verifiers: int = 3):
        """Run verification for all inference results"""
        self.logger.info("=" * 80)
        self.logger.info("EXPERIMENT 2 - STEP 2: VERIFICATION ONLY")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Number of verifiers per trial: {num_verifiers}")
        self.logger.info("=" * 80)

        # Find all metadata files
        metadata_files = sorted(self.input_dir.glob("*_metadata.json"))
        self.logger.info(f"Found {len(metadata_files)} inference results to verify")

        all_results = []

        for metadata_file in metadata_files:
            try:
                # Load inference data
                inference_data = self.load_inference_data(metadata_file)

                # Run verification
                result = self.run_verification(inference_data, num_verifiers)
                all_results.append(result)

                # Save individual result
                result_file = self.output_dir / f"exp2_verification_trial{result['trial_id']}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                self.logger.info(f"Result saved to: {result_file}")

            except Exception as e:
                self.logger.error(f"Failed to process {metadata_file}: {e}")
                import traceback
                traceback.print_exc()

        # Save summary
        if all_results:
            summary = {
                'experiment': 'exp2_heterogeneous',
                'model': self.model_name,
                'inference_device': all_results[0]['inference_device'],
                'verification_device': self.device,
                'num_trials': len(all_results),
                'num_verifiers_per_trial': num_verifiers,
                'aggregate': {
                    'avg_accept_rate': float(np.mean([r['summary']['avg_accept_rate'] for r in all_results])),
                    'avg_overhead_percentage': float(np.mean([r['summary']['overhead_percentage'] for r in all_results])),
                    'pass_rate': sum(1 for r in all_results if r['summary']['final_verdict'] == 'PASS') / len(all_results)
                },
                'timestamp': datetime.now().isoformat()
            }

            summary_file = self.output_dir / "exp2_verification_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            self.logger.info("\n" + "=" * 80)
            self.logger.info("VERIFICATION COMPLETE")
            self.logger.info(f"Completed {len(all_results)} trials")
            self.logger.info(f"Average accept rate: {summary['aggregate']['avg_accept_rate']*100:.2f}%")
            self.logger.info(f"Average overhead: {summary['aggregate']['avg_overhead_percentage']:.2f}%")
            self.logger.info(f"Pass rate: {summary['aggregate']['pass_rate']*100:.2f}%")
            self.logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Exp2 Step 2: Verification Only')
    parser.add_argument('--model', type=str, default='qwen2.5-7b',
                        help='Model name (default: qwen2.5-7b)')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device for verification (default: mps for Mac)')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing inference results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: data/raw/exp2/verification)')
    parser.add_argument('--num-verifiers', type=int, default=3,
                        help='Number of verification runs per trial (default: 3)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data/raw/exp2/verification"

    runner = Exp2VerificationOnly(
        model_name=args.model,
        device=args.device,
        input_dir=input_dir,
        output_dir=output_dir
    )

    runner.run(num_verifiers=args.num_verifiers)


if __name__ == "__main__":
    main()
