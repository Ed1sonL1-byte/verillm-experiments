"""
Experiment 3 - Step 2: Full Precision Verification

This script loads hidden states from QUANTIZED inference and performs
verification using a FULL-PRECISION model.

Goal: Detect if quantized inference can be caught by full-precision verification.
Expected Result: REJECT - quantization should cause detectable deviation.

Experiment Design:
- Step 1: NVIDIA server runs QUANTIZED inference, saves hidden states
- Step 2a: NVIDIA server runs FULL-PRECISION verification (homogeneous)
- Step 2b: Mac server runs FULL-PRECISION verification (heterogeneous)
Both verifiers should detect the quantization attack.

Usage:
    # On NVIDIA server (homogeneous verification):
    python scripts/exp3_step2_full_precision_verification.py \\
        --input-dir data/raw/exp3/quantized_inference \\
        --device cuda:0 \\
        --verifier-name nvidia

    # On Mac server (heterogeneous verification):
    python scripts/exp3_step2_full_precision_verification.py \\
        --input-dir data/raw/exp3/quantized_inference \\
        --device mps \\
        --verifier-name mac
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


class Exp3FullPrecisionVerification:
    """Run verification with FULL-PRECISION model on hidden states from QUANTIZED inference"""

    def __init__(self, model_name: str, device: str, input_dir: Path, output_dir: Path, verifier_name: str = None):
        self.model_name = model_name
        self.device = device
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Verifier name helps distinguish between NVIDIA and Mac verification results
        self.verifier_name = verifier_name or ("nvidia" if device.startswith("cuda") else "mac")
        self.logger = setup_logger(f"exp3_fp_verification_{model_name}_{self.verifier_name}")

        # Load experiment config for thresholds
        with open(project_root / "configs/experiments.yaml", 'r') as f:
            self.exp_config = yaml.safe_load(f)

        # Model will be loaded once and reused
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load FULL-PRECISION model once for all verifications"""
        if self.model is None:
            self.logger.info(f"Loading FULL-PRECISION model {self.model_name} to {self.device}...")
            model_loader = ModelLoader()
            self.model, self.tokenizer = model_loader.load_model(
                self.model_name,
                device=self.device,
                quantized=False  # Full precision
            )
            self.logger.info("Full-precision model loaded successfully")

    def unload_model(self):
        """Free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.tokenizer = None
            if self.device == 'mps':
                torch.mps.empty_cache()
            elif self.device.startswith('cuda'):
                torch.cuda.empty_cache()
            self.logger.info("Model unloaded, memory freed")

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
        """Compare hidden states from quantized inference and full-precision verification"""
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
        """Run full-precision verification on quantized inference data"""
        self.logger.info("=" * 80)
        self.logger.info(f"Verification for Trial {inference_data['trial_id']}")
        self.logger.info(f"Original inference: {inference_data['inference_device']} (QUANTIZED)")
        self.logger.info(f"Verification: {self.device} (FULL-PRECISION)")
        self.logger.info(f"Generated tokens: {inference_data['num_tokens_generated']}")
        self.logger.info("=" * 80)

        # Model should already be loaded
        if self.model is None:
            self.load_model()

        # Run multiple verifications
        verification_results = []

        for ver_id in range(num_verifiers):
            self.logger.info(f"Starting verification {ver_id + 1}/{num_verifiers} (FULL-PRECISION)...")

            verifier = Verifier(self.model, self.tokenizer, self.device, self.logger)

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

            # For quantized inference, we EXPECT low accept rate (detection)
            # Using relaxed threshold of 0.80 for "PASS" (meaning cheating NOT detected)
            verification_results.append({
                'verifier_id': ver_id,
                'timing': ver_result['timing'],
                'statistics': stats,
                'verdict': 'PASS' if stats['summary']['accept_rate'] > 0.80 else 'FAIL'
            })

            self.logger.info(f"Verifier {ver_id + 1}: Accept rate = {stats['summary']['accept_rate']*100:.2f}%")
            self.logger.info(f"Verifier {ver_id + 1}: Avg Pe = {stats['summary']['avg_Pe']:.6f}")
            self.logger.info(f"Verifier {ver_id + 1}: Avg Mean Error = {stats['summary']['avg_mean_error']:.6f}")
            self.logger.info(f"Verifier {ver_id + 1}: Verdict = {verification_results[-1]['verdict']}")

        # Aggregate results
        avg_accept_rate = np.mean([v['statistics']['summary']['accept_rate'] for v in verification_results])
        avg_verification_time = np.mean([v['timing']['total_time'] for v in verification_results])
        avg_Pe = np.mean([v['statistics']['summary']['avg_Pe'] for v in verification_results])
        avg_mean_error = np.mean([v['statistics']['summary']['avg_mean_error'] for v in verification_results])

        # Compute overhead (verification time vs inference time)
        inference_time = inference_data['timing']['total_time']
        overhead_percentage = (avg_verification_time / inference_time) * 100

        # Final verdict: For exp3, we EXPECT FAIL (quantization detected)
        final_verdict = 'PASS' if avg_accept_rate > 0.80 else 'FAIL'

        return {
            'experiment': 'exp3_quantization_detection',
            'step': 'full_precision_verification',
            'verifier_name': self.verifier_name,
            'trial_id': inference_data['trial_id'],
            'input_category': inference_data['input_category'],
            'model': self.model_name,
            'inference_device': inference_data['inference_device'],
            'inference_precision': 'quantized',
            'verification_device': self.device,
            'verification_precision': 'full',
            'inference_timing': inference_data['timing'],
            'verifiers': verification_results,
            'summary': {
                'avg_accept_rate': float(avg_accept_rate),
                'avg_verification_time': float(avg_verification_time),
                'avg_Pe': float(avg_Pe),
                'avg_mean_error': float(avg_mean_error),
                'inference_time': float(inference_time),
                'overhead_percentage': float(overhead_percentage),
                'final_verdict': final_verdict,
                'quantization_detected': final_verdict == 'FAIL'
            }
        }

    def run(self, num_verifiers: int = 3):
        """Run verification for all quantized inference results"""
        self.logger.info("=" * 80)
        self.logger.info("EXPERIMENT 3 - STEP 2: FULL-PRECISION VERIFICATION")
        self.logger.info(f"Verifier: {self.verifier_name.upper()}")
        self.logger.info(f"Model: {self.model_name} (FULL-PRECISION)")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Number of verifiers per trial: {num_verifiers}")
        self.logger.info("=" * 80)

        # Load model once before all verifications
        self.load_model()

        # Find all metadata files
        metadata_files = sorted(self.input_dir.glob("*_metadata.json"))
        self.logger.info(f"Found {len(metadata_files)} quantized inference results to verify")

        all_results = []

        for metadata_file in metadata_files:
            try:
                # Load inference data
                inference_data = self.load_inference_data(metadata_file)

                # Run verification
                result = self.run_verification(inference_data, num_verifiers)
                all_results.append(result)

                # Save individual result (include verifier name in filename)
                result_file = self.output_dir / f"exp3_verification_{self.verifier_name}_trial{result['trial_id']}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                self.logger.info(f"Result saved to: {result_file}")

            except Exception as e:
                self.logger.error(f"Failed to process {metadata_file}: {e}")
                import traceback
                traceback.print_exc()

        # Unload model after all verifications
        self.unload_model()

        # Save summary
        if all_results:
            detected_count = sum(1 for r in all_results if r['summary']['quantization_detected'])

            summary = {
                'experiment': 'exp3_quantization_detection',
                'verifier_name': self.verifier_name,
                'model': self.model_name,
                'inference_device': all_results[0]['inference_device'],
                'inference_precision': 'quantized',
                'verification_device': self.device,
                'verification_precision': 'full',
                'num_trials': len(all_results),
                'num_verifiers_per_trial': num_verifiers,
                'aggregate': {
                    'avg_accept_rate': float(np.mean([r['summary']['avg_accept_rate'] for r in all_results])),
                    'avg_Pe': float(np.mean([r['summary']['avg_Pe'] for r in all_results])),
                    'avg_mean_error': float(np.mean([r['summary']['avg_mean_error'] for r in all_results])),
                    'avg_overhead_percentage': float(np.mean([r['summary']['overhead_percentage'] for r in all_results])),
                    'quantization_detection_rate': detected_count / len(all_results),
                    'pass_rate': sum(1 for r in all_results if r['summary']['final_verdict'] == 'PASS') / len(all_results)
                },
                'timestamp': datetime.now().isoformat()
            }

            # Include verifier name in summary filename
            summary_file = self.output_dir / f"exp3_verification_{self.verifier_name}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            self.logger.info("\n" + "=" * 80)
            self.logger.info("EXPERIMENT 3 VERIFICATION COMPLETE")
            self.logger.info(f"Completed {len(all_results)} trials")
            self.logger.info(f"Average accept rate: {summary['aggregate']['avg_accept_rate']*100:.2f}%")
            self.logger.info(f"Average Pe: {summary['aggregate']['avg_Pe']:.6f}")
            self.logger.info(f"Average Mean Error: {summary['aggregate']['avg_mean_error']:.6f}")
            self.logger.info(f"Average overhead: {summary['aggregate']['avg_overhead_percentage']:.2f}%")
            self.logger.info(f"Quantization detection rate: {summary['aggregate']['quantization_detection_rate']*100:.2f}%")
            self.logger.info("=" * 80)

            if summary['aggregate']['quantization_detection_rate'] > 0.8:
                self.logger.info("SUCCESS: Quantization attack detected in most trials!")
            else:
                self.logger.info("WARNING: Quantization attack NOT reliably detected!")


def main():
    parser = argparse.ArgumentParser(description='Exp3 Step 2: Full Precision Verification')
    parser.add_argument('--model', type=str, default='qwen2.5-7b',
                        help='Model name (default: qwen2.5-7b)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for verification (default: cuda:0, use mps for Mac)')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing quantized inference results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: data/raw/exp3/verification)')
    parser.add_argument('--num-verifiers', type=int, default=3,
                        help='Number of verification runs per trial (default: 3)')
    parser.add_argument('--verifier-name', type=str, default=None,
                        help='Name for this verifier (default: nvidia or mac based on device)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data/raw/exp3/verification"

    runner = Exp3FullPrecisionVerification(
        model_name=args.model,
        device=args.device,
        input_dir=input_dir,
        output_dir=output_dir,
        verifier_name=args.verifier_name
    )

    runner.run(num_verifiers=args.num_verifiers)


if __name__ == "__main__":
    main()
