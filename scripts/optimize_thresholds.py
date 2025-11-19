"""
Threshold Optimization Script

This script runs multiple trials across different experiment configurations
to collect statistical data and determine optimal thresholds for verification.

Usage:
    python scripts/optimize_thresholds.py --experiment exp1 --trials 20
    python scripts/optimize_thresholds.py --experiment all --trials 10
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import argparse
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import experiment classes
from experiments.exp1_homogeneous import Exp1Homogeneous
from experiments.exp2_heterogeneous import Exp2Heterogeneous
from experiments.exp3_quantized_inference_homogeneous import Exp3QuantizedInferenceHomogeneous
from experiments.exp4_quantized_inference_heterogeneous import Exp4QuantizedInferenceHeterogeneous
from experiments.exp5_full_inference_quantized_verification import Exp5FullInferenceQuantizedVerification


class ThresholdOptimizer:
    """Optimize verification thresholds through extensive testing"""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else project_root / "data/threshold_optimization"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_statistics(self, experiment_class, experiment_config: dict, num_trials: int = 10) -> Dict:
        """Run multiple trials and collect all statistics"""
        print(f"\n{'='*80}")
        print(f"Collecting statistics for {experiment_config['name']}")
        print(f"Number of trials: {num_trials}")
        print(f"{'='*80}\n")

        all_stats = {
            'Pe': [],
            'Pm': [],
            'Pw': [],
            'mean_error': [],
            'max_error': [],
            'accept_rate': []
        }

        for trial in range(num_trials):
            print(f"\n--- Trial {trial + 1}/{num_trials} ---")

            # Create experiment instance
            exp = experiment_class(**experiment_config['params'])

            # Get a single prompt with config for consistency
            prompts_with_config = exp.get_prompts_with_config(num_prompts=1)
            prompt, max_tokens, min_tokens = prompts_with_config[0]

            # Run single trial
            result = exp.run_single_trial(prompt, trial_id=trial + 1, max_tokens=max_tokens, min_tokens=min_tokens)

            # Extract statistics from all verifiers
            for verifier_data in result['verifiers']:
                stats = verifier_data['statistics']
                per_layer = stats['per_layer']

                # Collect per-layer statistics
                for layer_key, layer_stats in per_layer.items():
                    all_stats['Pe'].append(layer_stats['Pe'])
                    all_stats['Pm'].append(layer_stats.get('Pm', 0.0))
                    all_stats['Pw'].append(layer_stats.get('Pw', 0.0))
                    all_stats['mean_error'].append(layer_stats['mean_error'])
                    all_stats['max_error'].append(layer_stats['max_error'])

                # Collect summary accept rate
                all_stats['accept_rate'].append(stats['summary']['accept_rate'])

        return all_stats

    def analyze_statistics(self, stats: Dict) -> Dict:
        """Analyze collected statistics and recommend thresholds"""
        analysis = {}

        for metric, values in stats.items():
            values_array = np.array(values)

            analysis[metric] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'percentile_50': float(np.percentile(values_array, 50)),
                'percentile_90': float(np.percentile(values_array, 90)),
                'percentile_95': float(np.percentile(values_array, 95)),
                'percentile_99': float(np.percentile(values_array, 99)),
                'count': len(values_array)
            }

        return analysis

    def recommend_thresholds(self, analysis: Dict, experiment_type: str) -> Dict:
        """Recommend optimal thresholds based on analysis"""

        # Conservative approach: use 95th percentile for error metrics
        # and 5th percentile for accept_rate

        recommended = {
            'Pe_max': min(0.10, analysis['Pe']['percentile_95'] * 1.2),  # 95th percentile + 20% margin
            'Pm_min': max(0.50, analysis['Pm']['percentile_50'] * 0.8),  # 50th percentile - 20% margin
            'Pw_min': max(0.30, analysis['Pw']['percentile_50'] * 0.8),  # 50th percentile - 20% margin
            'mean_error_max': min(0.10, analysis['mean_error']['percentile_95'] * 1.2),
            'accept_rate_min': max(0.70, analysis['accept_rate']['percentile_50'] * 0.9),  # Conservative
        }

        # Adjust based on experiment type
        if 'homogeneous' in experiment_type.lower() and 'quantized' not in experiment_type.lower():
            # Stricter for homogeneous FP16
            recommended['accept_rate_min'] = max(0.95, analysis['accept_rate']['percentile_50'] * 0.95)
            recommended['Pe_max'] = min(0.02, analysis['Pe']['percentile_90'])

        elif 'heterogeneous' in experiment_type.lower() and 'quantized' not in experiment_type.lower():
            # Moderate for heterogeneous FP16
            recommended['accept_rate_min'] = max(0.90, analysis['accept_rate']['percentile_50'] * 0.90)
            recommended['Pe_max'] = min(0.05, analysis['Pe']['percentile_95'])

        elif 'quantized' in experiment_type.lower() and 'homogeneous' in experiment_type.lower():
            # Relaxed for quantized + homogeneous
            recommended['accept_rate_min'] = max(0.80, analysis['accept_rate']['percentile_50'] * 0.85)
            recommended['Pe_max'] = min(0.10, analysis['Pe']['percentile_95'])

        elif 'quantized' in experiment_type.lower() and 'heterogeneous' in experiment_type.lower():
            # Most relaxed for quantized + heterogeneous
            recommended['accept_rate_min'] = max(0.70, analysis['accept_rate']['percentile_50'] * 0.80)
            recommended['Pe_max'] = min(0.15, analysis['Pe']['percentile_99'])

        return recommended

    def plot_distributions(self, stats: Dict, experiment_name: str):
        """Plot distributions of collected statistics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Statistics Distribution - {experiment_name}', fontsize=16)

        metrics = ['Pe', 'Pm', 'Pw', 'mean_error', 'max_error', 'accept_rate']

        for idx, metric in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            values = stats[metric]
            ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(f'{metric} Distribution')
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.axvline(np.mean(values), color='r', linestyle='--', label=f'Mean: {np.mean(values):.4f}')
            ax.axvline(np.percentile(values, 95), color='g', linestyle='--', label=f'95th: {np.percentile(values, 95):.4f}')
            ax.legend()

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"{experiment_name}_distribution.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")
        plt.close()

    def save_results(self, experiment_name: str, stats: Dict, analysis: Dict, recommended: Dict):
        """Save all results to JSON"""
        results = {
            'experiment': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'raw_statistics': {k: [float(v) for v in vals] for k, vals in stats.items()},
            'analysis': analysis,
            'recommended_thresholds': recommended
        }

        output_file = self.output_dir / f"{experiment_name}_threshold_optimization.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Optimize verification thresholds')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'all'],
                       help='Which experiment to optimize')
    parser.add_argument('--trials', type=int, default=10,
                       help='Number of trials to run')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for experiments')

    args = parser.parse_args()

    # Define experiment configurations
    experiments = []

    if args.experiment in ['exp1', 'all']:
        experiments.append({
            'name': 'exp1_homogeneous',
            'class': Exp1Homogeneous,
            'params': {
                'model_name': 'qwen2.5-7b',
                'device': args.device,
                'num_verifiers': 3
            }
        })

    if args.experiment in ['exp2', 'all']:
        experiments.append({
            'name': 'exp2_heterogeneous',
            'class': Exp2Heterogeneous,
            'params': {
                'model_name': 'qwen2.5-7b',
                'inference_device': args.device,
                'verification_device': 'mps' if 'cuda' in args.device else 'cuda:0',
                'num_verifiers': 3
            }
        })

    if args.experiment in ['exp3', 'all']:
        experiments.append({
            'name': 'exp3_quantized_homogeneous',
            'class': Exp3QuantizedInferenceHomogeneous,
            'params': {
                'model_name': 'qwen2.5-7b',
                'inference_device': args.device,
                'verification_device': args.device,
                'num_verifiers': 3
            }
        })

    if args.experiment in ['exp4', 'all']:
        experiments.append({
            'name': 'exp4_quantized_heterogeneous',
            'class': Exp4QuantizedInferenceHeterogeneous,
            'params': {
                'model_name': 'qwen2.5-7b',
                'inference_device': args.device,
                'verification_device': 'mps' if 'cuda' in args.device else 'cuda:0',
                'num_verifiers': 3
            }
        })

    if args.experiment in ['exp5', 'all']:
        experiments.append({
            'name': 'exp5_full_quantized_verification',
            'class': Exp5FullInferenceQuantizedVerification,
            'params': {
                'model_name': 'qwen2.5-7b',
                'inference_device': args.device,
                'verification_device': args.device,
                'num_verifiers': 3
            }
        })

    # Run optimization
    optimizer = ThresholdOptimizer()

    all_recommendations = {}

    for exp_config in experiments:
        try:
            # Collect statistics
            stats = optimizer.collect_statistics(
                exp_config['class'],
                exp_config,
                num_trials=args.trials
            )

            # Analyze statistics
            analysis = optimizer.analyze_statistics(stats)

            # Get recommendations
            recommended = optimizer.recommend_thresholds(analysis, exp_config['name'])

            # Plot distributions
            optimizer.plot_distributions(stats, exp_config['name'])

            # Save results
            optimizer.save_results(exp_config['name'], stats, analysis, recommended)

            # Store recommendations
            all_recommendations[exp_config['name']] = recommended

            # Print summary
            print(f"\n{'='*80}")
            print(f"RECOMMENDED THRESHOLDS FOR {exp_config['name'].upper()}")
            print(f"{'='*80}")
            for key, value in recommended.items():
                print(f"  {key}: {value:.4f}")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\nError processing {exp_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary of all recommendations
    summary_file = optimizer.output_dir / "threshold_recommendations_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'trials_per_experiment': args.trials,
            'recommendations': all_recommendations
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"SUMMARY SAVED TO: {summary_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
