"""
Threshold Optimization for VeriLLM Experiment 1
Runs multiple trials to determine optimal verification thresholds
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

import json
import numpy as np
from datetime import datetime
from experiments.exp1_homogeneous import Exp1Homogeneous


class ThresholdOptimizer:
    def __init__(self):
        self.output_dir = Path(__file__).parent.parent / "data/threshold_optimization"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_trials(self, num_trials, device):
        print(f"\n{'='*80}")
        print(f"Running {num_trials} trials on {device}")
        print(f"{'='*80}\n")

        all_stats = {
            'Pe': [], 'Pm': [], 'Pw': [],
            'mean_error': [], 'max_error': [], 'accept_rate': []
        }

        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}:")
            
            exp = Exp1Homogeneous(
                model_name='qwen2.5-7b', 
                device=device, 
                num_verifiers=3
            )
            prompts = exp.get_prompts(num_prompts=1)
            result = exp.run_single_trial(prompts[0], trial_id=trial + 1)

            for verifier_data in result['verifiers']:
                stats = verifier_data['statistics']
                for layer_key, layer_stats in stats['per_layer'].items():
                    all_stats['Pe'].append(layer_stats['Pe'])
                    all_stats['Pm'].append(layer_stats.get('Pm', 0.0))
                    all_stats['Pw'].append(layer_stats.get('Pw', 0.0))
                    all_stats['mean_error'].append(layer_stats['mean_error'])
                    all_stats['max_error'].append(layer_stats['max_error'])
                all_stats['accept_rate'].append(stats['summary']['accept_rate'])

            accept_rate = stats['summary']['accept_rate']
            print(f"  Accept rate: {accept_rate*100:.2f}%\n")

        return all_stats

    def analyze_and_save(self, stats):
        # Calculate statistics
        analysis = {}
        for metric, values in stats.items():
            arr = np.array(values)
            analysis[metric] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'p50': float(np.percentile(arr, 50)),
                'p90': float(np.percentile(arr, 90)),
                'p95': float(np.percentile(arr, 95)),
                'count': len(arr)
            }

        # Recommend thresholds based on percentiles
        recommended = {
            'Pe_max': min(0.02, analysis['Pe']['p90']),
            'Pm_min': max(0.75, analysis['Pm']['p50'] * 0.8),
            'Pw_min': max(0.50, analysis['Pw']['p50'] * 0.8),
            'mean_error_max': min(0.01, analysis['mean_error']['p95'] * 1.2),
            'accept_rate_min': max(0.95, analysis['accept_rate']['p50'] * 0.95),
        }

        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'trials': len(stats['accept_rate']) // 3,
            'analysis': analysis,
            'recommended_thresholds': recommended
        }
        
        output_file = self.output_dir / "exp1_threshold_optimization.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Print summary
        print(f"\n{'='*80}")
        print(f"Recommended thresholds (based on {results['trials']} trials):")
        print(f"{'='*80}")
        for key, value in recommended.items():
            print(f"  {key:20s}: {value:.6f}")
        
        print(f"\nStatistical summary:")
        print(f"  Accept rate: mean={analysis['accept_rate']['mean']:.4f}, "
              f"min={analysis['accept_rate']['min']:.4f}, "
              f"max={analysis['accept_rate']['max']:.4f}")
        print(f"  Pe: mean={analysis['Pe']['mean']:.6f}, "
              f"p90={analysis['Pe']['p90']:.6f}")
        
        print(f"\nResults saved to: {output_file}")
        print(f"{'='*80}\n")

        return recommended


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimize verification thresholds for Experiment 1'
    )
    parser.add_argument('--trials', type=int, default=10, 
                        help='Number of trials to run (default: 10)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='GPU device to use (default: cuda:0)')
    args = parser.parse_args()

    print(f"\nVeriLLM Experiment 1 - Threshold Optimization")
    print(f"Trials: {args.trials}")
    print(f"Device: {args.device}")
    print(f"Verifiers per trial: 3")
    print(f"Estimated time: {args.trials * 2}-{args.trials * 3} minutes\n")

    optimizer = ThresholdOptimizer()
    
    try:
        stats = optimizer.run_trials(args.trials, args.device)
        optimizer.analyze_and_save(stats)
        print("Optimization completed successfully.")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
