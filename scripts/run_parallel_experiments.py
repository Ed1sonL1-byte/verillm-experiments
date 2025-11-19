"""
Multi-GPU Parallel Experiment Execution

This script enables parallel execution of experiments across multiple GPUs
to maximize hardware utilization.

Usage:
    # Run all experiments in parallel across 3 GPUs
    python scripts/run_parallel_experiments.py --mode all --gpus 0 1 2

    # Run multiple trials of exp1 in parallel
    python scripts/run_parallel_experiments.py --mode trials --experiment exp1 --trials 10 --gpus 0 1 2

    # Run specific experiments
    python scripts/run_parallel_experiments.py --mode experiments --experiments exp1 exp2 exp3 --gpus 0 1 2
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments"))

import argparse
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict
import json

# Import experiment classes
from exp1_homogeneous import Exp1Homogeneous
from exp2_heterogeneous import Exp2Heterogeneous
from exp3_quantized_inference_homogeneous import Exp3QuantizedInferenceHomogeneous
from exp4_quantized_inference_heterogeneous import Exp4QuantizedInferenceHeterogeneous
from exp5_full_inference_quantized_verification import Exp5FullInferenceQuantizedVerification


class ParallelExperimentRunner:
    """Run experiments in parallel across multiple GPUs"""

    def __init__(self, gpu_ids: List[int], model_name: str = "qwen2.5-7b"):
        self.gpu_ids = gpu_ids
        self.model_name = model_name
        self.num_gpus = len(gpu_ids)

    def run_experiment_on_gpu(self, exp_class, exp_name: str, gpu_id: int, **kwargs) -> Dict:
        """Run a single experiment on specified GPU"""
        print(f"\n[GPU {gpu_id}] Starting {exp_name}...")
        start_time = time.time()

        try:
            device = f"cuda:{gpu_id}"
            exp = exp_class(
                model_name=self.model_name,
                device=device,
                **kwargs
            )
            exp.run()

            elapsed_time = time.time() - start_time
            print(f"\n[GPU {gpu_id}] {exp_name} completed in {elapsed_time:.2f}s")

            return {
                'experiment': exp_name,
                'gpu_id': gpu_id,
                'status': 'success',
                'elapsed_time': elapsed_time
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"\n[GPU {gpu_id}] {exp_name} failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                'experiment': exp_name,
                'gpu_id': gpu_id,
                'status': 'failed',
                'error': str(e),
                'elapsed_time': elapsed_time
            }

    def run_trial_on_gpu(self, exp_class, exp_name: str, gpu_id: int,
                         prompt: str, trial_id: int, max_tokens: int,
                         min_tokens: int, **kwargs) -> Dict:
        """Run a single trial on specified GPU"""
        print(f"\n[GPU {gpu_id}] Starting {exp_name} trial {trial_id}...")
        start_time = time.time()

        try:
            device = f"cuda:{gpu_id}"
            exp = exp_class(
                model_name=self.model_name,
                device=device,
                **kwargs
            )

            result = exp.run_single_trial(prompt, trial_id, max_tokens, min_tokens)

            # Save result
            filename = f"{exp_name}_gpu{gpu_id}_trial{trial_id}.json"
            exp.save_result(result, filename)

            elapsed_time = time.time() - start_time
            print(f"\n[GPU {gpu_id}] {exp_name} trial {trial_id} completed in {elapsed_time:.2f}s")

            return {
                'experiment': exp_name,
                'trial_id': trial_id,
                'gpu_id': gpu_id,
                'status': 'success',
                'elapsed_time': elapsed_time,
                'accept_rate': result['summary']['avg_accept_rate']
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"\n[GPU {gpu_id}] {exp_name} trial {trial_id} failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                'experiment': exp_name,
                'trial_id': trial_id,
                'gpu_id': gpu_id,
                'status': 'failed',
                'error': str(e),
                'elapsed_time': elapsed_time
            }

    def run_experiments_parallel(self, experiments: List[str]) -> List[Dict]:
        """Run different experiments in parallel across GPUs"""
        print(f"\n{'='*80}")
        print(f"Running {len(experiments)} experiments in parallel on {self.num_gpus} GPUs")
        print(f"GPUs: {self.gpu_ids}")
        print(f"Experiments: {experiments}")
        print(f"{'='*80}\n")

        exp_configs = {
            'exp1': (Exp1Homogeneous, {'num_verifiers': 3}),
            'exp2': (Exp2Heterogeneous, {
                'inference_device': 'cuda:0',
                'verification_device': 'cuda:1',
                'num_verifiers': 3
            }),
            'exp3': (Exp3QuantizedInferenceHomogeneous, {
                'inference_device': 'cuda:0',
                'verification_device': 'cuda:0',
                'num_verifiers': 3
            }),
            'exp4': (Exp4QuantizedInferenceHeterogeneous, {
                'inference_device': 'cuda:0',
                'verification_device': 'cuda:1',
                'num_verifiers': 3
            }),
            'exp5': (Exp5FullInferenceQuantizedVerification, {
                'inference_device': 'cuda:0',
                'verification_device': 'cuda:0',
                'num_verifiers': 3
            }),
        }

        tasks = []
        for idx, exp_name in enumerate(experiments):
            gpu_id = self.gpu_ids[idx % self.num_gpus]
            exp_class, kwargs = exp_configs[exp_name]

            # Override device in kwargs for homogeneous experiments
            if exp_name in ['exp1', 'exp3', 'exp5']:
                kwargs = {**kwargs, 'device': f"cuda:{gpu_id}"}
                if 'inference_device' in kwargs:
                    kwargs['inference_device'] = f"cuda:{gpu_id}"
                if 'verification_device' in kwargs:
                    kwargs['verification_device'] = f"cuda:{gpu_id}"

            tasks.append((exp_class, exp_name, gpu_id, kwargs))

        results = []
        with ProcessPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = {
                executor.submit(
                    self.run_experiment_on_gpu,
                    exp_class, exp_name, gpu_id, **kwargs
                ): exp_name
                for exp_class, exp_name, gpu_id, kwargs in tasks
            }

            for future in as_completed(futures):
                exp_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Exception in {exp_name}: {e}")
                    results.append({
                        'experiment': exp_name,
                        'status': 'exception',
                        'error': str(e)
                    })

        return results

    def run_trials_parallel(self, experiment: str, num_trials: int) -> List[Dict]:
        """Run multiple trials of the same experiment in parallel"""
        print(f"\n{'='*80}")
        print(f"Running {num_trials} trials of {experiment} in parallel on {self.num_gpus} GPUs")
        print(f"GPUs: {self.gpu_ids}")
        print(f"{'='*80}\n")

        exp_configs = {
            'exp1': (Exp1Homogeneous, {}),
            'exp3': (Exp3QuantizedInferenceHomogeneous, {}),
            'exp5': (Exp5FullInferenceQuantizedVerification, {}),
        }

        if experiment not in exp_configs:
            raise ValueError(f"Experiment {experiment} not supported for parallel trials. Use exp1, exp3, or exp5.")

        exp_class, base_kwargs = exp_configs[experiment]

        # Get prompts with config
        temp_exp = exp_class(model_name=self.model_name, device='cuda:0', num_verifiers=3, **base_kwargs)
        prompts_with_config = temp_exp.get_prompts_with_config(num_prompts=num_trials)

        tasks = []
        for trial_id in range(1, num_trials + 1):
            gpu_id = self.gpu_ids[(trial_id - 1) % self.num_gpus]
            prompt, max_tokens, min_tokens = prompts_with_config[trial_id - 1]

            kwargs = {**base_kwargs, 'num_verifiers': 3}
            tasks.append((
                exp_class, experiment, gpu_id, prompt,
                trial_id, max_tokens, min_tokens, kwargs
            ))

        results = []
        with ProcessPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = {
                executor.submit(
                    self.run_trial_on_gpu,
                    exp_class, exp_name, gpu_id, prompt,
                    trial_id, max_tokens, min_tokens, **kwargs
                ): trial_id
                for exp_class, exp_name, gpu_id, prompt, trial_id, max_tokens, min_tokens, kwargs in tasks
            }

            for future in as_completed(futures):
                trial_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Exception in trial {trial_id}: {e}")
                    results.append({
                        'trial_id': trial_id,
                        'status': 'exception',
                        'error': str(e)
                    })

        return results

    def print_summary(self, results: List[Dict]):
        """Print execution summary"""
        print(f"\n{'='*80}")
        print("PARALLEL EXECUTION SUMMARY")
        print(f"{'='*80}")

        total_time = sum(r.get('elapsed_time', 0) for r in results)
        success_count = sum(1 for r in results if r.get('status') == 'success')
        failed_count = len(results) - success_count

        print(f"\nTotal tasks: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Average time per task: {total_time/len(results):.2f}s")

        print(f"\nPer-task results:")
        for result in sorted(results, key=lambda x: x.get('gpu_id', 0)):
            exp_name = result.get('experiment', 'unknown')
            trial_id = result.get('trial_id', '')
            gpu_id = result.get('gpu_id', '?')
            status = result.get('status', 'unknown')
            elapsed = result.get('elapsed_time', 0)
            accept_rate = result.get('accept_rate', None)

            trial_str = f" trial {trial_id}" if trial_id else ""
            accept_str = f" (accept rate: {accept_rate:.2%})" if accept_rate else ""

            print(f"  [GPU {gpu_id}] {exp_name}{trial_str}: {status} ({elapsed:.2f}s){accept_str}")

        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Run experiments in parallel across multiple GPUs')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['all', 'experiments', 'trials'],
                       help='Execution mode: all (run all 5 experiments), experiments (run specific experiments), trials (run multiple trials of one experiment)')
    parser.add_argument('--experiments', type=str, nargs='+',
                       choices=['exp1', 'exp2', 'exp3', 'exp4', 'exp5'],
                       help='Experiments to run (for experiments mode)')
    parser.add_argument('--experiment', type=str,
                       choices=['exp1', 'exp3', 'exp5'],
                       help='Experiment for trials mode (only homogeneous experiments supported)')
    parser.add_argument('--trials', type=int, default=10,
                       help='Number of trials to run (for trials mode)')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2],
                       help='GPU IDs to use')
    parser.add_argument('--model', type=str, default='qwen2.5-7b',
                       help='Model to use')

    args = parser.parse_args()

    runner = ParallelExperimentRunner(gpu_ids=args.gpus, model_name=args.model)

    if args.mode == 'all':
        experiments = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']
        results = runner.run_experiments_parallel(experiments)

    elif args.mode == 'experiments':
        if not args.experiments:
            print("Error: --experiments required for experiments mode")
            return
        results = runner.run_experiments_parallel(args.experiments)

    elif args.mode == 'trials':
        if not args.experiment:
            print("Error: --experiment required for trials mode")
            return
        results = runner.run_trials_parallel(args.experiment, args.trials)

    runner.print_summary(results)

    # Save results
    output_dir = project_root / "data/parallel_execution"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"parallel_execution_{args.mode}_{int(time.time())}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'mode': args.mode,
            'gpus': args.gpus,
            'model': args.model,
            'results': results
        }, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
