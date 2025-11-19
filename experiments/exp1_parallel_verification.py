"""
Experiment 1 (Parallel Verification): Homogeneous Hardware with Multi-GPU Parallel Verification

Goal: Same as Exp1, but utilize multiple GPUs to run verifiers in parallel for faster execution.

Setup:
- Inference on GPU 0
- Verification distributed across GPU 0, 1, 2 (parallel execution)
- Same full-precision model on all GPUs
- Multiple diverse prompts

Benefits:
- 3x speedup in verification phase (3 verifiers run in parallel vs sequential)
- Full utilization of all 3 GPUs
- Same accuracy as sequential verification

Metrics:
- Verification overhead (now with parallel speedup)
- Hidden state comparison statistics (Pe, Pm, Pw, mean error)
- Acceptance rate
- Wall-clock time comparison vs sequential execution
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from base_experiment import BaseExperiment
from src.models.model_loader import ModelLoader
from src.inference.inferencer import Inferencer
from src.verification.verifier import Verifier
import torch

# Get project root
project_root = Path(__file__).parent.parent


class Exp1ParallelVerification(BaseExperiment):
    def __init__(self, model_name: str, gpu_ids: list, num_verifiers: int = 3):
        """
        Args:
            model_name: Model to use
            gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2])
            num_verifiers: Number of verification runs (should match len(gpu_ids) for full parallelism)
        """
        super().__init__(
            exp_name=f"exp1_parallel_verification_{model_name}_gpus_{'_'.join(map(str, gpu_ids))}",
            output_dir=str(project_root / "data/raw/exp1_parallel")
        )
        self.model_name = model_name
        self.gpu_ids = gpu_ids
        self.num_verifiers = num_verifiers
        self.inference_device = f"cuda:{gpu_ids[0]}"  # Use first GPU for inference

    def run_single_verifier(self, ver_id: int, gpu_id: int,
                           prompt: str, generated_tokens: list) -> dict:
        """Run a single verifier on specified GPU"""
        device = f"cuda:{gpu_id}"

        # Load model on this GPU
        model_loader = ModelLoader()
        model, tokenizer = model_loader.load_model(self.model_name, device=device)

        # Run verification
        verifier = Verifier(model, tokenizer, device, self.logger)
        ver_result = verifier.verify_with_prefill(
            prompt=prompt,
            generated_tokens=generated_tokens,
            sample_layers_every=8
        )

        # Clean up
        del model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

        return {
            'verifier_id': ver_id,
            'gpu_id': gpu_id,
            'result': ver_result
        }

    def run_single_trial(self, prompt: str, trial_id: int, max_tokens: int = 4000,
                         min_tokens: int = 500, input_category: str = "medium") -> dict:
        """Run a single inference + parallel verification trial"""
        self.logger.info(f"=" * 80)
        self.logger.info(f"Trial {trial_id}: {self.model_name} on GPUs {self.gpu_ids}")
        self.logger.info(f"Input category: {input_category}")
        self.logger.info(f"Prompt length: {len(prompt)} chars")
        self.logger.info(f"Max output tokens: {max_tokens}, Min output tokens: {min_tokens}")
        self.logger.info(f"Inference device: {self.inference_device}")
        self.logger.info(f"Verification GPUs: {self.gpu_ids[:self.num_verifiers]}")
        self.logger.info(f"=" * 80)

        # === INFERENCE PHASE (on first GPU) ===
        self.logger.info(f"Loading model {self.model_name} to {self.inference_device}...")
        model_loader = ModelLoader()
        model, tokenizer = model_loader.load_model(self.model_name, device=self.inference_device)

        inferencer = Inferencer(model, tokenizer, self.inference_device, self.logger)
        self.logger.info("Starting inference (Prefill + Decode)...")

        inference_result = inferencer.generate_with_hidden_states(
            prompt=prompt,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            temperature=0.7,
            top_p=0.9,
            sample_layers_every=8
        )

        self.logger.info(f"Inference complete: generated {len(inference_result['generated_tokens'])} tokens")
        self.logger.info(f"Inference time: {inference_result['timing']['total_time']:.2f}s")

        # Free inference model (will be reloaded on each GPU for verification)
        del model
        torch.cuda.empty_cache()

        # === PARALLEL VERIFICATION PHASE ===
        self.logger.info(f"Starting {self.num_verifiers} verifiers in PARALLEL across {self.num_verifiers} GPUs...")
        import time
        parallel_start = time.time()

        # Distribute verifiers across GPUs
        verification_tasks = []
        with ThreadPoolExecutor(max_workers=self.num_verifiers) as executor:
            for ver_id in range(self.num_verifiers):
                gpu_id = self.gpu_ids[ver_id % len(self.gpu_ids)]
                self.logger.info(f"Submitting verifier {ver_id+1} to GPU {gpu_id}...")

                future = executor.submit(
                    self.run_single_verifier,
                    ver_id, gpu_id,
                    inference_result['prompt'],
                    inference_result['generated_tokens']
                )
                verification_tasks.append(future)

            # Collect results as they complete
            verification_results = []
            for future in as_completed(verification_tasks):
                result = future.result()
                ver_id = result['verifier_id']
                gpu_id = result['gpu_id']
                self.logger.info(f"Verifier {ver_id+1} (GPU {gpu_id}) complete: "
                               f"{result['result']['timing']['total_time']:.2f}s")
                verification_results.append(result)

        parallel_end = time.time()
        parallel_verification_time = parallel_end - parallel_start

        # Sort by verifier_id for consistent ordering
        verification_results = sorted(verification_results, key=lambda x: x['verifier_id'])

        self.logger.info(f"All verifiers complete in {parallel_verification_time:.2f}s (parallel)")
        self.logger.info(f"Sequential time would have been: "
                        f"{sum(v['result']['timing']['total_time'] for v in verification_results):.2f}s")
        self.logger.info(f"Speedup: {sum(v['result']['timing']['total_time'] for v in verification_results) / parallel_verification_time:.2f}x")

        # === COMPARISON & ANALYSIS ===
        all_verifier_stats = []

        for ver_data in verification_results:
            ver_id = ver_data['verifier_id']
            ver_result = ver_data['result']
            gpu_id = ver_data['gpu_id']

            stats = self.compare_hidden_states(
                inference_result['hidden_states'],
                ver_result['hidden_states']
            )

            overhead = self.compute_verification_overhead(
                inference_result['timing']['total_time'],
                ver_result['timing']['total_time']
            )

            all_verifier_stats.append({
                'verifier_id': ver_id,
                'gpu_id': gpu_id,
                'statistics': stats,
                'overhead': overhead,
                'verdict': 'PASS' if stats['summary']['accept_rate'] > 0.95 else 'FAIL'
            })

            self.logger.info(f"Verifier {ver_id+1} (GPU {gpu_id}): Accept rate = {stats['summary']['accept_rate']*100:.2f}%")
            self.logger.info(f"Verifier {ver_id+1} (GPU {gpu_id}): Overhead = {overhead['percentage']:.2f}%")

        # Aggregate results
        avg_accept_rate = sum(v['statistics']['summary']['accept_rate'] for v in all_verifier_stats) / len(all_verifier_stats)
        avg_overhead = sum(v['overhead']['percentage'] for v in all_verifier_stats) / len(all_verifier_stats)

        return {
            'experiment': 'exp1_parallel_verification',
            'trial_id': trial_id,
            'model': self.model_name,
            'gpu_ids': self.gpu_ids,
            'inference_device': self.inference_device,
            'prompt': prompt[:200] + "...",
            'inference': {
                'generated_length': len(inference_result['generated_tokens']),
                'generated_text_preview': inference_result['generated_text'][:500] + "...",
                'timing': inference_result['timing']
            },
            'parallel_verification': {
                'wall_clock_time': parallel_verification_time,
                'sequential_equivalent_time': sum(v['result']['timing']['total_time'] for v in verification_results),
                'speedup': sum(v['result']['timing']['total_time'] for v in verification_results) / parallel_verification_time
            },
            'verifiers': all_verifier_stats,
            'summary': {
                'avg_accept_rate': avg_accept_rate,
                'avg_overhead_percentage': avg_overhead,
                'final_verdict': 'PASS' if avg_accept_rate > 0.95 else 'FAIL'
            }
        }

    def run(self):
        """Run full experiment with multiple prompts"""
        self.logger.info("=" * 80)
        self.logger.info(f"EXPERIMENT 1 (PARALLEL VERIFICATION): Homogeneous Hardware with Multi-GPU Verification")
        self.logger.info(f"Model: {self.model_name}, GPUs: {self.gpu_ids}")
        self.logger.info(f"Number of verifiers per trial: {self.num_verifiers}")
        self.logger.info(f"Inference device: {self.inference_device}")
        self.logger.info("=" * 80)

        prompts_with_config = self.get_prompts_with_config(num_prompts=3)
        all_results = []

        for trial_id, (prompt, max_tokens, min_tokens, input_category) in enumerate(prompts_with_config, start=1):
            result = self.run_single_trial(prompt, trial_id, max_tokens, min_tokens, input_category)
            all_results.append(result)

            # Save individual trial result
            filename = f"{self.model_name}_gpus_{'_'.join(map(str, self.gpu_ids))}_trial{trial_id}.json"
            self.save_result(result, filename)

        # Save aggregated summary
        summary = {
            'experiment': 'exp1_parallel_verification',
            'model': self.model_name,
            'gpu_ids': self.gpu_ids,
            'num_trials': len(all_results),
            'num_verifiers_per_trial': self.num_verifiers,
            'trials': all_results,
            'aggregate': {
                'avg_accept_rate': sum(r['summary']['avg_accept_rate'] for r in all_results) / len(all_results),
                'avg_overhead_percentage': sum(r['summary']['avg_overhead_percentage'] for r in all_results) / len(all_results),
                'avg_speedup': sum(r['parallel_verification']['speedup'] for r in all_results) / len(all_results),
                'pass_rate': sum(1 for r in all_results if r['summary']['final_verdict'] == 'PASS') / len(all_results)
            }
        }

        self.save_result(summary, f"{self.model_name}_gpus_{'_'.join(map(str, self.gpu_ids))}_summary.json")

        self.logger.info("=" * 80)
        self.logger.info("EXPERIMENT 1 (PARALLEL VERIFICATION) COMPLETE")
        self.logger.info(f"Average accept rate: {summary['aggregate']['avg_accept_rate']*100:.2f}%")
        self.logger.info(f"Average overhead: {summary['aggregate']['avg_overhead_percentage']:.2f}%")
        self.logger.info(f"Average speedup: {summary['aggregate']['avg_speedup']:.2f}x")
        self.logger.info(f"Trial pass rate: {summary['aggregate']['pass_rate']*100:.2f}%")
        self.logger.info("=" * 80)


def main():
    # ========== RTX PRO 6000 Configuration (3 GPUs) ==========
    MODEL_NAME = "qwen2.5-7b"

    # Use all 3 GPUs: GPU 0 for inference, GPUs 0/1/2 for parallel verification
    GPU_IDS = [0, 1, 2]

    # Number of verification runs (set to 3 to match number of GPUs)
    NUM_VERIFIERS = 3

    print("=" * 80)
    print(f"Experiment 1 (Parallel Verification) Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  GPUs: {GPU_IDS}")
    print(f"  Inference device: cuda:{GPU_IDS[0]}")
    print(f"  Verification GPUs: {GPU_IDS[:NUM_VERIFIERS]}")
    print(f"  Verifiers per trial: {NUM_VERIFIERS}")
    print(f"  Expected speedup: ~{NUM_VERIFIERS}x for verification phase")
    print("=" * 80)

    exp = Exp1ParallelVerification(
        model_name=MODEL_NAME,
        gpu_ids=GPU_IDS,
        num_verifiers=NUM_VERIFIERS
    )
    exp.run()


if __name__ == "__main__":
    main()
