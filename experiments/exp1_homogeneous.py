"""
Experiment 1: Homogeneous Hardware Baseline Verification

Goal: Establish baseline floating-point deviation when both inference
and verification run on identical hardware.

Setup:
- Same hardware for inference and verification (NVIDIA GPU or Mac M4)
- Same full-precision model
- Multiple diverse prompts
- Record hidden states every 8 layers

Metrics:
- Verification overhead (prefill time / full inference time)
- Hidden state comparison statistics (Pe, Pm, Pw, mean error)
- Acceptance rate
"""

from pathlib import Path
from base_experiment import BaseExperiment
from src.models.model_loader import ModelLoader
from src.inference.inferencer import Inferencer
from src.verification.verifier import Verifier

# Get project root
project_root = Path(__file__).parent.parent


class Exp1Homogeneous(BaseExperiment):
    def __init__(self, model_name: str, device: str, num_verifiers: int = 3):
        super().__init__(
            exp_name=f"exp1_homogeneous_{model_name}_{device}",
            output_dir=str(project_root / "data/raw/exp1")
        )
        self.model_name = model_name
        self.device = device
        self.num_verifiers = num_verifiers

    def run_single_trial(self, prompt: str, trial_id: int) -> dict:
        """Run a single inference + verification trial"""
        self.logger.info(f"=" * 80)
        self.logger.info(f"Trial {trial_id}: {self.model_name} on {self.device}")
        self.logger.info(f"Prompt length: {len(prompt)} chars")
        self.logger.info(f"=" * 80)

        # Load model
        self.logger.info(f"Loading model {self.model_name} to {self.device}...")
        model_loader = ModelLoader()
        model, tokenizer = model_loader.load_model(self.model_name, device=self.device)

        # === INFERENCE PHASE ===
        inferencer = Inferencer(model, tokenizer, self.device, self.logger)
        self.logger.info("Starting inference (Prefill + Decode)...")

        inference_result = inferencer.generate_with_hidden_states(
            prompt=prompt,
            max_new_tokens=1500,
            temperature=0.7,
            top_p=0.9,
            sample_layers_every=8
        )

        self.logger.info(f"Inference complete: generated {len(inference_result['generated_tokens'])} tokens")
        self.logger.info(f"Inference time: {inference_result['timing']['total_time']:.2f}s")

        # === VERIFICATION PHASE (Multiple verifiers) ===
        verification_results = []

        for ver_id in range(self.num_verifiers):
            self.logger.info(f"Starting verification {ver_id+1}/{self.num_verifiers} (Prefill only)...")
            verifier = Verifier(model, tokenizer, self.device, self.logger)

            ver_result = verifier.verify_with_prefill(
                prompt=inference_result['prompt'],
                generated_tokens=inference_result['generated_tokens'],
                sample_layers_every=8
            )

            self.logger.info(f"Verification {ver_id+1} complete: {ver_result['timing']['total_time']:.2f}s")
            verification_results.append(ver_result)

        # === COMPARISON & ANALYSIS ===
        all_verifier_stats = []

        for ver_id, ver_result in enumerate(verification_results):
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
                'statistics': stats,
                'overhead': overhead,
                'verdict': 'PASS' if stats['summary']['accept_rate'] > 0.95 else 'FAIL'
            })

            self.logger.info(f"Verifier {ver_id+1}: Accept rate = {stats['summary']['accept_rate']*100:.2f}%")
            self.logger.info(f"Verifier {ver_id+1}: Overhead = {overhead['percentage']:.2f}%")

        # Aggregate results
        avg_accept_rate = sum(v['statistics']['summary']['accept_rate'] for v in all_verifier_stats) / len(all_verifier_stats)
        avg_overhead = sum(v['overhead']['percentage'] for v in all_verifier_stats) / len(all_verifier_stats)

        return {
            'experiment': 'exp1_homogeneous',
            'trial_id': trial_id,
            'model': self.model_name,
            'device': self.device,
            'prompt': prompt[:200] + "...",  # Truncate for readability
            'inference': {
                'generated_length': len(inference_result['generated_tokens']),
                'generated_text_preview': inference_result['generated_text'][:500] + "...",
                'timing': inference_result['timing']
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
        self.logger.info(f"EXPERIMENT 1: Homogeneous Hardware Baseline")
        self.logger.info(f"Model: {self.model_name}, Device: {self.device}")
        self.logger.info(f"Number of verifiers per trial: {self.num_verifiers}")
        self.logger.info("=" * 80)

        prompts = self.get_prompts(num_prompts=3)
        all_results = []

        for trial_id, prompt in enumerate(prompts, start=1):
            result = self.run_single_trial(prompt, trial_id)
            all_results.append(result)

            # Save individual trial result
            filename = f"{self.model_name}_{self.device}_trial{trial_id}.json"
            self.save_result(result, filename)

        # Save aggregated summary
        summary = {
            'experiment': 'exp1_homogeneous',
            'model': self.model_name,
            'device': self.device,
            'num_trials': len(all_results),
            'num_verifiers_per_trial': self.num_verifiers,
            'trials': all_results,
            'aggregate': {
                'avg_accept_rate': sum(r['summary']['avg_accept_rate'] for r in all_results) / len(all_results),
                'avg_overhead_percentage': sum(r['summary']['avg_overhead_percentage'] for r in all_results) / len(all_results),
                'pass_rate': sum(1 for r in all_results if r['summary']['final_verdict'] == 'PASS') / len(all_results)
            }
        }

        self.save_result(summary, f"{self.model_name}_{self.device}_summary.json")

        self.logger.info("=" * 80)
        self.logger.info("EXPERIMENT 1 COMPLETE")
        self.logger.info(f"Average accept rate: {summary['aggregate']['avg_accept_rate']*100:.2f}%")
        self.logger.info(f"Average overhead: {summary['aggregate']['avg_overhead_percentage']:.2f}%")
        self.logger.info(f"Trial pass rate: {summary['aggregate']['pass_rate']*100:.2f}%")
        self.logger.info("=" * 80)


def main():
    # ========== A100 Configuration ==========
    # Model: Use any model from configs/models.yaml
    # Options: "qwen2.5-7b", "llama-3.1-8b", "mistral-7b"
    MODEL_NAME = "qwen2.5-7b"

    # Device: Use cuda:0 for first A100
    # For multi-GPU experiments, can use cuda:0, cuda:1, etc.
    DEVICE = "cuda:0"

    # Number of verification runs per trial (论文中用3个verifier)
    NUM_VERIFIERS = 3

    # Number of different prompts to test
    NUM_PROMPTS = 3

    print("=" * 80)
    print(f"Experiment 1 Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Device: {DEVICE}")
    print(f"  Verifiers per trial: {NUM_VERIFIERS}")
    print(f"  Number of prompts: {NUM_PROMPTS}")
    print("=" * 80)

    exp = Exp1Homogeneous(
        model_name=MODEL_NAME,
        device=DEVICE,
        num_verifiers=NUM_VERIFIERS
    )
    exp.run()


if __name__ == "__main__":
    main()
