"""
Experiment 5: Full-Precision Inference with Quantized Model Verification

Goal: Test verification mechanism when inference uses full-precision model
      and verification uses quantized model (reverse of Exp3).

Setup:
- Inference: Full-precision model (FP16) on Device A
- Verification: Quantized model (INT4/AWQ) on Device B (same vendor)
- Homogeneous hardware (same or similar devices)
- Multiple diverse prompts
- Record hidden states every 8 layers
- Multiple verifiers per trial

Metrics:
- Verification overhead (prefill time / full inference time)
- Hidden state comparison statistics (Pe, Pm, Pw, mean error)
- Acceptance rate (expected: similar to Exp3, >80%)

Key Research Question:
Can a quantized verifier reliably verify full-precision inference?
Is this direction (FP16→INT4) better or worse than Exp3 (INT4→FP16)?
"""

from pathlib import Path
from base_experiment import BaseExperiment
from src.models.model_loader import ModelLoader
from src.inference.inferencer import Inferencer
from src.verification.verifier import Verifier

# Get project root
project_root = Path(__file__).parent.parent


class Exp5FullInferenceQuantizedVerification(BaseExperiment):
    def __init__(self, model_name: str, inference_device: str, verification_device: str, num_verifiers: int = 3):
        """
        Args:
            model_name: Model to use (e.g., "qwen2.5-7b")
            inference_device: Device for inference (e.g., "cuda:0")
            verification_device: Device for verification (e.g., "cuda:1" or same as inference)
            num_verifiers: Number of verification runs per trial
        """
        super().__init__(
            exp_name=f"exp5_full_inference_quantized_verification_{model_name}_{inference_device}_to_{verification_device}",
            output_dir=str(project_root / "data/raw/exp5")
        )
        self.model_name = model_name
        self.inference_device = inference_device
        self.verification_device = verification_device
        self.num_verifiers = num_verifiers

    def run_single_trial(self, prompt: str, trial_id: int) -> dict:
        """Run a single inference + verification trial with full-precision inference, quantized verification"""
        self.logger.info(f"=" * 80)
        self.logger.info(f"Trial {trial_id}: {self.model_name}")
        self.logger.info(f"Inference Device: {self.inference_device} (FULL-PRECISION)")
        self.logger.info(f"Verification Device: {self.verification_device} (QUANTIZED)")
        self.logger.info(f"Prompt length: {len(prompt)} chars")
        self.logger.info(f"=" * 80)

        # === LOAD FULL-PRECISION INFERENCE MODEL ===
        self.logger.info(f"Loading FULL-PRECISION inference model on {self.inference_device}...")
        model_loader_inf = ModelLoader()
        inf_model, inf_tokenizer = model_loader_inf.load_model(
            self.model_name,
            device=self.inference_device,
            quantized=False  # Use full-precision model
        )

        # === INFERENCE PHASE (with full-precision model) ===
        inferencer = Inferencer(inf_model, inf_tokenizer, self.inference_device, self.logger)
        self.logger.info("Starting inference with FULL-PRECISION model (Prefill + Decode)...")

        inference_result = inferencer.generate_with_hidden_states(
            prompt=prompt,
            max_new_tokens=2000,
            temperature=0.7,
            top_p=0.9,
            sample_layers_every=8
        )

        self.logger.info(f"Inference complete: generated {len(inference_result['generated_tokens'])} tokens")
        self.logger.info(f"Inference time: {inference_result['timing']['total_time']:.2f}s")

        # Free inference model from memory
        del inf_model
        import torch
        if self.inference_device.startswith('cuda'):
            torch.cuda.empty_cache()

        # === LOAD QUANTIZED VERIFICATION MODEL ===
        self.logger.info(f"Loading QUANTIZED verification model on {self.verification_device}...")
        model_loader_ver = ModelLoader()
        ver_model, ver_tokenizer = model_loader_ver.load_model(
            self.model_name,
            device=self.verification_device,
            quantized=True  # Use quantized model
        )

        # === VERIFICATION PHASE (Multiple verifiers) ===
        verification_results = []

        for ver_id in range(self.num_verifiers):
            self.logger.info(f"Starting verification {ver_id+1}/{self.num_verifiers} with QUANTIZED model (Prefill only)...")
            verifier = Verifier(ver_model, ver_tokenizer, self.verification_device, self.logger)

            ver_result = verifier.verify_with_prefill(
                prompt=inference_result['prompt'],
                generated_tokens=inference_result['generated_tokens'],
                sample_layers_every=8
            )

            self.logger.info(f"Verification {ver_id+1} complete: {ver_result['timing']['total_time']:.2f}s")
            verification_results.append(ver_result)

        # Free verification model from memory
        del ver_model
        if self.verification_device.startswith('cuda'):
            torch.cuda.empty_cache()

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

            # Relaxed threshold for quantized verification (80% instead of 95%)
            accept_threshold = 0.80
            all_verifier_stats.append({
                'verifier_id': ver_id,
                'statistics': stats,
                'overhead': overhead,
                'verdict': 'PASS' if stats['summary']['accept_rate'] > accept_threshold else 'FAIL'
            })

            self.logger.info(f"Verifier {ver_id+1}: Accept rate = {stats['summary']['accept_rate']*100:.2f}%")
            self.logger.info(f"Verifier {ver_id+1}: Overhead = {overhead['percentage']:.2f}%")
            self.logger.info(f"Verifier {ver_id+1}: Verdict = {all_verifier_stats[-1]['verdict']}")

        # Aggregate results
        avg_accept_rate = sum(v['statistics']['summary']['accept_rate'] for v in all_verifier_stats) / len(all_verifier_stats)
        avg_overhead = sum(v['overhead']['percentage'] for v in all_verifier_stats) / len(all_verifier_stats)

        return {
            'experiment': 'exp5_full_inference_quantized_verification',
            'trial_id': trial_id,
            'model': self.model_name,
            'inference_device': self.inference_device,
            'inference_precision': 'full',
            'verification_device': self.verification_device,
            'verification_precision': 'quantized',
            'prompt': prompt[:200] + "...",
            'inference': {
                'generated_length': len(inference_result['generated_tokens']),
                'generated_text_preview': inference_result['generated_text'][:500] + "...",
                'timing': inference_result['timing']
            },
            'verifiers': all_verifier_stats,
            'summary': {
                'avg_accept_rate': avg_accept_rate,
                'avg_overhead_percentage': avg_overhead,
                'final_verdict': 'PASS' if avg_accept_rate > 0.80 else 'FAIL'
            }
        }

    def run(self):
        """Run full experiment with multiple prompts"""
        self.logger.info("=" * 80)
        self.logger.info(f"EXPERIMENT 5: Full-Precision Inference + Quantized Verification")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Inference Device: {self.inference_device} (FULL-PRECISION)")
        self.logger.info(f"Verification Device: {self.verification_device} (QUANTIZED)")
        self.logger.info(f"Number of verifiers per trial: {self.num_verifiers}")
        self.logger.info("=" * 80)

        prompts = self.get_prompts(num_prompts=3)
        all_results = []

        for trial_id, prompt in enumerate(prompts, start=1):
            result = self.run_single_trial(prompt, trial_id)
            all_results.append(result)

            # Save individual trial result
            filename = f"{self.model_name}_fp_{self.inference_device}_to_quantized_{self.verification_device}_trial{trial_id}.json"
            self.save_result(result, filename)

        # Save aggregated summary
        summary = {
            'experiment': 'exp5_full_inference_quantized_verification',
            'model': self.model_name,
            'inference_device': self.inference_device,
            'inference_precision': 'full',
            'verification_device': self.verification_device,
            'verification_precision': 'quantized',
            'num_trials': len(all_results),
            'num_verifiers_per_trial': self.num_verifiers,
            'trials': all_results,
            'aggregate': {
                'avg_accept_rate': sum(r['summary']['avg_accept_rate'] for r in all_results) / len(all_results),
                'avg_overhead_percentage': sum(r['summary']['avg_overhead_percentage'] for r in all_results) / len(all_results),
                'pass_rate': sum(1 for r in all_results if r['summary']['final_verdict'] == 'PASS') / len(all_results)
            }
        }

        self.save_result(summary, f"{self.model_name}_fp_{self.inference_device}_to_quantized_{self.verification_device}_summary.json")

        self.logger.info("=" * 80)
        self.logger.info("EXPERIMENT 5 COMPLETE")
        self.logger.info(f"Average accept rate: {summary['aggregate']['avg_accept_rate']*100:.2f}%")
        self.logger.info(f"Average overhead: {summary['aggregate']['avg_overhead_percentage']:.2f}%")
        self.logger.info(f"Trial pass rate: {summary['aggregate']['pass_rate']*100:.2f}%")
        self.logger.info("=" * 80)


def main():
    # ========== Configuration Options ==========

    # Model: Use any model from configs/models.yaml that has quantized version
    MODEL_NAME = "qwen2.5-7b"

    # Scenario 1: Same device (CUDA:0 → CUDA:0)
    # INFERENCE_DEVICE = "cuda:0"
    # VERIFICATION_DEVICE = "cuda:0"

    # Scenario 2: Different CUDA devices (CUDA:0 → CUDA:1)
    INFERENCE_DEVICE = "cuda:0"
    VERIFICATION_DEVICE = "cuda:1"

    # Scenario 3: Mac homogeneous (MPS → MPS, need to load/unload)
    # INFERENCE_DEVICE = "mps"
    # VERIFICATION_DEVICE = "mps"

    # Number of verification runs per trial
    NUM_VERIFIERS = 3

    # Number of different prompts to test
    NUM_PROMPTS = 3

    print("=" * 80)
    print(f"Experiment 5 Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Inference Device: {INFERENCE_DEVICE} (FULL-PRECISION)")
    print(f"  Verification Device: {VERIFICATION_DEVICE} (QUANTIZED)")
    print(f"  Verifiers per trial: {NUM_VERIFIERS}")
    print(f"  Number of prompts: {NUM_PROMPTS}")
    print("=" * 80)

    exp = Exp5FullInferenceQuantizedVerification(
        model_name=MODEL_NAME,
        inference_device=INFERENCE_DEVICE,
        verification_device=VERIFICATION_DEVICE,
        num_verifiers=NUM_VERIFIERS
    )
    exp.run()


if __name__ == "__main__":
    main()
