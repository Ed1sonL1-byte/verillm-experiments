"""
Run both heterogeneous scenarios:
1. NVIDIA → Mac (inference on GPU, verification on Mac)
2. Mac → NVIDIA (inference on Mac, verification on GPU)

This script runs both scenarios sequentially to compare cross-device verification performance.
"""

from exp2_heterogeneous import Exp2Heterogeneous

def main():
    MODEL_NAME = "qwen2.5-7b"
    NUM_VERIFIERS = 3

    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT 2: HETEROGENEOUS HARDWARE VERIFICATION")
    print("Both scenarios will be tested:")
    print("  1. NVIDIA → Mac")
    print("  2. Mac → NVIDIA")
    print("=" * 80 + "\n")

    # ========== Scenario 1: NVIDIA → Mac ==========
    print("\n" + "=" * 80)
    print("SCENARIO 1: NVIDIA (Inference) → Mac (Verification)")
    print("=" * 80 + "\n")

    exp1 = Exp2Heterogeneous(
        model_name=MODEL_NAME,
        inference_device="cuda:0",
        verification_device="mps",
        num_verifiers=NUM_VERIFIERS
    )
    exp1.run()

    # ========== Scenario 2: Mac → NVIDIA ==========
    print("\n" + "=" * 80)
    print("SCENARIO 2: Mac (Inference) → NVIDIA (Verification)")
    print("=" * 80 + "\n")

    exp2 = Exp2Heterogeneous(
        model_name=MODEL_NAME,
        inference_device="mps",
        verification_device="cuda:0",
        num_verifiers=NUM_VERIFIERS
    )
    exp2.run()

    print("\n" + "=" * 80)
    print("EXPERIMENT 2 COMPLETE - Both Scenarios Finished")
    print("Check data/raw/exp2/ for results")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
