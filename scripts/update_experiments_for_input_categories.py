"""
Batch update all experiment files to support new input categories (short/medium/long)
"""
import re
from pathlib import Path

project_root = Path(__file__).parent.parent
experiment_files = [
    project_root / "experiments/exp1_homogeneous.py",
    project_root / "experiments/exp2_heterogeneous.py",
    project_root / "experiments/exp3_quantized_inference_homogeneous.py",
    project_root / "experiments/exp4_quantized_inference_heterogeneous.py",
    project_root / "experiments/exp5_full_inference_quantized_verification.py",
]

for exp_file in experiment_files:
    print(f"Updating {exp_file.name}...")
    content = exp_file.read_text()

    # Update run_single_trial signature
    # Pattern 1: def run_single_trial(self, prompt: str, trial_id: int, max_tokens: int = 3000, min_tokens: int = 500)
    content = re.sub(
        r'def run_single_trial\(self, prompt: str, trial_id: int, max_tokens: int = \d+, min_tokens: int = \d+\)',
        'def run_single_trial(self, prompt: str, trial_id: int, max_tokens: int = 4000,\n                         min_tokens: int = 500, input_category: str = "medium")',
        content
    )

    # Add input_category logging after "Trial {trial_id}:" line
    content = re.sub(
        r'(self\.logger\.info\(f"Trial \{trial_id\}:.*\)\n)',
        r'\1        self.logger.info(f"Input category: {input_category}")\n',
        content
    )

    # Update run() method: unpack 4-tuple instead of 3-tuple
    # Pattern: for trial_id, (prompt, max_tokens, min_tokens) in enumerate(prompts_with_config, start=1):
    content = re.sub(
        r'for trial_id, \(prompt, max_tokens, min_tokens\) in enumerate\(prompts_with_config, start=1\):',
        'for trial_id, (prompt, max_tokens, min_tokens, input_category) in enumerate(prompts_with_config, start=1):',
        content
    )

    # Update run_single_trial() calls
    # Pattern: result = self.run_single_trial(prompt, trial_id, max_tokens, min_tokens)
    content = re.sub(
        r'result = self\.run_single_trial\(prompt, trial_id, max_tokens, min_tokens\)',
        'result = self.run_single_trial(prompt, trial_id, max_tokens, min_tokens, input_category)',
        content
    )

    # Write back
    exp_file.write_text(content)
    print(f"  ✓ Updated {exp_file.name}")

# Update optimize_thresholds.py
optimize_file = project_root / "scripts/optimize_thresholds.py"
print(f"\nUpdating {optimize_file.name}...")
content = optimize_file.read_text()

# Update unpacking
content = re.sub(
    r'prompt, max_tokens, min_tokens = prompts_with_config\[0\]',
    'prompt, max_tokens, min_tokens, input_category = prompts_with_config[0]',
    content
)

# Update run_single_trial call
content = re.sub(
    r'result = exp\.run_single_trial\(prompt, trial_id=trial \+ 1, max_tokens=max_tokens, min_tokens=min_tokens\)',
    'result = exp.run_single_trial(prompt, trial_id=trial + 1, max_tokens=max_tokens, min_tokens=min_tokens, input_category=input_category)',
    content
)

optimize_file.write_text(content)
print(f"  ✓ Updated {optimize_file.name}")

print("\n✅ All files updated successfully!")
