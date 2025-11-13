"""
Experiment Verification Script

This script verifies that all experiments can run successfully by:
1. Checking Python environment and dependencies
2. Checking GPU/device availability
3. Checking if models are downloaded
4. Running quick tests of each experiment

Usage:
    python scripts/verify_experiments.py
    python scripts/verify_experiments.py --quick  # Skip model loading tests
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import importlib
from typing import Dict, List, Tuple


class ExperimentVerifier:
    """Verify all experiments are runnable"""

    def __init__(self):
        self.results = {}
        self.issues = []

    def check_python_version(self) -> Tuple[bool, str]:
        """Check Python version"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
        else:
            return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

    def check_dependencies(self) -> Dict[str, Tuple[bool, str]]:
        """Check if required packages are installed"""
        required_packages = {
            'torch': 'PyTorch',
            'transformers': 'Transformers',
            'numpy': 'NumPy',
            'yaml': 'PyYAML',
            'tqdm': 'tqdm',
        }

        optional_packages = {
            'matplotlib': 'Matplotlib (for visualization)',
            'seaborn': 'Seaborn (for visualization)',
        }

        results = {}

        # Check required packages
        for package, name in required_packages.items():
            try:
                mod = importlib.import_module(package)
                version = getattr(mod, '__version__', 'unknown')
                results[name] = (True, f"✅ {name} {version}")
            except ImportError:
                results[name] = (False, f"❌ {name} NOT INSTALLED")
                self.issues.append(f"Missing required package: {name} (pip install {package})")

        # Check optional packages
        for package, name in optional_packages.items():
            try:
                mod = importlib.import_module(package)
                version = getattr(mod, '__version__', 'unknown')
                results[name] = (True, f"✅ {name} {version}")
            except ImportError:
                results[name] = (False, f"⚠️  {name} NOT INSTALLED (optional)")

        return results

    def check_devices(self) -> Dict[str, Tuple[bool, str]]:
        """Check available compute devices"""
        results = {}

        try:
            import torch

            # Check CUDA
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
                results['CUDA'] = (True, f"✅ CUDA available: {device_count} device(s) - {', '.join(device_names)}")
            else:
                results['CUDA'] = (False, "❌ CUDA not available")

            # Check MPS (Mac M-series)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                results['MPS'] = (True, "✅ MPS (Mac) available")
            else:
                results['MPS'] = (False, "❌ MPS not available")

            # CPU is always available
            results['CPU'] = (True, "✅ CPU available")

        except Exception as e:
            results['Error'] = (False, f"❌ Error checking devices: {e}")

        return results

    def check_model_files(self) -> Dict[str, Tuple[bool, str]]:
        """Check if model files exist"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import yaml

            # Load model config
            config_path = project_root / "configs/models.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            results = {}

            for model_key, model_config in config['models'].items():
                model_name = model_config['name']
                local_path = Path(model_config.get('local_path', ''))

                # Check if model exists locally
                if local_path.exists():
                    results[f"{model_key} (local)"] = (True, f"✅ {model_key} found at {local_path}")
                else:
                    # Try to load from HuggingFace
                    try:
                        # Just check if tokenizer is accessible (much faster than loading model)
                        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                        results[f"{model_key} (HF)"] = (True, f"✅ {model_key} accessible from HuggingFace")
                    except Exception as e:
                        results[f"{model_key}"] = (False, f"❌ {model_key} not found: {str(e)[:50]}...")
                        self.issues.append(f"Model {model_key} not available. Run: python scripts/download_models.sh")

            return results

        except Exception as e:
            return {'Error': (False, f"❌ Error checking models: {e}")}

    def check_experiment_imports(self) -> Dict[str, Tuple[bool, str]]:
        """Check if all experiment modules can be imported"""
        experiments = {
            'Experiment 1': 'experiments.exp1_homogeneous',
            'Experiment 2': 'experiments.exp2_heterogeneous',
            'Experiment 3': 'experiments.exp3_quantized_inference_homogeneous',
            'Experiment 4': 'experiments.exp4_quantized_inference_heterogeneous',
            'Experiment 5': 'experiments.exp5_full_inference_quantized_verification',
        }

        results = {}

        for name, module_path in experiments.items():
            try:
                importlib.import_module(module_path)
                results[name] = (True, f"✅ {name} imports successfully")
            except Exception as e:
                results[name] = (False, f"❌ {name} import failed: {str(e)[:50]}...")
                self.issues.append(f"{name} cannot be imported: {e}")

        return results

    def quick_inference_test(self, device: str = 'cpu') -> Tuple[bool, str]:
        """Quick test to verify inference works"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"\n  Running quick inference test on {device}...")

            # Use a tiny model for quick testing
            model_name = "gpt2"  # Small model for quick testing

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            if device != 'cpu':
                model = model.to(device)

            # Quick inference
            inputs = tokenizer("Hello, world!", return_tensors="pt")
            if device != 'cpu':
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Clean up
            del model
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

            return True, f"✅ Quick inference test passed on {device}"

        except Exception as e:
            return False, f"❌ Quick inference test failed on {device}: {str(e)[:100]}..."

    def print_summary(self):
        """Print verification summary"""
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)

        all_passed = True

        for category, results in self.results.items():
            print(f"\n{category}:")
            if isinstance(results, dict):
                for name, (passed, message) in results.items():
                    print(f"  {message}")
                    if not passed and 'optional' not in message.lower():
                        all_passed = False
            else:
                passed, message = results
                print(f"  {message}")
                if not passed:
                    all_passed = False

        if self.issues:
            print("\n" + "="*80)
            print("ISSUES FOUND:")
            print("="*80)
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. {issue}")

        print("\n" + "="*80)
        if all_passed:
            print("✅ ALL CRITICAL CHECKS PASSED - Experiments are ready to run!")
        else:
            print("❌ SOME CHECKS FAILED - Please fix issues above before running experiments")
        print("="*80 + "\n")

        return all_passed


def main():
    parser = argparse.ArgumentParser(description='Verify experiment environment')
    parser.add_argument('--quick', action='store_true',
                       help='Skip model loading tests (faster)')
    parser.add_argument('--test-inference', action='store_true',
                       help='Run quick inference test')

    args = parser.parse_args()

    verifier = ExperimentVerifier()

    print("="*80)
    print("VERIFYING VERILLM EXPERIMENTS ENVIRONMENT")
    print("="*80)

    # Check Python version
    print("\nChecking Python version...")
    verifier.results['Python'] = verifier.check_python_version()

    # Check dependencies
    print("\nChecking dependencies...")
    verifier.results['Dependencies'] = verifier.check_dependencies()

    # Check devices
    print("\nChecking compute devices...")
    verifier.results['Devices'] = verifier.check_devices()

    # Check model files (unless --quick)
    if not args.quick:
        print("\nChecking model availability...")
        verifier.results['Models'] = verifier.check_model_files()
    else:
        print("\nSkipping model checks (--quick mode)")

    # Check experiment imports
    print("\nChecking experiment modules...")
    verifier.results['Experiments'] = verifier.check_experiment_imports()

    # Run inference test if requested
    if args.test_inference:
        # Try CUDA first, then MPS, then CPU
        device_results = verifier.results.get('Devices', {})

        if device_results.get('CUDA', (False, ''))[0]:
            verifier.results['Inference Test'] = verifier.quick_inference_test('cuda')
        elif device_results.get('MPS', (False, ''))[0]:
            verifier.results['Inference Test'] = verifier.quick_inference_test('mps')
        else:
            verifier.results['Inference Test'] = verifier.quick_inference_test('cpu')

    # Print summary
    all_passed = verifier.print_summary()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
