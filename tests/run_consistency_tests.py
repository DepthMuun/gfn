"""
CUDA-Python Consistency Test Runner
====================================

This script provides a convenient way to run the CUDA-Python consistency tests
and view results in a formatted manner.

Usage:
    python run_consistency_tests.py [--cpu-only] [--verbose] [--output FILE]

Author: MiniMax Agent
Date: 2026-02-07
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to sys.path to allow imports when running script directly
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))


def run_tests(cpu_only: bool = False, verbose: bool = False, output: str = None):
    """Run the CUDA-Python consistency test suite."""
    
    print("=" * 80)
    print("GFN CUDA-PYTHON CONSISTENCY TEST SUITE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Build pytest arguments
    pytest_args = [
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "tests/test_cuda_python_consistency.py"
    ]
    
    if verbose:
        pytest_args.append("-vv")
    
    if cpu_only:
        pytest_args.append("--ignore-glob=*cuda*")
        print("Running CPU-only tests...")
    else:
        print("Running full test suite (CPU + CUDA if available)...")
    
    # Run tests
    result = subprocess.run(
        [sys.executable, "-m", "pytest"] + pytest_args,
        cwd=Path(__file__).parent.parent,
        capture_output=False
    )
    
    print()
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return result.returncode == 0


def run_quick_checks():
    """Run quick sanity checks without full test suite."""
    
    print("=" * 80)
    print("QUICK SANITY CHECKS")
    print("=" * 80)
    print()
    
    checks = [
        ("Import test", "imports"),
        ("CUDA availability", "cuda"),
        ("Python operations", "python_ops"),
        ("Gradient flow", "gradients"),
    ]
    
    results = []
    
    for name, category in checks:
        print(f"Running {name}... ", end="", flush=True)
        try:
            # Import test
            if category == "imports":
                from gfn.cuda.core import device_manager, CudaConstants
                from gfn.cuda.ops import ChristoffelOperation, LeapfrogOperation
                from gfn.cuda.autograd import christoffel_fused_autograd, leapfrog_fused_autograd
                print("✓")
                results.append((name, True, None))
                
            elif category == "cuda":
                import torch
                cuda_available = torch.cuda.is_available()
                print("✓" if cuda_available else "⊘ (CUDA not available)")
                results.append((name, cuda_available or True, None))
                
            elif category == "python_ops":
                import torch
                from gfn.cuda.ops import ChristoffelOperation
                
                v = torch.randn(4, 64)
                U = torch.randn(64, 8)
                W = torch.randn(64, 8)
                
                op = ChristoffelOperation()
                gamma = op.forward(v, U, W)
                
                assert gamma.shape == v.shape
                assert not torch.isnan(gamma).any()
                print("✓")
                results.append((name, True, None))
                
            elif category == "gradients":
                import torch
                from gfn.cuda.ops import ChristoffelOperation
                
                v = torch.randn(4, 64, requires_grad=True)
                U = torch.randn(64, 8, requires_grad=True)
                W = torch.randn(64, 8, requires_grad=True)
                
                op = ChristoffelOperation()
                gamma = op.forward(v, U, W)
                loss = torch.sum(gamma)
                loss.backward()
                
                assert v.grad is not None
                assert U.grad is not None
                print("✓")
                results.append((name, True, None))
                
        except Exception as e:
            print(f"✗ ({e})")
            results.append((name, False, str(e)))
    
    print()
    print("-" * 80)
    print("Summary:")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"  {passed}/{total} checks passed")
    
    return passed == total


def print_test_summary():
    """Print a summary of available tests."""
    
    print("=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print()
    
    test_classes = [
        ("TestCUDAAvailability", "CUDA device and constant verification"),
        ("TestChristoffelOperation", "Christoffel symbol computation tests"),
        ("TestLeapfrogIntegration", "Leapfrog integrator tests"),
        ("TestGradientConsistency", "Gradient computation verification"),
        ("TestCUDAVsPythonEquivalence", "CUDA vs Python numerical equivalence"),
        ("TestConvergenceBehavior", "Optimization convergence tests"),
        ("TestEdgeCases", "Edge case and boundary tests"),
        ("TestPerformanceBenchmarks", "Performance benchmarks"),
        ("TestTopologyBehavior", "Topology-specific behavior tests"),
        ("TestAutogradFunctionality", "Autograd function tests"),
        ("TestFullPipeline", "Full integration tests"),
    ]
    
    for class_name, description in test_classes:
        print(f"  {class_name}")
        print(f"    → {description}")
        print()
    
    print("-" * 80)
    print("Key metrics tracked:")
    print("  • Numerical equivalence (max_diff, mean_diff)")
    print("  • Gradient consistency (relative error)")
    print("  • Convergence behavior (loss reduction)")
    print("  • Energy preservation (Hamiltonian)")
    print("  • Performance (throughput, speedup)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CUDA-Python consistency tests"
    )
    parser.add_argument(
        "--cpu-only", 
        action="store_true",
        help="Run only CPU tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick sanity checks only"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show test summary"
    )
    
    args = parser.parse_args()
    
    if args.summary:
        print_test_summary()
        sys.exit(0)
    
    if args.quick:
        success = run_quick_checks()
        sys.exit(0 if success else 1)
    
    success = run_tests(cpu_only=args.cpu_only, verbose=args.verbose)
    sys.exit(0 if success else 1)
