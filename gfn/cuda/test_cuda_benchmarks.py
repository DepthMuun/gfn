"""
CUDA Performance Benchmark Suite
Performance comparison between CUDA and Python implementations.

Benchmarks:
- Christoffel computation speed
- Integrator performance
- Batch size scaling
- Dimension scaling
"""

import torch
import time
import sys
import os
import numpy as np

# Add parent directories to path
cuda_dir = os.path.dirname(os.path.abspath(__file__))
gfn_dir = os.path.dirname(cuda_dir)
manifold_dir = os.path.dirname(gfn_dir)
sys.path.insert(0, manifold_dir)

# Import modules
try:
    import gfn_cuda
    from gfn.geometry.lowrank import LowRankChristoffel
    from gfn.integrators.symplectic.leapfrog import LeapfrogIntegrator
    from gfn.integrators.runge_kutta.heun import HeunIntegrator
    from test_config import *
    from test_utils import *
    print("âœ“ Benchmark modules imported")
except ImportError as e:
    print(f"âœ— Import failed: {e}")
    sys.exit(1)


def benchmark_function(func, warmup=BENCHMARK_WARMUP, iterations=BENCHMARK_ITERATIONS):
    """
    Benchmark a function with warmup and multiple iterations.
    
    Returns:
        Tuple of (mean_time, std_time) in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        func()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)


class TestPerformance:
    """CUDA vs Python performance comparison."""
    
    def test_christoffel_performance(self):
        """Benchmark Christoffel computation."""
        print_test_header("Benchmark 1: Christoffel Computation")
        
        set_random_seed()
        batch, dim, rank = 64, 128, 16
        
        # Create matched instances
        christ_py, U, W = create_matched_christoffel(dim, rank, 'euclidean')
        
        # Generate test data
        data = generate_test_data(batch, dim, rank)
        v = data['v']
        
        x_empty = torch.empty(0, device=DEVICE, dtype=DTYPE)
        V_w_empty = torch.empty(0, device=DEVICE, dtype=DTYPE)
        
        # Benchmark Python
        def python_func():
            with torch.no_grad():
                return christ_py(v, None)
        
        py_mean, py_std = benchmark_function(python_func)
        
        # Benchmark CUDA
        def cuda_func():
            return gfn_cuda.lowrank_christoffel_fused(
                v, U, W, x_empty, V_w_empty,
                0.0, 1.0, 1.0, TOPOLOGY_EUCLIDEAN, DEFAULT_R, DEFAULT_r
            )
        
        cuda_mean, cuda_std = benchmark_function(cuda_func)
        
        speedup = py_mean / cuda_mean
        
        print(f"\nPerformance Results (batch={batch}, dim={dim}, rank={rank}):")
        print(f"  Python: {py_mean:.3f} Â± {py_std:.3f} ms")
        print(f"  CUDA:   {cuda_mean:.3f} Â± {cuda_std:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        assert speedup > 1.0, "CUDA should be faster than Python"
        print_test_result(True, f"CUDA {speedup:.2f}x faster")
    
    def test_heun_performance(self):
        """Benchmark Heun integrator."""
        print_test_header("Benchmark 2: Heun Integrator")
        
        set_random_seed()
        batch, dim, rank = 64, 128, 16
        dt = 0.01
        steps = 10
        
        # Create matched instances
        christ_py, U, W = create_matched_christoffel(dim, rank, 'euclidean')
        integrator_py = HeunIntegrator(christoffel=christ_py, dt=dt)
        
        # Generate test data
        data = generate_test_data(batch, dim, rank)
        x, v, force = data['x'], data['v'], data['force']
        
        # Benchmark Python
        def python_func():
            with torch.no_grad():
                return integrator_py.forward(x.clone(), v.clone(), force, steps=steps)
        
        py_mean, py_std = benchmark_function(python_func, warmup=5, iterations=50)
        
        # Benchmark CUDA
        def cuda_func():
            return gfn_cuda.heun_fused(
                x.clone(), v.clone(), force, U, W,
                dt, 1.0, steps, TOPOLOGY_EUCLIDEAN, DEFAULT_R, DEFAULT_r
            )
        
        cuda_mean, cuda_std = benchmark_function(cuda_func, warmup=5, iterations=50)
        
        speedup = py_mean / cuda_mean
        
        print(f"\nPerformance Results (batch={batch}, dim={dim}, steps={steps}):")
        print(f"  Python: {py_mean:.3f} Â± {py_std:.3f} ms")
        print(f"  CUDA:   {cuda_mean:.3f} Â± {cuda_std:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        assert speedup > 1.0, "CUDA should be faster than Python"
        print_test_result(True, f"CUDA {speedup:.2f}x faster")
    
    def test_batch_scaling(self):
        """Test performance scaling with batch size."""
        print_test_header("Benchmark 3: Batch Size Scaling")
        
        set_random_seed()
        dim, rank = 64, 8
        batch_sizes = [1, 4, 16, 64, 256]
        
        # Create test setup
        U = torch.randn(dim, rank, device=DEVICE, dtype=DTYPE)
        W = torch.randn(dim, rank, device=DEVICE, dtype=DTYPE)
        x_empty = torch.empty(0, device=DEVICE, dtype=DTYPE)
        V_w_empty = torch.empty(0, device=DEVICE, dtype=DTYPE)
        
        print(f"\nBatch Size Scaling (dim={dim}, rank={rank}):")
        print(f"{'Batch':>8} {'Time (ms)':>12} {'Throughput':>15}")
        print("-" * 40)
        
        for batch in batch_sizes:
            v = torch.randn(batch, dim, device=DEVICE, dtype=DTYPE)
            
            def cuda_func():
                return gfn_cuda.lowrank_christoffel_fused(
                    v, U, W, x_empty, V_w_empty,
                    0.0, 1.0, 1.0, TOPOLOGY_EUCLIDEAN, DEFAULT_R, DEFAULT_r
                )
            
            mean_time, _ = benchmark_function(cuda_func, warmup=5, iterations=50)
            throughput = batch / mean_time * 1000  # samples/sec
            
            print(f"{batch:>8} {mean_time:>12.3f} {throughput:>12.1f} samp/s")
        
        print_test_result(True, "Batch scaling measured")
    
    def test_dimension_scaling(self):
        """Test performance scaling with dimension."""
        print_test_header("Benchmark 4: Dimension Scaling")
        
        set_random_seed()
        batch, rank = 32, 8
        dimensions = [16, 32, 64, 128, 256]
        
        x_empty = torch.empty(0, device=DEVICE, dtype=DTYPE)
        V_w_empty = torch.empty(0, device=DEVICE, dtype=DTYPE)
        
        print(f"\nDimension Scaling (batch={batch}, rank={rank}):")
        print(f"{'Dim':>8} {'Time (ms)':>12} {'Time/Dim':>12}")
        print("-" * 40)
        
        for dim in dimensions:
            v = torch.randn(batch, dim, device=DEVICE, dtype=DTYPE)
            U = torch.randn(dim, rank, device=DEVICE, dtype=DTYPE)
            W = torch.randn(dim, rank, device=DEVICE, dtype=DTYPE)
            
            def cuda_func():
                return gfn_cuda.lowrank_christoffel_fused(
                    v, U, W, x_empty, V_w_empty,
                    0.0, 1.0, 1.0, TOPOLOGY_EUCLIDEAN, DEFAULT_R, DEFAULT_r
                )
            
            mean_time, _ = benchmark_function(cuda_func, warmup=5, iterations=50)
            time_per_dim = mean_time / dim
            
            print(f"{dim:>8} {mean_time:>12.3f} {time_per_dim:>12.6f}")
        
        print_test_result(True, "Dimension scaling measured")


# ============================================================================
# Main Benchmark Runner
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CUDA PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Warmup iterations: {BENCHMARK_WARMUP}")
    print(f"Benchmark iterations: {BENCHMARK_ITERATIONS}")
    print("=" * 80)
    
    # Run benchmarks
    suite = TestPerformance()
    
    suite.test_christoffel_performance()
    suite.test_heun_performance()
    suite.test_batch_scaling()
    suite.test_dimension_scaling()
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 80)

