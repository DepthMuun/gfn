
import torch
import time
import sys
import os

# Add project root to path

from gfn.cuda.ops import leapfrog_fused, CUDA_AVAILABLE
# Fallback import if needed
try:
    from gfn.integrators.symplectic.leapfrog import LeapfrogIntegrator
    from gfn.geometry.toroidal import ToroidalRiemannianGeometry
except ImportError:
    pass

def benchmark_live():
    print("================================================================")
    print("   GFN CUDA LIVE BENCHMARK")
    print("================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Torch Device: {device}")
    print(f"[*] GFN CUDA Extension Available: {CUDA_AVAILABLE}")
    
    if not CUDA_AVAILABLE:
        print("[!] CUDA Extension NOT LOADED. Running in Python/Fallout mode.")
    
    # Setup
    batch_size = 8192
    dim = 128
    steps = 100
    
    x = torch.randn(batch_size, dim, device=device)
    v = torch.randn(batch_size, dim, device=device)
    f = torch.randn(batch_size, dim, device=device)
    
    # Dummy Params for Kernel
    rank = 16
    U = torch.randn(dim, rank, device=device) * 0.01
    W = torch.randn(dim, rank, device=device) * 0.01
    
    # Warmup
    print("[*] Warming up...")
    for _ in range(5):
        if CUDA_AVAILABLE and device.type == 'cuda':
             # Direct kernel call for benchmark
             leapfrog_fused(x, v, f, U, W, 0.01, 1.0, 10, topology=1)
        else:
             # Just some math to warm up
             torch.matmul(x, U)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    print(f"[*] Starting Benchmark Loop (Batch={batch_size}, Dim={dim})")
    print("----------------------------------------------------------------")
    
    start_time = time.time()
    iter_count = 0
    
    try:
        while True:
            t0 = time.time()
            
            # RUN OP
            # We run a block of 'steps' integration steps per iteration
            if CUDA_AVAILABLE and device.type == 'cuda':
                # Returns x, v
                x, v = leapfrog_fused(x, v, f, U, W, 0.01, 1.0, steps, topology=1)
            else:
                # CPU Simulation (Naive)
                # Just doing matrix mults to simulate load if fallback
                h = x @ U
                gamma = (h**2) @ W.t()
                x = x + 0.01 * v
                v = v + 0.01 * (f - gamma)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            t1 = time.time()
            dt = t1 - t0
            iter_count += 1
            
            total_time = t1 - start_time
            avg_ips = iter_count / total_time
            
            # Physics Steps per Second = iter_count * steps / total_time
            sps = (iter_count * steps) / total_time
            
            # Live Print
            status = "CUDA ACTIVE" if (CUDA_AVAILABLE and device.type == 'cuda') else "CPU FALLBACK"
            sys.stdout.write(f"\r[{status}] Iter: {iter_count} | Speed: {avg_ips:.2f} calls/s | Physics: {sps:.0f} steps/s | Last Batch: {dt*1000:.1f}ms   ")
            sys.stdout.flush()
            
            if total_time > 10.0 and iter_count > 50: # Run for at least 10s
                break
                
    except KeyboardInterrupt:
        print("\n[!] User Interrupted")
    
    print("\n----------------------------------------------------------------")
    print(f"Done. Average Speed: {(iter_count * steps) / (time.time() - start_time):.0f} steps/s")

if __name__ == "__main__":
    benchmark_live()
