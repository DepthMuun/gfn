import sys
import os
from pathlib import Path
import time
import torch

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
sys.path.append(str(PROJECT_ROOT))

import gfn

def benchmark_op(name, op, args, kwargs=None, device="cuda", iters=100, warmup=10):
    if kwargs is None:
        kwargs = {}
    
    # Warmup
    for _ in range(warmup):
        op(*args, **kwargs)
        
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iters):
            op(*args, **kwargs)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        start_t = time.perf_counter()
        for _ in range(iters):
            op(*args, **kwargs)
        end_t = time.perf_counter()
        elapsed_ms = (end_t - start_t) * 1000

    avg_ms = elapsed_ms / iters
    return {"Component": name, "Avg Latency (ms)": f"{avg_ms:.3f}", "Iters": iters, "Device": device}

def run_speed_tests():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running MANIFOLD V5 Speed Tests on {device.upper()}...")
    
    B, L, H, D = 32, 100, 4, 16
    dim_total = H * D
    
    # Dummy tensors
    x = torch.randn(B, L, H, D, device=device)
    v = torch.randn(B, L, H, D, device=device)
    force = torch.randn(B, L, dim_total, device=device)

    # Just instantiate a standard model and extract its components to avoid init signature errors.
    model = gfn.create(preset_name='stable-torus', vocab_size=100, dim=dim_total, rank=16, depth=1, heads=H).to(device)
                       
    # Components extracted directly from the engine
    manifold_layer = model.layers[0]
    integrator = manifold_layer.integrator
    dynamics = integrator.physics_engine
    geo = dynamics.geometry
    
    # Loss
    toroidal_loss = gfn.losses.toroidal.ToroidalLoss()
    target = torch.randn(B, L, dim_total, device=device)
    token_ids = torch.randint(0, 100, (B, L), device=device)
    
    results = []
    
    # Run benchmarks
    results.append(benchmark_op("ManifoldLayer (Forward X,V -> X',V')", manifold_layer, (x, v), {"force": force}, device))
    results.append(benchmark_op("MANIFOLD Model (L=100 Sequence Forward)", model, (token_ids,), {}, device, iters=20))
    results.append(benchmark_op("ToroidalLoss (Compute)", toroidal_loss, (force, target), {}, device))
    
    print("\nMANIFOLD V5 Component Latency Report")
    print("-" * 75)
    print(f"{'Component':<40} | {'Device':<6} | {'Iters':<6} | {'Avg Latency (ms)':<15}")
    print("-" * 75)
    for r in results:
        print(f"{r['Component']:<40} | {r['Device']:<6} | {r['Iters']:<6} | {r['Avg Latency (ms)']:<15}")
    print("-" * 75)

if __name__ == "__main__":
    run_speed_tests()
