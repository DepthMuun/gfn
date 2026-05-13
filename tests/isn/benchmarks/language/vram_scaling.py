"""
isn.(Inertial State Network) VRAM Scaling Benchmark
Measures VRAM usage across different sequence lengths for Inference and Training.
"""

import torch
import torch.nn as nn
import json
import os
import sys
import time

# Ensure the local gfn package is in path
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_dir))
from gfn import isn

def measure_vram(model, input_ids, mode='inference'):
    device = input_ids.device
    model.to(device)
    
    # User rule: reset_peak_memory_stats() antes de cada medición VRAM
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    if mode == 'inference':
        model.eval()
        with torch.no_grad():
            # Warmup to stabilize allocations
            _ = model(input_ids)
            torch.cuda.reset_peak_memory_stats()
            # Actual measurement
            _ = model(input_ids)
    elif mode == 'forward':
        model.train()
        # Warmup
        _ = model(input_ids)
        torch.cuda.reset_peak_memory_stats()
        # Actual measurement
        _ = model(input_ids)
    elif mode == 'backward':
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Warmup
        outputs = model(input_ids)
        loss = outputs['logits'].sum()
        loss.backward()
        optimizer.zero_grad()
        torch.cuda.reset_peak_memory_stats()
        
        # Actual measurement
        outputs = model(input_ids)
        loss = outputs['logits'].sum()
        loss.backward()
        
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
    return peak_vram

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("Error: CUDA is required for VRAM benchmarking.")
        return

    config_path = base_dir / "configs" / "shakespeare_config.json"
    if not os.path.exists(config_path):
        # Fallback to default params if config missing
        d_model, d_emb, d_prop = 256, 256, 128
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
        d_model = config['model']['d_model']
        d_emb = config['model']['d_embedding']
        d_prop = config['model']['d_properties']
        
    vocab_size = 65 
    
    # Test lengths
    lengths = [32, 128, 512, 1024, 2048, 4096, 8192]
    batch_size = 1 
    
    print(f"\n{'='*60}")
    print(f"isn.VRAM SCALING BENCHMARK")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model Configuration: d_model={d_model}, d_emb={d_emb}, d_prop={d_prop}")
    print(f"{'='*60}\n")
    
    print(f"{'L':<10} | {'Inference (MB)':<15} | {'Forward (MB)':<15} | {'Backward (MB)':<15}")
    print("-" * 65)
    
    # Create model
    model = isn.create(
        vocab_size=vocab_size,
        d_model=d_model,
        d_embedding=d_emb,
        d_properties=d_prop,
        scanner_cls=isn.GFNScanner,
        world_cls=isn.GFNWorld,
        emitter_cls=isn.GFNEmitter
    ).to(device)
    
    for L in lengths:
        input_ids = torch.randint(0, vocab_size, (batch_size, L)).to(device)
        
        try:
            vram_inf = measure_vram(model, input_ids, 'inference')
            vram_fwd = measure_vram(model, input_ids, 'forward')
            vram_bwd = measure_vram(model, input_ids, 'backward')
            
            print(f"{L:<10} | {vram_inf:<15.2f} | {vram_fwd:<15.2f} | {vram_bwd:<15.2f}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{L:<10} | {'OOM':<15} | {'OOM':<15} | {'OOM':<15}")
                torch.cuda.empty_cache()
            else:
                raise e
        
    print("\nBenchmark Complete.")

if __name__ == "__main__":
    run_benchmark()
