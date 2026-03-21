"""
isn.Language Benchmark
Compares Transformer vs SSM vs GFN backbones on language modeling tasks.
"""

import torch
import time
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_dir))
from gfn import isn

def run_benchmark(
    name: str,
    scanner_cls,
    emitter_cls,
    world_cls=isn.TopologicalWorld,
    vocab_size: int = 100,
    d_model: int = 128,
    seq_len: int = 1024,
    batch_size: int = 8
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Benchmarking: {name} on {device} ---")
    
    # Create model
    model = isn.create(
        vocab_size=vocab_size,
        d_model=d_model,
        scanner_cls=scanner_cls,
        world_cls=world_cls,
        emitter_cls=emitter_cls
    )
    model.to(device)
    
    # Dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # Measure Latency
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    with torch.no_grad():
        output = model(input_ids)
    
    end_time = time.time()
    vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    
    print(f"Latency ({seq_len} tokens): {end_time - start_time:.4f}s")
    print(f"Peak VRAM: {vram:.2f} MB")
    print(f"Output Logic Shape: {output['logits'].shape}")
    print("-" * 30)

if __name__ == "__main__":
    # Standard Vocab
    V = 100
    D = 128
    
    # Test different lengths to verify O(1)
    for L in [128, 1024, 4096]:
        print(f"\n>>>> SEQUENCE LENGTH L={L} <<<<")
        
        # 1. Baseline: Transformer Scanner
        run_benchmark("Transformer Scanner (Quadratic)", 
                      isn.TransformerScanner, isn.ThresholdEmitter, 
                      vocab_size=V, d_model=D, seq_len=L)
        
        # 2. O(1) Candidate: SSM Scanner
        run_benchmark("SSM Scanner (O(1))", 
                      isn.SSMScanner, isn.SSMEmitter, 
                      vocab_size=V, d_model=D, seq_len=L)
        
        # 3. Pure GFN: GFN Scanner + GFN World + GFN Emitter
        run_benchmark("Pure GFN (Flow O(1))", 
                      isn.GFNScanner, isn.GFNEmitter, 
                      world_cls=isn.GFNWorld,
                      vocab_size=V, d_model=D, seq_len=L)
