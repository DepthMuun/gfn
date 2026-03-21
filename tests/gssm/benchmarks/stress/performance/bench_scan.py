import torch
import time
from gfn import ManifoldLayer
from gfn.realizations.gssm.models.parallel_manifold_network.parallel_manifold_model import ParallelMLayer

def benchmark_mlayers():
    dim = 256
    heads = 4
    seq_lengths = [128, 512, 1024, 2048]
    batch_size = 4
    
    physics_config = {
        'active_inference': {
            'dynamic_time': {'enabled': True}
        }
    }
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on: {device}")
    
    m_seq = MLayer(config=config) if 'config' in locals() else MLayer(config).to(device)
    m_par = ParallelMLayer(config=config) if 'config' in locals() else MLayer(config).to(device)
    
    for L in seq_lengths:
        x = torch.randn(batch_size, L, dim).to(device)
        v = torch.randn(batch_size, L, dim).to(device)
        force = torch.randn(batch_size, L, dim).to(device)
        
        # Warmup
        m_par(x, v, force=force)
        
        # Benchmark Parallel
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()
        for _ in range(5):
            m_par(x, v, force=force)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        par_time = (time.time() - t0) / 5
        
        print(f"L={L:4d} | Parallel: {par_time*1000:.2f}ms")

if __name__ == "__main__":
    benchmark_mlayers()
