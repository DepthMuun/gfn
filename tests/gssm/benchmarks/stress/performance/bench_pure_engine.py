import torch
import sys
from pathlib import Path
from rich.console import Console

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Import GFN V2 Configs & Models
from gfn import Manifold
from gfn.realizations.gssm.config.schema import ManifoldConfig, PhysicsConfig
from gfn import constants
from tests.benchmarks.infra.utils.logger import ResultsLogger

console = Console()

def run_pure_inference_benchmark():
    logger = ResultsLogger("perf_pure_inf_v2", category="performance")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = ManifoldConfig(vocab_size=256, dim=256, depth=12, heads=8)
    model = Manifold(config).to(device)
    
    x = torch.randint(0, 256, (1, 1024)).to(device)
    
    console.print(f"\n[bold magenta]GFN Pure Inference Performance (API V2)[/]")

    # 1. Standard Eval
    constants.GEODESIC_PURE_INFERENCE = False
    model.eval()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    std_time = (time.time() - start) / 50
    
    # 2. Pure Inference Engine
    constants.GEODESIC_PURE_INFERENCE = True
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        # In Pure Inference, no_grad is intrinsic but keeping it for safety
        _ = model(x)
    torch.cuda.synchronize()
    pure_time = (time.time() - start) / 50
    
    speedup = (std_time / pure_time - 1) * 100
    
    console.print(f"Standard Eval: {std_time*1000:.2f} ms")
    console.print(f"Pure Inference: {pure_time*1000:.2f} ms")
    console.print(f"Speedup: [bold green]{speedup:.1f}%[/]")

    console.print(f"\n[bold green]Success![/] Pure inference metrics generated.")

if __name__ == "__main__":
    import time
    run_pure_inference_benchmark()
