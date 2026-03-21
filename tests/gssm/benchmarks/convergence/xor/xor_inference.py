"""
MANIFOLD XOR Generalization / Inference Script
==============================================
Loads a trained Model from .bin and tests it on varying sequence lengths.
Evaluates O(1) memory scaling and zero-shot generalization.
"""
import sys
import math
from pathlib import Path
import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Standalone execution support
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gfn

console = Console()

# ============================================================================
# PARITY MATH (matches training logic)
# ============================================================================
def compute_accuracy(x_pred: torch.Tensor, targets_class: torch.Tensor) -> float:
    """Toroidal nearest-class classification."""
    PI = math.pi
    TWO_PI = 2.0 * PI
    half_pi = PI * 0.5
    dist_pos = torch.min(
        torch.abs(x_pred - half_pi) % TWO_PI,
        TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI)
    )
    dist_neg = torch.min(
        torch.abs(x_pred + half_pi) % TWO_PI,
        TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI)
    )
    preds = (dist_pos.mean(dim=-1) < dist_neg.mean(dim=-1)).long()
    return (preds == targets_class).float().mean().item()

# ============================================================================
# INFERENCE RUNNER
# ============================================================================
def run_xor_inference(model_path: str = "results/xor_best_model.bin", lengths = [20, 50, 100, 1000]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model with Training Config
    # Note: These parameters MUST match the saved checkpoint exactly.
    # From logic_xor.py: dim=16, depth=1, heads=1, integrator='yoshida'
    # PRODUCTION_PHYSICS_CONFIG used in training:
    PHYSICS_CONFIG = {
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16, 'impulse_scale': 80.0},
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {
            'enabled': True,
            'dynamic_time': {'enabled': True},
            'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
            'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8},
        },
        'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 1.0},
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.4},
    }

    model = gfn.create(
        preset_name='stable-torus',
        vocab_size=2,
        dim=8, # Matched to xor_best_model.bin
        depth=2,
        heads=2,
        integrator='yoshida',
        impulse_scale=80.0,
        holographic=True,
    ).to(device)

    # Resolve absolute path to checkpoint
    abs_model_path = Path(__file__).resolve().parent / model_path
    if not abs_model_path.exists():
        console.print(f"[bold red]Error:[/] Model not found at {abs_model_path}. Run logic_xor.py first.")
        return

    model.load_state_dict(torch.load(abs_model_path, weights_only=True, map_location=device))
    model.eval()
    
    console.print(f"\n[bold cyan]MANIFOLD XOR Inference Script[/]")
    console.print(f"Loaded: [green]{abs_model_path.name}[/]")
    console.print(f"Device: {device}\n")

    # 2. Generalization Table
    results_table = Table(title="Zero-Shot Length Generalization")
    results_table.add_column("Length", style="cyan")
    results_table.add_column("Accuracy", justify="right")
    results_table.add_column("VRAM (MB)", justify="right")
    results_table.add_column("Latency (ms)", justify="right")

    with torch.no_grad():
        for L in lengths:
            try:
                # Reset stats for fresh measurement
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                
                # Input Generation (Batch = 32 for statistical relevance)
                batch_size = 32
                x_in = torch.randint(0, 2, (batch_size, L), device=device)
                y_target = torch.cumsum(x_in, dim=1) % 2
                
                # Timing
                start = time.time()
                output = model(x_in)
                elapsed = (time.time() - start) * 1000.0 / batch_size # ms per sample
                
                x_pred = output[0] # [B, T, D]
                
                # Accuracy on LAST TOKEN (True memory test)
                acc = compute_accuracy(x_pred[:, -1, :], y_target[:, -1])
                
                # Memory
                if device.type == 'cuda':
                    vram = torch.cuda.max_memory_allocated() / 1e6
                    vram_str = f"{vram:.1f}"
                else:
                    vram_str = "CPU"
                
                color = "green" if acc > 0.9 else "yellow" if acc > 0.5 else "red"
                results_table.add_row(
                    str(L), 
                    f"[{color}]{acc*100:.1f}%[/]", 
                    vram_str, 
                    f"{elapsed:.2f}"
                )
            except Exception as e:
                results_table.add_row(str(L), "FAILED", "-", "-")
                console.print(f"[dim red]Error at length {L}: {e}[/]")

    console.print(results_table)

if __name__ == "__main__":
    import time
    run_xor_inference()
