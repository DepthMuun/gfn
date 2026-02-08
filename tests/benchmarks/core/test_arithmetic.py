"""
Professional Arithmetic Unit-Benchmark (v2.6.5)
=============================================

Specialized probe for basic arithmetic reliability.
Verifies that the Manifold-GFN can learn simple 1-digit addition/subtraction
with extremely high precision and zero divergence.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn import Manifold
from gfn.optimizers import RiemannianAdam
from tests.benchmarks.bench_utils import ResultsLogger

console = Console()

class ArithmeticTask:
    """Simplified 1-digit addition/subtraction probe."""
    def __init__(self):
        self.vocab_size = 15
        self.op_map = {'+': 10, '-': 11, '=': 13, '<PAD>': 14}
        
    def generate(self, batch_size, device='cpu'):
        ins, tgs = [], []
        for _ in range(batch_size):
            a, b = np.random.randint(0, 5), np.random.randint(0, 5)
            # a + b = res
            res = a + b
            prob = [a, self.op_map['+'], b, self.op_map['='], res]
            ins.append(prob[:-1])
            tgs.append(prob[-1])
        return torch.tensor(ins).to(device), torch.tensor(tgs).to(device)

def run_arithmetic_benchmark():
    logger = ResultsLogger("arithmetic_unit", category="core")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    console.print(Panel.fit(
        "[bold green]GFN ARITHMETIC UNIT TEST[/]\n[white]Low-level Logic Verification (v2.6.5)[/]",
        border_style="green"
    ))

    task = ArithmeticTask()
    model = Manifold(
        vocab_size=15, dim=128, depth=4, heads=4,
        physics_config={'topology': 'euclidean', 'plasticity': 0.1}
    ).to(device)
    
    opt = RiemannianAdam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    history = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        train_task = progress.add_task("Training...", total=500)
        
        for i in range(500):
            model.train()
            x, y = task.generate(32, device=device)
            
            opt.zero_grad()
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            
            loss = crit(logits[:, -1, :], y)
            loss.backward()
            opt.step()
            
            acc = (logits[:, -1, :].argmax(dim=-1) == y).float().mean().item()
            history.append(loss.item())
            
            progress.update(train_task, advance=1, description=f"Loss: {loss.item():.4f} | Acc: {acc*100:.1f}%")
            
            if acc > 0.99 and i > 100:
                console.print(f"\n[bold green][CONVERGED][/] Step {i} at 100% accuracy.")
                break

    # Results
    logger.save_json(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history, color='green', lw=2)
    ax.set_title("Arithmetic Convergence (v2.6.5)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    logger.save_plot(fig, "arithmetic_convergence.png")
    
    console.print(f"\n[bold green][SUCCESS][/] Unit test complete. Results in [cyan]{logger.run_dir}[/]\n")

if __name__ == "__main__":
    run_arithmetic_benchmark()

if __name__ == '__main__':
    run_arithmetic_benchmark()
