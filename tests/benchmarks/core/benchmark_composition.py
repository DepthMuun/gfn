"""
Professional Function Composition Benchmark (v2.6.5)
==================================================

THE ULTIMATE TEST: Can GFN learn to COMPOSE functions?
Objective:
- Compare Manifold-GFN with Transformer (MicroGPT) on unseen compositions.
- Evaluate the hypothesis that GFN learns continuous flows that compose naturally.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn import Manifold
from gfn.optimizers import RiemannianAdam
from tests.benchmarks.baselines import MicroGPT
from tests.benchmarks.bench_utils import ResultsLogger

console = Console()

class FunctionCompositionDataset:
    """Dataset of function applications and compositions."""
    
    def __init__(self, train_mode=True):
        self.train_mode = train_mode
        chars = [str(i) for i in range(10)] + ['+', '-', '*', '=', 'f', 'g', 'h', '(', ')', '<PAD>', '<EOS>']
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(chars)
        
        self.funcs = {
            'f': lambda x: x + 2,
            'g': lambda x: x * 3,
            'h': lambda x: x - 1,
        }
    
    def apply_composition(self, x, composition):
        result = x
        for func_name in reversed(composition):
            result = self.funcs[func_name](result)
        return result
    
    def generate_problem(self):
        if self.train_mode:
            func_name = np.random.choice(['f', 'g', 'h'])
            x = np.random.randint(0, 30)
            result = self.funcs[func_name](x)
            problem = f"{func_name}({x})={result}"
        else:
            length = np.random.choice([2, 3])
            composition = ''.join(np.random.choice(['f', 'g', 'h'], size=length))
            x = np.random.randint(0, 5)
            result = self.apply_composition(x, composition)
            
            nested = f"{composition[0]}("
            for i in range(1, len(composition)):
                nested += f"{composition[i]}("
            nested += str(x)
            nested += ")" * len(composition)
            problem = f"{nested}={result}"
        
        return problem
    
    def encode(self, text):
        return [self.char_to_id[c] for c in text]
    
    def decode(self, ids):
        return ''.join([self.id_to_char.get(i, '?') for i in ids if i not in [self.char_to_id['<PAD>'], self.char_to_id['<EOS>']]])

def evaluate_model(model, dataset, num_samples=100, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    
    pad_id = dataset.char_to_id['<PAD>']
    eos_id = dataset.char_to_id['<EOS>']
    
    with torch.no_grad():
        for _ in range(num_samples):
            problem = dataset.generate_problem()
            parts = problem.split('=')
            prompt = parts[0] + '='
            target = parts[1]
            
            ids = dataset.encode(prompt)
            input_seq = torch.tensor([ids]).to(device)
            
            generated = list(ids)
            curr_input = input_seq
            state = None
            
            for _ in range(5): # Expected result length
                if isinstance(model, Manifold):
                    out = model(curr_input, state=state)
                    logits = out[0] if isinstance(out, tuple) else out
                    state = out[1] if isinstance(out, tuple) and len(out) > 1 else None
                else:
                    logits = model(curr_input)
                
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                tok_id = next_token.item()
                if tok_id == eos_id: break
                generated.append(tok_id)
                curr_input = next_token.unsqueeze(0).unsqueeze(0) if isinstance(model, Manifold) else torch.tensor([generated]).to(device)
            
            pred = dataset.decode(generated).split('=')[-1].strip()
            if pred == target.strip():
                correct += 1
            total += 1
    
    return (correct / total) * 100

def run_composition_benchmark():
    logger = ResultsLogger("composition_benchmark", category="core")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = FunctionCompositionDataset(train_mode=True)
    test_dataset = FunctionCompositionDataset(train_mode=False)
    
    vocab_size = train_dataset.vocab_size
    dim = 256
    depth = 4
    
    physics_config = {
        'plasticity': 0.1, 
        'singularity_thresh': 0.8, 
        'singularity_strength': 5.0,
        'R': 2.0, 'r': 1.0
    }

    gfn = Manifold(
        vocab_size=vocab_size, dim=dim, depth=depth, heads=4, 
        integrator_type='leapfrog', physics_config=physics_config, holographic=True
    ).to(device)
    
    gpt = MicroGPT(vocab_size=vocab_size, dim=dim, depth=depth, heads=4).to(device)
    
    console.print(f"\n[bold]GFN FUNCTION COMPOSITION AUDIT[/] (Manifold v2.6.5)\n")

    # Training
    steps = 1500
    batch_size = 32
    gfn_opt = RiemannianAdam(gfn.parameters(), lr=1e-3)
    gpt_opt = torch.optim.Adam(gpt.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.char_to_id['<PAD>'])

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        m_task = progress.add_task("[cyan]Manifold Training", total=steps)
        t_task = progress.add_task("[pink1]Transformer Training", total=steps)
        
        for step in range(steps):
            # Batch Gen
            batch = [train_dataset.generate_problem() for _ in range(batch_size)]
            inputs, targets = [], []
            for p in batch:
                ids = train_dataset.encode(p + '<EOS>')
                inputs.append(ids[:-1])
                targets.append(ids[1:])
                
            max_len = max(len(s) for s in inputs)
            padded_in = torch.tensor([s + [train_dataset.char_to_id['<PAD>']] * (max_len - len(s)) for s in inputs]).to(device)
            padded_tg = torch.tensor([s + [train_dataset.char_to_id['<PAD>']] * (max_len - len(s)) for s in targets]).to(device)
            
            # GFN Step
            gfn_opt.zero_grad()
            out = gfn(padded_in)
            gfn_logits = out[0] if isinstance(out, tuple) else out
            gfn_loss = criterion(gfn_logits.reshape(-1, vocab_size), padded_tg.reshape(-1))
            gfn_loss.backward()
            gfn_opt.step()
            
            # GPT Step
            gpt_opt.zero_grad()
            gpt_logits = gpt(padded_in)
            gpt_loss = criterion(gpt_logits.reshape(-1, vocab_size), padded_tg.reshape(-1))
            gpt_loss.backward()
            gpt_opt.step()
            
            progress.update(m_task, advance=1, description=f"[cyan]Manifold L:{gfn_loss.item():.3f}")
            progress.update(t_task, advance=1, description=f"[pink1]Transformer L:{gpt_loss.item():.3f}")

    # Eval
    console.print("\n[bold yellow]Evaluating zero-shot composition...[/]")
    gfn_train_acc = evaluate_model(gfn, train_dataset, device=device)
    gpt_train_acc = evaluate_model(gpt, train_dataset, device=device)
    gfn_test_acc = evaluate_model(gfn, test_dataset, device=device)
    gpt_test_acc = evaluate_model(gpt, test_dataset, device=device)

    summary_table = Table(title="Composition Accuracy Matrix", border_style="cyan")
    summary_table.add_column("Distribution", style="bold")
    summary_table.add_column("Manifold-GFN", justify="center")
    summary_table.add_column("Transformer", justify="center")
    
    summary_table.add_row("Seen (Simple Funcs)", f"{gfn_train_acc:.1f}%", f"{gpt_train_acc:.1f}%")
    summary_table.add_row("Unseen (Composition)", f"[bold green]{gfn_test_acc:.1f}%[/]", f"{gpt_test_acc:.1f}%")
    
    console.print(summary_table)

    # Plot
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "grid.color": "#2a2a2a"})
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#121212')
    labels = ['Seen (Train)', 'Unseen (OOD Composition)']
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, [gfn_train_acc, gfn_test_acc], width, label='Manifold-GFN', color='#00ADB5')
    ax.bar(x + width/2, [gpt_train_acc, gpt_test_acc], width, label='Transformer', color='#FF2E63')
    
    ax.set_title("Function Composition Advantage", color='white', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='white')
    ax.set_ylabel("Accuracy (%)", color='white')
    ax.legend()
    
    plt.tight_layout()
    logger.save_plot(fig, "composition_comparison.png")
    logger.save_json({"manifold": {"train": gfn_train_acc, "test": gfn_test_acc}, "transformer": {"train": gpt_train_acc, "test": gpt_test_acc}})
    
    console.print(f"\n[bold green][SUCCESS][/] Plot saved to [cyan]{logger.run_dir}[/]\n")

if __name__ == "__main__":
    run_composition_benchmark()


if __name__ == "__main__":
    run_composition_benchmark()
