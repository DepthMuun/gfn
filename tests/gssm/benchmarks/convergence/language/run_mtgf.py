#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import argparse
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TimeRemainingColumn, TimeElapsedColumn, TaskProgressColumn
)

# 0. Setup Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gfn

console = Console()
OUTPUT_DIR = SCRIPT_DIR / "results_mtgf"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Tokenizer & Data Loader (Same as run.py)
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
    def encode(self, s): return [self.stoi[c] for c in s]
    def decode(self, l): return ''.join([self.itos[i] for i in l])

def get_data():
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path = SCRIPT_DIR / "tinyshakespeare.txt"
    if not data_path.exists():
        console.print("[yellow]Downloading TinyShakespeare...[/]")
        r = requests.get(data_url)
        with open(data_path, "w") as f: f.write(r.text)
    
    with open(data_path, 'r', encoding='utf-8') as f: data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    tokenizer = CharTokenizer(data)
    train_ids = torch.tensor(tokenizer.encode(train_data), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_data), dtype=torch.long)
    return train_ids, val_ids, tokenizer

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# 2. Benchmark Logic
def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    if args.quick:
        args.steps = 200
        args.batch_size = 8
        args.block_size = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ids, val_ids, tokenizer = get_data()
    
    DIM, DEPTH, HEADS = 64, 4, 4
    
    console.print(f"\n[bold magenta]MTGF vs Standard GFN Baseline[/]")
    console.print(f"Dataset: TinyShakespeare | Vocab: {tokenizer.vocab_size}")

    models = {}
    
    # 1. Standard GFN (Partition Mode)
    # Reemplazamos preset_name por configuración explícita V5
    models["GFN-Standard"] = gfn.create(
        vocab_size=tokenizer.vocab_size,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        topology_type='torus',
        readout_type='categorical',
        holographic=False,  # Forzar False
        integrator='leapfrog'
    ).to(device)
    
    # Debug info para verificar dimensiones
    if hasattr(models['GFN-Standard'].readout_plugin.readout, 'linear'):
        console.print(f"[yellow]GFN-Standard Readout Output Dim:[/] {models['GFN-Standard'].readout_plugin.readout.linear.out_features}")
    else:
        console.print(f"[yellow]GFN-Standard Readout Type:[/] {type(models['GFN-Standard'].readout_plugin.readout).__name__}")
    
    # 2. MTGF Ensemble (Ensemble Mode)
    from gfn.realizations.gssm.config.schema import PhysicsConfig, ManifoldConfig
    physics_mtgf = PhysicsConfig()
    physics_mtgf.trajectory_mode = 'ensemble'
    physics_mtgf.mixture.coupler_mode = 'mean_field'
    physics_mtgf.topology.type = 'torus'
    physics_mtgf.stability.base_dt = 0.05
    
    physics_mtgf.readout.type = 'categorical' # Configuración correcta
    
    config_mtgf = ManifoldConfig(
        vocab_size=tokenizer.vocab_size,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        physics=physics_mtgf,
        holographic=False
    )
    models["GFN-MTGF"] = gfn.create(config=config_mtgf).to(device)
    
    results = {}
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        console.print(f"\n[bold cyan]Training {name}[/] ({params:,} params)")
        
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        hist = {"loss": [], "val_ppl": []}
        
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]{name:>12}[/]"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TextColumn("[bold yellow]{task.fields[info]}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Training", total=args.steps, info="")
            
            for step in range(args.steps):
                model.train()
                xb, yb = get_batch(train_ids, args.batch_size, args.block_size, device)
                
                # Forward pass return logits and (x,v) and others
                logits, *_ = model(xb)
                
                # Cross entropy loss on averaged/final logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                hist["loss"].append(loss.item())
                
                if step % 10 == 0:
                    description = f"L: {loss.item():.3f}"
                    if step % 50 == 0:
                        model.eval()
                        with torch.no_grad():
                            xv, yv = get_batch(val_ids, args.batch_size, args.block_size, device)
                            v_logits, *_ = model(xv)
                            v_ce = F.cross_entropy(v_logits.view(-1, v_logits.size(-1)), yv.view(-1))
                            ppl = math.exp(min(v_ce.item(), 10.0)) # Clamp to avoid huge values
                            hist["val_ppl"].append(ppl)
                            description += f" | PPL: {ppl:.1f}"
                    
                    progress.update(task, advance=1, info=description)
        
        duration = time.time() - start_time
        console.print(f"{name} took {duration:.1f}s")
        results[name] = hist

    # 3. Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for name, hist in results.items():
        plt.plot(hist['loss'], label=name, alpha=0.5)
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for name, hist in results.items():
        plt.plot(hist['val_ppl'], label=name, marker='o')
    plt.title('Validation Perplexity (Steps)')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mtgf_comparison.png")
    console.print(f"\n[bold green]Benchmark Complete![/] Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_benchmark()
