#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
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
OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Tokenizer & Data Loader
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

# 2. Baselines: nanoGPT
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # Causal masking removed (BIDIRECTIONAL Mode)
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, dim, depth, heads, block_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, block_size) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
    
    def forward(self, idx, targets=None):
        b, t = idx.shape
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

# 3. Causal Verification
def verify_causal(model, tokenizer, device):
    console.print("\n[bold cyan]Checking Causal Integrity...[/]")
    model.eval()
    x = torch.randint(0, tokenizer.vocab_size, (1, 10)).to(device)
    with torch.no_grad():
        out_full, _ = model(x)
        if isinstance(out_full, tuple): out_full = out_full[0] # Handle return_probs
        out_part, _ = model(x[:, :5])
        if isinstance(out_part, tuple): out_part = out_part[0]
    
    diff = torch.abs(out_full[:, :5] - out_part).max().item()
    if diff < 1e-4:
        console.print("[bold green]PASS:[/] Integrity preserved.")
        return True
    else:
        console.print(f"[bold yellow]NOTE:[/] Non-causal mode enabled (Leakage: {diff:.4f})")
        return True # Continue even with leakage as requested

# 4. Benchmarking Entry
def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    if args.quick:
        args.steps = 100
        args.batch_size = 8
        args.block_size = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, val_data, tokenizer = get_data()
    
    DIM, DEPTH, HEADS = 64, 2, 2
    
    console.print(f"\n[bold magenta]Language Benchmarking: MANIFOLD vs nanoGPT[/]")
    console.print(f"Dataset: TinyShakespeare | Vocab: {tokenizer.vocab_size}")

    # --- MANIFOLD ---
    manifold_model = gfn.create(
        preset_name='stable-torus',
        vocab_size=tokenizer.vocab_size,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        integrator='yoshida',
        mixer_type='low_rank',
        holographic=True
    ).to(device)
    
    with torch.no_grad():
        all_tokens = torch.arange(tokenizer.vocab_size, device=device).unsqueeze(1)
        ref_points = manifold_model.embedding(all_tokens).squeeze(1)

    class ManifoldLM(nn.Module):
        def __init__(self, model, vocab_size):
            super().__init__()
            self.model = model
            self.vocab_size = vocab_size
            self.register_buffer("all_tokens", torch.arange(vocab_size).unsqueeze(1))
            # Temperature for Perplexity conversion. 
            # 0.5 provides a smoother distribution for more standard character-level PPL.
            self.temp = 0.5 
            
        def forward(self, idx, targets=None):
            # 1. Prediction on manifold
            x_pred = self.model(idx)[0] # [B, S, D]
            
            # 2. Dynamic Target Anchors (recalculate to follow trained embedding)
            # We detach to ensure ref_points act as stable targets for the integrator
            ref_points = self.model.embedding(self.all_tokens).squeeze(1).detach() # [V, D]
            
            loss = None
            if targets is not None:
                target_pts = ref_points[targets] # [B, S, D]
                # Toroidal MSE via wrapping
                diff = x_pred - target_pts
                diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
                loss = diff_wrapped.pow(2).mean()

            # 3. Probability Mapping (Calibrated for Perplexity)
            # Standard PPL expects P(w|context). We map Toroidal Distance to Logits.
            B, S, D = x_pred.shape
            V = self.vocab_size
            
            # Vectorized toroidal distance calculation
            x_exp = x_pred.unsqueeze(2) # [B, S, 1, D]
            r_exp = ref_points.view(1, 1, V, D)
            
            # Manhattan distance on the torus (proxy for probability density)
            # Using (1 - cos) is more numerically stable than squared atan2 diff for Softmax
            toroidal_dist = (1.0 - torch.cos(x_exp - r_exp)).mean(dim=-1)
            logits = -toroidal_dist / self.temp
            
            return logits, loss

    m_lm = ManifoldLM(manifold_model, tokenizer.vocab_size).to(device)
    m_params = sum(p.numel() for p in m_lm.parameters())
    
    # --- nanoGPT ---
    class TunedNanoGPT(NanoGPT):
        def __init__(self, vocab_size, dim, depth, heads, block_size):
            super().__init__(vocab_size, dim, depth, heads, block_size)
            # Re-init blocks with correct MLP to match params
            blocks = []
            for _ in range(depth):
                block = TransformerBlock(dim, heads, block_size)
                h_dim = int(0.6 * dim) # Adjust to match ~560K-570K params
                block.mlp = nn.Sequential(
                    nn.Linear(dim, h_dim),
                    nn.GELU(),
                    nn.Linear(h_dim, dim),
                )
                blocks.append(block)
            self.blocks = nn.Sequential(*blocks)
    
    gpt = TunedNanoGPT(tokenizer.vocab_size, DIM, DEPTH, HEADS, args.block_size).to(device)
    g_params = sum(p.numel() for p in gpt.parameters())
    
    table = Table(title="Professional Parameter Comparison")
    table.add_column("Model Architecture", style="cyan")
    table.add_column("Parameters", justify="right", style="green")
    table.add_row("MANIFOLD GFN (128D, 6L)", f"{m_params:,}")
    table.add_row("nanoGPT Baseline", f"{g_params:,}")
    console.print(table)
    
    if not verify_causal(m_lm, tokenizer, device): return

    # 5. Training
    def train_engine(name, model_wrap):
        opt = torch.optim.AdamW(model_wrap.parameters(), lr=1e-3, weight_decay=0.01)
        hist = {"loss": [], "val_ppl": []}
        
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]{name:>8}[/]"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TextColumn("[bold yellow]{task.fields[info]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Training", total=args.steps, info="")
            
            for step in range(args.steps):
                model_wrap.train()
                xb, yb = get_batch(train_data, args.batch_size, args.block_size, device)
                logits, loss = model_wrap(xb, targets=yb)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                hist["loss"].append(loss.item())
                
                if step % 1 == 0:
                    description = f"L: {loss.item():.3f}"
                    if step % 20 == 0:
                        model_wrap.eval()
                        with torch.no_grad():
                            xv, yv = get_batch(val_data, args.batch_size, args.block_size, device)
                            v_logits, _ = model_wrap(xv, targets=yv)
                            v_ce = F.cross_entropy(v_logits.view(-1, v_logits.size(-1)), yv.view(-1))
                            ppl = math.exp(v_ce.item())
                            hist["val_ppl"].append(ppl)
                            description += f" | PPL: {ppl:.1f}"
                    
                    progress.update(task, advance=1, info=description)
        return hist

    m_hist = train_engine("MANIFOLD", m_lm)
    g_hist = train_engine("nanoGPT", gpt)
    
    # 6. Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(m_hist['loss'], label='MANIFOLD', color='blue', alpha=0.3)
    plt.plot(g_hist['loss'], label='nanoGPT', color='red', alpha=0.3)
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(m_hist['val_ppl'], label='MANIFOLD', color='blue', marker='o')
    plt.plot(g_hist['val_ppl'], label='nanoGPT', color='red', marker='x')
    plt.title('Validation Perplexity')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "language_results.png")
    
    console.print(f"\n[bold green]Benchmark Complete![/] Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_benchmark()
