#!/usr/bin/env python3
"""
MANIFOLD SHA256 Benchmark
=========================

Demonstrates learning cryptographic hashing (SHA256) over arbitrary sequence lengths
with the MANIFOLD architecture.

PHYSICAL INTUITION
------------------
The SHA256 task demonstrates the capability of MANIFOLD's geodesic physics to process
information and build complex deterministic representations.

TASK
----
Input:   bit sequence of a message, followed by 256 zero bits.
Target:  the 256 bits of the SHA256 hash of the message at the last 256 positions.
Angles:  class 0 → -π/2, class 1 → +π/2

Usage:
  python run.py [--quicktest] [--seed SEED]
"""

import argparse
import json
import math
import sys
import time
import hashlib
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from tqdm import tqdm

# ── Bootstrap ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gfn
from gfn.realizations.gssm.models.g_ssm import GSSM as Model


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PHYSICS_CONFIG = {
    'embedding': {
        'type': 'functional',
        'mode': 'linear',
        'coord_dim': 16,
    },
    'readout': {'type': 'implicit', 'coord_dim': 16},
    'active_inference': {
        'enabled': True,
        'dynamic_time': {'enabled': True},
        'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
        'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8},
    },
    'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
    'topology': {'type': 'torus'},
    'stability': {'base_dt': 0.4},   # Faster exploration
}


@dataclass
class BenchmarkConfig:
    # Model
    dim: int = 32
    depth: int = 4
    heads: int = 2
    integrator: str = 'leapfrog'
    impulse_scale: float = 80.0

    # Training
    train_message_bytes: int = 4       # Sequences for training
    train_batch_size: int = 64
    train_steps: int = 600
    checkpoint_every: int = 100
    lr: float = 1e-3
    max_lr: float = 2e-3

    # Evaluation
    eval_lengths: List[int] = None
    eval_batch_size: int = 8

    # Convergence
    acc_threshold: float = 0.95
    patience: int = 30
    min_steps: int = 150

    def __post_init__(self):
        if self.eval_lengths is None:
            self.eval_lengths = [16, 64, 256, 1024, 4096, 16384, 65536, 262144]


@dataclass
class QuickTestConfig(BenchmarkConfig):
    """Fast config to verify all logic in < 5 minutes."""
    dim: int = 128
    depth: int = 4
    heads: int = 4
    train_message_bytes: int = 4
    train_batch_size: int = 16
    train_steps: int = 1000
    checkpoint_every: int = 40
    eval_lengths: List[int] = None
    min_steps: int = 30
    patience: int = 100

    def __post_init__(self):
        if self.eval_lengths is None:
            self.eval_lengths = [16, 64, 256, 1024, 4096, 16384, 65536, 262144]


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

PI = math.pi


def make_sha256_batch(
    batch_size: int,
    message_bytes: int,
    device: torch.device,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a SHA256 batch.
    
    Returns:
        x       [B, message_bytes * 8 + 256] – input tokens (bits)
        y_angle [B, message_bytes * 8 + 256] – target angles
        y_class [B, message_bytes * 8 + 256] – target classes (0 or 1)
    """
    x_list = []
    y_class_list = []
    
    for _ in range(batch_size):
        b = np.random.bytes(message_bytes)
        h = hashlib.sha256(b).digest()
        
        msg_bits = torch.tensor(np.unpackbits(np.frombuffer(b, dtype=np.uint8)), dtype=torch.long)
        hash_bits = torch.tensor(np.unpackbits(np.frombuffer(h, dtype=np.uint8)), dtype=torch.long)
        
        # Input is message bits followed by 256 zero bits
        x = torch.cat([msg_bits, torch.zeros(256, dtype=torch.long)])
        # Target classes is 0 for the message part, and the hash bits for the end
        y_class = torch.cat([torch.zeros_like(msg_bits), hash_bits])
        
        x_list.append(x)
        y_class_list.append(y_class)
        
    x = torch.stack(x_list).to(device)
    y_class = torch.stack(y_class_list).to(device)
    
    # Map class → toroidal angle target
    y_angle = (y_class.float() * 2.0 - 1.0) * (PI * 0.5)  # {-π/2, +π/2}
    return x, y_angle, y_class


# ══════════════════════════════════════════════════════════════════════════════
# ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

def toroidal_accuracy(x_pred: torch.Tensor, y_class: torch.Tensor) -> float:
    """
    Classify each position via shortest toroidal distance to ±π/2.
    """
    TWO_PI = 2.0 * PI
    half_pi = PI * 0.5
    dist_pos = torch.min(
        torch.abs(x_pred - half_pi) % TWO_PI,
        TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI),
    )
    dist_neg = torch.min(
        torch.abs(x_pred + half_pi) % TWO_PI,
        TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI),
    )
    preds = (dist_pos.mean(dim=-1) < dist_neg.mean(dim=-1)).long()
    return (preds == y_class).float().mean().item()


def sha256_accuracy(
    x_pred: torch.Tensor,    # [B, L, D]
    y_class: torch.Tensor,   # [B, L]
) -> Dict[str, float]:
    """
    Break down accuracy. Mostly interested in the final 256 bits representing the hash.
    """
    x_pred_hash = x_pred[:, -256:]
    y_class_hash = y_class[:, -256:]
    
    return {
        'acc_hash_bits': toroidal_accuracy(x_pred_hash, y_class_hash),
        'acc_overall': toroidal_accuracy(x_pred, y_class),
    }


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS MANAGEMENT  (incremental JSON)
# ══════════════════════════════════════════════════════════════════════════════

class ResultsWriter:
    """Writes benchmark results to JSON incrementally — never loses data."""

    def __init__(self, path: Path, config_dict: Dict):
        self.path = path
        self._data = {
            'benchmark': 'MANIFOLD SHA256 Benchmark',
            'timestamp_start': datetime.now().isoformat(),
            'config': config_dict,
            'training': {},
            'eval_lengths': {},
        }
        self._flush()

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self._data, f, indent=2)

    def write_training(self, result: Dict):
        self._data['training'] = result
        self._data['timestamp_train_done'] = datetime.now().isoformat()
        self._flush()
        print(f"[Results] Training results saved -> {self.path.name}")

    def write_length(self, length: int, result: Dict):
        self._data['eval_lengths'][str(length)] = {
            **result,
            'timestamp': datetime.now().isoformat(),
        }
        self._data['timestamp_last_eval'] = datetime.now().isoformat()
        self._flush()
        print(f"[Results] Length {length:>6,} saved -> {self.path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(
    model: Model,
    cfg: BenchmarkConfig,
    device: torch.device,
    ckpt_dir: Path,
    rng: torch.Generator,
) -> Dict:
    """
    Train the MANIFOLD model on SHA256 sequences.
    """
    optimizer = optim.AdamW([
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
            'lr': cfg.lr, 'weight_decay': 1e-4,
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
            'lr': cfg.lr * 10, 'weight_decay': 0,
        },
    ])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.max_lr, total_steps=cfg.train_steps, pct_start=0.2
    )
    criterion = gfn.loss('toroidal')
    model.train()

    best_acc = 0.0
    hits = 0
    converged_at = None
    history = {'loss': [], 'acc': []}

    pbar = tqdm(range(cfg.train_steps), desc='Training SHA256 model')

    for step in pbar:
        x, y_angle, y_class = make_sha256_batch(
            cfg.train_batch_size, cfg.train_message_bytes, device, rng=rng
        )

        optimizer.zero_grad()
        
        out = model(x)
        x_pred = out[0]                             # [B, L, D]
        
        # Loss only on the hash part
        x_pred_hash = x_pred[:, -256:]
        y_exp_hash = y_angle[:, -256:].unsqueeze(-1).expand_as(x_pred_hash)
        loss = criterion(x_pred_hash, y_exp_hash)
            
        if torch.isnan(loss):
            print(f"[WARN] NaN loss at step {step}, skipping")
            continue

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            acc = toroidal_accuracy(x_pred_hash, y_class[:, -256:])

        history['loss'].append(loss.item())
        history['acc'].append(acc)

        if step % 5 == 0:
            pbar.set_postfix(loss=f'{loss.item():.4f}', hash_acc=f'{acc*100:.1f}%')

        best_acc = max(best_acc, acc)

        if step >= cfg.min_steps and acc >= cfg.acc_threshold:
            hits += 1
        else:
            hits = 0
            
        if hits >= cfg.patience and converged_at is None:
            converged_at = step
            print(f'\n[Train] Converged at step {step} | hash_acc={acc*100:.1f}%')
            break

        if (step + 1) % cfg.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f'model_step{step+1:05d}.pt'
            torch.save({'step': step + 1, 'model': model.state_dict(),
                        'loss': loss.item(), 'acc': acc}, ckpt_path)
            print(f'[Checkpoint] Step {step+1} -> {ckpt_path.name}')

    final_path = ckpt_dir / 'model_final.pt'
    torch.save({
        'config': cfg.__dict__,
        'physics_config': PHYSICS_CONFIG,
        'model': model.state_dict(),
        'history': history,
        'converged_at': converged_at,
        'best_acc': best_acc,
    }, final_path)
    print(f'[Train] Final model saved -> {final_path.name}')

    return {
        'steps_run': len(history['loss']),
        'converged_at': converged_at,
        'final_loss': history['loss'][-1] if history['loss'] else None,
        'final_acc': history['acc'][-1] if history['acc'] else None,
        'best_acc': best_acc,
        'checkpoint': str(final_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_length(
    model: Model,
    message_bytes: int,
    eval_batch_size: int,
    device: torch.device,
) -> Dict:
    """
    Evaluate SHA256 at a specific message length.
    """
    model.eval()

    with torch.no_grad():
        x, y_angle, y_class = make_sha256_batch(
            eval_batch_size, message_bytes, device
        )

        t0 = time.time()
        out = model(x)
        elapsed = time.time() - t0

        x_pred = out[0]  # [B, L, D]
        metrics = sha256_accuracy(x_pred, y_class)
        metrics['inference_time_s'] = round(elapsed, 3)

    model.train()

    return {
        'message_bytes': message_bytes,
        'mean_acc_hash_bits': metrics['acc_hash_bits'],
        'mean_acc_overall': metrics['acc_overall'],
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: ResultsWriter):
    """Print a clean summary table of all evaluated lengths."""
    data = results._data
    train = data.get('training', {})
    evals = data.get('eval_lengths', {})

    print('\n' + '=' * 68)
    print(' MANIFOLD SHA256 BENCHMARK - RESULTS')
    print('=' * 68)
    print(f' Converged at step: {train.get("converged_at", "N/A")}')
    print(f' Best training hash acc: {train.get("best_acc", 0)*100:.1f}%')
    print()
    print(f' {"Message Bytes":>16}  {"Acc (Hash)":>11}  {"Acc (Overall)":>13}')
    print(' ' + '-' * 48)
    for length_str, ev in sorted(evals.items(), key=lambda x: int(x[0])):
        L = int(length_str)
        ah = ev.get('mean_acc_hash_bits', 0)
        ao = ev.get('mean_acc_overall', 0)
        flag = 'OK' if ah >= 0.9 else ('?' if ah >= 0.7 else 'FAIL')
        print(f' {L:>16,}  {ah*100:>10.1f}%  {ao*100:>12.1f}%  {flag}')
    print('=' * 68 + '\n')


def run_benchmark(cfg: BenchmarkConfig, quicktest: bool, seed: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # Directories
    results_dir = HERE / 'results'
    ckpt_dir = HERE / 'checkpoints'
    results_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    mode_tag = 'quicktest' if quicktest else 'full'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'sha256_{mode_tag}_{ts}.json'

    print('\n' + '=' * 68)
    print(' MANIFOLD SHA256 BENCHMARK')
    print(f' Mode: {"QUICKTEST" if quicktest else "FULL BENCHMARK"}')
    print(f' Device: {device}')
    print('=' * 68)

    model = gfn.create(
        preset_name='stable-torus',
        vocab_size=2,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        integrator=cfg.integrator,
        impulse_scale=cfg.impulse_scale,
        holographic=True,
        dynamics_type='direct',
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f' Parameters: {n_params:,}')
    print(f' Dim={cfg.dim}, Depth={cfg.depth}, Heads={cfg.heads}')
    print(f' Train msg_bytes={cfg.train_message_bytes}, steps={cfg.train_steps}')
    print(f' Eval lengths (bytes): {cfg.eval_lengths}')
    print('=' * 68 + '\n')

    cfg_dict = {k: v for k, v in asdict(cfg).items()}
    cfg_dict['device'] = str(device)
    cfg_dict['n_params'] = n_params
    cfg_dict['seed'] = seed
    writer = ResultsWriter(results_path, cfg_dict)

    print('[Phase 1] Training on short SHA256 sequences...')
    train_result = train(model, cfg, device, ckpt_dir, rng)
    writer.write_training(train_result)

    if train_result.get('converged_at') is None:
        print(f"[WARN] Model did not converge in {cfg.train_steps} steps.")

    print('\n[Phase 2] Evaluating at increasing message lengths...')
    
    for length in cfg.eval_lengths:
        print(f'\n── Message length: {length:,} bytes ─────────────────────────')
        t0 = time.time()
        result = evaluate_length(
            model, length,
            cfg.eval_batch_size,
            device,
        )
        elapsed = time.time() - t0
        result['eval_time_s'] = round(elapsed, 2)

        print(f'  Acc Hash: {result["mean_acc_hash_bits"]*100:.1f}%  '
              f'| Overall: {result["mean_acc_overall"]*100:.1f}%  '
              f'| Time: {elapsed:.1f}s')

        writer.write_length(length, result)

    print_summary(writer)
    print(f'Full results -> {results_path}')
    print(f'Final model  -> {ckpt_dir / "model_final.pt"}\n')


def main():
    parser = argparse.ArgumentParser(
        description='MANIFOLD SHA256 Benchmark'
    )
    parser.add_argument(
        '--quicktest', action='store_true',
        help='Run fast logic verification (< 5min)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    args = parser.parse_args()

    cfg = QuickTestConfig() if args.quicktest else BenchmarkConfig()
    run_benchmark(cfg, quicktest=args.quicktest, seed=args.seed)


if __name__ == '__main__':
    main()
