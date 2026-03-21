#!/usr/bin/env python3
"""
MANIFOLD Needle-in-a-Haystack (NIAH) Benchmark
===============================================

Demonstrates O(1) memory scaling of the MANIFOLD architecture.

PHYSICAL INTUITION
------------------
The NIAH task demonstrates a key property of MANIFOLD's geodesic physics:
  - Haystack token (0) → force = 0 → x stays put (inertia)
  - Needle token   (1) → force = 80 → x kicked to +π/2
  - Low friction (GATE_BIAS_CLOSED=-3.0) → kick persists across arbitrarily
    many subsequent haystack tokens
  - Memory is FREE: O(1) state (x, v) regardless of sequence length

Unlike Transformers (which need O(N²) attention to relate tokens N apart),
MANIFOLD needs only to propagate (x, v) through one timestep per token.

TASK
----
Input:  sequence of length L, all zeros except ONE "1" at position p
Target: y[t] = 1 for t >= p (needle seen), else 0
Angles: class 0 → -π/2, class 1 → +π/2

MODES
-----
  python run.py               # Full benchmark (model=large, lengths up to 32k)
  python run.py --quicktest   # Logic verification (fast, ~3min)

RESULTS
-------
  results/niah_results.json   # Written incrementally after each length
  checkpoints/model_*.pt      # Checkpoints during training + final

Usage:
  cd tests/system/convergence/needle_haystack
  python run.py [--quicktest] [--seed SEED]
"""

import argparse
import json
import math
import sys
import time
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

# Production physics — proven by xor_old.py (100% acc @ step 119)
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
    'stability': {'base_dt': 0.4},
}


@dataclass
class BenchmarkConfig:
    # Model
    dim: int = 192
    depth: int = 6
    heads: int = 6
    integrator: str = 'yoshida'
    impulse_scale: float = 80.0

    # Training
    train_seq_len: int = 64       # Short sequences for training
    train_batch_size: int = 64
    train_steps: int = 600
    checkpoint_every: int = 100
    lr: float = 1e-3
    max_lr: float = 2e-3

    # Evaluation lengths (context lengths to test O(1) memory)
    eval_lengths: List[int] = None
    # Needle positions to test per length (fraction of sequence)
    needle_fractions: List[float] = None
    eval_batch_size: int = 8      # Test multiple needle positions at once

    # Convergence
    acc_threshold: float = 0.95
    patience: int = 30            # Steps of acc > threshold to declare convergence
    min_steps: int = 150

    def __post_init__(self):
        if self.eval_lengths is None:
            self.eval_lengths = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000]
        if self.needle_fractions is None:
            self.needle_fractions = [0.05, 0.25, 0.50, 0.75, 0.95]


@dataclass
class QuickTestConfig(BenchmarkConfig):
    """Fast config to verify all logic in < 5 minutes."""
    dim: int = 128
    depth: int = 4
    heads: int = 4
    train_seq_len: int = 20
    train_batch_size: int = 32
    train_steps: int = 80
    checkpoint_every: int = 40
    eval_lengths: List[int] = None
    needle_fractions: List[float] = None
    min_steps: int = 30
    patience: int = 10

    def __post_init__(self):
        if self.eval_lengths is None:
            self.eval_lengths = [64, 256, 512, 1024]
        if self.needle_fractions is None:
            self.needle_fractions = [0.1, 0.5, 0.9]


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

PI = math.pi


def make_niah_batch(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    needle_positions: Optional[List[int]] = None,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Generate a Needle-in-a-Haystack batch.

    Returns:
        x       [B, L] – input tokens (0=haystack, 1=needle)
        y_angle [B, L] – target angles (-π/2 or +π/2)
        y_class [B, L] – target classes (0 or 1)
        positions       – needle position for each sample
    """
    # All haystack (zeros)
    x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    y_class = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    if needle_positions is None:
        # Random needle positions, keeping needle away from extremes
        lo = max(1, int(0.02 * seq_len))
        hi = max(lo + 1, int(0.98 * seq_len))
        if rng is not None:
            raw = torch.randint(lo, hi, (batch_size,), generator=rng, device=device)
        else:
            raw = torch.randint(lo, hi, (batch_size,), device=device)
        positions = raw.tolist()
    else:
        positions = needle_positions

    for i, p in enumerate(positions):
        x[i, p] = 1                         # Place needle
        y_class[i, p:] = 1                  # Target: seen-needle signal

    # Map class → toroidal angle target
    y_angle = (y_class.float() * 2.0 - 1.0) * (PI * 0.5)  # {-π/2, +π/2}
    return x, y_angle, y_class, positions


# ══════════════════════════════════════════════════════════════════════════════
# ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

def toroidal_accuracy(x_pred: torch.Tensor, y_class: torch.Tensor) -> float:
    """
    Classify each position via shortest toroidal distance to ±π/2.
    Matches xor_old.py formula exactly.
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


def detailed_accuracy(
    x_pred: torch.Tensor,    # [B, L, D]
    y_class: torch.Tensor,   # [B, L]
    positions: List[int],
) -> Dict[str, float]:
    """
    Break down accuracy into before/at/after needle segments.
    This is the core NIAH metric: does the model remember the needle?
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
    preds = (dist_pos.mean(dim=-1) < dist_neg.mean(dim=-1)).long()  # [B, L]

    acc_before, acc_at, acc_after = [], [], []
    B, L = y_class.shape

    for i, p in enumerate(positions):
        if p > 0:
            acc_before.append((preds[i, :p] == y_class[i, :p]).float().mean().item())
        acc_at.append((preds[i, p:p+1] == y_class[i, p:p+1]).float().mean().item())
        if p + 1 < L:
            acc_after.append((preds[i, p+1:] == y_class[i, p+1:]).float().mean().item())

    return {
        'acc_before': float(torch.tensor(acc_before).mean()) if acc_before else 1.0,
        'acc_at':     float(torch.tensor(acc_at).mean())     if acc_at else 0.0,
        'acc_after':  float(torch.tensor(acc_after).mean())  if acc_after else 0.0,
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
            'benchmark': 'MANIFOLD Needle-in-a-Haystack',
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
        """Called immediately after each eval length — progressive results."""
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
    Train the MANIFOLD model on NIAH sequences.
    Uses the exact optimizer setup proven by xor_old.py.
    """
    # Optimizer: AdamW + OneCycleLR with physics-aware parameter groups
    optimizer = optim.AdamW([
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
            'lr': cfg.lr, 'weight_decay': 1e-4,
        },
        {
            # Physics params (initial state, impulse scale, gates) update faster
            'params': [p for n, p in model.named_parameters()
                       if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
            'lr': cfg.lr * 10, 'weight_decay': 0,
        },
    ])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.max_lr, total_steps=cfg.train_steps, pct_start=0.2
    )
    criterion = ToroidalDistanceLoss()
    model.train()

    best_acc = 0.0
    hits = 0
    converged_at = None
    history = {'loss': [], 'acc': []}

    pbar = tqdm(range(cfg.train_steps), desc='Training NIAH model')

    for step in pbar:
        x, y_angle, y_class, _ = make_niah_batch(
            cfg.train_batch_size, cfg.train_seq_len, device, rng=rng
        )

        optimizer.zero_grad()
        out = model(x)
        x_pred = out[0]                             # [B, L, D]
        y_exp = y_angle.unsqueeze(-1).expand_as(x_pred)

        loss = criterion(x_pred, y_exp)
        if torch.isnan(loss):
            print(f"[WARN] NaN loss at step {step}, skipping")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            acc = toroidal_accuracy(x_pred, y_class)

        history['loss'].append(loss.item())
        history['acc'].append(acc)

        if step % 5 == 0:
            pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{acc*100:.1f}%')

        best_acc = max(best_acc, acc)

        # Convergence check
        if step >= cfg.min_steps and acc >= cfg.acc_threshold:
            hits += 1
        else:
            hits = 0
        if hits >= cfg.patience and converged_at is None:
            converged_at = step
            print(f'\n[Train] Converged at step {step} | acc={acc*100:.1f}%')
            break

        # Checkpoint
        if (step + 1) % cfg.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f'model_step{step+1:05d}.pt'
            torch.save({'step': step + 1, 'model': model.state_dict(),
                        'loss': loss.item(), 'acc': acc}, ckpt_path)
            print(f'[Checkpoint] Step {step+1} -> {ckpt_path.name}')

    # Save final model
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
    length: int,
    needle_fractions: List[float],
    eval_batch_size: int,
    device: torch.device,
) -> Dict:
    """
    Evaluate NIAH at a specific context length.

    Tests the model with the needle at each specified fraction of the sequence.
    Reports:
      - acc_before: positions before needle (should always be ~1.0)
      - acc_at:     the needle position itself
      - acc_after:  ALL positions AFTER needle (the memory test!)
      - acc_overall: full sequence accuracy

    The key metric is acc_after: does the model remember the needle
    for (length - needle_pos) subsequent tokens?
    """
    model.eval()
    needle_results = {}

    with torch.no_grad():
        for frac in needle_fractions:
            needle_pos = max(1, int(frac * length))
            # Use batch of size eval_batch_size, all with the SAME needle position
            positions = [needle_pos] * eval_batch_size

            x, y_angle, y_class, _ = make_niah_batch(
                eval_batch_size, length, device, needle_positions=positions
            )

            t0 = time.time()
            out = model(x)
            elapsed = time.time() - t0

            x_pred = out[0]  # [B, L, D]
            metrics = detailed_accuracy(x_pred, y_class, positions)
            metrics['needle_pos'] = needle_pos
            metrics['distance_to_end'] = length - needle_pos
            metrics['inference_time_s'] = round(elapsed, 3)
            needle_results[f'needle_{int(frac*100):02d}pct'] = metrics

    # Aggregate across needle positions
    acc_afters = [v['acc_after'] for v in needle_results.values()]
    acc_overalls = [v['acc_overall'] for v in needle_results.values()]

    model.train()

    return {
        'context_length': length,
        'mean_acc_after': round(float(torch.tensor(acc_afters).mean()), 4),
        'mean_acc_overall': round(float(torch.tensor(acc_overalls).mean()), 4),
        'needle_positions': needle_results,
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
    print(' MANIFOLD NIAH BENCHMARK - RESULTS')
    print('=' * 68)
    print(f' Converged at step: {train.get("converged_at", "N/A")}')
    print(f' Best training acc: {train.get("best_acc", 0)*100:.1f}%')
    print()
    print(f' {"Context Length":>16}  {"Acc (After)":>11}  {"Acc (Overall)":>13}')
    print(' ' + '-' * 48)
    for length_str, ev in sorted(evals.items(), key=lambda x: int(x[0])):
        L = int(length_str)
        aa = ev.get('mean_acc_after', 0)
        ao = ev.get('mean_acc_overall', 0)
        flag = 'OK' if aa >= 0.9 else ('?' if aa >= 0.7 else 'FAIL')
        print(f' {L:>16,}  {aa*100:>10.1f}%  {ao*100:>12.1f}%  {flag}')
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
    results_path = results_dir / f'niah_{mode_tag}_{ts}.json'

    print('\n' + '=' * 68)
    print(' MANIFOLD NEEDLE-IN-A-HAYSTACK (NIAH) BENCHMARK')
    print(' Physical memory: O(1) state - inertia preserves needle signal')
    print(f' Mode: {"QUICKTEST" if quicktest else "FULL BENCHMARK"}')
    print(f' Device: {device}')
    print('=' * 68)

    # ── Build model ─────────────────────────────────────────────────────────
    model = gfn.create(
        preset_name='stable-torus',
        vocab_size=2,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        integrator=cfg.integrator,
        impulse_scale=cfg.impulse_scale,
        holographic=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f' Parameters: {n_params:,}')
    print(f' Dim={cfg.dim}, Depth={cfg.depth}, Heads={cfg.heads}')
    print(f' Train seq_len={cfg.train_seq_len}, steps={cfg.train_steps}')
    print(f' Eval lengths: {cfg.eval_lengths}')
    print('=' * 68 + '\n')

    # Results writer
    cfg_dict = {k: v for k, v in asdict(cfg).items()}
    cfg_dict['device'] = str(device)
    cfg_dict['n_params'] = n_params
    cfg_dict['seed'] = seed
    writer = ResultsWriter(results_path, cfg_dict)

    # ── Phase 1: Training ───────────────────────────────────────────────────
    print('[Phase 1] Training on short NIAH sequences...')
    train_result = train(model, cfg, device, ckpt_dir, rng)
    writer.write_training(train_result)

    if train_result.get('converged_at') is None:
        print(f"[WARN] Model did not converge in {cfg.train_steps} steps. "
              "Evaluation may show poor results.")

    # ── Phase 2: Evaluation at scale ────────────────────────────────────────
    print('\n[Phase 2] Evaluating O(1) memory at increasing context lengths...')
    print('  (Results saved after each length — never lose data)\n')

    for length in cfg.eval_lengths:
        print(f'\n── Context length: {length:,} tokens ─────────────────────────')
        t0 = time.time()
        result = evaluate_length(
            model, length,
            cfg.needle_fractions,
            cfg.eval_batch_size,
            device,
        )
        elapsed = time.time() - t0

        result['eval_time_s'] = round(elapsed, 2)

        # Print immediate summary
        print(f'  Acc after needle: {result["mean_acc_after"]*100:.1f}%  '
              f'| Overall: {result["mean_acc_overall"]*100:.1f}%  '
              f'| Time: {elapsed:.1f}s')
        for k, v in result['needle_positions'].items():
            pos = v['needle_pos']
            dist = v['distance_to_end']
            aa = v['acc_after']
            print(f'    Needle @ pos {pos:>6,} (dist {dist:>6,} to end): '
                  f'acc_after={aa*100:.1f}%')

        # Write immediately
        writer.write_length(length, result)

    # ── Summary ─────────────────────────────────────────────────────────────
    print_summary(writer)
    print(f'Full results -> {results_path}')
    print(f'Final model  -> {ckpt_dir / "model_final.pt"}\n')


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MANIFOLD Needle-in-a-Haystack Benchmark'
    )
    parser.add_argument(
        '--quicktest', action='store_true',
        help='Run fast logic verification (< 5min) instead of full benchmark'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    args = parser.parse_args()

    cfg = QuickTestConfig() if args.quicktest else BenchmarkConfig()
    run_benchmark(cfg, quicktest=args.quicktest, seed=args.seed)


if __name__ == '__main__':
    main()
