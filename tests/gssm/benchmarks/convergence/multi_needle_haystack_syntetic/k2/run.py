#!/usr/bin/env python3
"""
MANIFOLD Multi-Needle-in-a-Haystack (MNIAH) Benchmark
======================================================

Extension of the NIAH benchmark with K needles per sequence.

PHYSICAL INTUITION
------------------
Multi-needle uses AND semantics:
  - K needles are placed at random positions in the sequence
  - Target: y[t] = 1 only after ALL K needles have been seen
  - The model must accumulate state across K distinct events
  - This is strictly harder than single NIAH: the model can't just
    "latch on first 1" — it must count/track K kicks before flipping

TASK
----
Input:  sequence of length L with exactly K ones at random positions
Target: y[t] = 1 for t >= last_needle_pos, else 0
Angles: class 0 → -π/2, class 1 → +π/2

The key challenge vs single NIAH:
  - Single NIAH: one kick → stay latched
  - Multi NIAH:  K kicks must accumulate before final state flips
  - Tests whether geodesic dynamics can compose multiple impulses

MODES
-----
  python run_multi_needle.py               # Full benchmark
  python run_multi_needle.py --quicktest   # Fast logic check (~5min)
  python run_multi_needle.py --needles 3   # Override needle count

RESULTS
-------
  results/mniah_results.json   # Written incrementally after each length
  checkpoints/model_*.pt       # Checkpoints during training + final

Usage:
  cd tests/system/convergence/needle_haystack
  python run_multi_needle.py [--quicktest] [--needles K] [--seed SEED]
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from tqdm import tqdm

# ── Bootstrap ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = HERE.parents[3]  # convergente/multi_needle.../run.py -> benchmarks -> convergence -> tests -> ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gfn
from gfn import Model, create, loss
from gfn.realizations.gssm.losses.toroidal import ToroidalDistanceLoss


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
    'topology': {'type': 'torus', 'riemannian_type': 'low_rank'},
    'stability': {
        'enable_trace_normalization': True,
        'base_dt': 0.4,
        'velocity_saturation': 15.0,
        'friction': 2.0,
        'toroidal_curvature_scale': 0.01
      },

    'hysteresis': {
        'enabled': False
    }
}


@dataclass
class BenchmarkConfig:
    # Model
    dim: int = 16
    depth: int = 2
    heads: int = 2
    integrator: str = 'leapfrog'
    impulse_scale: float = 80.0
    dynamics_type: str = 'direct',

    # Multi-needle
    num_needles: int = 2           # K needles per sequence (AND semantics)

    # Training
    train_seq_len: int = 64
    train_batch_size: int = 64
    train_steps: int = 800
    checkpoint_every: int = 100
    lr: float = 1e-3
    max_lr: float = 2e-3

    # Evaluation
    eval_lengths: List[int] = None
    eval_batch_size: int = 8

    # Convergence
    acc_threshold: float = 0.995
    patience: int = 100
    min_steps: int = 200

    def __post_init__(self):
        if self.eval_lengths is None:
            self.eval_lengths = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000]


@dataclass
class QuickTestConfig(BenchmarkConfig):
    """Fast config to verify all logic in < 5 minutes."""
    dim: int = 128
    depth: int = 4
    heads: int = 4
    num_needles: int = 2
    train_seq_len: int = 24
    train_batch_size: int = 32
    train_steps: int = 120
    checkpoint_every: int = 40
    eval_lengths: List[int] = None
    min_steps: int = 40
    patience: int = 10

    def __post_init__(self):
        if self.eval_lengths is None:
            self.eval_lengths = [64, 256, 512, 1024]


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

PI = math.pi


def make_mniah_batch(
    batch_size: int,
    seq_len: int,
    num_needles: int,
    device: torch.device,
    needle_positions: Optional[List[List[int]]] = None,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]], List[int]]:
    """
    Generate a Multi-Needle-in-a-Haystack batch.

    AND semantics: target flips to 1 only after the LAST needle.

    Returns:
        x            [B, L]  – input tokens (0=haystack, 1=needle)
        y_angle      [B, L]  – target angles (-π/2 or +π/2)
        y_class      [B, L]  – target classes (0 or 1)
        all_positions        – list of K needle positions per sample
        last_positions       – position of the last needle per sample
    """
    assert num_needles >= 1
    assert seq_len > num_needles + 2, \
        f"seq_len={seq_len} too short for {num_needles} needles"

    x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    y_class = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    all_positions: List[List[int]] = []
    last_positions: List[int] = []

    for i in range(batch_size):
        if needle_positions is not None:
            positions = sorted(needle_positions[i])
        else:
            # Sample K unique positions from [2%, 98%] of the sequence
            lo = max(1, int(0.02 * seq_len))
            hi = max(lo + num_needles + 1, int(0.98 * seq_len))

            if rng is not None:
                # Sample without replacement using a manual approach
                pool = torch.randperm(hi - lo, generator=rng, device=device)[:num_needles]
            else:
                pool = torch.randperm(hi - lo, device=device)[:num_needles]

            positions = sorted((pool + lo).tolist())

        last_pos = positions[-1]
        all_positions.append(positions)
        last_positions.append(last_pos)

        # Place needles
        for p in positions:
            x[i, p] = 1

        # Target: 1 only after ALL needles seen (AND semantics)
        y_class[i, last_pos:] = 1

    y_angle = (y_class.float() * 2.0 - 1.0) * (PI * 0.5)
    return x, y_angle, y_class, all_positions, last_positions


# ══════════════════════════════════════════════════════════════════════════════
# ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

def toroidal_accuracy(x_pred: torch.Tensor, y_class: torch.Tensor) -> float:
    # x_pred can be [B, L, D] or [B, L, H, D]
    if x_pred.ndim == 4:
        # Average over heads/ensemble
        x_pred = x_pred.mean(dim=2)
    
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
    # Average over last (spatial) dim for classification
    preds = (dist_pos.mean(dim=-1) < dist_neg.mean(dim=-1)).long()
    return (preds == y_class).float().mean().item()


def detailed_accuracy(
    x_pred: torch.Tensor,      # [B, L, D]
    y_class: torch.Tensor,     # [B, L]
    all_positions: List[List[int]],
    last_positions: List[int],
) -> Dict[str, float]:
    """
    Breakdown per region:
      acc_before_first : tokens before the first needle (all should be 0)
      acc_between      : tokens between first and last needle (should be 0)
      acc_after_last   : tokens after the last needle (should be 1) — THE KEY METRIC
      acc_overall      : full sequence accuracy
    Also reports false_positive_rate: fraction of between-needle positions
    that were incorrectly predicted as 1 (premature trigger).
    """
    # Average heads if present
    if x_pred.ndim == 4:
        x_pred = x_pred.mean(dim=2)

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

    acc_before, acc_between, acc_after = [], [], []
    false_positives = []
    B, L = y_class.shape

    for i in range(B):
        positions = all_positions[i]
        first_pos = positions[0]
        last_pos = last_positions[i]

        if first_pos > 0:
            acc_before.append(
                (preds[i, :first_pos] == y_class[i, :first_pos]).float().mean().item()
            )

        # Between first and last needle (exclusive): model should output 0
        if last_pos > first_pos + 1:
            between_preds = preds[i, first_pos + 1:last_pos]
            between_true = y_class[i, first_pos + 1:last_pos]  # all 0
            acc_between.append((between_preds == between_true).float().mean().item())
            # False positives: predicted 1 before all needles seen
            false_positives.append(between_preds.float().mean().item())

        # After last needle: model should output 1
        if last_pos + 1 < L:
            acc_after.append(
                (preds[i, last_pos + 1:] == y_class[i, last_pos + 1:]).float().mean().item()
            )

    return {
        'acc_before_first': float(torch.tensor(acc_before).mean())    if acc_before    else 1.0,
        'acc_between':      float(torch.tensor(acc_between).mean())   if acc_between   else 1.0,
        'acc_after_last':   float(torch.tensor(acc_after).mean())     if acc_after     else 0.0,
        'false_positive_rate': float(torch.tensor(false_positives).mean()) if false_positives else 0.0,
        'acc_overall':      toroidal_accuracy(x_pred, y_class),
    }


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

class ResultsWriter:
    def __init__(self, path: Path, config_dict: Dict):
        self.path = path
        self._data = {
            'benchmark': 'MANIFOLD Multi-Needle-in-a-Haystack',
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
    criterion = ToroidalDistanceLoss()
    model.train()

    best_acc = 0.0
    hits = 0
    converged_at = None
    history = {'loss': [], 'acc': []}

    pbar = tqdm(range(cfg.train_steps), desc=f'Training MNIAH (K={cfg.num_needles})')

    for step in pbar:
        x, y_angle, y_class, _, _ = make_mniah_batch(
            cfg.train_batch_size, cfg.train_seq_len, cfg.num_needles, device, rng=rng
        )

        optimizer.zero_grad()
        out = model(x)
        x_pred = out[0]  # [B, L, D] or [B, L, H, D]
        
        # Ensure target matches prediction shape
        # y_angle starts at [B, L]
        y_exp = y_angle.view(y_angle.shape + (1,) * (x_pred.ndim - y_angle.ndim))
        y_exp = y_exp.expand_as(x_pred)

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

        if step >= cfg.min_steps and acc >= cfg.acc_threshold:
            hits += 1
        else:
            hits = 0
        if hits >= cfg.patience and converged_at is None:
            converged_at = step
            print(f'\n[Train] Converged at step {step} | acc={acc*100:.1f}%')
            break

        if (step + 1) % cfg.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f'mniah_model_step{step+1:05d}.pt'
            torch.save({'step': step + 1, 'model': model.state_dict(),
                        'loss': loss.item(), 'acc': acc}, ckpt_path)
            print(f'[Checkpoint] Step {step+1} -> {ckpt_path.name}')

    final_path = ckpt_dir / 'mniah_model_final.pt'
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
    num_needles: int,
    eval_batch_size: int,
    device: torch.device,
    num_trials: int = 3,
) -> Dict:
    """
    Evaluate MNIAH at a specific context length.

    Runs num_trials batches with random needle placements and averages.

    Key metrics:
      acc_after_last     : accuracy AFTER all needles seen (AND memory test)
      acc_between        : accuracy between first and last needle (false trigger test)
      false_positive_rate: fraction of premature positive predictions
    """
    model.eval()
    trial_results = []

    with torch.no_grad():
        for trial in range(num_trials):
            x, y_angle, y_class, all_pos, last_pos = make_mniah_batch(
                eval_batch_size, length, num_needles, device
            )

            t0 = time.time()
            out = model(x)
            elapsed = time.time() - t0

            x_pred = out[0]
            metrics = detailed_accuracy(x_pred, y_class, all_pos, last_pos)
            metrics['inference_time_s'] = round(elapsed, 3)

            # Report needle spread (avg distance between first and last needle)
            spreads = [last_pos[i] - all_pos[i][0] for i in range(eval_batch_size)]
            metrics['avg_needle_spread'] = round(sum(spreads) / len(spreads), 1)
            metrics['avg_last_needle_pos'] = round(sum(last_pos) / len(last_pos), 1)

            trial_results.append(metrics)

    # Average across trials
    keys = ['acc_before_first', 'acc_between', 'acc_after_last',
            'false_positive_rate', 'acc_overall']
    aggregated = {}
    for k in keys:
        vals = [r[k] for r in trial_results]
        aggregated[k] = round(sum(vals) / len(vals), 4)

    aggregated['avg_needle_spread'] = trial_results[-1]['avg_needle_spread']
    aggregated['avg_last_needle_pos'] = trial_results[-1]['avg_last_needle_pos']
    aggregated['context_length'] = length
    aggregated['num_trials'] = num_trials

    model.train()
    return aggregated


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: ResultsWriter, num_needles: int):
    data = results._data
    train = data.get('training', {})
    evals = data.get('eval_lengths', {})

    print('\n' + '=' * 76)
    print(f' MANIFOLD MULTI-NIAH BENCHMARK (K={num_needles} needles) - RESULTS')
    print('=' * 76)
    print(f' Converged at step : {train.get("converged_at", "N/A")}')
    print(f' Best training acc : {train.get("best_acc", 0)*100:.1f}%')
    print()
    print(f' {"Length":>10}  {"Acc After":>10}  {"Acc Between":>12}  {"FP Rate":>8}  {"Overall":>8}')
    print(' ' + '-' * 60)
    for length_str, ev in sorted(evals.items(), key=lambda x: int(x[0])):
        L = int(length_str)
        aa = ev.get('acc_after_last', 0)
        ab = ev.get('acc_between', 0)
        fp = ev.get('false_positive_rate', 0)
        ao = ev.get('acc_overall', 0)
        flag = 'OK' if aa >= 0.9 else ('?' if aa >= 0.7 else 'FAIL')
        print(f' {L:>10,}  {aa*100:>9.1f}%  {ab*100:>11.1f}%  {fp*100:>7.1f}%  {ao*100:>7.1f}%  {flag}')
    print('=' * 76 + '\n')


def run_benchmark(cfg: BenchmarkConfig, quicktest: bool, seed: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    results_dir = HERE / 'results'
    ckpt_dir = HERE / 'checkpoints'
    results_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    mode_tag = 'quicktest' if quicktest else 'full'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'mniah_k{cfg.num_needles}_{mode_tag}_{ts}.json'

    print('\n' + '=' * 68)
    print(' MANIFOLD MULTI-NEEDLE-IN-A-HAYSTACK (MNIAH) BENCHMARK')
    print(f' Semantics: AND — output 1 only after ALL {cfg.num_needles} needles seen')
    print(f' Mode: {"QUICKTEST" if quicktest else "FULL BENCHMARK"}')
    print(f' Device: {device}')
    print('=' * 68)

    model = gfn.create(
        vocab_size=2,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        integrator=cfg.integrator,
        impulse_scale=cfg.impulse_scale,
        holographic=True,
        physics=PHYSICS_CONFIG,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f' Parameters  : {n_params:,}')
    print(f' Dim={cfg.dim}, Depth={cfg.depth}, Heads={cfg.heads}')
    print(f' Needles K   : {cfg.num_needles}')
    print(f' Train L     : {cfg.train_seq_len}, steps={cfg.train_steps}')
    print(f' Eval lengths: {cfg.eval_lengths}')
    print('=' * 68 + '\n')

    cfg_dict = {k: v for k, v in asdict(cfg).items()}
    cfg_dict['device'] = str(device)
    cfg_dict['n_params'] = n_params
    cfg_dict['seed'] = seed
    writer = ResultsWriter(results_path, cfg_dict)

    # ── Phase 1: Training ────────────────────────────────────────────────────
    print('[Phase 1] Training on short MNIAH sequences...')
    train_result = train(model, cfg, device, ckpt_dir, rng)
    writer.write_training(train_result)

    if train_result.get('converged_at') is None:
        print(f"[WARN] Model did not converge in {cfg.train_steps} steps.")

    # ── Phase 2: Evaluation ──────────────────────────────────────────────────
    print('\n[Phase 2] Evaluating O(1) memory at increasing context lengths...')
    print(f'  (K={cfg.num_needles} needles, AND semantics)\n')

    for length in cfg.eval_lengths:
        # Skip lengths too short for K needles
        if length <= cfg.num_needles + 4:
            print(f'\n── Skipping length {length} (too short for K={cfg.num_needles})')
            continue

        print(f'\n── Context length: {length:,} tokens (K={cfg.num_needles} needles) ───')
        t0 = time.time()
        result = evaluate_length(
            model, length, cfg.num_needles, cfg.eval_batch_size, device
        )
        elapsed = time.time() - t0
        result['eval_time_s'] = round(elapsed, 2)

        aa = result['acc_after_last']
        ab = result['acc_between']
        fp = result['false_positive_rate']
        print(f'  Acc after last needle : {aa*100:.1f}%')
        print(f'  Acc between needles   : {ab*100:.1f}%  (FP rate: {fp*100:.1f}%)')
        print(f'  Overall               : {result["acc_overall"]*100:.1f}%')
        print(f'  Avg needle spread     : {result["avg_needle_spread"]:.0f} tokens')
        print(f'  Time                  : {elapsed:.1f}s')

        writer.write_length(length, result)

    print_summary(writer, cfg.num_needles)
    print(f'Full results -> {results_path}')
    print(f'Final model  -> {ckpt_dir / "mniah_model_final.pt"}\n')


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MANIFOLD Multi-Needle-in-a-Haystack Benchmark'
    )
    parser.add_argument(
        '--quicktest', action='store_true',
        help='Run fast logic verification instead of full benchmark'
    )
    parser.add_argument(
        '--needles', type=int, default=None,
        help='Number of needles K (default: 2 full, 2 quicktest)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    args = parser.parse_args()

    cfg = QuickTestConfig() if args.quicktest else BenchmarkConfig()
    if args.needles is not None:
        cfg.num_needles = args.needles

    run_benchmark(cfg, quicktest=args.quicktest, seed=args.seed)


if __name__ == '__main__':
    main()
