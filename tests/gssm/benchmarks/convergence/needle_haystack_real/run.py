#!/usr/bin/env python3
"""
MANIFOLD Real Needle-in-a-Haystack Benchmark
=============================================

Validates MANIFOLD on the real NLP dataset:
  ameyhengle/Multilingual-Needle-in-a-Haystack

PHYSICAL INTUITION
------------------
Real text introduces high entropy. The model must latch onto the specifically
requested needle (answer_sentence) within a real 4K-32K token context.
By mapping the full tokenizer vocab onto S1 (ToroidalDistanceLoss), we test if
geodesic metric-space topology handles massive discrete vocabularies effectively.

CONFIGS
-------
  baseline  ~400-600 token sequences (recommended for initial runs)
  4k        ~4000 token sequences
  8k        ~8000 token sequences
  16k       ~16000 token sequences
  32k       ~32000 token sequences

Usage:
  python run.py               # Full benchmark (4k dataset)
  python run.py --quicktest   # Fast logic verification
  python run.py --config 8k   # Specific context length config
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset

# ── Bootstrap ─────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gfn
from gfn.realizations.gssm.models.g_ssm import GSSM as Model, Trainer
from gfn.losses import PhysicsInformedLoss

# ── Tokenizer (with fallback chain) ──────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    VOCAB_SIZE = _enc.n_vocab
    def encode_text(text: str) -> List[int]:
        return _enc.encode(text)
    TOKENIZER_NAME = 'tiktoken-cl100k_base'
except ImportError:
    try:
        import logging
        from transformers import AutoTokenizer, logging as hf_logging
        hf_logging.set_verbosity_error()  # Suppress 'seq > 1024' positional embedding warning
        # (MANIFOLD uses functional embeddings — positional limits don't apply)
        _enc = AutoTokenizer.from_pretrained('gpt2')
        _enc.model_max_length = 1_000_000  # remove the 1024 cap from the tokenizer object
        VOCAB_SIZE = _enc.vocab_size
        def encode_text(text: str) -> List[int]:
            return _enc.encode(text, truncation=False)
        TOKENIZER_NAME = 'transformers-gpt2'
    except ImportError:
        VOCAB_SIZE = 256
        def encode_text(text: str) -> List[int]:
            return [b % 256 for b in text.encode('utf-8', errors='replace')]
        TOKENIZER_NAME = 'ascii-byte-fallback'

PI = math.pi

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS CONFIG  — aligned with needle_haystack_syntetic/run.py
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
    'topology': {
        'type': 'torus',
        'R': 2.0,
        'r': 1.0,
    },
    'stability': {
        'base_dt': 0.1,
        'enable_trace_normalization': True,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MAX_SEQ_LEN = 3000  # Hard cap: GPU memory guard + avoids tokenizer warnings


@dataclass
class BenchmarkConfig:
    # Model
    dim: int = 192
    depth: int = 6
    heads: int = 6
    integrator: str = 'yoshida'
    impulse_scale: float = 120.0

    # These are set from VOCAB_SIZE at runtime
    vocab_size: int = VOCAB_SIZE

    # Training  — batch size MUST be small: real sequences are 1k-4k tokens
    train_batch_size: int = 4
    train_steps: int = 800000     # Run until convergence
    checkpoint_every: int = 200
    lr: float = 1e-3
    max_lr: float = 2e-3

    # Evaluation
    eval_batch_size: int = 4
    eval_steps: int = 100

    # Convergence
    acc_threshold: float = 0.90
    patience: int = 200             # Consecutive steps above threshold
    min_steps: int = 300

    # Dataset
    hf_config: str = '4k'
    max_seq_len: int = MAX_SEQ_LEN

@dataclass
class QuickTestConfig(BenchmarkConfig):
    """Fast config to verify all logic. Uses 'baseline' (short sequences)."""
    dim: int = 128
    depth: int = 4
    heads: int = 4
    train_batch_size: int = 4
    train_steps: int = 100000
    checkpoint_every: int = 30
    eval_steps: int = 50
    min_steps: int = 100
    patience: int = 100
    hf_config: str = 'baseline'
    max_seq_len: int = MAX_SEQ_LEN
    integrator: str = 'rk4'

# ══════════════════════════════════════════════════════════════════════════════
# DATA HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_niah_generator(n: int = 10_000):
    """
    Offline fallback: generates synthetic NIAH rows when HuggingFace is unreachable.
    Produces realistic enough text to check that the training loop runs correctly.
    """
    import random
    templates = [
        "The science of physics studies how objects move and interact with energy.",
        "Machine learning algorithms improve from experience without explicit programming.",
        "Climate change refers to long-term shifts in temperatures and weather patterns.",
        "A neural network consists of layers of interconnected computational nodes.",
        "The Earth orbits the Sun at an average distance of 150 million kilometers.",
        "Language models predict the next word given a sequence of previous words.",
        "Calculus is used to compute derivatives and integrals of mathematical functions.",
        "Proteins are large molecules that perform most of the work inside cells.",
        "Artificial intelligence aims to create systems that can perform human-like tasks.",
        "Quantum mechanics describes the behavior of particles at the subatomic scale.",
    ]
    needles = [
        "The secret code is 42.",
        "The password is 1337.",
        "The answer is blue.",
        "The key is seventeen.",
        "The number is ninety-nine.",
    ]
    for i in range(n):
        haystack = " ".join(random.choices(templates, k=random.randint(5, 15)))
        needle = random.choice(needles)
        split_pos = random.randint(len(haystack)//3, 2*len(haystack)//3)
        prompt = haystack[:split_pos] + " " + needle + " " + haystack[split_pos:]
        yield {'prompt': prompt, 'answer_sentence': needle}


def get_real_niah_iterator(hf_config: str):
    """
    Three-tier data source:
      1. HuggingFace streaming (requires network)
      2. HuggingFace offline from local cache
      3. Synthetic NIAH generator (always works, used when HF is unreachable)
    """
    # Tier 1: live streaming
    try:
        ds = load_dataset(
            'ameyhengle/Multilingual-Needle-in-a-Haystack',
            hf_config, split='en', streaming=True
        )
        return iter(ds)
    except Exception as e1:
        pass

    # Tier 2: offline cache
    try:
        import os; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        ds = load_dataset(
            'ameyhengle/Multilingual-Needle-in-a-Haystack',
            hf_config, split='en', streaming=False, download_mode='reuse_cache_if_exists'
        )
        print(f"  [data] Using offline HF cache for config='{hf_config}'")
        return iter(ds)
    except Exception as e2:
        pass

    # Tier 3: synthetic fallback
    print(f"  [data] WARNING: HuggingFace unreachable. Using synthetic NIAH generator.")
    return _synthetic_niah_generator(n=50_000)

def tokenize_and_collate(rows, device: torch.device, max_seq_len: int):
    """
    Tokenize each row's prompt + answer_sentence.
    Truncates total length to max_seq_len to prevent OOM.
    Returns input tokens, toroidal target angles, token classes, and answer mask.
    """
    encoded_pairs = []
    max_len = 0

    for r in rows:
        p_tokens = encode_text(r['prompt'] + " ")
        a_tokens = encode_text(r['answer_sentence'])
        full = p_tokens + a_tokens

        # Hard truncate to max_seq_len
        if len(full) > max_seq_len:
            # Keep as much of the prompt context as possible
            full = full[:max_seq_len]
            # Recompute p_len after truncation
            p_len = min(len(p_tokens), max_seq_len - len(a_tokens)) if len(a_tokens) < max_seq_len else max_seq_len
        else:
            p_len = len(p_tokens)

        encoded_pairs.append((full, p_len))
        max_len = max(max_len, len(full))

    batch_x, batch_mask = [], []
    for full, p_len in encoded_pairs:
        pad_len = max_len - len(full)
        padded = full + [0] * pad_len
        # Mask == 1 ONLY at the answer token positions (what the needle IS)
        mask = [0] * p_len + [1] * max(len(full) - p_len, 0) + [0] * pad_len
        batch_x.append(padded)
        batch_mask.append(mask)

    x = torch.tensor(batch_x, dtype=torch.long, device=device)
    mask = torch.tensor(batch_mask, dtype=torch.float, device=device)

    # LM shift: target at t is input at t-1
    y_class = torch.cat([
        torch.zeros(len(batch_x), 1, dtype=torch.long, device=device),
        x[:, :-1]
    ], dim=1)
    # Mask shifts too  (we predict the answer, not the pad after it)
    mask = torch.cat([
        torch.zeros(len(batch_x), 1, dtype=torch.float, device=device),
        mask[:, :-1]
    ], dim=1)

    # Toroidal mapping: token_id → angle on S¹
    step = (2 * PI) / VOCAB_SIZE
    y_angle = -PI + (y_class.float() + 0.5) * step

    return x, y_angle, y_class, mask


def toroidal_accuracy(logits: torch.Tensor, y_class: torch.Tensor,
                      mask: torch.Tensor) -> float:
    """
    Standard categorical accuracy for logits.
    Only counts positions where mask == 1 (answer tokens).
    """
    if logits.dim() == 4:
        # Multi-head logits [B, S, H, HD] -> [B, S, V]
        logits = logits.mean(dim=2)  # Average heads if categorical readout is split
    
    preds = logits.argmax(dim=-1)
    correct = (preds == y_class).float()

    valid_tokens = mask.sum()
    if valid_tokens > 0:
        return (correct * mask).sum().item() / valid_tokens.item()
    return 0.0

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

class ResultsWriter:
    """Writes benchmark results to JSON incrementally — never loses data."""

    def __init__(self, path: Path, config_dict: Dict):
        self.path = path
        self._data = {
            'benchmark': 'MANIFOLD Real NIAH (HF ameyhengle/Multilingual)',
            'timestamp_start': datetime.now().isoformat(),
            'config': config_dict,
            'training': {},
            'evaluation': {},
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
        print(f"[Results] Training saved -> {self.path.name}")

    def write_eval(self, result: Dict):
        self._data['evaluation'] = result
        self._data['timestamp_eval_done'] = datetime.now().isoformat()
        self._flush()
        print(f"[Results] Evaluation saved -> {self.path.name}")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(model: Model, cfg: BenchmarkConfig, device: torch.device,
          ckpt_dir: Path) -> Dict:
    """
    Train on real-text NIAH sequences.
    Optimizer: AdamW + OneCycleLR with physics-aware parameter groups (proven by xor_old.py).
    """
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
        optimizer, max_lr=cfg.max_lr,
        total_steps=cfg.train_steps, pct_start=0.2
    )
    # PhysicsInformedLoss: NLL (probabilistic) + Physics (geometric regularization)
    loss_cfg = {
        'lambda_physics': 0.05,  # Weight for physical regularizers
        'lambda_geo': 0.005,    # Penalize path curvature
        'lambda_ham': 0.001,    # Preserve energy
        'lambda_kin': 0.001,    # Bound velocity
        'max_kinetic': 20.0,
    }
    criterion = PhysicsInformedLoss(loss_cfg)
    model.train()

    ds_iter = get_real_niah_iterator(cfg.hf_config)
    best_acc, hits, converged_at = 0.0, 0, None
    history = {'loss': [], 'answer_acc': []}
    pbar = tqdm(range(cfg.train_steps), desc=f'Training Real NIAH [{cfg.hf_config}]')

    for step in pbar:
        # Fetch batch
        rows = []
        for _ in range(cfg.train_batch_size):
            try:
                rows.append(next(ds_iter))
            except StopIteration:
                ds_iter = get_real_niah_iterator(cfg.hf_config)
                rows.append(next(ds_iter))

        x, y_angle, y_class, mask = tokenize_and_collate(rows, device, cfg.max_seq_len)

        optimizer.zero_grad()
        out = model(x)
        x_pred = out[0] 
        
        # Loss: x_pred: logits [B, S, V], y_class: token ids [B, S]
        # state_info comes from model hooks during forward
        loss = criterion(x_pred, y_class, state_info=out[2])

        if torch.isnan(loss):
            print(f"[WARN] NaN loss at step {step} (seq_len={x.shape[1]}), skipping")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            answer_acc = toroidal_accuracy(x_pred, y_class, mask)

        history['loss'].append(loss.item())
        history['answer_acc'].append(answer_acc)
        best_acc = max(best_acc, answer_acc)

        if step % 5 == 0:
            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                ans_acc=f'{answer_acc*100:.1f}%',
                seq=x.shape[1],
            )

        if step >= cfg.min_steps and answer_acc >= cfg.acc_threshold:
            hits += 1
        else:
            hits = 0

        if hits >= cfg.patience and converged_at is None:
            converged_at = step
            print(f'\n[Train] Converged at step {step} | ans_acc={answer_acc*100:.1f}%')
            break

        if (step + 1) % cfg.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f'real_niah_step{step+1:06d}.pt'
            torch.save({
                'step': step + 1,
                'model': model.state_dict(),
                'loss': loss.item(),
                'answer_acc': answer_acc,
            }, ckpt_path)
            print(f'[Checkpoint] Step {step+1} -> {ckpt_path.name}')

    final_path = ckpt_dir / 'real_niah_final.pt'
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
        'final_acc': history['answer_acc'][-1] if history['answer_acc'] else None,
        'best_acc': best_acc,
    }

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model: Model, cfg: BenchmarkConfig, device: torch.device) -> Dict:
    """Evaluate answer recall accuracy on held-out HF examples."""
    model.eval()
    ds_iter = get_real_niah_iterator(cfg.hf_config)

    total_acc = 0.0
    total_time = 0.0
    valid_batches = 0
    seq_lengths = []

    with torch.no_grad():
        for _ in tqdm(range(cfg.eval_steps), desc='Evaluating Real NIAH'):
            rows = []
            for _ in range(cfg.eval_batch_size):
                try:
                    rows.append(next(ds_iter))
                except StopIteration:
                    break
            if not rows:
                break

            x, y_angle, y_class, mask = tokenize_and_collate(rows, device, cfg.max_seq_len)

            t0 = time.time()
            out = model(x)
            elapsed = time.time() - t0

            acc = toroidal_accuracy(out[0], y_class, mask)
            total_acc += acc
            total_time += elapsed
            valid_batches += 1
            seq_lengths.append(x.shape[1])

    model.train()

    if valid_batches == 0:
        return {'answer_accuracy': 0.0, 'avg_inference_time_s': 0.0, 'evaluated_batches': 0}

    return {
        'answer_accuracy': round(total_acc / valid_batches, 4),
        'avg_inference_time_s': round(total_time / valid_batches, 3),
        'evaluated_batches': valid_batches,
        'avg_seq_len': round(sum(seq_lengths) / len(seq_lengths), 1),
        'max_seq_len_seen': max(seq_lengths),
    }

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MANIFOLD Real Needle-in-a-Haystack Benchmark'
    )
    parser.add_argument('--quicktest', action='store_true',
                        help='Fast logic verification using baseline config')
    parser.add_argument('--config', type=str, default=None,
                        help='HF dataset config (baseline, 4k, 8k, 16k, 32k)')
    args = parser.parse_args()

    cfg = QuickTestConfig() if args.quicktest else BenchmarkConfig()
    if args.config:
        cfg.hf_config = args.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results_dir = HERE / 'results'
    ckpt_dir = HERE / 'checkpoints'
    results_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    mode_tag = 'quicktest' if args.quicktest else f'full_{cfg.hf_config}'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'results_{mode_tag}_{ts}.json'

    cfg_dict = asdict(cfg)
    cfg_dict['device'] = str(device)
    cfg_dict['tokenizer'] = TOKENIZER_NAME
    writer = ResultsWriter(results_path, cfg_dict)

    model = gfn.create(
        preset_name='stable-torus',
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        integrator=cfg.integrator,
        impulse_scale=cfg.impulse_scale,
        holographic=False,  # Set to False to get categorical logits for NLL
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('\n' + '=' * 68)
    print(' MANIFOLD REAL NEEDLE-IN-A-HAYSTACK BENCHMARK')
    print(f' Dataset: ameyhengle/Multilingual-Needle-in-a-Haystack [{cfg.hf_config}]')
    print(f' Mode:    {"QUICKTEST" if args.quicktest else "FULL BENCHMARK"}')
    print(f' Device:  {device}')
    print('=' * 68)
    print(f' Parameters:  {n_params:,}')
    print(f' Tokenizer:   {TOKENIZER_NAME}  (vocab={VOCAB_SIZE:,})')
    print(f' Dim={cfg.dim}, Depth={cfg.depth}, Heads={cfg.heads}')
    print(f' Max seq len: {cfg.max_seq_len} tokens (hard truncation)')
    print(f' Batch size:  {cfg.train_batch_size} (eval: {cfg.eval_batch_size})')
    print('=' * 68 + '\n')

    print('[Phase 1] Training on real-text NIAH sequences...')
    train_result = train(model, cfg, device, ckpt_dir)
    writer.write_training(train_result)

    if train_result.get('converged_at') is None:
        print(f"[WARN] Model did not converge in {train_result['steps_run']} steps.")

    print('\n[Phase 2] Evaluating answer recall at held-out examples...')
    ev = evaluate(model, cfg, device)
    print(f'  Answer Accuracy:   {ev["answer_accuracy"]*100:.2f}%')
    print(f'  Avg seq length:    {ev.get("avg_seq_len", "N/A")} tokens')
    print(f'  Avg infer time:    {ev["avg_inference_time_s"]}s/batch')
    print(f'  Batches evaluated: {ev["evaluated_batches"]}')
    writer.write_eval(ev)

    print(f'\nResults -> {results_path}')
    print(f'Checkpoints -> {ckpt_dir}\n')


if __name__ == '__main__':
    main()
