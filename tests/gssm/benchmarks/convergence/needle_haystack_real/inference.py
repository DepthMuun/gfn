#!/usr/bin/env python3
"""
MANIFOLD Real NIAH — Inference Script
======================================
Load a saved checkpoint from needle_haystack_real/run.py and probe what the
model learned about language. Provides:

  1. Perplexity estimation on custom text
  2. Greedy next-token generation
  3. Toroidal angle visualization (what the model "sees" in toroidal space)
  4. Answer recall probe (feed prompt → check if model predicts needle tokens)

Usage:
    python inference.py
    python inference.py --ckpt checkpoints/real_niah_final.pt
    python inference.py --ckpt checkpoints/real_niah_final.pt --generate "The capital of France is"
    python inference.py --ckpt checkpoints/real_niah_final.pt --perplexity "The cat sat on the mat."
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ── Bootstrap ─────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gfn.realizations.gssm.models.manifolds.adjoint import AdjointManifold

# ── Physics config — must match training ─────────────────────────────────────
PHYSICS_CONFIG = {
    'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16},
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

PI = math.pi

# ── Tokenizer (GPT-2, matching the training run) ──────────────────────────────
def load_tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        vocab_size = enc.n_vocab
        encode = enc.encode
        decode = enc.decode
        name = 'tiktoken-cl100k_base'
    except ImportError:
        from transformers import AutoTokenizer
        enc = AutoTokenizer.from_pretrained('gpt2')
        vocab_size = enc.vocab_size
        encode = enc.encode
        decode = lambda ids: enc.decode(ids)  # noqa
        name = 'transformers-gpt2'
    return encode, decode, vocab_size, name


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device) -> AdjointManifold:
    print(f"  Loading: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Extract config from checkpoint (saved by training loop)
    cfg = ckpt.get('config', {})
    dim = cfg.get('dim', 128)
    depth = cfg.get('depth', 4)
    heads = cfg.get('heads', 4)
    vocab_size = cfg.get('vocab_size', 50257)
    integrator = cfg.get('integrator', 'yoshida')

    print(f"  Config: dim={dim}, depth={depth}, heads={heads}, vocab={vocab_size}")

    model = AdjointManifold(
        vocab_size=vocab_size, dim=dim, depth=depth, heads=heads,
        integrator_type=integrator,
        impulse_scale=80.0, holographic=True,
        adjoint_method='rk4',
    ).to(device)

    state_key = 'model' if 'model' in ckpt else 'state_dict'
    model.load_state_dict(ckpt[state_key])
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, vocab_size


# ── Inference helpers ─────────────────────────────────────────────────────────

def tokens_to_angles(pred: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Collapse feature dim D → angle via circular mean."""
    return torch.atan2(torch.sin(pred).mean(-1), torch.cos(pred).mean(-1))  # [B, L]


def angles_to_tokens(angles: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Map angles ∈ (-π, π) → token id ∈ [0, V-1]."""
    TWO_PI = 2 * PI
    step = TWO_PI / vocab_size
    shifted = (angles + PI) % TWO_PI
    return (shifted / step).long().clamp(0, vocab_size - 1)


@torch.no_grad()
def predict_next(model: AdjointManifold, tokens: list, device: torch.device,
                 vocab_size: int, temperature: float = 1.0, top_k: int = 10):
    """
    Forward pass on `tokens`, return top-k predictions for the NEXT token
    (position -1 of the sequence).
    """
    x = torch.tensor([tokens], dtype=torch.long, device=device)  # [1, L]
    out = model(x)
    x_pred = out[0]  # [1, L, D]

    # Last position prediction
    angles = tokens_to_angles(x_pred[:, -1], vocab_size)   # [1]
    pred_id = angles_to_tokens(angles, vocab_size)[0].item()

    # Toroidal top-k: rank tokens by angular distance
    TWO_PI = 2 * PI
    step = TWO_PI / vocab_size
    # All token angles on S1
    all_ids = torch.arange(vocab_size, device=device)
    all_angles = -PI + (all_ids.float() + 0.5) * step

    last_angle = angles[0]
    dists = torch.abs(torch.atan2(
        torch.sin(all_angles - last_angle),
        torch.cos(all_angles - last_angle)
    ))
    topk_ids = dists.topk(top_k, largest=False).indices.tolist()

    return pred_id, topk_ids, x_pred, angles


@torch.no_grad()
def greedy_generate(model: AdjointManifold, prompt_tokens: list,
                    device: torch.device, vocab_size: int,
                    n_tokens: int = 20, temperature: float = 1.0):
    """Autoregressively generate `n_tokens` tokens after `prompt_tokens`."""
    tokens = list(prompt_tokens)
    generated = []

    for _ in range(n_tokens):
        pred_id, _, _, _ = predict_next(model, tokens, device, vocab_size, temperature)
        tokens.append(pred_id)
        generated.append(pred_id)

    return generated


@torch.no_grad()
def perplexity(model: AdjointManifold, tokens: list,
               device: torch.device, vocab_size: int) -> float:
    """
    Estimate per-token perplexity using toroidal angular distance
    as a proxy for token prediction confidence.
    H = -log P(next) where P is approximated by 1 - (dist / π)
    """
    if len(tokens) < 2:
        return float('inf')

    x = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model(x)
    x_pred = out[0]  # [1, L, D]

    angles = tokens_to_angles(x_pred[0], vocab_size)  # [L]

    TWO_PI = 2 * PI
    step = TWO_PI / vocab_size

    # Ground truth angles for shifted target (LM convention: predict t from t-1)
    targets = torch.tensor(tokens[1:], dtype=torch.long, device=device)
    target_angles = -PI + (targets.float() + 0.5) * step

    # Predicted angles for positions 0..L-2
    pred_angles = angles[:-1]

    # Toroidal angular distance ∈ [0, π]
    dists = torch.abs(torch.atan2(
        torch.sin(target_angles - pred_angles),
        torch.cos(target_angles - pred_angles)
    ))

    # Approximate probability: p ≈ 1 - dist/π (1 = perfect, 0 = antipodal)
    p = (1.0 - dists / PI).clamp(1e-6, 1.0)
    nll = -torch.log(p).mean().item()
    ppl = math.exp(nll)
    return ppl


@torch.no_grad()
def visualize_toroidal_angles(model: AdjointManifold, tokens: list,
                               decode, device: torch.device, vocab_size: int,
                               max_show: int = 20):
    """Show what angle the model outputs per token and what it predicts."""
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model(x)
    x_pred = out[0][0]  # [L, D]

    angles = tokens_to_angles(x_pred.unsqueeze(0), vocab_size)[0]   # [L]
    pred_ids = angles_to_tokens(angles, vocab_size)  # [L]

    TWO_PI = 2 * PI
    step = TWO_PI / vocab_size
    target_angles = -PI + (torch.tensor(tokens, device=device).float() + 0.5) * step
    dists = torch.abs(torch.atan2(
        torch.sin(target_angles - angles),
        torch.cos(target_angles - angles)
    ))

    print(f"\n{'Pos':>4}  {'True token':>20}  {'Pred token':>20}  {'Angle':>8}  {'AngDist':>8}")
    print("─" * 72)
    for i in range(min(len(tokens), max_show)):
        true_tok = tokens[i]
        pred_tok = pred_ids[i].item()
        ang = angles[i].item()
        dist = dists[i].item()
        true_str = repr(decode([true_tok]))[:18]
        pred_str = repr(decode([pred_tok]))[:18]
        print(f"{i:>4}  {true_str:>20}  {pred_str:>20}  {ang:>+8.3f}  {dist:>8.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MANIFOLD Real NIAH — Inference & Language Probe'
    )
    parser.add_argument('--ckpt', type=str,
                        default='checkpoints/real_niah_final.pt',
                        help='Path to checkpoint .pt file')
    parser.add_argument('--generate', type=str, default=None,
                        help='Text prompt to continue with greedy generation')
    parser.add_argument('--n-tokens', type=int, default=25,
                        help='Number of tokens to generate (default: 25)')
    parser.add_argument('--perplexity', type=str, default=None,
                        help='Text to measure perplexity on')
    parser.add_argument('--probe', type=str, default=None,
                        help='Text to visualize per-token toroidal angles')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encode, decode, _, tokenizer_name = load_tokenizer()

    ckpt_path = Path(args.ckpt) if Path(args.ckpt).is_absolute() else HERE / args.ckpt
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print('\n' + '=' * 60)
    print(' MANIFOLD Real NIAH -- Inference')
    print('=' * 60)
    print(f'  Device:    {device}')
    print(f'  Tokenizer: {tokenizer_name}')
    model, vocab_size = load_model(ckpt_path, device)
    print('=' * 60 + '\n')

    # ── Default demo if no flags given ─────────────────────────────────────
    if not any([args.generate, args.perplexity, args.probe]):
        demo_texts = [
            "france",
            "In mathematics, a prime number is",
            "The quick brown fox",
        ]
        args.generate = demo_texts[0]
        args.perplexity = "The quick brown fox jumps over the lazy dog."
        args.probe = "Language modeling is the task of"

    # ── Generation ─────────────────────────────────────────────────────────
    if args.generate:
        print(f'[GENERATION] Prompt: "{args.generate}"')
        prompt_tokens = encode(args.generate)
        generated_ids = greedy_generate(
            model, prompt_tokens, device, vocab_size,
            n_tokens=args.n_tokens, temperature=args.temperature
        )
        continuation = decode(generated_ids)
        print(f'  Continuation: "{continuation}"')
        print(f'  Full text:    "{args.generate}{continuation}"\n')

        # Also show top-5 next tokens after prompt
        _, topk, _, _ = predict_next(model, prompt_tokens, device, vocab_size, top_k=5)
        print('  Top-5 next token candidates (by toroidal proximity):')
        for rank, tid in enumerate(topk):
            tok_str = repr(decode([tid]))
            print(f'    {rank+1}. {tok_str:>20}  (id={tid})')
        print()

    # ── Perplexity ─────────────────────────────────────────────────────────
    if args.perplexity:
        print(f'[PERPLEXITY] Text: "{args.perplexity}"')
        tokens = encode(args.perplexity)
        print(f'  Tokens: {len(tokens)}')
        ppl = perplexity(model, tokens, device, vocab_size)
        print(f'  Toroidal Perplexity: {ppl:.2f}')
        print(f'  (Lower = model is more "confident" about next token)\n')

    # ── Toroidal angle visualization ────────────────────────────────────────
    if args.probe:
        print(f'[ANGLE PROBE] Text: "{args.probe}"')
        tokens = encode(args.probe)
        print(f'  Tokens: {len(tokens)}')
        visualize_toroidal_angles(model, tokens, decode, device, vocab_size, max_show=25)
        print()


if __name__ == '__main__':
    main()
