#!/usr/bin/env python3
"""
Performance and Speed Profiler (speedbench) for GSSM/GFN Components.
"""

import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

# Setup paths to import gfn
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[4]  # Navigate to the manifold_mini root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Adjust path to find dev/dev/gfn
GFN_PATH = PROJECT_ROOT / "dev" / "dev" / "gfn"
if str(GFN_PATH) not in sys.path:
    sys.path.insert(0, str(GFN_PATH))

from gfn.realizations import gssm
from gfn.realizations.gssm.config.schema import PhysicsConfig
from gfn.realizations.gssm.models.components.embedding import FunctionalEmbedding
from gfn.realizations.gssm.geometry.euclidean import EuclideanGeometry
from gfn.realizations.gssm.geometry.torus import ToroidalRiemannianGeometry
from gfn.realizations.gssm.geometry.low_rank import LowRankRiemannianGeometry
from gfn.realizations.gssm.physics.integrators.symplectic.leapfrog import LeapfrogIntegrator
from gfn.realizations.gssm.physics.integrators.adaptive import AdaptiveIntegrator
from gfn.realizations.gssm.models.components.mixer import GeodesicAttentionMixer, FlowMixer
from gfn.realizations.gssm.models.components.readout import CategoricalReadout, ImplicitReadout
from gfn.realizations.gssm.physics.engine import ManifoldPhysicsEngine


def time_cuda_op(f, warmup=10, runs=50) -> float:
    """Helper to time a function using PyTorch CUDA Events for precision."""
    # Warmup
    for _ in range(warmup):
        res = f()
        if isinstance(res, torch.Tensor):
            res.sum().backward(retain_graph=True)
        elif isinstance(res, dict):
            loss = sum(v.sum() for v in res.values() if isinstance(v, torch.Tensor))
            loss.backward(retain_graph=True)
            
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(runs):
        res = f()
        if isinstance(res, tuple):
            loss = sum(r.sum() for r in res if isinstance(r, torch.Tensor))
        elif isinstance(res, dict):
            loss = sum(v.sum() for v in res.values() if isinstance(v, torch.Tensor))
        else:
            loss = res.sum()
        loss.backward(retain_graph=True)
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / runs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running speedbench on device: {device}")
    if device.type != 'cuda':
        print("[WARN] For accurate profiling, running on a CUDA GPU is highly recommended.")

    # Configurations for profiling
    batch_size = 32
    seq_len = 32
    dim = 64
    heads = 4
    head_dim = dim // heads
    vocab_size = 32
    rank = 16

    print("\n" + "=" * 60)
    print(" 1) PROFILING INDIVIDUAL COMPONENTS (FORWARD + BACKWARD)")
    print("=" * 60)
    print(f"Settings: Batch size={batch_size}, Seq len={seq_len}, Dim={dim}, Heads={heads}")
    print("-" * 60)

    # 1. Embedding
    emb_lookup = FunctionalEmbedding(vocab_size, dim, mode='lookup').to(device)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    t_emb = time_cuda_op(lambda: emb_lookup(token_ids))
    print(f"Embedding (Lookup)           : {t_emb:.3f} ms")

    # 2. Geometry
    # Create mock state inputs: [B, H, HD]
    x = torch.randn(batch_size, heads, head_dim, device=device, requires_grad=True)
    v = torch.randn(batch_size, heads, head_dim, device=device, requires_grad=True)
    force = torch.randn(batch_size, heads, head_dim, device=device)
    
    phys_cfg = PhysicsConfig()
    geom_euclidean = EuclideanGeometry(config=phys_cfg).to(device)
    geom_torus = ToroidalRiemannianGeometry(dim=head_dim, num_heads=heads, config=phys_cfg).to(device)
    geom_low_rank = LowRankRiemannianGeometry(dim=head_dim, rank=rank, num_heads=heads, config=phys_cfg).to(device)

    t_geom_euc = time_cuda_op(lambda: geom_euclidean(x, v, force))
    print(f"Geometry (Euclidean)         : {t_geom_euc:.3f} ms")

    t_geom_tor = time_cuda_op(lambda: geom_torus(x, v, force))
    print(f"Geometry (Toroidal)          : {t_geom_tor:.3f} ms")

    t_geom_lr = time_cuda_op(lambda: geom_low_rank(x, v, force))
    print(f"Geometry (Low Rank / metric) : {t_geom_lr:.3f} ms")

    # 3. Integrator
    engine_euc = ManifoldPhysicsEngine(geom_euclidean, config=phys_cfg)
    engine_tor = ManifoldPhysicsEngine(geom_torus, config=phys_cfg)
    
    int_leapfrog = LeapfrogIntegrator(engine_tor, config=phys_cfg)
    int_adaptive = AdaptiveIntegrator(engine_tor, config=phys_cfg)

    t_int_lf = time_cuda_op(lambda: int_leapfrog.step(x, v, force))
    print(f"Integrator (Leapfrog)        : {t_int_lf:.3f} ms")

    t_int_adapt = time_cuda_op(lambda: int_adaptive.step(x, v, force))
    print(f"Integrator (Adaptive)        : {t_int_adapt:.3f} ms")

    # 4. Mixer
    x_in_mixer = torch.randn(batch_size, heads, head_dim, device=device, requires_grad=True)
    v_in_mixer = torch.randn(batch_size, heads, head_dim, device=device, requires_grad=True)
    
    mixer_lr = FlowMixer(dim, heads=heads).to(device)
    mixer_attn = GeodesicAttentionMixer(dim, heads=heads, topology='torus').to(device)

    t_mix_lr = time_cuda_op(lambda: mixer_lr(x_in_mixer, v_in_mixer))
    print(f"Mixer (Low Rank Flow)        : {t_mix_lr:.3f} ms")

    t_mix_attn = time_cuda_op(lambda: mixer_attn(x_in_mixer, v_in_mixer))
    print(f"Mixer (Geodesic Attention)   : {t_mix_attn:.3f} ms")

    # 5. Readout
    x_readout = torch.randn(batch_size, dim, device=device, requires_grad=True)
    readout_cat = CategoricalReadout(dim, vocab_size, topology_type='torus').to(device)
    readout_impl = ImplicitReadout(dim, vocab_size, topology_type='torus').to(device)

    t_read_cat = time_cuda_op(lambda: readout_cat(x_readout))
    print(f"Readout (Categorical/Torus)  : {t_read_cat:.3f} ms")

    t_read_impl = time_cuda_op(lambda: readout_impl(x_readout))
    print(f"Readout (Implicit/MLP)       : {t_read_impl:.3f} ms")


    print("\n" + "=" * 60)
    print(" 2) END-TO-END MODEL PROFILING (SEQ_LEN = 32)")
    print("=" * 60)

    # Configuration A: Euclidean, Leapfrog, Standard
    cfg_euc = {
        "vocab_size": vocab_size,
        "dim": dim,
        "depth": 2,
        "heads": heads,
        "integrator": "leapfrog",
        "physics": {
            "topology": {"type": "euclidean"},
            "stability": {"adaptive": False, "integrator_type": "leapfrog"},
            "embedding": {"mode": "lookup"},
            "readout": {"type": "standard"}
        }
    }
    model_euc = gssm.create(config=cfg_euc).to(device)
    t_model_euc = time_cuda_op(lambda: model_euc(token_ids))
    print(f"End-to-End Model (Euclidean) : {t_model_euc:.3f} ms")

    # Configuration B: Torus, Leapfrog (Non-Adaptive)
    cfg_tor_lf = {
        "vocab_size": vocab_size,
        "dim": dim,
        "depth": 2,
        "heads": heads,
        "integrator": "leapfrog",
        "physics": {
            "topology": {"type": "torus", "riemannian_type": "reactive"},
            "stability": {"adaptive": False, "integrator_type": "leapfrog"},
            "embedding": {"mode": "lookup"},
            "readout": {"type": "standard"}
        }
    }
    model_tor_lf = gssm.create(config=cfg_tor_lf).to(device)
    t_model_tor_lf = time_cuda_op(lambda: model_tor_lf(token_ids))
    print(f"End-to-End Model (Torus, Leap): {t_model_tor_lf:.3f} ms")

    # Configuration C: Torus, Adaptive Integrator
    cfg_tor_ad = {
        "vocab_size": vocab_size,
        "dim": dim,
        "depth": 2,
        "heads": heads,
        "integrator": "adaptive",
        "physics": {
            "topology": {"type": "torus", "riemannian_type": "reactive"},
            "stability": {"adaptive": True, "integrator_type": "adaptive"},
            "embedding": {"mode": "lookup"},
            "readout": {"type": "standard"}
        }
    }
    model_tor_ad = gssm.create(config=cfg_tor_ad).to(device)
    t_model_tor_ad = time_cuda_op(lambda: model_tor_ad(token_ids))
    print(f"End-to-End Model (Torus, Adap): {t_model_tor_ad:.3f} ms")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
