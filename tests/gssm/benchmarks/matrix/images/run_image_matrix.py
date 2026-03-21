import sys
import os
from pathlib import Path

# Add current dir to path to import local matrix framework
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

import torch
from tests.benchmarks.matrix.framework import MatrixRunner
from tests.benchmarks.matrix.images.task_image import train_image, eval_image

def run_geometry_matrix():
    print("\n>>> Running IMAGE MATRIX: Geometry (Scope & Rank)")
    runner = MatrixRunner("Image_Geometry")
    
    axes = {
        "physics.topology.geometry_scope": ["local", "global"],
        "physics.topology.riemannian_type": ["low_rank", "reactive"],
        "rank": [32, 64]
    }
    
    base = {
        "dim": 128,
        "depth": 2,
        "heads": 8,
        "vocab_size": 2, # for detector
        "holographic": True,
        "initial_spread": 0.01
    }
    
    # partial to inject steps
    def train_wrap(model, overrides=None):
        return train_image(model, steps=500, img_size=32, overrides=overrides)
        
    def eval_wrap(model, adapter=None):
        return eval_image(model, img_size=32, adapter=adapter)

    runner.run_axes(axes, train_wrap, eval_wrap, base_overrides=base)

def run_topology_matrix():
    print("\n>>> Running IMAGE MATRIX: Topology & Stability")
    runner = MatrixRunner("Image_Topology")
    
    axes = {
        "physics.topology.type": ["torus", "euclidean"],
        "physics.stability.enable_trace_normalization": [True, False],
        "physics.dynamics.type": ["direct", "residual", "mix"]
    }
    
    base = {
        "dim": 128,
        "depth": 2,
        "heads": 4,
        "vocab_size": 2,
        "holographic": True
    }
    
    def train_wrap(model, overrides=None):
        return train_image(model, steps=500, img_size=32, overrides=overrides)
        
    def eval_wrap(model):
        return eval_image(model, img_size=32)

    runner.run_axes(axes, train_wrap, eval_wrap, base_overrides=base)

def run_readout_matrix():
    print("\n>>> Running IMAGE MATRIX: Readout & Plugins")
    runner = MatrixRunner("Image_Readout")
    
    axes = {
        "physics.readout.type": ["standard", "implicit", "identity"],
        "holographic": [True, False],
        "physics.mixture.coupler_mode": ["mean_field", "low_rank"]
    }
    
    base = {
        "dim": 128,
        "depth": 2,
        "heads": 4,
        "vocab_size": 2,
        "initial_spread": 0.01
    }
    
    def train_wrap(model, overrides=None):
        return train_image(model, steps=500, img_size=32, overrides=overrides)
        
    def eval_wrap(model):
        return eval_image(model, img_size=32)

    runner.run_axes(axes, train_wrap, eval_wrap, base_overrides=base)

def run_dynamics_matrix():
    print("\n>>> Running IMAGE MATRIX: Dynamics Modes")
    runner = MatrixRunner("Image_Dynamics")
    axes = {
        "physics.dynamics.type": ["direct", "residual", "mix", "gated"],
        "physics.active_inference.enabled": [True, False],
        "physics.stability.integrator_type": ["leapfrog", "rk4"]
    }
    base = {"dim": 128, "depth": 2, "heads": 4, "vocab_size": 2}
    def train_wrap(model, overrides=None): 
        return train_image(model, steps=500, loss_mode='mse', overrides=overrides)
    def eval_wrap(model, adapter=None): return eval_image(model, img_size=32, adapter=adapter)
    runner.run_axes(axes, train_wrap, eval_wrap, base_overrides=base)

def run_embeds_matrix():
    print("\n>>> Running IMAGE MATRIX: Embedding Modes")
    runner = MatrixRunner("Image_Embeds")
    axes = {
        "physics.embedding.type": ["standard", "functional"],
        "physics.embedding.mode": ["linear", "binary", "sinusoidal", "siren"],
        "holographic": [True, False]
    }
    base = {"dim": 128, "depth": 2, "heads": 4, "vocab_size": 2}
    def train_wrap(model, overrides=None): 
        return train_image(model, steps=500, loss_mode='mse', overrides=overrides)
    def eval_wrap(model, adapter=None): return eval_image(model, img_size=32, adapter=adapter)
    runner.run_axes(axes, train_wrap, eval_wrap, base_overrides=base)

def run_stability_matrix():
    print("\n>>> Running IMAGE MATRIX: Stability & Integrators")
    runner = MatrixRunner("Image_Stability")
    axes = {
        "physics.stability.integrator_type": ["leapfrog", "yoshida", "rk4", "verlet", "heun"],
        "physics.stability.enable_trace_normalization": [True, False],
        "physics.stability.adaptive": [True, False]
    }
    base = {"dim": 128, "depth": 2, "heads": 4, "vocab_size": 2}
    def train_wrap(model, overrides=None): 
        return train_image(model, steps=500, loss_mode='mse', overrides=overrides)
    def eval_wrap(model, adapter=None): return eval_image(model, img_size=32, adapter=adapter)
    runner.run_axes(axes, train_wrap, eval_wrap, base_overrides=base)

def run_optimization_matrix():
    print("\n>>> Running IMAGE MATRIX: Optimization & Losses")
    runner = MatrixRunner("Image_Optimization")
    axes = {
        "loss_mode": ["mse", "circular", "riemannian"], # passed to train_wrap
        "lambda_geo": [0.0, 0.001, 0.01],              # passed to train_wrap
        "physics.stability.integrator_type": ["leapfrog", "rk4"]
    }
    base = {"dim": 128, "depth": 2, "heads": 4, "vocab_size": 2}
    def train_wrap(model, overrides=None): 
        return train_image(model, steps=500, overrides=overrides)
        
    def eval_wrap(model, adapter=None): return eval_image(model, img_size=32, adapter=adapter)
    runner.run_axes(axes, train_wrap, eval_wrap, base_overrides=base)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, choices=["geometry", "topology", "readout", "dynamics", "embeds", "stability", "optimization", "all"], default="all")
    args = parser.parse_args()
    
    test_map = {
        "geometry": run_geometry_matrix,
        "topology": run_topology_matrix,
        "readout": run_readout_matrix,
        "dynamics": run_dynamics_matrix,
        "embeds": run_embeds_matrix,
        "stability": run_stability_matrix,
        "optimization": run_optimization_matrix
    }
    
    if args.test == "all":
        for fn in test_map.values(): fn()
    else:
        test_map[args.test]()
