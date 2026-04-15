"""
trajectories.py — Ultra-Premium GFN Latent State Visualizer
==========================================================
Visualizes the physical evolution of position (x) and velocity (v) 
tensors with high fidelity, spline interpolation, and 300 DPI.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple
from scipy.interpolate import make_interp_spline
import math

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from gfn.realizations.gssm.models.factory import ModelFactory
    from gfn.realizations.gssm.config.schema import ManifoldConfig
    from gfn.realizations.gssm.utils.coords import torus_to_box
except ImportError as e:
    print(f"Error importing GFN components: {e}")
    sys.exit(1)

def smooth_trajectory(coords: np.ndarray, num_points: int = 200) -> np.ndarray:
    """Smoothes a trajectory using spline interpolation."""
    if len(coords) < 4:
        return coords
    t = np.linspace(0, 1, len(coords))
    t_new = np.linspace(0, 1, num_points)
    
    # Handle x and y separately
    spl_x = make_interp_spline(t, coords[:, 0], k=3)
    spl_y = make_interp_spline(t, coords[:, 1], k=3)
    
    return np.stack([spl_x(t_new), spl_y(t_new)], axis=1)

def plot_latent_trajectories(
    x_seq: torch.Tensor, 
    v_seq: torch.Tensor, 
    tokens: List[str],
    output_path: str,
    title: str = "GFN Latent Trajectory Evolution",
    manifold_type: str = "torus"
):
    """
    Plots the trajectories of x and v in 2D space with premium aesthetics.
    x_seq, v_seq: [Seq, Heads, HeadDim]
    """
    seq_len, n_heads, head_dim = x_seq.shape
    
    # Premium Style Settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Roboto', 'Arial'],
        'text.color': 'white',
        'axes.labelcolor': '#8b949e',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
    })

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor='#0d1117', dpi=300)
    plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.25)
    
    # Use Magma for a more professional scientific look
    colors = plt.cm.magma(np.linspace(0.3, 0.9, seq_len))
    
    for ax, data, label, sub_title in zip(
        axes, [x_seq, v_seq], ["x", "v"], ["Position (Semantic State)", "Velocity (Momentum/Internal Logic)"]
    ):
        ax.set_facecolor('#0d1117')
        
        # Use first two dimensions
        coords = data[:, 0, :2].detach().cpu().numpy() # [Seq, 2]
        
        # 1. Smooth the path for a "flow" look
        path_coords = smooth_trajectory(coords)
        
        # 2. Draw the continuous flow line with gradient-like segments
        for i in range(len(path_coords) - 1):
            # Calculate color interpolation for the smooth line
            color_idx = int((i / len(path_coords)) * seq_len)
            c = colors[min(color_idx, seq_len-1)]
            ax.plot(path_coords[i:i+2, 0], path_coords[i:i+2, 1], color=c, alpha=0.9, linewidth=2.5, zorder=1)
            # Add a subtle glow
            ax.plot(path_coords[i:i+2, 0], path_coords[i:i+2, 1], color=c, alpha=0.1, linewidth=6, zorder=0)
        
        # 3. Plot markers for actual tokens
        for i in range(seq_len):
            ax.scatter(coords[i, 0], coords[i, 1], color=colors[i], s=120, edgecolors='white', linewidth=1.5, zorder=3)
            
            # Use a slightly offset box for labels to ensure readability
            ax.annotate(
                tokens[i], 
                (coords[i, 0], coords[i, 1]),
                xytext=(8, 8), textcoords='offset points',
                color='white', fontsize=11, fontweight='bold', alpha=1.0,
                bbox=dict(boxstyle='round,pad=0.3', fc='#161b22', alpha=0.6, ec='#30363d', lw=1),
                zorder=4
            )
            
        ax.set_title(sub_title, color='white', fontsize=16, fontweight='bold', pad=15)
        ax.tick_params(labelsize=10)
        ax.grid(True, color='#30363d', alpha=0.2, linestyle=':')
        
        for spine in ax.spines.values():
            spine.set_color('#30363d')
            
        if manifold_type == "torus":
            padding = 1.0
            ax.set_xlim(-math.pi - padding, math.pi + padding)
            ax.set_ylim(-math.pi - padding, math.pi + padding)
            # Draw boundary box for torus [-pi, pi]
            rect = plt.Rectangle((-math.pi, -math.pi), 2*math.pi, 2*math.pi, fill=False, color='#238636', linestyle='--', linewidth=2, alpha=0.4)
            ax.add_patch(rect)
            ax.annotate("Manifold Boundary", (-math.pi, math.pi + 0.2), color='#238636', alpha=0.6, fontsize=10)
    
    fig.suptitle(title, color='white', fontsize=22, fontweight='bold', y=0.95)
    
    # Add a professional colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.magma, norm=plt.Normalize(vmin=0, vmax=seq_len-1))
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.025])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Sequence Progress (Informational Flow)', color='#8b949e', fontsize=12, labelpad=10)
    cbar.ax.xaxis.set_tick_params(color='#8b949e', labelcolor='#8b949e')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Premium visualization saved to {output_path}")

def run_quicktest():
    """Runs an ultra-premium trajectory visualization."""
    print("🚀 Generating Ultra-Premium Trajectory Visualization...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup Model (Standard config from Benchmarks)
    model = ModelFactory.create(
        vocab_size=1000,
        dim=128,
        depth=4,
        heads=4,
        topology_type='torus',
        integrator='yoshida'
    ).to(device)
    
    # 2. Prepare Input
    # Symbolic logic sequence for compelling visualization
    tokens = ["Input", "Entropy", "Processing", "Filtering", "Coherence", "Logic", "Synthesis", "Output"]
    input_ids = torch.randint(0, 1000, (1, len(tokens))).to(device)
    
    # 3. Run Forward pass
    with torch.no_grad():
        logits, (x_final, v_final), state_info = model(input_ids=input_ids)
        
    # 4. Extract sequences [Seq, Heads, HeadDim]
    x_seq = state_info['x_seq'][0]
    v_seq = state_info['v_seq'][0]
    
    # 5. Plot
    output_dir = PROJECT_ROOT / "tests" / "results" / "visualizers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_latent_trajectories(
        x_seq, v_seq, tokens, 
        str(output_dir / "state_trajectories.png"),
        title="Physical Emergence: Latent Flow in GFN Manifold",
        manifold_type="torus"
    )

if __name__ == "__main__":
    run_quicktest()
