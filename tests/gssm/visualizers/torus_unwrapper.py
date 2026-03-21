"""
torus_unwrapper.py — Ultra-Premium GFN Toroidal Flow Visualizer
==============================================================
Visualizes toroidal wrapping with explicit phase-jump handling,
professional grid mapping, and 300 DPI.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple
import math

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from gfn.realizations.gssm.models.factory import ModelFactory
    from gfn.realizations.gssm.utils.coords import wrap_angles
except ImportError as e:
    print(f"Error importing GFN components: {e}")
    sys.exit(1)

def plot_torus_wrap(
    x_seq: torch.Tensor, 
    tokens: List[str],
    output_path: str,
    title: str = "GFN Toroidal Boundary Wrapping"
):
    """
    Plots trajectories on a flat [-pi, pi] square with jump handling and premium aesthetics.
    x_seq: [Seq, Heads, HeadDim]
    """
    seq_len, n_heads, head_dim = x_seq.shape
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Roboto', 'Arial'],
        'text.color': 'white',
        'axes.labelcolor': '#8b949e',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
    })

    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#0d1117', dpi=300)
    ax.set_facecolor('#0d1117')
    
    # Use first two dimensions of Head 0
    coords = x_seq[:, 0, :2].detach().cpu().numpy() # [Seq, 2]
    
    # Professional Grid
    grid_res = 8
    grid_ticks = np.linspace(-math.pi, math.pi, grid_res)
    for tick in grid_ticks:
        ax.axhline(tick, color='#30363d', linestyle='-', alpha=0.15, linewidth=0.5)
        ax.axvline(tick, color='#30363d', linestyle='-', alpha=0.15, linewidth=0.5)

    # Boundary box [-pi, pi]
    rect = plt.Rectangle((-math.pi, -math.pi), 2*math.pi, 2*math.pi, fill=True, color='#238636', alpha=0.03, zorder=0)
    ax.add_patch(rect)
    ax.add_patch(plt.Rectangle((-math.pi, -math.pi), 2*math.pi, 2*math.pi, fill=False, color='#238636', alpha=0.3, linewidth=2, linestyle='--', zorder=5))
    
    # Labels for boundaries
    ax.text(-math.pi, math.pi + 0.1, "-π", color='gray', fontsize=12, ha='center')
    ax.text(math.pi, math.pi + 0.1, "π", color='gray', fontsize=12, ha='center')
    ax.text(-math.pi - 0.2, -math.pi, "-π", color='gray', fontsize=12, va='center')
    ax.text(-math.pi - 0.2, math.pi, "π", color='gray', fontsize=12, va='center')

    # Plot segments to handle wrapping
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, seq_len))
    
    for i in range(seq_len - 1):
        p1 = coords[i]
        p2 = coords[i+1]
        
        # Check for wrap-around (jump)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if np.abs(dx) > math.pi or np.abs(dy) > math.pi:
            # It's a jump! Show the continuity with dashed lines to the edge
            # This is a bit complex, let's simplify by drawing to the "projected" edge
            ax.plot([p1[0], p1[0]+dx*0.1], [p1[1], p1[1]+dy*0.1], color=colors[i], linestyle=':', alpha=0.4, linewidth=1.5)
            # Annotate jump sequence
            ax.annotate("Φ-Jump", (p1[0], p1[1]), xytext=(10, -10), textcoords='offset points', 
                        color='cyan', fontsize=9, fontstyle='italic', alpha=0.6)
        else:
            # Regular flow
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors[i], linewidth=3, alpha=0.9, zorder=2)
            # Subtle Glow
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors[i], linewidth=7, alpha=0.1, zorder=1)

        ax.scatter(p1[0], p1[1], color=colors[i], s=140, edgecolors='white', linewidth=1.5, zorder=4)
        ax.annotate(
            tokens[i], 
            (p1[0], p1[1]),
            xytext=(5, 5), textcoords='offset points',
            color='white', fontsize=11, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.2', fc='#161b22', alpha=0.7, ec='none'),
            zorder=6
        )

    # Final point with special marker
    ax.scatter(coords[-1, 0], coords[-1, 1], color=colors[-1], marker='h', s=300, edgecolors='white', linewidth=2, zorder=10)
    ax.annotate(
        tokens[-1], 
        (coords[-1, 0], coords[-1, 1]),
        xytext=(5, 5), textcoords='offset points',
        color='white', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='#238636', alpha=0.8, ec='white'),
        zorder=11
    )
    
    ax.set_xlim(-math.pi - 1.0, math.pi + 1.0)
    ax.set_ylim(-math.pi - 1.0, math.pi + 1.0)
    ax.set_title(title, color='white', fontsize=22, fontweight='bold', pad=30)
    ax.set_xlabel("Latent Dim φ", color='#8b949e', fontsize=14)
    ax.set_ylabel("Latent Dim ψ", color='#8b949e', fontsize=14)
    
    ax.set_axis_off() # Hidden axis for cleaner topology look
    
    # Legend description
    ax.text(0, -math.pi - 0.8, "The 2D Periodic Grid represents an 'unrolled' n-Torus manifold.\nStates jump from π to -π without loss of continuity in higher D.", 
            color='gray', fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Premium Torus visualization saved to {output_path}")

def run_quicktest():
    """Runs a torus wrapping visualization with high impulse."""
    print("🚀 Generating Ultra-Premium Torus Wrap Visualization...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ModelFactory.create(
        vocab_size=100,
        dim=64,
        depth=2,
        heads=4,
        topology_type='torus',
        impulse_scale=7.0 # High impulse to ensure wraps
    ).to(device)
    
    tokens = ["Loop", "A", "Loop", "B", "Circle", "Wrap", "Return", "Infinity"]
    input_ids = torch.randint(0, 100, (1, len(tokens))).to(device)
    
    with torch.no_grad():
        _, _, state_info = model(input_ids=input_ids)
        
    x_seq = state_info['x_seq'][0]
    
    output_dir = PROJECT_ROOT / "tests" / "results" / "visualizers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_torus_wrap(
        x_seq, tokens, 
        str(output_dir / "torus_wrapping.png"),
        title="Topology Mapping: Toroidal Boundary Unrolling"
    )

if __name__ == "__main__":
    run_quicktest()
