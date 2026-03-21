"""
logic_as_physics.py — Ultra-Premium Reasoning as Collision
==========================================================
Visualizes logic as a physical interaction with glow effects,
professional typography, and 300 DPI.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def plot_logic_collision(output_path: str):
    """
    Simulates a conceptual 2D collision in the manifold with premium aesthetics.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Display', 'Roboto', 'Arial'],
        'text.color': 'white',
    })

    fig, ax = plt.subplots(figsize=(14, 12), facecolor='#0d1117', dpi=300)
    ax.set_facecolor('#0d1117')
    
    # Coordinates for the "Arithmetic Collision"
    p1_start = np.array([-4.0, 2.0])
    p1_end   = np.array([0.0, 0.0])
    
    p2_start = np.array([-4.0, -2.0])
    p2_end   = np.array([0.0, 0.0])
    
    p_res_start = np.array([0.0, 0.0])
    p_res_end   = np.array([5.0, 0.0])
    
    t = np.linspace(0, 1, 30)
    traj1 = p1_start + (p1_end - p1_start) * t[:, None]
    traj2 = p2_start + (p2_end - p2_start) * t[:, None]
    traj_res = p_res_start + (p_res_end - p_res_start) * t[:, None]
    
    # 1. Plot Trajectories with GLOW
    for traj, color in [(traj1, '#58a6ff'), (traj2, '#d2a8ff'), (traj_res, '#3fb950')]:
        # Background Glow
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=10, alpha=0.05, zorder=1)
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, alpha=0.6, linestyle='--', zorder=2)
    
    # 2. Plot Entities
    # Entity A (The '2')
    ax.scatter(p1_start[0], p1_start[1], color='#58a6ff', s=400, edgecolors='white', linewidth=2, zorder=10)
    ax.annotate("Entity: '2'\n(Mass/Property Vector)", p1_start, xytext=(-20, 30), textcoords='offset points', 
                color='white', fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.4', fc='#161b22', alpha=0.8, ec='#58a6ff'))

    # Entity B (The '3')
    ax.scatter(p2_start[0], p2_start[1], color='#d2a8ff', s=400, edgecolors='white', linewidth=2, zorder=10)
    ax.annotate("Entity: '3'\n(Mass/Property Vector)", p2_start, xytext=(-20, -60), textcoords='offset points', 
                color='white', fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.4', fc='#161b22', alpha=0.8, ec='#d2a8ff'))

    # 3. Collision Point (The Fusion Event)
    # Layered glow for fusion
    for r, a in [(1.5, 0.05), (1.0, 0.1), (0.5, 0.2)]:
        ax.add_patch(plt.Circle((0, 0), r, color='#f1e05a', fill=True, alpha=a, zorder=1))
    
    ax.scatter(0, 0, color='#f1e05a', marker='*', s=800, edgecolors='white', linewidth=2, zorder=15)
    ax.annotate("Geometric Intersection\nConservation of Semantic Momentum", (0, 0), xytext=(0, 40), textcoords='offset points', 
                color='#f1e05a', ha='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', fc='#0d1117', alpha=0.8, ec='none'))

    # 4. Result Entity (The '5')
    ax.scatter(p_res_end[0], p_res_end[1], color='#3fb950', s=700, edgecolors='white', linewidth=3, zorder=10)
    ax.annotate("Determined Result: '5'\nInvariant Logical State", p_res_end, xytext=(20, 0), textcoords='offset points', 
                color='#3fb950', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='#161b22', alpha=0.9, ec='#3fb950', lw=2))

    ax.set_title("Logic as Physics: Entity Fusion in Latent Space", color='white', 
                 fontsize=26, fontweight='bold', pad=50)

    # Conceptual Grid
    ax.grid(True, color='#30363d', alpha=0.1, linestyle=':')
    ax.set_xlim(-6, 8)
    ax.set_ylim(-4, 4)
    ax.axis('off')
    
    # Subtitle footnote
    ax.text(0.5, -0.05, "In a geometric model, 2 + 3 = 5 is not a 'prediction',\n it is the only state that satisfies the conservation of semantic properties.", 
            transform=ax.transAxes, color='gray', fontsize=11, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Premium logic-as-physics visualization saved to {output_path}")

if __name__ == "__main__":
    print("🚀 Generating Ultra-Premium Logic-as-Physics Visualization...")
    output_dir = PROJECT_ROOT / "tests" / "results" / "visualizers"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_logic_collision(str(output_dir / "logic_as_physics.png"))
