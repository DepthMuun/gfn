"""
holographic_interference.py — Ultra-Premium Force Interaction Waves
==================================================================
Visualizes holographic wave interference between manifold heads 
with high resolution, professional color scales, and 300 DPI.
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

def plot_interference(output_path: str, title: str = "GFN: Holographic Force Interference"):
    """
    Simulates and visualizes wave-like interference between multi-head forces.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Roboto', 'Arial'],
        'text.color': 'white',
    })

    # High-resolution grid
    res = 500
    x = np.linspace(-6, 6, res)
    y = np.linspace(-6, 6, res)
    X, Y = np.meshgrid(x, y)
    
    # Source A (Multi-head Component 1)
    s1_x, s1_y = -2.5, 0.5
    dist1 = np.sqrt((X - s1_x)**2 + (Y - s1_y)**2)
    # Complex wave: sine with decay and frequency modulation
    wave1 = np.sin(2 * np.pi * dist1 * 0.7) * np.exp(-dist1 * 0.15)
    
    # Source B (Multi-head Component 2)
    s2_x, s2_y = 2.5, -0.5
    dist2 = np.sqrt((X - s2_x)**2 + (Y - s2_y)**2)
    wave2 = np.sin(2 * np.pi * dist2 * 0.7) * np.exp(-dist2 * 0.15)
    
    # Interaction Field
    field = wave1 + wave2
    
    fig, ax = plt.subplots(figsize=(14, 14), facecolor='#0d1117', dpi=300)
    ax.set_facecolor('#0d1117')
    
    # Plot pattern with Twilight (Cyclic scientific colormap)
    im = ax.imshow(field, extent=[-6, 6, -6, 6], cmap='twilight', origin='lower', interpolation='bicubic')
    
    # Professional Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Holographic Resonance Energy (H)', color='#8b949e', fontsize=12, labelpad=15)
    cbar.ax.yaxis.set_tick_params(color='#8b949e', labelcolor='#8b949e')
    
    # Overlays: Component Sources
    ax.scatter([s1_x, s2_x], [s1_y, s2_y], color='white', s=300, marker='*', edgecolors='cyan', 
               linewidth=1.5, label='Manifold Force Heads', zorder=10)
    
    ax.set_title(title, color='white', fontsize=26, fontweight='bold', pad=40)
    ax.set_axis_off()
    
    # Scientific Annotations
    ax.annotate("Constructive Interference\n(Coherent Information Synthesis)", xy=(0, 0), xytext=(0, 100), 
                textcoords='offset points', color='cyan', fontsize=13, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5, alpha=0.7, connectionstyle="arc3,rad=.2"))
    
    # Technical Footnote
    ax.text(0.5, -0.02, "Visualization of multi-head force interactions on a 2D Manifold Slice.\nPatterns represent stable semantic oscillations.", 
            transform=ax.transAxes, color='gray', fontsize=11, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Premium holographic interference saved to {output_path}")

if __name__ == "__main__":
    print("🚀 Generating Ultra-Premium Holographic Interference Visualization...")
    output_dir = PROJECT_ROOT / "tests" / "results" / "visualizers"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_interference(str(output_dir / "holographic_interference.png"))
