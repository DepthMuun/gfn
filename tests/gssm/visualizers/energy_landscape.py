"""
energy_landscape.py — Ultra-Premium GFN Force Field Visualizer
==============================================================
Visualizes the potential energy and curiosity/steering forces 
with high resolution, smooth contours, and 300 DPI.
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
    from gfn.realizations.gssm.config.schema import ManifoldConfig
except ImportError as e:
    print(f"Error importing GFN components: {e}")
    sys.exit(1)

def plot_force_field(
    model, 
    output_path: str,
    title: str = "GFN Energy Landscape & Force Field",
    resolution: int = 80 # High resolution for smooth contours
):
    """
    Probes the manifold over a high-res 2D grid and plots the acceleration field.
    """
    device = next(model.parameters()).device
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Roboto', 'Arial'],
        'text.color': 'white',
        'axes.labelcolor': '#8b949e',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
    })

    # Create a grid [-pi, pi]
    grid_range = np.linspace(-math.pi, math.pi, resolution)
    x_grid, y_grid = np.meshgrid(grid_range, grid_range)
    
    flat_x = x_grid.flatten()
    flat_y = y_grid.flatten()
    
    v_batch = torch.zeros(len(flat_x), model.config.heads, model.config.dim // model.config.heads).to(device)
    x_batch = torch.zeros_like(v_batch)
    x_batch[:, 0, 0] = torch.tensor(flat_x).to(device)
    x_batch[:, 0, 1] = torch.tensor(flat_y).to(device)
    
    layer = model.layers[0]
    engine = layer.integrator.physics_engine
    
    with torch.no_grad():
        # Simulate a complex field with multiple attractors
        ext_force = torch.zeros_like(x_batch)
        # Adding a central attractor and a rotatational component
        ext_force[:, 0, 0] = -torch.tensor(flat_x).to(device) * 0.4 + torch.tensor(flat_y).to(device) * 0.1
        ext_force[:, 0, 1] = -torch.tensor(flat_y).to(device) * 0.4 - torch.tensor(flat_x).to(device) * 0.1
        
        accel = engine.compute_acceleration(x_batch, v_batch, force=ext_force)
        
    u = accel[:, 0, 0].cpu().numpy().reshape(resolution, resolution)
    v = accel[:, 0, 1].cpu().numpy().reshape(resolution, resolution)
    
    # Energy = Log-magnitude for better dynamic range visualization
    energy = np.log1p(np.sqrt(u**2 + v**2))
    
    fig, ax = plt.subplots(figsize=(14, 12), facecolor='#0d1117', dpi=300)
    ax.set_facecolor('#0d1117')
    
    # 1. Plot Energy Heatmap with many levels for smoothness
    contour = ax.contourf(x_grid, y_grid, energy, 100, cmap='inferno', alpha=0.9)
    
    # 2. Add subtle contour lines
    ax.contour(x_grid, y_grid, energy, 15, colors='white', alpha=0.08, linewidths=0.5)
    
    # 3. Plot Quiver (arrows) - downsample for arrows so it's not too crowded
    skip = resolution // 25
    ax.quiver(x_grid[::skip, ::skip], y_grid[::skip, ::skip], 
              u[::skip, ::skip], v[::skip, ::skip], 
              color='white', alpha=0.4, width=0.0015, headwidth=4, headlength=5, pivot='mid')
    
    ax.set_title(title, color='white', fontsize=22, fontweight='bold', pad=30)
    ax.set_xlabel("Latent Coordinate x₁", color='#8b949e', fontsize=14)
    ax.set_ylabel("Latent Coordinate x₂", color='#8b949e', fontsize=14)
    
    # Professional Colorbar
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Log Geometric Force Magnitude (‖Φ‖)', color='#8b949e', fontsize=12, labelpad=15)
    cbar.ax.yaxis.set_tick_params(color='#8b949e', labelcolor='#8b949e')
    
    # Legend overlay for physical components
    ax.annotate("Stable Region (Potential Well)", (0, 0), xytext=(20, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='cyan', connectionstyle="arc3,rad=.2"),
                color='cyan', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='#0d1117', alpha=0.5, ec='none'))

    # Add boundary
    rect = plt.Rectangle((-math.pi, -math.pi), 2*math.pi, 2*math.pi, fill=False, color='#238636', linestyle='--', linewidth=2, alpha=0.3)
    ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Premium force field visualization saved to {output_path}")

def run_quicktest():
    """Generates an ultra-premium energy landscape plot."""
    print("🚀 Generating Ultra-Premium Energy Landscape Visualization...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ModelFactory.create(
        vocab_size=100,
        dim=64,
        depth=1,
        heads=4,
        topology_type='torus',
        physics={'singularities': {'enabled': True, 'strength': 8.0}}
    ).to(device)
    
    output_dir = PROJECT_ROOT / "tests" / "results" / "visualizers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_force_field(
        model, 
        str(output_dir / "energy_landscape.png"),
        title="Field of Reason: Toroidal Energy Manifold & Force Vectors"
    )

if __name__ == "__main__":
    run_quicktest()
