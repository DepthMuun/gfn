"""
landscape_3d.py — Ultra-Premium 3D Geometric Potential Landscape
================================================================
Visualizes the "valleys of logic" as a 3D surface plot 
with bottom-contour projection, high resolution, and 300 DPI.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import math

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from gfn.realizations.gssm.models.factory import ModelFactory
except ImportError as e:
    print(f"Error importing GFN components: {e}")
    sys.exit(1)

def plot_landscape_3d(
    model, 
    output_path: str,
    title: str = "GFN: 3D Geometric Potential Landscape",
    resolution: int = 120 # Ultra-High resolution
):
    """
    Renders the energy landscape as a premium 3D surface with projections.
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

    # 1. Generate High-Res Grid
    grid_range = np.linspace(-math.pi, math.pi, resolution)
    x_grid, y_grid = np.meshgrid(grid_range, grid_range)
    
    flat_x = x_grid.flatten()
    flat_y = y_grid.flatten()
    
    # 2. Probe Engine Physics
    v_batch = torch.zeros(len(flat_x), model.config.heads, model.config.dim // model.config.heads).to(device)
    x_batch = torch.zeros_like(v_batch)
    x_batch[:, 0, 0] = torch.tensor(flat_x).to(device)
    x_batch[:, 0, 1] = torch.tensor(flat_y).to(device)
    
    layer = model.layers[0]
    engine = layer.integrator.physics_engine
    
    with torch.no_grad():
        # Complex multi-attractor field
        ext_force = torch.zeros_like(x_batch)
        for ax_off, ay_off, scale in [(-1.5, -1.5, 0.8), (1.5, 1.5, 0.8), (-1.5, 1.5, 0.5), (1.5, -1.5, 0.5), (0, 0, 1.2)]:
             dist_sq = (flat_x - ax_off)**2 + (flat_y - ay_off)**2
             f_mag = scale / (dist_sq + 0.1)
             ext_force[:, 0, 0] += torch.tensor(-f_mag * (flat_x - ax_off)).to(device)
             ext_force[:, 0, 1] += torch.tensor(-f_mag * (flat_y - ay_off)).to(device)

        accel = engine.compute_acceleration(x_batch, v_batch, force=ext_force)
    
    # Z = Negative magnitude (Valleys of Logic)
    z_vals = torch.norm(accel, dim=-1).mean(dim=1).cpu().numpy()
    z_mesh = -np.log1p(z_vals.reshape(resolution, resolution)) # Log for dynamic range
    
    # 3. Render Suite
    fig = plt.figure(figsize=(18, 14), facecolor='#0d1117', dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d1117')
    
    # Main Surface with Professional Shading
    surf = ax.plot_surface(x_grid, y_grid, z_mesh, cmap='terrain', alpha=0.95, 
                          linewidth=0, antialiased=True, shade=True, lightsource=None)
    
    # Bottom Projection (Show the 'map' underneath)
    offset = z_mesh.min() - 0.5
    cset = ax.contourf(x_grid, y_grid, z_mesh, zdir='z', offset=offset, cmap='terrain', alpha=0.4)
    
    # 4. Global Styling
    ax.set_title(title, color='white', fontsize=28, fontweight='bold', pad=60)
    ax.set_xlabel("Latent Axiom φ", color='#8b949e', fontsize=14, labelpad=20)
    ax.set_ylabel("Latent Axiom ψ", color='#8b949e', fontsize=14, labelpad=20)
    ax.set_zlabel("Potential Wells (-Φ)", color='#8b949e', fontsize=14, labelpad=10)
    
    ax.tick_params(colors='#8b949e', labelsize=10)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False) 
    
    # View and Framing
    ax.view_init(elev=35, azim=-55)
    ax.dist = 8 # Zoom in slightly more
    
    # Professional Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.4, aspect=15, pad=0.1)
    cbar.set_label('Informational Depth', color='#8b949e', fontsize=12, labelpad=15)
    cbar.ax.yaxis.set_tick_params(color='#8b949e', labelcolor='#8b949e')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Ultra-Premium 3D Landscape saved to {output_path}")

def run():
    print("🚀 Generating Ultra-Premium 3D Landscape Visualization...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ModelFactory.create(
        vocab_size=100, dim=64, depth=2, heads=4, 
        physics={'singularities': {'enabled': True, 'strength': 10.0}}
    ).to(device)
    
    output_dir = PROJECT_ROOT / "tests" / "results" / "visualizers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_landscape_3d(
        model, 
        str(output_dir / "landscape_3D.png"),
        title="Ontological Relief: 3D Geometric Potential Manifold"
    )

if __name__ == "__main__":
    run()
