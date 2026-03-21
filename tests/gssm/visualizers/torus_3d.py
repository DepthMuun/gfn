"""
torus_3d.py — Ultra-Premium 3D Torus Topology Projection
========================================================
Visualizes trajectories wrapping around a 3D donut (Torus) 
with enhanced lighting, 300 DPI, and professional aesthetics.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import math
from typing import Optional, List, Tuple

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

def plot_torus_3d(
    x_seq: torch.Tensor, 
    tokens: List[str],
    output_path: str,
    title: str = "GFN: Trajectory on 3D Torus Manifold"
):
    """
    Renders a 3D Torus and plots the trajectory on its surface with premium cosmetics.
    """
    seq_len = len(tokens)
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Roboto', 'Arial'],
        'text.color': 'white',
    })

    # Torus parameters
    R = 3.5 # Major radius
    r = 1.2 # Minor radius
    
    # 1. Generate High-Res Torus Mesh
    n_mesh = 100
    theta_mesh = np.linspace(0, 2*np.pi, n_mesh)
    phi_mesh = np.linspace(0, 2*np.pi, n_mesh)
    theta_mesh, phi_mesh = np.meshgrid(theta_mesh, phi_mesh)
    
    X_mesh = (R + r * np.cos(theta_mesh)) * np.cos(phi_mesh)
    Y_mesh = (R + r * np.cos(theta_mesh)) * np.sin(phi_mesh)
    Z_mesh = r * np.sin(theta_mesh)
    
    fig = plt.figure(figsize=(14, 14), facecolor='#0d1117', dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d1117')
    
    # Draw Torus Surface with sophisticated shading
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='winter', alpha=0.1, linewidth=0, antialiased=True, shade=True)
    ax.plot_wireframe(X_mesh, Y_mesh, Z_mesh, color='cyan', alpha=0.03, linewidth=0.3)
    
    # 2. Map Latent Trajectory to Torus
    # x_seq is in [-pi, pi]. Map to [0, 2pi]
    coords = (x_seq[:, 0, :2].detach().cpu().numpy() + np.pi) 
    
    phi_t = coords[:, 0]
    theta_t = coords[:, 1]
    
    x_t = (R + r * np.cos(theta_t)) * np.cos(phi_t)
    y_t = (R + r * np.cos(theta_t)) * np.sin(phi_t)
    z_t = r * np.sin(theta_t)
    
    # Plot Trajectory with GLOW
    # Color mapping for sequence evolution
    colors = plt.cm.spring(np.linspace(0.1, 1, seq_len))
    
    # Draw trajectory segments
    for i in range(seq_len - 1):
        # Core line
        ax.plot(x_t[i:i+2], y_t[i:i+2], z_t[i:i+2], color=colors[i], linewidth=5, alpha=0.9, zorder=20)
        # Glow layer
        ax.plot(x_t[i:i+2], y_t[i:i+2], z_t[i:i+2], color=colors[i], linewidth=12, alpha=0.15, zorder=19)
    
    # Draw entities (Points)
    for i in range(seq_len):
        ax.scatter(x_t[i], y_t[i], z_t[i], color=colors[i], s=200, edgecolors='white', linewidth=1.5, depthshade=False, zorder=25)
        # High-contrast annotation
        ax.text(x_t[i]*1.15, y_t[i]*1.15, z_t[i]*1.15, tokens[i], color='white', fontsize=14, fontweight='bold', 
                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='#161b22', alpha=0.5, ec='none'), zorder=30)

    ax.set_title(title, color='white', fontsize=26, fontweight='bold', pad=50)
    
    # Professional 3D Framing
    ax.set_axis_off()
    ax.view_init(elev=30, azim=55)
    
    # Lighting and Aspect
    ax.set_box_aspect([1, 1, 0.4]) # Flattened torus look
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Premium 3D Torus visualization saved to {output_path}")

def run():
    print("🚀 Generating Ultra-Premium 3D Torus Projection...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Heavy impulse for visual complexity
    model = ModelFactory.create(vocab_size=1000, dim=128, depth=2, heads=4, impulse_scale=10.0).to(device)
    
    tokens = ["Origin", "Force", "Momentum", "Topology", "Wrap", "Recursion", "Self", "State", "Sink"]
    input_ids = torch.randint(0, 1000, (1, len(tokens))).to(device)
    
    with torch.no_grad():
        _, _, state_info = model(input_ids=input_ids)
    
    x_seq = state_info['x_seq'][0]
    
    output_dir = PROJECT_ROOT / "tests" / "results" / "visualizers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_torus_3d(
        x_seq, tokens, 
        str(output_dir / "torus_3D.png"),
        title="Emergent Topology: n-Torus Geodesic Mapping"
    )

if __name__ == "__main__":
    run()
