"""
semantic_clusters.py — Ultra-Premium GFN Embedding Visualizer
=============================================================
Visualizes how token embeddings (entities) are organized 
geometrically with professional categorization and 300 DPI.
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

def plot_semantic_clusters(
    model, 
    token_labels: List[str],
    categories: List[str],
    output_path: str,
    title: str = "GFN Semantic Geometry: Token Organization"
):
    """
    Plots the static embeddings of a set of tokens with categorical grouping.
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

    # 1. Get embeddings for tokens
    input_ids = torch.arange(len(token_labels)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        forces = model.embedding(input_ids)
        
    coords = forces[0, :, :2].cpu().numpy() # [T, 2]
    
    fig, ax = plt.subplots(figsize=(14, 14), facecolor='#0d1117', dpi=300)
    ax.set_facecolor('#0d1117')
    
    # Define color palette for categories
    unique_cats = sorted(list(set(categories)))
    cat_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
    color_map = {cat: color for cat, color in zip(unique_cats, cat_colors)}
    
    # 2. Plot categorical clusters
    for i, (label, cat) in enumerate(zip(token_labels, categories)):
        color = color_map[cat]
        ax.scatter(coords[i, 0], coords[i, 1], color=color, s=200, edgecolors='white', linewidth=1.5, zorder=3, label=cat if cat not in ax.get_legend_handles_labels()[1] else "")
        
        # Professional Annotation
        ax.annotate(
            label, 
            (coords[i, 0], coords[i, 1]),
            xytext=(10, 10), textcoords='offset points',
            color='white', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#161b22', alpha=0.9, ec=color, lw=1.5),
            zorder=4
        )
        
    ax.set_title(title, color='white', fontsize=24, fontweight='bold', pad=35)
    ax.set_xlabel("Latent Feature Axis 1", color='#8b949e', fontsize=14)
    ax.set_ylabel("Latent Feature Axis 2", color='#8b949e', fontsize=14)
    
    ax.tick_params(colors='#8b949e', labelsize=10)
    ax.grid(alpha=0.1, color='white', linestyle=':')
    
    # 3. Aesthetics: Influence Spheres
    max_r = np.max(np.abs(coords))
    for r_scale in [0.5, 0.75, 1.0]:
        circle = plt.Circle((0, 0), max_r * r_scale * 1.2, color='#238636', fill=False, linestyle='--', alpha=0.1, linewidth=1)
        ax.add_artist(circle)
    
    # Styled Legend
    legend = ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=14, loc='upper right', framealpha=0.8)
    legend.set_title("Entity Categories", prop={'size': 14, 'weight': 'bold'})
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Premium Cluster visualization saved to {output_path}")

def run_quicktest():
    """Generates an ultra-premium semantic cluster plot."""
    print("🚀 Generating Ultra-Premium Semantic Cluster Visualization...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ModelFactory.create(
        vocab_size=100,
        dim=128,
        depth=2,
        heads=4,
        physics={'embedding': {'mode': 'siren', 'coord_dim': 32}}
    ).to(device)
    
    # Tokens with explicit categories
    data = [
        ("0", "Numeric"), ("1", "Numeric"), ("2", "Numeric"), ("3", "Numeric"), ("4", "Numeric"),
        ("+", "Operator"), ("-", "Operator"), ("*", "Operator"), ("/", "Operator"), ("=", "Operator"),
        ("cat", "Entity"), ("dog", "Entity"), ("bird", "Entity"), ("fish", "Entity"),
        ("red", "Property"), ("blue", "Property"), ("green", "Property"), ("yellow", "Property"),
        ("run", "Action"), ("jump", "Action"), ("sleep", "Action"), ("eat", "Action")
    ]
    tokens = [d[0] for d in data]
    categories = [d[1] for d in data]
    
    output_dir = PROJECT_ROOT / "tests" / "results" / "visualizers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_semantic_clusters(
        model, tokens, categories,
        str(output_dir / "semantic_clusters.png"),
        title="Ontological Grounding: Geometric Clustering of Concepts"
    )

if __name__ == "__main__":
    run_quicktest()
