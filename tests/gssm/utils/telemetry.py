import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

class TelemetryAnalyzer:
    """
    MANIFOLD Telemetry System
    Records simulation metrics and generates visualizations 
    for the unified testing suites in V5.
    """
    def __init__(self, output_dir: str = "tests/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.metrics_file = self.output_dir / f"metrics_{self.run_id}.json"
        self._data = {}
        
    def save_metric(self, name: str, value: Any, prefix: str = ""):
        """Records a scalar, list, or string metric associated with a test."""
        key = f"{prefix}_{name}" if prefix else name
        self._data[key] = value
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self._data, f, indent=4)
            
    def plot_trajectories(self, x_seq: torch.Tensor, v_seq: torch.Tensor, 
                          title: str = "Manifold Flow Trajectory", 
                          prefix: str = "test", num_samples: int = 1) -> str:
        """
        Plots evolution of latent trajectories and velocity norms.
        x_seq: [B, Seq, Heads, HeadDim] or [B, Seq, Dim]
        v_seq: [B, Seq, Heads, HeadDim] or [B, Seq, Dim]
        """
        # Ensure numpy
        x_np = x_seq.detach().cpu().numpy()
        v_np = v_seq.detach().cpu().numpy()
        
        # Calculate Norms over the last (spatial) dimension
        v_norm = np.linalg.norm(v_np, axis=-1)
        x_norm = np.linalg.norm(x_np, axis=-1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Determine sequence length (dim=1)
        seq_len = x_np.shape[1]
        steps = np.arange(seq_len)
        
        for b in range(min(num_samples, x_np.shape[0])):
            if x_np.ndim == 4:
                # It has a Heads dimension: plot only Head 0 to avoid clutter
                ax1.plot(steps, v_norm[b, :, 0], marker='o', label=f"Batch {b} (Head 0)")
                ax2.plot(steps, x_norm[b, :, 0], marker='o', label=f"Batch {b} (Head 0)")
            elif x_np.ndim == 3:
                # [B, Seq, Dim]
                ax1.plot(steps, v_norm[b, :], marker='o', label=f"Batch {b}")
                ax2.plot(steps, x_norm[b, :], marker='o', label=f"Batch {b}")
                
        ax1.set_title("Norma de la Velocidad ||v||")
        ax1.set_xlabel("Secuencia / Evolución")
        ax1.set_ylabel("Magnitud")
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        ax2.set_title("Norma del Estado ||x||")
        ax2.set_xlabel("Secuencia / Evolución")
        ax2.set_ylabel("Magnitud")
        ax2.grid(alpha=0.3)
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        safe_title = title.replace(" ", "_").replace(":", "").replace("|", "and").lower()
        out_path = self.output_dir / f"{prefix}_{safe_title}_{self.run_id}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        
        return str(out_path)
