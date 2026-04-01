"""
gfn/api.py — GFN V5
Simplified public interface and high-level orchestration.
Centralizes model creation, loading and evaluation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union

from .models.factory import ModelFactory
from .models.manifold import ManifoldModel
from .losses.factory import LossFactory
from .training.trainer import GFNTrainer
from .training.evaluation import ManifoldMetricEvaluator

# -- Main aliases
Model = ManifoldModel
Manifold = ManifoldModel
Trainer = GFNTrainer

def create(*args, **kwargs):
    """Factory for Manifold models (V5)."""
    return ModelFactory.create(*args, **kwargs)

def loss(config, **kwargs):
    """Factory for loss functions (V5)."""
    return LossFactory.create(config, **kwargs)

def save(model: nn.Module, path: str):
    """
    Saves the model and its configuration (HuggingFace Style).
    """
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(path)
    else:
        # Fallback for models that don't inherit from BaseModel
        torch.save({'state_dict': model.state_dict()}, path)

def load(path: str, device: Optional[str] = None):
    """
    Loads a saved model along with its configuration.
    Supports directories (HF Style) or legacy .pth/.bin files.
    """
    import os
    if os.path.isdir(path):
        return ModelFactory.from_pretrained(path)
    
    # Fallback for isolated legacy files
    checkpoint = torch.load(path, map_location=device or 'cpu', weights_only=True)
    config = checkpoint.get('config')
    if config is None:
        raise ValueError(f"Configuration not found in checkpoint {path}. Use HF directories for full loading.")
        
    model = create(config=config)
    
    # Robust state_dict extraction (handles different saving conventions)
    state_dict = checkpoint.get('state_dict') or checkpoint.get('model') or checkpoint
    
    # Filter state_dict against the model's actual parameters
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() if k in model_state}
    
    # Log filtered keys for debugging (optional)
    n_filtered = len(state_dict) - len(filtered_state)
    if n_filtered > 0:
        import logging
        logging.getLogger("gssm.api").info(f"Filtered {n_filtered} unexpected keys from state_dict (legacy or auxiliary data).")

    # Load with strict=False to handle potential missing non-essential parameters
    model.load_state_dict(filtered_state, strict=False)
    return model

def benchmark(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
              device: Optional[str] = None) -> Dict[str, float]:
    """
    Ejecuta una evaluación rápida de métricas geométricas y de tarea.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    evaluator = ManifoldMetricEvaluator(model)
    all_x, all_v, all_y = [], [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, (xf, vf), info = model(x)
            
            all_x.append(xf.detach().cpu())
            all_v.append(vf.detach().cpu())
            all_y.append(y.detach().cpu())
            
    if not all_x:
        return {}

    x_total = torch.cat(all_x, dim=0)
    v_total = torch.cat(all_v, dim=0)
    y_total = torch.cat(all_y, dim=0)
    
    return evaluator.full_report(x_total, v_total, y_total)
