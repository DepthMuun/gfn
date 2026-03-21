import torch
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def check_model_health(model: torch.nn.Module) -> Dict[str, Any]:
    """Checks for NaNs in parameters and gradients."""
    results: Dict[str, List[str]] = {"param_nans": [], "grad_nans": []}
    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            results["param_nans"].append(name)
        if p.grad is not None and torch.isnan(p.grad).any():
            results["grad_nans"].append(name)
    return results

def trace_forward(model: torch.nn.Module, input_sample: torch.Tensor):
    """Diagnostic trace to inspect shapes during forward pass."""
    logger.info("--- Forward Trace ---")
    logger.info(f"Input: {input_sample.shape}")
    
    hooks = []
    def hook_fn(module, input, output):
        name = module.__class__.__name__
        if isinstance(output, torch.Tensor):
            logger.info(f"{name}: {output.shape}")
        elif isinstance(output, tuple):
             logger.info(f"{name}: tuple of {len(output)}")
             
    for m in model.modules():
        hooks.append(m.register_forward_hook(hook_fn))
        
    try:
        model(input_sample)
    finally:
        for h in hooks:
            h.remove()
    logger.info("--- Trace End ---")
