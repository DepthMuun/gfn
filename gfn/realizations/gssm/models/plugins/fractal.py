"""
gfn/realizations/gssm/models/plugins/fractal.py
Fractal sub-manifold tunneling plugin for ManifoldLayer.
Extracted from manifold_layer.py - replaces the fractal step logic.
"""
import torch
import torch.nn as nn
from typing import Optional, Any, Tuple

from . import LayerPlugin, register_layer_plugin


@register_layer_plugin('fractal')
class FractalPlugin(LayerPlugin):
    """
    Plugin for fractal sub-manifold tunneling.
    
    When curvature (velocity norm) exceeds threshold, blends state
    with micro-manifold evolution for finer-scale dynamics.
    """
    
    def __init__(self, layer: nn.Module, config: Any):
        super().__init__(layer, config)
        
        self.threshold = getattr(config, 'threshold', 1.0)
        self.alpha = getattr(config, 'alpha', 0.1)
        self.slope = getattr(config, 'slope', 1.0)
        
        # Micro-manifold will be created in setup() if needed
        self.micro_manifold: Optional[nn.Module] = None
    
    def setup(self) -> None:
        """Create micro-manifold if fractal is enabled."""
        # Check if we should create a micro-manifold
        # This could be a smaller version of the main manifold
        # For now, we just store the parameters - actual micro_manifold
        # would be set externally or created based on config
        pass
    
    def finalize(
        self,
        x: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply fractal tunneling step.
        
        Estimates curvature from velocity norm and blends with
        micro-manifold state if threshold exceeded.
        """
        if not self.enabled or self.micro_manifold is None:
            return x, v
        
        # Estimate curvature: average norm of velocity per head
        # x, v: [B_eff, H, D]
        curvature_est = v.norm(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
        
        # Compute tunnel gate: sigmoid of (curvature - threshold)
        tunnel_gate = torch.sigmoid((curvature_est - self.threshold) * self.slope)  # [B, 1, 1]
        
        # Run micro-manifold evolution
        # Note: force is not passed here as it's stored in layer state
        x_f, v_f = self.micro_manifold(x, v)
        
        # Blend states with alpha scaling
        x_out = x + tunnel_gate * (x_f - x) * self.alpha
        v_out = v + tunnel_gate * (v_f - v) * self.alpha
        
        return x_out, v_out
