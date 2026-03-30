"""
gfn/realizations/gssm/models/plugins/mixing.py
Head mixing plugin for ManifoldLayer.
Extracted from manifold_layer.py - replaces the mixer logic.
"""
import torch
import torch.nn as nn
from typing import Optional, Any, Tuple

from . import LayerPlugin, register_layer_plugin


@register_layer_plugin('mixing')
class MixingPlugin(LayerPlugin):
    """
    Plugin for head mixing in ManifoldLayer.
    
    Replaces the mixer component and handles both:
    - FlowMixer (default)
    - GeodesicAttentionMixer
    
    Mixing happens after integration and before dynamics routing.
    """
    
    def __init__(self, layer: nn.Module, config: Any):
        super().__init__(layer, config)
        
        self.mixer_type = getattr(config, 'type', 'flow')
        self.mixer: Optional[nn.Module] = None
    
    def setup(self) -> None:
        """Initialize the mixer based on configuration."""
        layer = self.layer
        
        # Import here to avoid circular dependencies
        from ..components.mixer import FlowMixer, GeodesicAttentionMixer
        
        heads = getattr(layer, 'heads', 1)
        head_dim = getattr(layer, 'head_dim', 64)
        mixer_dim = heads * head_dim
        
        # Get mixer type from config
        if self.mixer_type == 'geodesic_attention':
            self.mixer = GeodesicAttentionMixer(mixer_dim)
        else:
            self.mixer = FlowMixer(mixer_dim)
    
    def post_integrate(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        x_prev: torch.Tensor,
        v_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixing after integration step.
        
        Args:
            x: [B_eff, H, D] positions after integration
            v: [B_eff, H, D] velocities after integration
            x_prev, v_prev: previous states (unused but kept for API)
            
        Returns:
            (x_mix, v_mix) mixed states
        """
        if self.mixer is None:
            return x, v
        
        # Flatten heads for mixing: [B_eff, H*D]
        B_eff = x.shape[0]
        x_flat = x.reshape(B_eff, -1)
        v_flat = v.reshape(B_eff, -1)
        
        # Apply mixer
        x_mix, v_mix = self.mixer(x_flat, v_flat)
        
        # Reshape back if mixer outputs flat
        if x_mix.dim() == 2:
            # Reshape back to [B_eff, H, D]
            heads = x.shape[1]
            head_dim = x.shape[2]
            x_mix = x_mix.view(B_eff, heads, head_dim)
            v_mix = v_mix.view(B_eff, heads, head_dim)
        
        return x_mix, v_mix
