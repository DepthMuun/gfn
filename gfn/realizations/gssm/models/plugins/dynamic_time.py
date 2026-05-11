"""
gfn/realizations/gssm/models/plugins/dynamic_time.py
Dynamic time gating plugin for ManifoldLayer.
Extracted from manifold_layer.py - replaces the dynamic time logic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Tuple, List

from . import LayerPlugin, register_layer_plugin
from ...physics.gating import RiemannianGating


@register_layer_plugin('dynamic_time')
class DynamicTimePlugin(LayerPlugin):
    """
    Plugin for per-head dynamic time-step gating.
    
    Replaces the dynamic time logic in ManifoldLayer that:
    1. Creates trainable dt_params per head
    2. Applies softplus scaling with configurable min/max
    3. Supports 'thermo' or 'standard' gating modes
    """
    
    def __init__(self, layer: nn.Module, config: Any):
        super().__init__(layer, config)
        
        self.gating_type = getattr(config, 'dynamic_time_type', 'standard')
        self.use_thermo_gating = self.gating_type == 'thermo'
        
        # Will be initialized in setup()
        self.dt_params: Optional[nn.Parameter] = None
        self.gatings: Optional[nn.ModuleList] = None
        self.base_dt: float = 0.0
        self.dt_min: float = 1e-4
        self.dt_max: float = 0.5
        
    def setup(self) -> None:
        """Initialize parameters per head."""
        layer = self.layer
        
        # Get stability config
        stability_cfg = getattr(layer.config, 'stability', None)
        if stability_cfg:
            self.base_dt = getattr(stability_cfg, 'base_dt', 0.1)
            self.dt_min = getattr(stability_cfg, 'dt_min', 1e-4)
            self.dt_max = getattr(stability_cfg, 'dt_max', 0.5)
        
        # Get topology and dimension info
        heads = getattr(layer, 'heads', 1)
        head_dim = getattr(layer, 'head_dim', 64)
        topology = getattr(layer, 'topology', None)
        
        # Create per-head trainable dt parameters with softplus scaling
        scale_vals: List[torch.Tensor] = []
        for i in range(heads):
            target_dt = self.base_dt / 0.9
            val_init = torch.log(torch.exp(torch.tensor(target_dt)) - 1.0)
            dt_increment = 0.05
            scale_vals.append(val_init + i * dt_increment)
        
        self.dt_params = nn.Parameter(torch.stack(scale_vals))
        
        # Create gating networks per head
        if self.use_thermo_gating:
            # Thermodynamic gating takes both x and v
            self.gatings = nn.ModuleList([
                RiemannianGating(head_dim, topology=topology, gating_type='thermo')
                for _ in range(heads)
            ])
        else:
            # Standard gating takes only x
            self.gatings = nn.ModuleList([
                RiemannianGating(head_dim, topology=topology)
                for _ in range(heads)
            ])
    
    def pre_integrate(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        dt: torch.Tensor,
        force: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute dynamic dt per head before integrator step.
        
        Args:
            x: [B_eff, H, D] current positions
            v: [B_eff, H, D] current velocities
            dt: original dt (ignored, we compute our own)
            force: optional external force
            
        Returns:
            (x, v, dt_eff) where dt_eff is [1, H, 1]
        """
        B_eff, H, D = x.shape
        
        # Base dt with softplus scaling: [1, H, 1]
        dt_base = F.softplus(self.dt_params).view(1, H, 1)
        dt_base = torch.clamp(dt_base, self.dt_min, self.dt_max)
        
        if self.use_thermo_gating:
            # Thermodynamic gating: use both x and v
            gates_list = [
                self.gatings[i](x[:, i], v[:, i])
                for i in range(H)
            ]
        else:
            # Standard gating: use only x
            gates_list = [
                self.gatings[i](x[:, i])
                for i in range(H)
            ]
        
        gates = torch.stack(gates_list, dim=1)  # [B_eff, H, 1]
        dt_eff = dt_base * gates  # [B_eff, H, 1] broadcast
        
        return x, v, dt_eff
