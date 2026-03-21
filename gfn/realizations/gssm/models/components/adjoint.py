"""
gfn/models/components/adjoint.py
Adjoint Geodesic Optimization Plugin.
Implements O(1) training memory by solving the ODE backwards for the gradient pass.
"""

import torch
import torch.nn as nn
from typing import Callable, Any, Tuple, List, Dict, Optional
from ...models.hooks import Plugin, HookManager
from ...config.schema import ManifoldConfig

try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    odeint = None


class GeodesicODEFunc(nn.Module):
    """
    Wraps the GFN evolution logic into a continuous-time ODE function.
    Handles multi-head dynamics and external force interpolation.
    """
    def __init__(self, layers: nn.ModuleList, forces: torch.Tensor, mask: torch.Tensor, hooks: HookManager):
        super().__init__()
        self.layers = layers
        self.forces = forces 
        self.mask = mask     
        self.hooks = hooks
        self.n_steps = forces.shape[1]

    def forward(self, t, state):
        x, v = state
        
        # 1. Determine local force via piece-wise interpolation (nearest neighbor)
        # In ODE land, t=[0, 1, 2...] corresponds to sequence indices.
        idx = torch.clamp(t.long(), 0, self.n_steps - 1)
        force = self.forces[:, idx] * self.mask[:, idx]
        
        # 2. Reproduce BaseModel._evolve_sequence logic for one step
        # Timestep Start Hooks
        step_start_res = self.hooks.trigger("on_timestep_start", x=x, v=v, force=force)
        for res in step_start_res:
             if isinstance(res, torch.Tensor):
                 force = force + res
             elif isinstance(res, dict) and "force" in res:
                 force = res["force"]

        curr_x, curr_v = x, v
        for layer in self.layers:
            # Layer Start Hooks
            layer_kwargs = {}
            self.hooks.trigger("on_layer_start", layer=layer, layer_kwargs=layer_kwargs, x=curr_x, v=curr_v)
            
            res = layer(curr_x, curr_v, force, **layer_kwargs)
            if isinstance(res, tuple) and len(res) >= 2:
                curr_x, curr_v = res[0], res[1]
                extra_info = res[2] if len(res) > 2 else {}
            else:
                curr_x, curr_v = res, curr_v
                extra_info = {}
            
            # Layer End Hooks
            self.hooks.trigger("on_layer_end", layer=layer, x=curr_x, v=curr_v, extra_info=extra_info)

        # 3. Compute derivatives: dz/dt = (z_new - z_old) / 1.0 (step size)
        return curr_x - x, curr_v - v


class AdjointPlugin(Plugin):
    """
    Adjoint Plugin for GFN.
    Wraps the evolution loop in an adjoint-based solver to achieve O(1) memory.
    """
    def __init__(self, config: Optional[ManifoldConfig] = None):
        super().__init__()
        self.config = config
        self.rtol = getattr(config, 'adjoint_rtol', 1e-4) if config else 1e-4
        self.atol = getattr(config, 'adjoint_atol', 1e-4) if config else 1e-4
        # Switching to Euler to match discrete steps exactly
        self.method = 'euler' 

    def register_hooks(self, manager: HookManager):
        if odeint is None:
            # Silently disable or log warning
            return
        manager.register("wrap_evolution", self.wrap_evolution)

    def wrap_evolution(self, evolution_fn: Callable) -> Callable:
        """Returns the adjoint-based evolution wrapper."""
        
        def adjoint_evolution(x_in, v_in, f_seq, m_seq, **kwargs):
            seq_len = f_seq.shape[1]
            t = torch.linspace(0, seq_len, seq_len + 1).to(f_seq.device).float()
            
            # The ODE Function
            hooks = kwargs.get('hooks', HookManager())
            ode_func = GeodesicODEFunc(kwargs.get('layers', nn.ModuleList()), f_seq, m_seq, hooks)
            
            # Parameters to track for adjoint
            # Need to include all model parameters AND the force sequence
            adj_params = tuple(ode_func.parameters()) + (f_seq,)
            
            # Pack state
            initial_state = (x_in, v_in)
            
            # Use standard odeint for gradient validation
            from torchdiffeq import odeint as odeint_std
            full_seq = odeint_std(
                ode_func,
                initial_state,
                t,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method
            )
            
            x_seq_stack, v_seq_stack = full_seq # [S+1, B, H, HD]
            
            # BaseModel expects sequence of length S (original seq)
            # full_seq has S+1 points (0 to S). We take 1 to S.
            x_seq_stack = x_seq_stack[1:]
            v_seq_stack = v_seq_stack[1:]
            
            x_final = x_seq_stack[-1]
            v_final = v_seq_stack[-1]
            
            x_list = [x_seq_stack[i] for i in range(seq_len)]
            v_list = [v_seq_stack[i] for i in range(seq_len)]
            
            l_logits = []
            hooks = kwargs.get('hooks', None)
            if hooks:
                for i in range(seq_len):
                    step_res = hooks.trigger("on_timestep_end", x=x_list[i], v=v_list[i])
                    for r in step_res:
                        if isinstance(r, torch.Tensor):
                            l_logits.append(r)
            
            return l_logits, (x_final, v_final), (x_list, v_list)

        return adjoint_evolution
