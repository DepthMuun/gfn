"""
CUDA Fusion Manager
==================

Manages CUDA kernel fusion for efficient forward passes.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import warnings


class CUDAFusionManager:
    """Manages CUDA kernel fusion for Manifold model.
    
    This class handles the complex logic of checking fusion availability,
    preparing parameters, and executing fused CUDA kernels.
    """
    
    def __init__(self, model):
        """Initialize fusion manager.
        
        Args:
            model: Parent Manifold model instance
        """
        self.model = model
        self._cuda_available = None
    
    def can_fuse(self, collect_christ: bool = False) -> bool:
        """Check if CUDA fusion is available and applicable.
        
        Args:
            collect_christ: Whether Christoffel collection is enabled
            
        Returns:
            True if fusion can be used
        """
        # Check basic requirements
        if self.model.use_scan:
            return False
        if self.model.depth == 0:
            return False
        if collect_christ:
            return False
        if self.model.integrator_type not in ['heun', 'leapfrog']:
            return False
        
        # Check CUDA availability
        try:
            from ..cuda.ops import CUDA_AVAILABLE
            return CUDA_AVAILABLE
        except ImportError:
            return False
    
    def prepare_parameters(self) -> Optional[dict]:
        """Prepare and stack parameters for CUDA kernel.
        
        Returns:
            Dictionary of stacked parameters or None if preparation fails
        """
        try:
            # Determine topology
            topo_cfg = self.model.physics_config.get('topology', {})
            topology_type = topo_cfg.get('type', 'euclidean')
            is_torus = (topology_type == 'torus')
            
            # Stack per-head parameters
            U_list = []
            W_list = []
            W_forget_list = []
            W_input_list = []
            b_forget_list = []
            W_potential_list = []
            b_potential_list = []
            mix_x_list = []
            mix_v_list = []
            mix_x_bias_list = []
            mix_v_bias_list = []
            norm_x_weight_list = []
            norm_x_bias_list = []
            norm_v_weight_list = []
            norm_v_bias_list = []
            gate_W1_list = []
            gate_b1_list = []
            gate_W2_list = []
            gate_b2_list = []
            dt_scales_list = []
            
            for layer in self.model.layers:
                device = next(self.model.parameters()).device
                # Handle fractal wrapper
                target_layer = layer
                if hasattr(layer, 'macro_manifold'):
                    target_layer = layer.macro_manifold

                dt_scales_list.append(torch.nn.functional.softplus(target_layer.dt_params))

                if self.model.heads > 1 and hasattr(target_layer, 'out_proj_x'):
                    mix_x_list.append(target_layer.out_proj_x.weight)
                    mix_v_list.append(target_layer.out_proj_v.weight)
                    mix_x_bias_list.append(target_layer.out_proj_x.bias)
                    mix_v_bias_list.append(target_layer.out_proj_v.bias)
                    norm_x_weight_list.append(target_layer.mixed_norm_x.weight)
                    norm_x_bias = getattr(target_layer.mixed_norm_x, 'bias', None)
                    norm_x_bias_list.append(norm_x_bias if norm_x_bias is not None else torch.empty(0, device=device))
                    
                    norm_v_weight_list.append(target_layer.mixed_norm_v.weight)
                    norm_v_bias = getattr(target_layer.mixed_norm_v, 'bias', None)
                    norm_v_bias_list.append(norm_v_bias if norm_v_bias is not None else torch.empty(0, device=device))
                else:
                    mix_x_list.append(torch.empty(0, device=device))
                    mix_v_list.append(torch.empty(0, device=device))
                    mix_x_bias_list.append(torch.empty(0, device=device))
                    mix_v_bias_list.append(torch.empty(0, device=device))
                    norm_x_weight_list.append(torch.empty(0, device=device))
                    norm_x_bias_list.append(torch.empty(0, device=device))
                    norm_v_weight_list.append(torch.empty(0, device=device))
                    norm_v_bias_list.append(torch.empty(0, device=device))

                gate_W1_layer = []
                gate_b1_layer = []
                gate_W2_layer = []
                gate_b2_layer = []
                if hasattr(target_layer, 'gatings'):
                    for g in target_layer.gatings:
                        gate_W1_layer.append(g.curvature_net[0].weight)
                        gate_b1_layer.append(g.curvature_net[0].bias)
                        gate_W2_layer.append(g.curvature_net[2].weight)
                        gate_b2_layer.append(g.curvature_net[2].bias)
                gate_W1_list.append(torch.stack(gate_W1_layer) if gate_W1_layer else torch.empty(0, device=next(self.model.parameters()).device))
                gate_b1_list.append(torch.stack(gate_b1_layer) if gate_b1_layer else torch.empty(0, device=next(self.model.parameters()).device))
                gate_W2_list.append(torch.stack(gate_W2_layer) if gate_W2_layer else torch.empty(0, device=next(self.model.parameters()).device))
                gate_b2_list.append(torch.stack(gate_b2_layer) if gate_b2_layer else torch.empty(0, device=next(self.model.parameters()).device))
                
                for head_idx in range(self.model.heads):
                    head_geo = target_layer.christoffels[head_idx]
                    
                    # Non-torus uses U/W matrices
                    if not is_torus:
                        if not hasattr(head_geo, 'U') or not hasattr(head_geo, 'W'):
                            return None
                        U_list.append(head_geo.U)
                        W_list.append(head_geo.W)
                    else:
                        # AUDIT WARNING (Component 2): TOROIDAL GEOMETRY BYPASS
                        # =====================================================
                        # CRITICAL BUG: When topology='torus', fusion manager creates DUMMY
                        # zero tensors instead of using actual toroidal Christoffel symbols.
                        # 
                        # IMPACT:
                        # - All toroidal curvature information is LOST in CUDA fused mode
                        # - Model degrades to flat Euclidean manifold (zero Christoffels)
                        # - Math tasks that rely on toroidal structure fail to converge
                        #
                        # ROOT CAUSE:
                        # - CUDA kernel 'christoffel_impl.cuh' has toroidal branch (lines 55-71)
                        # - But fusion manager never routes to it, passes zeros instead
                        #
                        # TODO (Requires CUDA Compilation):
                        # 1. Create dedicated toroidal_christoffel_fused.cu kernel
                        # 2. Add toroidal routing logic here to call dedicated kernel
                        # 3. Pass R, r (major/minor radii) to kernel
                        # 4. Add Python bindings in cuda_kernels.cpp
                        # 5. Update consistency tests to verify toroidal CUDA vs Python
                        #
                        # WORKAROUND (Current):
                        # - Use use_scan=True to force Python loop (bypasses fusion)
                        # - Or set physics_config['topology']['type'] = 'euclidean'
                        #
                        # REFERENCES:
                        # - technical_analysis.md: Line 55-72 (Toroidal Geometry Bypass)
                        # - implementation_plan.md: Component 2 (lines 96-120)
                        # - gfn/cuda/src/geometry/christoffel_impl.cuh: Lines 55-71
                        
                        # Dummy placeholders for torus mode (WILL BE REPLACED WITH ACTUAL KERNEL)
                        device = next(self.model.parameters()).device
                        U_list.append(torch.zeros(self.model.dim // self.model.heads, 1, device=device))
                        W_list.append(torch.zeros(self.model.dim // self.model.heads, 1, device=device))
                    
                    # Clutch parameters
                    if hasattr(head_geo, 'forget_gate'):
                        W_forget_list.append(head_geo.forget_gate.weight)
                        bias = getattr(head_geo.forget_gate, 'bias', None)
                        b_forget_list.append(bias if bias is not None else torch.zeros(head_geo.forget_gate.out_features, device=device))
                        W_input_list.append(head_geo.input_gate.weight if hasattr(head_geo, 'input_gate') else torch.zeros(head_geo.forget_gate.out_features, head_geo.forget_gate.in_features, device=device))
                    else:
                        # Fallback for legacy christoffels
                        h_dim = target_layer.head_dim
                        device = next(self.model.parameters()).device
                        W_forget_list.append(torch.zeros(self.model.dim//self.model.heads, h_dim, device=device))
                        b_forget_list.append(torch.zeros(self.model.dim//self.model.heads, device=device))
                        W_input_list.append(torch.zeros(self.model.dim//self.model.heads, h_dim, device=device))
                    
                    # Singularity parameters
                    if hasattr(head_geo, 'V') and head_geo.V is not None:
                        W_potential_list.append(head_geo.V.weight)
                        b_bias = head_geo.V.bias
                        if b_bias is None:
                            device = next(self.model.parameters()).device
                            b_bias = torch.zeros(1, device=device)
                        b_potential_list.append(b_bias)
                    else:
                        # Potential gate uses 2*head_dim for torus
                        device = next(self.model.parameters()).device
                        p_dim = 2 * (self.model.dim // self.model.heads) if is_torus else (self.model.dim // self.model.heads)
                        W_potential_list.append(torch.zeros(1, p_dim, device=device))
                        b_potential_list.append(torch.zeros(1, device=device))
            
            # Stack all parameters
            U_stack = torch.stack(U_list)
            W_stack = torch.stack(W_list)
            W_f_stack = torch.stack(W_forget_list)
            W_i_stack = torch.stack(W_input_list)
            b_f_stack = torch.stack(b_forget_list)
            W_p_stack = torch.stack(W_potential_list)
            b_p_stack = torch.stack(b_potential_list)
            
            # Get mixing weights
            device = next(self.model.parameters()).device
            mix_x = torch.stack(mix_x_list) if mix_x_list else torch.empty(0, device=device)
            mix_v = torch.stack(mix_v_list) if mix_v_list else torch.empty(0, device=device)
            mix_x_bias = torch.stack(mix_x_bias_list) if mix_x_bias_list else torch.empty(0, device=device)
            mix_v_bias = torch.stack(mix_v_bias_list) if mix_v_bias_list else torch.empty(0, device=device)
            norm_x_weight = torch.stack(norm_x_weight_list) if norm_x_weight_list else torch.empty(0, device=device)
            norm_x_bias = torch.stack(norm_x_bias_list) if norm_x_bias_list else torch.empty(0, device=device)
            norm_v_weight = torch.stack(norm_v_weight_list) if norm_v_weight_list else torch.empty(0, device=device)
            norm_v_bias = torch.stack(norm_v_bias_list) if norm_v_bias_list else torch.empty(0, device=device)
            gate_W1 = torch.stack(gate_W1_list) if gate_W1_list else torch.empty(0, device=device)
            gate_b1 = torch.stack(gate_b1_list) if gate_b1_list else torch.empty(0, device=device)
            gate_W2 = torch.stack(gate_W2_list) if gate_W2_list else torch.empty(0, device=device)
            gate_b2 = torch.stack(gate_b2_list) if gate_b2_list else torch.empty(0, device=device)
            dt_scales_stack = torch.stack(dt_scales_list) if dt_scales_list else torch.empty(0, device=device)
            
            # Get base dt
            first_layer = self.model.layers[0]
            if hasattr(first_layer, 'macro_manifold'):
                first_layer = first_layer.macro_manifold
            base_dt = first_layer.base_dt
            
            # Get physics parameters
            act_inf = self.model.physics_config.get('active_inference', {})
            plasticity = act_inf.get('plasticity', 0.0) if act_inf.get('enabled', False) else 0.0
            
            sing_cfg = self.model.physics_config.get('singularities', {})
            sing_enabled = sing_cfg.get('enabled', False)
            sing_thresh = sing_cfg.get('threshold', 0.9) if sing_enabled else 1.0
            sing_strength = sing_cfg.get('strength', 1.0) if sing_enabled else 1.0
            
            # Torus radii
            major_R = topo_cfg.get('major_radius', 2.0)
            minor_r = topo_cfg.get('minor_radius', 1.0)
            
            # AUDIT FIX (Component 2): Toroidal kernel routing
            # ================================================
            # If topology is toroidal, we need to use dedicated toroidal kernel
            # instead of low-rank approximation. The toroidal kernel computes
            # metric-derived Christoffel symbols directly.
            #
            # When is_torus=True:
            #   - U/W are dummy zeros (from lines 146-179)
            #   - execute_fused_forward will detect topology_id=1
            #   - Route to launch_toroidal_leapfrog_fused() instead
            #
            # This routing happens in execute_fused_forward() below.
            
            return {
                'U_stack': U_stack,
                'W_stack': W_stack,
                'W_f_stack': W_f_stack,
                'W_i_stack': W_i_stack,
                'b_f_stack': b_f_stack,
                'W_p_stack': W_p_stack,
                'b_p_stack': b_p_stack,
                'mix_x': mix_x,
                'mix_v': mix_v,
                'mix_x_bias': mix_x_bias,
                'mix_v_bias': mix_v_bias,
                'norm_x_weight': norm_x_weight,
                'norm_x_bias': norm_x_bias,
                'norm_v_weight': norm_v_weight,
                'norm_v_bias': norm_v_bias,
                'gate_W1': gate_W1,
                'gate_b1': gate_b1,
                'gate_W2': gate_W2,
                'gate_b2': gate_b2,
                'dt_scales_stack': dt_scales_stack,
                'base_dt': base_dt,
                'plasticity': plasticity,
                'sing_thresh': sing_thresh,
                'sing_strength': sing_strength,
                'major_R': major_R,
                'minor_r': minor_r,
                'topology_id': 1 if is_torus else 0,
                'is_torus': is_torus  # AUDIT FIX: Explicit flag for routing
            }
            
        except Exception as e:
            warnings.warn(f"[GFN:WARN] Failed to prepare fusion parameters: {e}")
            return None
    
    def execute_fused_forward(self, x: torch.Tensor, v: torch.Tensor, 
                             forces: torch.Tensor, mask: torch.Tensor,
                             params: dict,
                             hysteresis_state: Optional[torch.Tensor] = None,
                             hyst_update_w: Optional[torch.Tensor] = None,
                             hyst_update_b: Optional[torch.Tensor] = None,
                             hyst_readout_w: Optional[torch.Tensor] = None,
                             hyst_readout_b: Optional[torch.Tensor] = None,
                             hyst_decay: float = 0.9,
                             hyst_enabled: bool = False) -> Optional[Tuple]:
        """Execute fused CUDA kernel with Hysteresis support.
        
        Args:
            x: Initial position [batch, dim]
            v: Initial velocity [batch, dim]
            forces: Force sequence [batch, seq_len, dim]
            mask: Attention mask [batch, seq_len, 1]
            params: Stacked parameters from prepare_parameters()
            hysteresis_state: Hysteresis memory state
            hyst_update_w/b: Hysteresis update weights/bias
            hyst_readout_w/b: Hysteresis readout weights/bias
            hyst_decay: State decay factor
            hyst_enabled: Flag to enable hysteresis
            
        Returns:
            Tuple of (x_final, v_final, x_seq, reg_loss) or None if failed
        """
        try:
            from ..cuda.ops import recurrent_manifold_fused
            from ..cuda.autograd import recurrent_manifold_fused_autograd
            
            # AUDIT FIX (Component 2): Toroidal Kernel Routing
            # ================================================
            # If topology is toroidal, route to dedicated toroidal kernel
            # instead of using low-rank approximation with dummy zeros.
            #
            # Toroidal kernel: launch_toroidal_leapfrog_fused()
            #   - Computes metric-derived Christoffel symbols
            #   - Handles toroidal boundary wrapping
            #   - Direct computation, no U/W decomposition
            #
            # CUDA kernel location: gfn/cuda/src/integrators/toroidal/toroidal_christoffel_fused.cu
            # Python binding: gfn/cuda/ops.py::launch_toroidal_leapfrog_fused()
            
            if params.get('is_torus', False):
                try:
                    from ..cuda.ops import launch_toroidal_leapfrog_fused
                    
                    # Call dedicated toroidal kernel
                    result = launch_toroidal_leapfrog_fused(
                        x=x, v=v, f=forces * mask,
                        R=params['major_R'],
                        r=params['minor_r'],
                        dt=params['base_dt'],
                        batch=x.shape[0],
                        seq_len=forces.shape[1],
                        dim=x.shape[1]
                    )
                    
                    if result is not None:
                        return result
                    else:
                        warnings.warn("[GFN:WARN] Toroidal kernel returned None, falling back to Python loop")
                        return None
                        
                except ImportError:
                    warnings.warn("[GFN:WARN] Toroidal kernel not available (compilation needed), falling back to Python loop")
                    return None
            
            # Standard Euclidean path (low-rank approximation)
            # Get dt scales and forget rates
            f_layer = self.model.layers[0]
            if hasattr(f_layer, 'macro_manifold'):
                f_layer = f_layer.macro_manifold

            dt_scales = params['dt_scales_stack']
            forget_rates = torch.sigmoid(f_layer.christoffels[0].forget_gate.bias.mean())
            if forget_rates.numel() == 1:
                forget_rates = forget_rates.expand(self.model.heads)
            
            # Choose between training (autograd) and inference path
            func = recurrent_manifold_fused_autograd if self.model.training else recurrent_manifold_fused
            
            res = func(
                x=x, v=v, f=forces * mask,
                U_stack=params['U_stack'], W_stack=params['W_stack'],
                dt=params['base_dt'], dt_scales=dt_scales, forget_rates=forget_rates,
                num_heads=self.model.heads,
                plasticity=params['plasticity'],
                sing_thresh=params['sing_thresh'],
                sing_strength=params['sing_strength'],
                mix_x=params['mix_x'], mix_v=params['mix_v'],
                Wf=params['W_f_stack'],
                Wi=params['W_i_stack'],
                bf=params['b_f_stack'],
                Wp=params['W_p_stack'],
                bp=params['b_p_stack'],
                topology=params['topology_id'],
                R=params['major_R'], r=params['minor_r'],
                mix_x_bias=params['mix_x_bias'],
                mix_v_bias=params['mix_v_bias'],
                norm_x_weight=params['norm_x_weight'],
                norm_x_bias=params['norm_x_bias'],
                norm_v_weight=params['norm_v_weight'],
                norm_v_bias=params['norm_v_bias'],
                gate_W1=params['gate_W1'],
                gate_b1=params['gate_b1'],
                gate_W2=params['gate_W2'],
                gate_b2=params['gate_b2'],
                integrator_type=1 if self.model.integrator_type == 'leapfrog' else 0,
                # Integrator type mapping for CUDA kernel:
                # 0 = Heun (default)
                # 1 = Leapfrog
                # Note: Other integrators (rk4, euler, etc.) fall back to Python loop
                # --- NEW HYSTERESIS PARAMETERS ---
                hysteresis_state=hysteresis_state,
                hyst_update_w=hyst_update_w,
                hyst_update_b=hyst_update_b,
                hyst_readout_w=hyst_readout_w,
                hyst_readout_b=hyst_readout_b,
                hyst_decay=hyst_decay,
                hyst_enabled=hyst_enabled
            )
            
            if res is not None:
                # res is (x_final, v_final, x_seq, reg_loss, new_h_state)
                return res
            return None
            
        except Exception as e:
            warnings.warn(f"[GFN:WARN] Fused kernel failed: {e}, falling back to Python loop")
            return None
