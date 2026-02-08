
"""
Leapfrog (Kick-Drift-Kick) Symplectic Integrator.

IMPORTANT NOTES FROM AUDITORIA LOGICA (2026-02-06):

1. INTEGRATOR VARIANT:
   This is a VARIANT of the standard Stormer-Verlet / Leapfrog integrator,
   modified for systems with position-dependent friction.
   
   Standard Leapfrog:
   v(t+0.5h) = v(t) + 0.5h * a(x(t))
   x(t+h) = x(t) + h * v(t+0.5h)
   v(t+h) = v(t+0.5h) + 0.5h * a(x(t+h))
   
   Our variant (with friction mu(x)):
   v(t+0.5h) = (v(t) + 0.5h * (F - Gamma)) / (1 + 0.5h * mu(x(t)))
   x(t+h) = x(t) + h * v(t+0.5h)
   v(t+h) = (v(t+0.5h) + 0.5h * (F - Gamma)) / (1 + 0.5h * mu(x(t+h)))

2. CONSERVATION PROPERTIES:
   - In ABSENCE of friction (mu = 0), energy is conserved
   - With FRICTION, energy is DISSIPATED (as expected physically)
   - VOLUME preservation is LOST when friction != 0
     (this is correct: dissipative systems shrink phase space volume)

3. PRODUCTION FIX (2026-02-07):
   - FRICTION_SCALE reduced to 0.05 for proper symplectic behavior
   - EPSILON_STANDARD set to 1e-8 for cleaner gradients
   - Added smooth boundary wrapping for differentiable gradients

References:
- Hairer, E., Lubich, C., & Wanner, G. (2006). Geometric Numerical Integration
- Leimkuhler, B., & Reich, S. (2004). Simulating Hamiltonian Dynamics
"""

import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import leapfrog_fused, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

try:
    from gfn.geometry.boundaries import apply_boundary_python
except ImportError:
    def apply_boundary_python(x, tid): return x

try:
    from gfn.constants import FRICTION_SCALE, EPSILON_STANDARD, DEFAULT_DT
except ImportError:
    FRICTION_SCALE = 0.05
    EPSILON_STANDARD = 1e-8
    DEFAULT_DT = 0.02


def smooth_boundary_wrap(x, topology_id=1):
    """
    AUDIT FIX: Smooth boundary wrapping with differentiable gradients.
    
    Instead of using torch.remainder which has discontinuous gradients at
    boundaries, this function uses atan2(sin, cos) for smooth wrapping.
    
    For topology_id == 1 (torus): wraps to [-π, π] with smooth gradients
    For topology_id == 0 (euclidean): no wrapping
    
    Args:
        x: Input tensor
        topology_id: Topology identifier (0=euclidean, 1=torus)
    
    Returns:
        Wrapped tensor with smooth gradients
    """
    if topology_id == 0:
        return x
    
    # AUDIT FIX: Use atan2 for smooth, differentiable wrapping
    # atan2(sin(x), cos(x)) naturally wraps to [-π, π]
    x_wrapped = torch.atan2(torch.sin(x), torch.cos(x))
    
    return x_wrapped



class LeapfrogIntegrator(nn.Module):
    """
    Variant Stormer-Verlet integrator with position-dependent friction.
    
    For use with LowRankChristoffel and ReactiveChristoffel geometries.
    
    PRODUCTION FIX (2026-02-07):
    - Uses updated FRICTION_SCALE=0.05
    - Uses EPSILON_STANDARD=1e-8
    - Supports smooth boundary wrapping
    """
    def __init__(self, christoffel, dt=None):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt if dt is not None else DEFAULT_DT

    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
        if force is None:
            force = torch.zeros_like(x)
            
        # Try Professional Fused CUDA Kernel
        if CUDA_AVAILABLE and x.is_cuda and not collect_christ:
            try:
                # Logic matrices
                U = getattr(self.christoffel, 'U', None)
                W = getattr(self.christoffel, 'W', None)
                
                if U is not None and W is not None:
                    topology_id = kwargs.get('topology', getattr(self.christoffel, 'topology_id', 0))
                    Wf = kwargs.get('W_forget_stack', None)
                    bf = kwargs.get('b_forget_stack', None)
                    if Wf is not None and Wf.dim() == 3:
                        Wf = Wf[0]
                    if bf is not None and bf.dim() == 2:
                        bf = bf[0]
                    res_fused = leapfrog_fused(x, v, force, U, W, self.dt, dt_scale, steps=steps, topology=topology_id, Wf=Wf, bf=bf)
                    if isinstance(res_fused, tuple) and len(res_fused) == 3:
                        xf, vf, _ = res_fused
                        return xf, vf
                    return res_fused
            except Exception as e:
                print(f"[GFN:WARN] CUDA leapfrog_fused failed: {e}, falling back to PyTorch")
                # Fall through to PyTorch implementation

        curr_x, curr_v = x, v
        # Tell Christoffel to return friction separately for implicit update
        was_separate = getattr(self.christoffel, 'return_friction_separately', False)
        self.christoffel.return_friction_separately = True
        
        # Get topology_id
        topology_id = kwargs.get('topology', getattr(self.christoffel, 'topology_id', 0))
        if topology_id == 0 and hasattr(self.christoffel, 'is_torus') and self.christoffel.is_torus:
             topology_id = 1
        
        try:
            for _ in range(steps):
                effective_dt = self.dt * dt_scale
                h = 0.5 * effective_dt
                
                # 1. Kick (Implicit Friction)
                # PRODUCTION: The implicit update v_new = (v + h*a) / (1 + h*mu)
                # is equivalent to applying friction implicitly
                res = self.christoffel(curr_v, curr_x, force=force, **kwargs)
                if isinstance(res, tuple):
                    gamma, mu = res
                else:
                    gamma, mu = res, 0.0  # Fallback
                
                # Get friction from gate if needed
                Wf = kwargs.get('W_forget_stack', None)
                bf = kwargs.get('b_forget_stack', None)
                
                if Wf is not None and bf is not None:
                    if Wf.dim() == 3:
                        Wf = Wf[0]
                    if bf.dim() == 2:
                        bf = bf[0]
                    feat = curr_x
                    if topology_id == 1:
                        feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)
                    gate = torch.matmul(feat, Wf.t()) + bf
                    mu = torch.sigmoid(gate) * FRICTION_SCALE  # PRODUCTION: Use updated constant

                # PRODUCTION: Implicit Update with EPSILON_STANDARD=1e-8
                # v_half = (curr_v + h*(F - Gamma)) / (1 + h*mu + eps)
                v_half = (curr_v + h * (force - gamma)) / (1.0 + h * mu + EPSILON_STANDARD)
                
                # 2. Drift (full step position)
                curr_x = curr_x + effective_dt * v_half
                
                # Apply Boundary (Torus) - PRODUCTION: Use smooth wrapping
                curr_x = smooth_boundary_wrap(curr_x, topology_id)
                
                # 3. Kick (half step velocity at new pos)
                res_half = self.christoffel(v_half, curr_x, force=force, **kwargs)
                if isinstance(res_half, tuple):
                    gamma_half, mu_half = res_half
                else:
                    gamma_half, mu_half = res_half, 0.0
                
                if Wf is not None and bf is not None:
                    if Wf.dim() == 3:
                        Wf = Wf[0]
                    if bf.dim() == 2:
                        bf = bf[0]
                    feat = curr_x
                    if topology_id == 1:
                        feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)
                    gate = torch.matmul(feat, Wf.t()) + bf
                    mu_half = torch.sigmoid(gate) * FRICTION_SCALE  # PRODUCTION: Use updated constant
                    
                # PRODUCTION: Implicit Update with EPSILON_STANDARD=1e-8
                curr_v = (v_half + h * (force - gamma_half)) / (1.0 + h * mu_half + EPSILON_STANDARD)
        finally:
            self.christoffel.return_friction_separately = was_separate
        
        return curr_x, curr_v
