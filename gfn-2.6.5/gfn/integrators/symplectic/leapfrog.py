
"""
Leapfrog (Kick-Drift-Kick) Symplectic Integrator.
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

class LeapfrogIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt

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
                    return leapfrog_fused(x, v, force, U, W, self.dt, dt_scale, steps=steps, topology=topology_id, Wf=Wf, bf=bf)
            except Exception as e:
                print(f"[GFN:WARN] CUDA leapfrog_fused failed: {e}, falling back to PyTorch")
                # Fall through to PyTorch implementation

        curr_x, curr_v = x, v
        # Tell Christoffel to return friction separately for implicit update
        was_separate = getattr(self.christoffel, 'return_friction_separately', False)
        self.christoffel.return_friction_separately = True
        
        try:
            for _ in range(steps):
                effective_dt = self.dt * dt_scale
                h = 0.5 * effective_dt
                
                # 1. Kick (Implicit Friction)
                # v_next = (v_prev + h*(F - gamma)) / (1 + h*mu)
                res = self.christoffel(curr_v, curr_x, force=force, **kwargs)
                if isinstance(res, tuple):
                    gamma, mu = res
                else:
                    gamma, mu = res, 0.0 # Fallback
                Wf = kwargs.get('W_forget_stack', None)
                Wi = kwargs.get('W_input_stack', None)
                bf = kwargs.get('b_forget_stack', None)
                
                topology_id = kwargs.get('topology', getattr(self.christoffel, 'topology_id', 0))
                if topology_id == 0 and hasattr(self.christoffel, 'is_torus') and self.christoffel.is_torus:
                     topology_id = 1
                if Wf is not None and bf is not None:
                    if Wf.dim() == 3:
                        Wf = Wf[0]
                    if bf.dim() == 2:
                        bf = bf[0]
                    feat = curr_x
                    if topology_id == 1:
                        feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)
                    gate = torch.matmul(feat, Wf.t()) + bf
                    mu = torch.sigmoid(gate) * 5.0
                    
                # Match CUDA v5.0: Implicit Update
                # v_half = (curr_v + h * (force - gamma)) / (1.0 + h * mu)
                v_half = (curr_v + h * (force - gamma)) / (1.0 + h * mu)
                
                # 2. Drift (full step position)
                curr_x = curr_x + effective_dt * v_half
                
                # Apply Boundary (Torus)
                curr_x = apply_boundary_python(curr_x, topology_id)
                
                # 3. Kick (half step velocity at new pos)
                res_half = self.christoffel(v_half, curr_x, force=force, **kwargs)
                if isinstance(res_half, tuple):
                    gamma_half, mu_half = res_half
                else:
                    gamma_half, mu_half = res_half, 0.0
                Wf = kwargs.get('W_forget_stack', None)
                Wi = kwargs.get('W_input_stack', None)
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
                    mu_half = torch.sigmoid(gate) * 5.0
                    
                # Match CUDA v5.0: Implicit Update
                curr_v = (v_half + h * (force - gamma_half)) / (1.0 + h * mu_half)
        finally:
            self.christoffel.return_friction_separately = was_separate
        
        return curr_x, curr_v
