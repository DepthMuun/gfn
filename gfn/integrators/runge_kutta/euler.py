
"""
Euler Integrator (1st Order).
The simplest explicit integrator. 
Useful as a baseline to demonstrate the instability of low-order non-symplectic methods.
"""
import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import euler_fused, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

try:
    from gfn.geometry.boundaries import apply_boundary_python
except ImportError:
    def apply_boundary_python(x, tid): return x

class EulerIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt

    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
        # Try Professional Fused CUDA Kernel
        if CUDA_AVAILABLE and x.is_cuda and not collect_christ:
            try:
                # We need U, W from Christoffel
                U = getattr(self.christoffel, 'U', None)
                W = getattr(self.christoffel, 'W', None)
                if U is not None and W is not None:
                    topology = getattr(self.christoffel, 'topology_id', 0)
                    if hasattr(self.christoffel, 'is_torus') and self.christoffel.is_torus: topology = 1
                    
                    R = getattr(self.christoffel, 'R', 2.0)
                    r = getattr(self.christoffel, 'r', 1.0)
                    
                    return euler_fused(x, v, force, U, W, self.dt, dt_scale, steps=steps, topology=topology, R=R, r=r)
            except Exception as e:
                print(f"[GFN:WARN] CUDA euler_fused failed: {e}, falling back to PyTorch")
                # Fall through to PyTorch implementation

        curr_x, curr_v = x, v
        for _ in range(steps):
            dt = self.dt * dt_scale
            
            acc = -self.christoffel(curr_v, curr_x, force=force, **kwargs)
            if force is not None:
                acc = acc + force
                
            curr_x = curr_x + dt * curr_v
            curr_v = curr_v + dt * acc
            
            # Apply Boundary (Torus)
            topo_id = getattr(self.christoffel, 'topology_id', 0)
            if topo_id == 0 and hasattr(self.christoffel, 'is_torus') and self.christoffel.is_torus:
                 topo_id = 1
            curr_x = apply_boundary_python(curr_x, topo_id)
        
        return curr_x, curr_v
