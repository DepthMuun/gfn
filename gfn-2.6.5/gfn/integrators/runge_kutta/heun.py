
"""
Heun's Method (Improved Euler / RK2).
2nd order accuracy with only 2 evaluations per step.
Great balance between accuracy and speed.
"""
import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import heun_fused, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

try:
    from gfn.geometry.boundaries import apply_boundary_python
except ImportError:
    def apply_boundary_python(x, tid): return x

class HeunIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt
        
    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
        # Try Professional Fused CUDA Kernel
        if CUDA_AVAILABLE and x.is_cuda and not collect_christ:
            try:
                U = getattr(self.christoffel, 'U', None)
                W = getattr(self.christoffel, 'W', None)
                if U is not None and W is not None:
                    topology = getattr(self.christoffel, 'topology_id', 0)
                    if hasattr(self.christoffel, 'is_torus') and self.christoffel.is_torus: topology = 1
                    
                    R = getattr(self.christoffel, 'R', 2.0)
                    r = getattr(self.christoffel, 'r', 1.0)
                    
                    return heun_fused(x, v, force, U, W, self.dt, dt_scale, steps=steps, topology=topology, R=R, r=r)
            except Exception:
                pass

        curr_x, curr_v = x, v
        for _ in range(steps):
            dt = self.dt * dt_scale
            
            def dynamics(current_x, current_v):
                # LEVEL 25: CLUTCH CONNECTION
                acc = -self.christoffel(current_v, current_x, force=force, **kwargs)
                if force is not None:
                    acc = acc + force
                return acc
                
            # Determine Topology
            topo_id = getattr(self.christoffel, 'topology_id', 0)
            if topo_id == 0 and hasattr(self.christoffel, 'is_torus') and self.christoffel.is_torus:
                 topo_id = 1

            # k1
            dx1 = curr_v
            dv1 = dynamics(curr_x, curr_v)
            
            # Predictor step (Euler)
            v_pred = curr_v + dt * dv1
            x_pred = apply_boundary_python(curr_x + dt * dx1, topo_id)
            
            # k2 (using predicted velocity AND position)
            dx2 = v_pred
            dv2 = dynamics(x_pred, v_pred)
            
            # Corrector step
            curr_x = curr_x + (dt / 2.0) * (dx1 + dx2)
            curr_v = curr_v + (dt / 2.0) * (dv1 + dv2)
            
            # Apply Boundary (Torus)
            curr_x = apply_boundary_python(curr_x, topo_id)
        
        return curr_x, curr_v
