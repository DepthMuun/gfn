
"""
Symplectic Integrator (Velocity Verlet).
Kept separate for historical continuity and as a standard baseline.
"""
import torch
import torch.nn as nn

try:
    from gfn.geometry.boundaries import apply_boundary_python
except ImportError:
    def apply_boundary_python(x, tid): return x

class SymplecticIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt

    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
        # Try Professional Fused CUDA Kernel
        if x.is_cuda and not collect_christ:
            try:
                from gfn.cuda.ops import verlet_fused, CUDA_AVAILABLE
                if CUDA_AVAILABLE:
                    U = getattr(self.christoffel, 'U', None)
                    W = getattr(self.christoffel, 'W', None)
                    if U is not None and W is not None:
                        topology = getattr(self.christoffel, 'topology_id', 0)
                        if hasattr(self.christoffel, 'is_torus') and self.christoffel.is_torus: topology = 1
                        
                        R = getattr(self.christoffel, 'R', 2.0)
                        r = getattr(self.christoffel, 'r', 1.0)
                        
                        return verlet_fused(x, v, force, U, W, self.dt, dt_scale, steps=steps, topology=topology, R=R, r=r)
            except Exception:
                pass

        for _ in range(steps):
            dt = self.dt * dt_scale
            
            # Determine Topology
            topo_id = getattr(self.christoffel, 'topology_id', 0)
            if topo_id == 0 and hasattr(self.christoffel, 'is_torus') and self.christoffel.is_torus:
                 topo_id = 1

            # Compute acceleration at current state
            gamma = self.christoffel(v, x, force=force, **kwargs)
            
            if force is None:
                a = -gamma
            else:
                a = -gamma + force
                
            # Velocity Verlet Step 1: v(t+0.5*dt)
            v_half = v + 0.5 * dt * a
            
            # Step 2: x(t+dt)
            x = x + dt * v_half
            
            # Apply Boundary (Torus)
            x = apply_boundary_python(x, topo_id)
            
            # Re-compute acceleration at x_next
            gamma_next = self.christoffel(v_half, x, force=force, **kwargs)
            if force is None:
                a_next = -gamma_next
            else:
                a_next = -gamma_next + force
                
            v = v_half + 0.5 * dt * a_next
        
        return x, v
