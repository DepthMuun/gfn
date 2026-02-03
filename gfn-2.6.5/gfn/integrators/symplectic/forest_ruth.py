
"""
Forest-Ruth 4th Order Symplectic Integrator.
Standard symplectic integrator, often considered an improved alternative to Yoshida 
for certain Hamiltonian systems due to different coefficient properties.

coeffs from: Forest, E., & Ruth, R. D. (1990). Fourth-order symplectic integration.
"""
import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import forest_ruth_fused, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

try:
    from gfn.geometry.boundaries import apply_boundary_python
except ImportError:
    def apply_boundary_python(x, tid): return x

class ForestRuthIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt
        
        # Forest-Ruth coefficients
        theta = 1.0 / (2.0 - 2.0**(1.0/3.0))
        
        self.c1 = theta / 2.0
        self.c2 = (1.0 - theta) / 2.0
        self.c3 = (1.0 - theta) / 2.0
        self.c4 = theta / 2.0
        
        self.d1 = theta
        self.d2 = 1.0 - 2.0*theta
        self.d3 = theta

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
                    return forest_ruth_fused(x, v, force, U, W, self.dt, dt_scale, steps=steps, topology=topology, R=R, r=r)
            except Exception:
                pass

        dt = self.dt * dt_scale
        
        # Determine Topology
        topo_id = getattr(self.christoffel, 'topology_id', 0)
        if topo_id == 0 and hasattr(self.christoffel, 'is_torus') and self.christoffel.is_torus:
                topo_id = 1

        # Python Implementation
        curr_x, curr_v = x, v
        for _ in range(steps):
            def acceleration(tx, tv):
                 a = -self.christoffel(tv, tx, force=force, **kwargs)
                 if force is not None:
                     a = a + force
                 return a
                 
            # Level 25: Stage-wise Wrapping
            # Step 1
            x1 = apply_boundary_python(curr_x + self.c1 * dt * curr_v, topo_id)
            v1 = curr_v + self.d1 * dt * acceleration(x1, curr_v) 
            
            # Step 2
            x2 = apply_boundary_python(x1 + self.c2 * dt * v1, topo_id)
            v2 = v1 + self.d2 * dt * acceleration(x2, v1)
            
            # Step 3
            x3 = apply_boundary_python(x2 + self.c3 * dt * v2, topo_id)
            v3 = v2 + self.d3 * dt * acceleration(x3, v2)
            
            # Step 4 (Final Drift)
            curr_x = apply_boundary_python(x3 + self.c4 * dt * v3, topo_id)
            curr_v = v3 + self.d3 * dt * acceleration(curr_x, v3)
        
        return curr_x, curr_v
