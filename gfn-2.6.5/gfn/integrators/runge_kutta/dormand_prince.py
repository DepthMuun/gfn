
"""
Dormand-Prince (RK45) integrator.

Uses RK45 coefficients with a fixed step size.
"""
import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import dormand_prince_fused, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

try:
    from gfn.geometry.boundaries import apply_boundary_python
except ImportError:
    def apply_boundary_python(x, tid): return x

class DormandPrinceIntegrator(nn.Module):
    r"""
    Dormand-Prince (DP5) Integrator.
    
    Uses the RK45 tableau and applies the 5th-order solution for updates.
    """
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.base_dt = dt
        
        # RK45 (Dormand-Prince) coefficients
        self.c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
        
        # a-coefficients (DP54)
        self.a21 = 1/5
        self.a31, self.a32 = 3/40, 9/40
        self.a41, self.a42, self.a43 = 44/45, -56/15, 32/9
        self.a51, self.a52, self.a53, self.a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        self.a61, self.a62, self.a63, self.a64, self.a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        
        # b5: 5th-order weights
        self.b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        
    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
        # Try fused CUDA kernel
        if CUDA_AVAILABLE and x.is_cuda and not collect_christ:
            try:
                U = getattr(self.christoffel, 'U', None)
                W = getattr(self.christoffel, 'W', None)
                if U is not None and W is not None:
                    topology = getattr(self.christoffel, 'topology_id', 0)
                    if hasattr(self.christoffel, 'is_torus') and self.christoffel.is_torus: topology = 1
                    
                    R = getattr(self.christoffel, 'R', 2.0)
                    r = getattr(self.christoffel, 'r', 1.0)
                    return dormand_prince_fused(x, v, force, U, W, self.base_dt, dt_scale, steps=steps, topology=topology, R=R, r=r)
            except Exception:
                pass

        dt = self.base_dt * dt_scale
        
        # Resolve topology
        topo_id = getattr(self.christoffel, 'topology_id', 0)
        if topo_id == 0 and hasattr(self.christoffel, 'is_torus') and self.christoffel.is_torus:
                topo_id = 1
        
        curr_x, curr_v = x, v
        k1_v_cache = None
        for _ in range(steps):
            dt = self.base_dt * dt_scale
            
            def dynamics(tx, tv):
                acc = -self.christoffel(tv, tx, force=force, **kwargs)
                if force is not None:
                    acc = acc + force
                return acc
                
            # k1
            k1_x = curr_v
            if k1_v_cache is None:
                k1_v = dynamics(curr_x, curr_v)
            else:
                k1_v = k1_v_cache
                k1_v_cache = None
            
            # k2
            x2 = apply_boundary_python(curr_x + dt * (self.a21*k1_x), topo_id)
            v2 = curr_v + dt * (self.a21*k1_v)
            k2_x = v2
            k2_v = dynamics(x2, v2)
            
            # k3
            x3 = apply_boundary_python(curr_x + dt * (self.a31*k1_x + self.a32*k2_x), topo_id)
            v3 = curr_v + dt * (self.a31*k1_v + self.a32*k2_v)
            k3_x = v3
            k3_v = dynamics(x3, v3)
            
            # k4
            x4 = apply_boundary_python(curr_x + dt * (self.a41*k1_x + self.a42*k2_x + self.a43*k3_x), topo_id)
            v4 = curr_v + dt * (self.a41*k1_v + self.a42*k2_v + self.a43*k3_v)
            k4_x = v4
            k4_v = dynamics(x4, v4)
            
            # k5
            x5 = apply_boundary_python(curr_x + dt * (self.a51*k1_x + self.a52*k2_x + self.a53*k3_x + self.a54*k4_x), topo_id)
            v5 = curr_v + dt * (self.a51*k1_v + self.a52*k2_v + self.a53*k3_v + self.a54*k4_v)
            k5_x = v5
            k5_v = dynamics(x5, v5)
            
            # k6
            x6 = apply_boundary_python(curr_x + dt * (self.a61*k1_x + self.a62*k2_x + self.a63*k3_x + self.a64*k4_x + self.a65*k5_x), topo_id)
            v6 = curr_v + dt * (self.a61*k1_v + self.a62*k2_v + self.a63*k3_v + self.a64*k4_v + self.a65*k5_v)
            k6_x = v6
            k6_v = dynamics(x6, v6)
            
            # Result using b5
            curr_x = curr_x + dt * (self.b5[0]*k1_x + self.b5[2]*k3_x + self.b5[3]*k4_x + self.b5[4]*k5_x + self.b5[5]*k6_x)
            curr_x = apply_boundary_python(curr_x, topo_id)
            curr_v = curr_v + dt * (self.b5[0]*k1_v + self.b5[2]*k3_v + self.b5[3]*k4_v + self.b5[4]*k5_v + self.b5[5]*k6_v)
            k1_v_cache = dynamics(curr_x, curr_v)
        
        return curr_x, curr_v
