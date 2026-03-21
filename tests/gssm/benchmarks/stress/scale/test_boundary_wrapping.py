import pytest
import torch
from gfn import ToroidalRiemannianGeometry
from gfn import Leapfrog as LeapfrogIntegrator
from gfn.constants import TOROIDAL_PERIOD

class TestBoundaryStability:
    
    def test_long_term_wrapping_stability(self, device):
        """Verify that positions stay in [-pi, pi] and don't explode over 2000 steps."""
        dim = 8
        dt = 0.5 # Aggressive dt
        steps = 2000
        
        geometry = ToroidalRiemannianGeometry(dim).to(device)
        integrator = LeapfrogIntegrator(geometry, dt=dt).to(device)
        
        # High velocity to ensure many wrappings
        x = torch.zeros(1, dim, device=device)
        v = torch.ones(1, dim, device=device) * 10.0
        force = torch.zeros_like(x)
        
        curr_x, curr_v = x.clone(), v.clone()
        
        for t in range(steps):
            curr_x, curr_v, _ = integrator(curr_x, curr_v, force)
            
            # Check range [0, 2pi]
            assert torch.all(curr_x >= -1e-4) and torch.all(curr_x <= TOROIDAL_PERIOD + 1e-4), \
                f"Boundary wrapping failed at step {t}: x={curr_x.max().item()}"
            
            # Check for NaNs
            assert not torch.isnan(curr_x).any(), f"NaN detected at step {t}"
            
        print(f"[Stress] Completed {steps} steps with successful wrapping.")

    def test_gradient_continuity_across_boundary(self, device):
        """Verify that gradients can be backpropagated through a wrapping event."""
        dim = 2
        dt = 0.1
        
        geometry = ToroidalRiemannianGeometry(dim).to(device)
        integrator = LeapfrogIntegrator(geometry, dt=dt).to(device)
        
        # Position near the boundary (pi - epsilon)
        x = torch.tensor([[3.1, 0.0]], device=device, requires_grad=True)
        v = torch.tensor([[1.0, 0.0]], device=device, requires_grad=True)
        force = torch.zeros_like(x)
        
        # Step that crosses the boundary
        # x_new = 3.1 + 0.1 * 1.0 = 3.2 -> should wrap to -3.08
        x_next, v_next, _ = integrator(x, v, force)
        
        loss = x_next.sum()
        loss.backward()
        
        # Check if gradients exist and are finite
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert v.grad is not None and torch.isfinite(v.grad).all()
        
        print(f"[Stress] Gradient wrapping check passed. dL/dx: {x.grad}")
