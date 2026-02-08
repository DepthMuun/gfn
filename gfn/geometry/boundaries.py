
import torch


def apply_boundary_python(x, topology_id):
    """
    Apply boundary conditions to position tensor.
    
    IMPORTANT NOTES FROM AUDITORIA LOGICA (2026-02-06):
    
    1. TOROIDAL TOPOLOGY:
       For topology_id == 1 (torus), positions are wrapped to [0, 2π).
       The wrapping is periodic: x = x % (2*π)
       
    2. VELOCITY HANDLING:
       Velocity vectors should NOT be wrapped!
       - Position: x is on the manifold, needs wrapping
       - Velocity: v is in the TANGENT SPACE, invariant under wrapping
       
       The wrapping of velocity would create artificial discontinuities
       that break the smoothness of geodesic flow.
       
       If you need to apply velocity corrections, use apply_velocity_correction().
    
    Topology IDs:
    0: Euclidean (None) - No boundary conditions
    1: Toroidal (Periodic [0, 2*PI)) - Positions wrapped
    
    Args:
        x: Position tensor [batch, dim]
        topology_id: Integer topology identifier
        
    Returns:
        Position tensor with boundaries applied
    """
    if topology_id == 1:
        # AUDIT FIX: Use atan2 for smooth, differentiable wrapping
        # This preserves gradient continuity at 0/2π boundaries
        # atan2(sin(x), cos(x)) wraps to [-π, π], shift to [0, 2π)
        PI = 3.14159265359
        TWO_PI = 2.0 * PI
        x_wrapped = torch.atan2(torch.sin(x), torch.cos(x))
        # Shift from [-π, π] to [0, 2π)
        x_wrapped = torch.where(x_wrapped < 0, x_wrapped + TWO_PI, x_wrapped)
        return x_wrapped
    return x


def apply_velocity_correction(v, x_old, x_new, topology_id):
    """
    Correct velocity for toroidal boundary crossings.
    
    When position crosses the boundary (e.g., from 6.28 to 0.01),
    the apparent velocity is wrong. This function computes the true
    velocity considering boundary crossings.
    
    AUDIT FIX: This function handles velocity correction for torus.
    
    Args:
        v: Velocity tensor [batch, dim]
        x_old: Previous position [batch, dim]
        x_new: Current position [batch, dim]
        topology_id: Topology identifier
        
    Returns:
        Corrected velocity tensor
    """
    if topology_id != 1:
        return v
    
    PI = 3.14159265359
    TWO_PI = 2.0 * PI
    
    # AUDIT FIX: Use atan2 for smooth displacement calculation
    # Compute apparent displacement
    apparent_disp = x_new - x_old
    
    # Compute wrapped displacement with smooth gradients
    wrapped_disp = torch.atan2(torch.sin(apparent_disp), torch.cos(apparent_disp))
    
    # True displacement should be the wrapped version
    # This captures boundary crossings correctly
    return wrapped_disp


def toroidal_dist_python(x1, x2):
    """
    Shortest angular distance on Torus.
    
    IMPORTANT (Auditoria 2026-02-06):
    This computes distance on a FLAT torus (product of circles).
    It does NOT account for the LEARNED Christoffel curvature.
    
    For tasks requiring true geodesic distance on the learned manifold,
    use Christoffel-based distance computation instead.
    
    Args:
        x1: Position tensor [batch, dim]
        x2: Position tensor [batch, dim]
        
    Returns:
        Shortest distance tensor [batch]
    """
    PI = 3.14159265359
    TWO_PI = 2.0 * PI
    # AUDIT FIX: Use atan2 for smooth distance with continuous gradients
    diff = x1 - x2
    diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
    # Return shortest path: min(|diff|, 2π - |diff|)
    return torch.abs(diff_wrapped)
