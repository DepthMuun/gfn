import torch
import torch.nn as nn

class EuclideanChristoffel(nn.Module):
    """
    Flat Geometry. Gamma = 0.
    Standard Deep Learning / ResNet behavior.
    """
    def __init__(self, dim, physics_config=None):
        super().__init__()
        self.dim = dim
        
    def forward(self, v, x=None, **kwargs):
        return torch.zeros_like(v)

class HyperbolicChristoffel(nn.Module):
    """
    Hyperbolic Geometry (Poincaré Ball Model).
    Constant Negative Curvature.
    
    Structure:
    Tree-like embeddings, ideal for Hierarchies and Syntax.
    
    Geodesic Accel: a = -Gamma(v,v)
    Approximation near origin or exact formula?
    Uses Conformal Factor lambda = 2 / (1 - |x|^2)
    """
    def __init__(self, dim, physics_config=None):
        super().__init__()
        self.dim = dim
        self.curvature = -1.0
        
    def forward(self, v, x, **kwargs):
        if x is None: return torch.zeros_like(v)
        
        # Conformal factor lambda(x) approx
        # For numeric stability with unconstrained x, we treat x as being in tangent space 
        # mapped to manifold, or we assume x is typically small.
        # Strict Poincaré requires |x| < 1.
        # We implementation a Soft-Poincaré:
        # Scale curvature effect by distance from origin.
        
        # Formula: a = 2 (<x,v>v - |v|^2 x) / (1 - |x|^2)  (roughly)
        # We simplify to: Gamma ~ - ( <x,v>v - |v|^2 x )
        # Negative curvature pushes paths APART (diverge).
        
        x_sq = torch.sum(x*x, dim=-1, keepdim=True)
        v_sq = torch.sum(v*v, dim=-1, keepdim=True)
        xv = torch.sum(x*v, dim=-1, keepdim=True)
        
        # Divergent force:
        gamma = 2 * xv * v - v_sq * x
        
        # Scale by 1/(1-x^2)? No, dangerous if x not bounded.
        # Let's just use the directionality for now as a "Hyperbolic Bias".
        return gamma * 0.1 # Small scale factor for stability

class SphericalChristoffel(nn.Module):
    """
    Spherical Geometry (Stereographic Projection).
    Constant Positive Curvature.
    
    Structure:
    Cyclic embeddings, valid for Rotations and Patterns.
    
    Positive curvature pulls paths TOGETHER (converge).
    """
    def __init__(self, dim, physics_config=None):
        super().__init__()
        self.dim = dim
        self.curvature = 1.0
        
    def forward(self, v, x, **kwargs):
        if x is None: return torch.zeros_like(v)
        
        x_sq = torch.sum(x*x, dim=-1, keepdim=True)
        v_sq = torch.sum(v*v, dim=-1, keepdim=True)
        xv = torch.sum(x*v, dim=-1, keepdim=True)
        
        # Convergent force (Sign flip vs Hyperbolic):
        # Gamma ~ ( <x,v>v - |v|^2 x )
        gamma = -(2 * xv * v - v_sq * x)
        
        return gamma * 0.1
