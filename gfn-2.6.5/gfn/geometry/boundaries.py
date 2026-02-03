
import torch

def apply_boundary_python(x, topology_id):
    """
    Python equivalent of boundaries.cuh
    Topology IDs:
    0: Euclidean (None)
    1: Toroidal (Periodic [0, 2*PI))
    """
    if topology_id == 1:
        PI = 3.14159265359
        TWO_PI = 2.0 * PI
        # Periodic wrapping: x = x % (2*PI)
        # This handles both positive and negative drifts.
        return torch.remainder(x, TWO_PI)
    return x

def toroidal_dist_python(x1, x2):
    """
    Shortest angular distance on Torus.
    """
    PI = 3.14159265359
    TWO_PI = 2.0 * PI
    diff = torch.abs(x1 - x2)
    diff = torch.remainder(diff, TWO_PI)
    return torch.min(diff, TWO_PI - diff)
