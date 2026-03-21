from dataclasses import dataclass
from typing import List, Any

@dataclass
class MatrixAxis:
    name: str
    options: List[Any]
    description: str

class MatrixAxes:
    """Defines the Grid of all possible component combinations."""
    
    INTEGRATORS = MatrixAxis(
        "integrator", 
        ["leapfrog", "rk4", "heun", "euler"], 
        "Solver dynamics logic"
    )
    
    TOPOLOGIES = MatrixAxis(
        "topology", 
        ["euclidean", "torus", "hyperbolic", "spherical"], 
        "Manifold geometry type"
    )
    
    EMBEDDINGS = MatrixAxis(
        "embedding", 
        ["standard", "functional", "implicit"], 
        "Input representation method"
    )
    
    READOUTS = MatrixAxis(
        "readout", 
        ["linear", "implicit", "identity"], 
        "Output projection method"
    )
    
    FRICTION = MatrixAxis(
        "friction",
        [0.0, 0.1],
        "Energy dissipation coefficient"
    )
    
    ACTIVE_INFERENCE = MatrixAxis(
        "active_inference",
        [True, False],
        "Curvature plasticity and singularities"
    )

    DEPTH = MatrixAxis(
        "depth",
        [2, 6],
        "Network depth"
    )

    @classmethod
    def get_all_axes(cls) -> List[MatrixAxis]:
        return [
            cls.INTEGRATORS,
            cls.TOPOLOGIES,
            cls.EMBEDDINGS,
            cls.READOUTS,
            cls.FRICTION,
            cls.ACTIVE_INFERENCE,
            cls.DEPTH
        ]
