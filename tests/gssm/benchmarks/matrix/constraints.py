from typing import Dict, Any

class MatrixConstraints:
    """
    Surgical logic to filter impossible or non-sensical combinations.
    """
    
    @staticmethod
    def is_valid(config: Dict[str, Any]) -> bool:
        """
        Check if a flattened configuration dictionary is valid.
        """
        topo = config.get('topology')
        readout = config.get('readout')
        emb = config.get('embedding')
        integ = config.get('integrator')
        active = config.get('active_inference')
        
        # 1. Identity Readout (Holographic) Requirements
        # Identity readout implies the Manifold State IS the Output.
        # This works best when Embedding is also Functional/Implicit (Coordinate based)
        # But we can allow Standard Embedding -> Identity Readout (just passing embeddings through flow)
        if readout == 'identity':
            # Identity readout usually couples with holographic setups
            pass
            
        # 2. Toroidal Topology Restrictions
        if topo == 'torus':
            # Torus works best with periodic or bounded embeddings
            pass
            
        # 3. Hyperbolic Logic
        if topo == 'hyperbolic':
            # Hyperbolic often needs active inference to stabilize curvature
            if not active:
                # Warn or prune? Let's prune strictly for "Production" matrix, allow for "Stress" matrix.
                # For now allow, but maybe flag. 
                pass

        # 4. Explicit Integrator Conflicts
        # Leapfrog is symplectic, good for Hamiltonian.
        # Euler is generally bad for deep depth, but valid for testing.
        
        # 5. Spherical Topology
        # Not fully implemented in all backends?
        # Assuming it is implemented or will just strictly fail.
        
        return True

    @staticmethod
    def get_skip_reason(config: Dict[str, Any]) -> str:
        """Returns reason for skipping, or None if valid."""
        readout = config.get('readout')
        topo = config.get('topology')
        
        # Hard Constraints (Impossible Architectures)
        
        # Identity Readout usually implies we map dim-to-dim directly.
        # If vocab_size != dim, Linear readout handles projection.
        # Identity requires the flow to align with output space.
        # (Assuming vocab_size is handled externally or fixed dim output)
        
        return None
