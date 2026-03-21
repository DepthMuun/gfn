import itertools
from typing import Iterator, Dict, Any
from .axes import MatrixAxes
from .constraints import MatrixConstraints
from gfn.realizations.gssm.config import ManifoldConfig, PhysicsConfig, TopologyConfig, EmbeddingConfig, ReadoutConfig, StabilityConfig, ActiveInferenceConfig

class MatrixGenerator:
    """
    Generates test configurations.
    """
    
    @staticmethod
    def generate_all() -> Iterator[ManifoldConfig]:
        axes = MatrixAxes.get_all_axes()
        
        # Create cartesian product of all options
        keys = [axis.name for axis in axes]
        option_lists = [axis.options for axis in axes]
        
        for values in itertools.product(*option_lists):
            flat_config = dict(zip(keys, values))
            
            if not MatrixConstraints.is_valid(flat_config):
                continue
                
            yield MatrixGenerator.build_config(flat_config)
            
    @staticmethod
    def build_config(flat: Dict[str, Any]) -> ManifoldConfig:
        """Hydrate a flat dictionary into a ManifoldConfig object."""
        
        physics = PhysicsConfig(
            topology=TopologyConfig(type=flat['topology']),
            embedding=EmbeddingConfig(type=flat['embedding']),
            readout=ReadoutConfig(type=flat['readout']),
            stability=StabilityConfig(
                friction=flat['friction'],
                base_dt=0.1, # Standard dt
                velocity_friction_scale=flat['friction'] # Match friction
            ),
            active_inference=ActiveInferenceConfig(
                enabled=flat['active_inference'],
                dynamic_time=flat['active_inference'] # Couple dynamics
            )
        )
        
        return ManifoldConfig(
            vocab_size=16, # Fixed for benchmark
            dim=32,       # Small dim for speed
            depth=flat['depth'],
            heads=4,
            integrator=flat['integrator'],
            physics=physics
        )
