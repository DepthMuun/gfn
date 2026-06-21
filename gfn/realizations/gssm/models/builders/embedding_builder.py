"""
gfn/realizations/gssm/models/builders/embedding_builder.py
Builder for FunctionalEmbedding component.
"""
import torch.nn as nn
from typing import Any

from . import ComponentBuilder, MODEL_BUILDER_REGISTRY
from ..components.embedding import FunctionalEmbedding


class EmbeddingBuilder(ComponentBuilder):
    """
    Builder for the token embedding component.
    
    Creates FunctionalEmbedding from ManifoldConfig parameters.
    """
    
    def build(self) -> nn.Module:
        """Build and return the embedding component."""
        config = self.config
        
        # BUG FIX: Added support for continuous_input_dim parameter
        # This is needed for mode='continuous' to properly handle input dimensions
        embedding_kwargs = {
            'vocab_size': config.vocab_size,
            'emb_dim': config.dim,
            'coord_dim': config.physics.embedding.coord_dim,
            'mode': config.physics.embedding.mode,
            'impulse_scale': config.impulse_scale,
            'omega_0': config.physics.embedding.omega_0,
        }
        
        # Pass continuous_input_dim if available in config
        if hasattr(config, 'continuous_input_dim') and config.continuous_input_dim is not None:
            embedding_kwargs['continuous_input_dim'] = config.continuous_input_dim
        
        embedding = FunctionalEmbedding(**embedding_kwargs)
        
        return embedding


# Register the builder
MODEL_BUILDER_REGISTRY.register('embedding', EmbeddingBuilder)
