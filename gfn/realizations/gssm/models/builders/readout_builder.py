"""
gfn/realizations/gssm/models/builders/readout_builder.py
Builder for readout components.
"""
import logging
import torch.nn as nn
from typing import Any

from . import ComponentBuilder, MODEL_BUILDER_REGISTRY
from ..components.readout import CategoricalReadout, ReadoutPlugin, IdentityReadout, ImplicitReadout

logger = logging.getLogger(__name__)


class ReadoutBuilder(ComponentBuilder):
    """
    Builder for the readout component.
    
    Creates the appropriate readout based on config.physics.readout.type.
    """
    
    def __init__(self, config: Any, dim_total: int, topology: str):
        super().__init__(config)
        self.dim_total = dim_total
        self.topology = topology
    
    def build(self) -> nn.Module:
        """Build and return the readout plugin."""
        readout_type = self.config.physics.readout.type
        
        if readout_type == 'implicit':
            readout = self._build_implicit_readout()
        elif readout_type == 'identity':
            readout = IdentityReadout()
        else:
            if readout_type == 'standard' and self.config.holographic:
                logger.warning(
                    "Config uses holographic=True with readout.type='standard'. "
                    "This no longer implies IdentityReadout; set readout.type='identity' explicitly."
                )
            readout = CategoricalReadout(
                self.dim_total, self.config.vocab_size, topology_type=self.topology
            )
        
        plugin = ReadoutPlugin(readout)
        return plugin
    
    def _build_implicit_readout(self) -> nn.Module:
        """Build implicit readout with custom output dimensions."""
        out_dim = getattr(self.config.physics.readout, 'out_dim', None)
        if out_dim is None:
            out_dim = self.config.vocab_size
            
        hidden_dim = getattr(self.config.physics.readout, 'hidden_dim', None)
        if hidden_dim is None:
            hidden_dim = 128
        
        return ImplicitReadout(
            self.dim_total, out_dim,
            hidden_dim=hidden_dim,
            topology_type=self.topology,
        )


# Register the builder
MODEL_BUILDER_REGISTRY.register('readout', ReadoutBuilder)
