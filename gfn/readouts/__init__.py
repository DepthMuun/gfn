"""
Readout Strategies
==================

Different strategies for converting hidden states to output logits.

Available Readouts:
    - ImplicitReadout: Temperature-annealed sigmoid with topology support

Future Readouts:
    - ExplicitReadout: Standard linear projection
    - AdaptiveReadout: Dynamic readout selection
    - AttentionReadout: Attention-based readout
"""

from .implicit import ImplicitReadout

__all__ = [
    'ImplicitReadout',
]
