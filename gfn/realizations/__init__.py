"""
GFN Realizations Subpackage
===========================
Contains specific implementations of the GFN paradigm.
Registration is handled dynamically by each realization module.
"""

# Trigger registration of standard realizations
try:
    from . import gssm
except ImportError:
    pass

try:
    from . import isn
except ImportError:
    pass

__all__ = ['gssm', 'isn']
