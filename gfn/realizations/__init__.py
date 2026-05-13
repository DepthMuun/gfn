"""
GFN Realizations Subpackage
===========================
Contains specific implementations of the GFN paradigm:
- G-SSM: Geodesic State Space Model (Riemannian/Symplectic)
- ISN: Inertial State Network (Physics-Informed Interaction Engine)
"""

from . import api
from .api import create, list_available

# Trigger registration of standard realizations
from . import gssm
from . import isn

# Future realizations can be added here or via external plugins
# from . import rt

__all__ = ['gssm', 'isn', 'api', 'create', 'list_available']
