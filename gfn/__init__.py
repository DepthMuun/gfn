"""
GFN (Geodesic Flow Network) Package
==================================
Unified framework for Geodesic State Space Models (G-SSM)
and Inertial State Networks (ISN).
"""

# ── API Entry Points ──────────────────────────────────────────────────────────
from .api import create, load, save, list_available, register
from .api import REALIZATIONS as _dummy_reg # Just for backward compatibility if needed

# ── Metadata ──────────────────────────────────────────────────────────────────
__version__ = "2.7.2" # Incrementing version due to refactor
__author__ = "DepthMuun"

# ── Realizations ──────────────────────────────────────────────────────────────
# We trigger registration by importing the realizations subpackage
from . import realizations

# ── Shortcuts for easier access ───────────────────────────────────────────────
# Allows: import gfn; model = gfn.gssm.create(...)
from .realizations import gssm, isn

__all__ = [
    "create",
    "load",
    "save",
    "list_available",
    "register",
    "realizations",
    "gssm",
    "isn",
]
