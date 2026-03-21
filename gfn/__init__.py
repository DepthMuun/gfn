"""
GFN (Geodesic Flow Network) Package
==================================
Unified framework for Geodesic State Space Models (G-SSM)
and Inertial State Networks (ISN).

This package implements the GFN paradigm as a platform for
physics-informed neural dynamics.
"""

# ── Realizations ──────────────────────────────────────────────────────────────
from .realizations import api, gssm, isn
from .realizations.api import create, load, save

# ── Dynamic Registry
REALIZATIONS = api.list_available()

# ── Package Metadata ──────────────────────────────────────────────────────────
__version__ = "2.7.1"
__author__ = "DepthMuun"

__all__ = [
    "gssm",
    "isn",
    "api",
    "create",
    "load",
    "save",
    "REALIZATIONS",
]
