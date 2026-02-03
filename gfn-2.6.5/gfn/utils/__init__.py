"""
Utilities
=========

Utility functions for GFN.

Available Utilities:
    - parallel_scan: Parallel scan implementation
    - GPUMonitor: GPU temperature monitoring
"""

from .scan import parallel_scan
from .safety import GPUMonitor

# Visualization functions (if they exist)
try:
    from .visualization import (
        visualize_manifold_flow,
        visualize_christoffel_field,
        visualize_geodesic_paths,
    )
    __all__ = [
        'parallel_scan',
        'GPUMonitor',
        'visualize_manifold_flow',
        'visualize_christoffel_field',
        'visualize_geodesic_paths',
    ]
except ImportError:
    __all__ = [
        'parallel_scan',
        'GPUMonitor',
    ]
