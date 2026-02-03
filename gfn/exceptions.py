"""
GFN Custom Exceptions
=====================

Custom exception hierarchy for better error handling and debugging.
"""


class GFNError(Exception):
    """Base exception for all GFN-related errors."""
    pass


class CUDAFallbackError(GFNError):
    """Raised when CUDA operation fails and fallback is used.
    
    This is typically a warning-level exception that indicates
    the system gracefully degraded to PyTorch implementation.
    """
    pass


class InvalidStateError(GFNError):
    """Raised when model state is invalid or malformed.
    
    Examples:
        - State is not a tuple
        - State tensors have wrong shapes
        - State tensors are on wrong device
    """
    pass


class GeometryError(GFNError):
    """Raised when geometry computation fails.
    
    Examples:
        - Division by zero in Christoffel symbols
        - Invalid manifold parameters
        - NaN/Inf in curvature computation
    """
    pass


class IntegrationError(GFNError):
    """Raised when numerical integration fails.
    
    Examples:
        - Integrator diverges
        - Invalid timestep
        - NaN/Inf in state evolution
    """
    pass


class ScanError(GFNError):
    """Raised when parallel scan operation fails.
    
    Examples:
        - Invalid input dimensions
        - Zero-length sequence
        - Shape mismatch between inputs
    """
    pass


class EmbeddingError(GFNError):
    """Raised when embedding computation fails.
    
    Examples:
        - Invalid token indices
        - Dimension mismatch
        - SIREN initialization failure
    """
    pass
