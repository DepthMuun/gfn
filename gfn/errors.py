# gfn/errors.py
# Proxy for G-SSM errors to maintain backward compatibility in tests.

try:
    from .realizations.gssm.errors import (
        GFNError,
        ConfigurationError,
        GeometryError,
        PhysicsError,
        IntegrationError,
        TrainingError
    )
except ImportError:
    # Minimal fallbacks if gssm is not available
    class GFNError(Exception): pass
    class ConfigurationError(GFNError): pass
    class GeometryError(GFNError): pass
    class PhysicsError(GFNError): pass
    class IntegrationError(GFNError): pass
    class TrainingError(GFNError): pass

__all__ = [
    "GFNError",
    "ConfigurationError",
    "GeometryError",
    "PhysicsError",
    "IntegrationError",
    "TrainingError"
]
