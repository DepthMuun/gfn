class GFNError(Exception):
    """Base exception for all GFN errors."""
    pass

class ConfigurationError(GFNError):
    """Raised when a configuration is invalid or inconsistent."""
    pass

class GeometryError(GFNError):
    """Raised when a geometric operation fails (e.g., out of manifold)."""
    pass

class PhysicsError(GFNError):
    """Raised during physics engine failures (e.g., NaN detected)."""
    pass

class IntegrationError(GFNError):
    """Raised during numerical integration failures."""
    pass

class TrainingError(GFNError):
    """Raised during model training or optimization failures."""
    pass
