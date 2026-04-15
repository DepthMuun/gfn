from gfn.errors import (
    GFNError as BaseGFNError,
    ConfigurationError as BaseConfigurationError,
    GeometryError as BaseGeometryError,
    PhysicsError as BasePhysicsError,
    IntegrationError as BaseIntegrationError,
    TrainingError as BaseTrainingError
)

class GFNError(BaseGFNError):
    """Base exception for G-SSM specific errors."""
    pass

class ConfigurationError(BaseConfigurationError, GFNError):
    """Raised when a G-SSM configuration is invalid."""
    pass

class GeometryError(BaseGeometryError, GFNError):
    """Raised when a G-SSM geometric operation fails."""
    pass

class PhysicsError(BasePhysicsError, GFNError):
    """Raised during G-SSM physics engine failures."""
    pass

class IntegrationError(BaseIntegrationError, GFNError):
    """Raised during G-SSM integration failures."""
    pass

class TrainingError(BaseTrainingError, GFNError):
    """Raised during G-SSM training failures."""
    pass
