# Export core API
from .api import create, save, load, Model, Manifold, loss, Trainer

# Register with central realization registry
try:
    from .. import api as central_api
    from . import api as gssm_api
    central_api.register('gssm', gssm_api)
except ImportError:
    pass # Fallback for standalone GSSM usage
