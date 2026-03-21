# gfn/losses/__init__.py
# Proxy for G-SSM losses to maintain backward compatibility in tests.

try:
    from ..realizations.gssm.losses.toroidal import ToroidalLoss
    from ..realizations.gssm.losses.generative import ManifoldGenerativeLoss
    from ..realizations.gssm.losses.factory import LossFactory
except ImportError:
    pass

__all__ = ["ToroidalLoss", "ManifoldGenerativeLoss", "LossFactory"]
