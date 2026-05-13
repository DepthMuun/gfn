"""
GFN V5 Optimizer Compatibility Shim
Migrated to gfn.training.optimizer
"""
from ..training.optimizer import RiemannianAdam, RiemannianSGD, ManifoldSGD

Adam = RiemannianAdam
SGD = ManifoldSGD

__all__ = ["Adam", "SGD", "RiemannianAdam", "ManifoldSGD"]
