"""Loss functions for ISN Model."""

from .coherence import MultiDimensionalLoss
from .semantic_distance import SemanticDistanceLoss
from .energy_threshold import ThresholdModulationLoss

__all__ = ['MultiDimensionalLoss', 'SemanticDistanceLoss', 'ThresholdModulationLoss']
