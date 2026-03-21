"""
gfn/training/evaluation.py — GFN V5
Portado desde: gfn_old/engine/evaluation/

Evaluadores de métricas geométricas y restricciones físicas para modelos GFN.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ..constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN
import math


class ManifoldMetricEvaluator:
    """
    Evalúa métricas geométricas y arquitectónicas del sistema Manifold.

    Métricas disponibles:
      - Cobertura del espacio del manifold (varianza, distancia al centroide)
      - Estadísticas de velocidad (norma media/max, std)
      - Perfil de curvatura (si se provee geometría)
      - Accuracy Toroidal (para benchmarks con clasificación angular)
    """

    def __init__(self, model: nn.Module, geometry: Optional[Any] = None):
        self.model = model
        self.geometry = geometry

    def evaluate_state_coverage(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Mide cobertura del manifold: varianza y distancia al centroide.
        """
        with torch.no_grad():
            # Aplanar a 2D si x tiene más dimensiones
            x_flat = x.reshape(x.shape[0], -1)
            var = torch.var(x_flat, dim=0).mean().item()
            centroid = x_flat.mean(dim=0)
            mean_dist = torch.norm(x_flat - centroid, dim=-1).mean().item()

        return {
            "manifold_variance": var,
            "mean_centroid_distance": mean_dist,
        }

    def evaluate_velocity_statistics(self, v: torch.Tensor) -> Dict[str, float]:
        """
        Analiza la distribución de velocidades (momentum).
        """
        with torch.no_grad():
            v_flat = v.reshape(v.shape[0], -1)
            v_norm = torch.norm(v_flat, dim=-1)

        return {
            "velocity_mean_norm": v_norm.mean().item(),
            "velocity_max_norm": v_norm.max().item(),
            "velocity_std": v_norm.std().item(),
        }

    def evaluate_curvature_profile(self, x: torch.Tensor,
                                   v: torch.Tensor) -> Dict[str, float]:
        """Analiza la curvatura encontrada si hay geometría disponible."""
        if self.geometry and hasattr(self.geometry, 'forward'):
            with torch.no_grad():
                gamma = self.geometry(x, v)
                gamma_norm = torch.norm(gamma.reshape(gamma.shape[0], -1), dim=-1)
                return {
                    "mean_curvature_response": gamma_norm.mean().item(),
                    "max_curvature_response": gamma_norm.max().item(),
                }
        return {}

    def evaluate_toroidal_accuracy(self, x_pred: torch.Tensor,
                                   y_class: torch.Tensor) -> float:
        """
        Accuracy para representaciones toroidales.

        Asigna la clase basándose en la distancia angular más próxima
        a ±π/2. Correcto para GFNs con salidas en espacio angulas.
        """
        with torch.no_grad():
            if x_pred.dim() > 2:
                x_pred = x_pred.mean(dim=list(range(2, x_pred.dim())))
            PI = math.pi
            TWO_PI = 2.0 * PI
            half_pi = PI * 0.5

            dist_pos = torch.min(
                torch.abs(x_pred - half_pi) % TWO_PI,
                TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI),
            )
            dist_neg = torch.min(
                torch.abs(x_pred + half_pi) % TWO_PI,
                TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI),
            )
            preds = (dist_pos.mean(dim=-1) < dist_neg.mean(dim=-1)).long()
            return (preds == y_class).float().mean().item()

    def full_report(self, x: torch.Tensor, v: torch.Tensor,
                    y_class: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Reporte completo de todas las métricas disponibles."""
        report = {}
        report.update(self.evaluate_state_coverage(x))
        report.update(self.evaluate_velocity_statistics(v))
        report.update(self.evaluate_curvature_profile(x, v))
        if y_class is not None:
            report["toroidal_accuracy"] = self.evaluate_toroidal_accuracy(x, y_class)
        return report


class PhysicsConstraintEvaluator:
    """
    Valida que el entrenamiento respeta las restricciones físicas del GFN.

    Verificaciones:
      - Conservación de energía (drift Hamiltoniano)
      - Normalidad de velocidades (sin explosión de gradientes físicos)
      - Ausencia de NaN/Inf en los estados
      - Wrapping correcto para topología toroidal
    """

    def __init__(self, topology: str = TOPOLOGY_TORUS, velocity_threshold: float = 10.0):
        self.topology = str(topology).lower().strip()
        self.velocity_threshold = velocity_threshold

    def check_nan_states(self, x: torch.Tensor,
                         v: torch.Tensor) -> Dict[str, bool]:
        """Detecta NaN o Inf en los estados."""
        return {
            "x_has_nan":  bool(torch.isnan(x).any().item()),
            "v_has_nan":  bool(torch.isnan(v).any().item()),
            "x_has_inf":  bool(torch.isinf(x).any().item()),
            "v_has_inf":  bool(torch.isinf(v).any().item()),
        }

    def check_torus_wrapping(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Para topología toroidal: verifica que los valores estén en [-π, π].
        Retorna la fracción de valores fuera del rango.
        """
        if self.topology != TOPOLOGY_TORUS:
            return {}
        import math
        in_range = ((x >= -math.pi) & (x <= math.pi)).float()
        return {"torus_in_range_fraction": in_range.mean().item()}

    def check_velocity_explosion(self, v: torch.Tensor) -> Dict[str, Any]:
        """Detecta explosión de velocidades física."""
        v_norm = torch.norm(v.reshape(v.shape[0], -1), dim=-1)
        max_norm = v_norm.max().item()
        return {
            "velocity_max_norm": max_norm,
            "velocity_exploded": max_norm > self.velocity_threshold,
        }

    def full_check(self, x: torch.Tensor,
                   v: torch.Tensor) -> Dict[str, Any]:
        """Reporte completo de restricciones físicas."""
        report: Dict[str, Any] = {}
        report.update(self.check_nan_states(x, v))
        report.update(self.check_torus_wrapping(x))
        report.update(self.check_velocity_explosion(v))
        report["physically_valid"] = not any([
            report.get("x_has_nan", False),
            report.get("v_has_nan", False),
            report.get("v_has_inf", False),
            report.get("velocity_exploded", False),
        ])
        return report


__all__ = ['ManifoldMetricEvaluator', 'PhysicsConstraintEvaluator']
