"""
gfn/physics/monitor.py — GFN V5
PhysicsMonitorPlugin: diagnóstico de energía hamiltoniana durante training.

Conecta al HookManager y trackea H(t) = T + V por batch,
calculando energy_drift relativo para detectar inestabilidad.

Uso:
    monitor = PhysicsMonitorPlugin(geometry, integrator, enabled=True)
    monitor.register_hooks(model.hooks)
    model.add_module('physics_monitor', monitor)

    # Después del forward pass, leer métricas:
    print(monitor.energy_drift)   # float — drift relativo H(t)/H(0) - 1
    print(monitor.mean_KE)        # float — energía cinética media
"""
import torch
import torch.nn as nn
from typing import Optional, Any, Dict, List
from ..models.hooks import Plugin, HookManager


class PhysicsMonitorPlugin(Plugin):
    """
    Plugin de diagnóstico para conservación de energía hamiltoniana.

    Se engacha en `on_timestep_end` para recolectar (x, v) de cada step,
    y calcula al final del batch:
      - energy_drift: |H(T) - H(0)| / |H(0)|  (debería ser ~0 para integradores simplécticos)
      - mean_KE: energía cinética media del batch  T = (1/2) g_ij v^i v^j
      - mean_speed: norma media de v
    
    Si la geometría implementa `compute_kinetic_energy(x, v)`, lo usa.
    De lo contrario, fallback a T = 0.5 * ||v||² (euclídeo).
    
    NO impacta el gradiente: toda la lógica de monitoreo corre bajo torch.no_grad().
    """

    def __init__(self, geometry: Optional[nn.Module] = None, enabled: bool = True,
                 window: int = 64):
        """
        Args:
            geometry: Objeto de geometría GFN (para T Riemanniana). Puede ser None.
            enabled:  Si False, el plugin es un no-op (sin overhead).
            window:   Máximo de steps almacenados por batch para el drift.
        """
        super().__init__()
        self.geometry = geometry
        self.enabled = enabled
        self.window = window

        # Resultados — accesibles después del forward pass
        self.energy_drift: float = 0.0
        self.mean_KE: float = 0.0
        self.mean_speed: float = 0.0
        self.H_history: List[float] = []

        # Buffer interno por batch
        self._x_buf: List[torch.Tensor] = []
        self._v_buf: List[torch.Tensor] = []

    def register_hooks(self, manager: HookManager):
        if not self.enabled:
            return
        manager.register("on_batch_start", self._on_batch_start)
        manager.register("on_timestep_end", self._on_timestep_end)
        manager.register("on_batch_end", self._on_batch_end)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_batch_start(self, **kwargs):
        self._x_buf.clear()
        self._v_buf.clear()

    def _on_timestep_end(self, x: torch.Tensor, v: torch.Tensor, **kwargs):
        """Recolecta snapshot (x, v) de cada timestep (sin gradiente)."""
        if len(self._x_buf) < self.window:
            with torch.no_grad():
                self._x_buf.append(x.detach().clone())
                self._v_buf.append(v.detach().clone())

    def _on_batch_end(self, **kwargs):
        """Calcula todas las métricas al final del batch."""
        if not self._x_buf:
            return
        with torch.no_grad():
            H_vals = []
            KE_vals = []
            speed_vals = []

            for x, v in zip(self._x_buf, self._v_buf):
                # Energía cinética — Riemanniana si la geometría lo soporta
                if self.geometry is not None and hasattr(self.geometry, 'compute_kinetic_energy'):
                    # T = (1/2) Σ_i g_ii v_i²
                    # x puede ser [B, H, D] — geometry espera eso
                    T = self.geometry.compute_kinetic_energy(x.reshape(x.shape[0], -1),
                                                              v.reshape(v.shape[0], -1))
                else:
                    # Fallback euclídeo: T = (1/2)||v||²
                    T = 0.5 * v.reshape(v.shape[0], -1).pow(2).sum(dim=-1)

                # Energía potencial — usa geometry.compute_potential_energy si existe
                if self.geometry is not None and hasattr(self.geometry, 'compute_potential_energy'):
                    V = self.geometry.compute_potential_energy(x.reshape(x.shape[0], -1))
                else:
                    V = torch.zeros_like(T)

                H = (T + V).mean().item()
                H_vals.append(H)
                KE_vals.append(T.mean().item())
                speed_vals.append(v.norm(dim=-1).mean().item())

            # Energy drift: |H(t) - H(0)| / (|H(0)| + eps)
            H0 = abs(H_vals[0]) + 1e-8
            self.energy_drift = max(abs(h - H_vals[0]) / H0 for h in H_vals)
            self.mean_KE = sum(KE_vals) / len(KE_vals)
            self.mean_speed = sum(speed_vals) / len(speed_vals)
            self.H_history = H_vals

    # ── API pública ────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, float]:
        """Retorna las métricas del último batch como dict."""
        return {
            "energy_drift": self.energy_drift,
            "mean_KE": self.mean_KE,
            "mean_speed": self.mean_speed,
        }

    def is_unstable(self, drift_threshold: float = 100.0) -> bool:
        """
        True si el drift de energía supera el umbral — señal de inestabilidad NUMÉRICA.

        IMPORTANTE: Durante training con fuerzas externas (imágenes, audio, texto),
        el drift naturalmente es alto porque las fuerzas inyectan energía al sistema.
        Esto es correcto y esperado — el manifold SE MUEVE gracias a las fuerzas.

        Guía de interpretación:
          drift < 1:    Sistema en equilibrio o fuerzas débiles
          drift 1-50:   Training normal con fuerzas moderadas (rango saludable)
          drift 50-100: Fuerzas muy fuertes o dt grande — monitorear
          drift > 100:  Posible inestabilidad numérica — revisar impulse_scale y dt
        """
        return self.energy_drift > drift_threshold



__all__ = ['PhysicsMonitorPlugin']
