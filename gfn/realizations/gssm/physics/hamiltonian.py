"""
gfn/physics/hamiltonian.py — GFN V5
Portado desde: gfn_old/nn/physics/dynamics/hamiltonian.py

HamiltonianTrajectorySolver: genera trayectorias físicamente consistentes
calculando la energía total del sistema H = T + V.
"""
import torch
from typing import Optional, Tuple, Any


class HamiltonianTrajectorySolver:
    """
    Solver de dinámica Hamiltoniana en el espacio de fases (x, p).

    Integra las ecuaciones de movimiento para generar trayectorias
    físicamente consistentes con conservación de energía (sujeto al
    drift numérico del integrador elegido).

    Compatible con cualquier integrador de GFN V5 que acepte
    `step(x, v, force, dt)` → Dict["x", "v"].
    """

    def __init__(self, geometry: Any, integrator: Any, dt: float = 0.01):
        """
        Args:
            geometry: Objeto de geometría (debe tener `compute_kinetic_energy` si es posible).
            integrator: Integrador de GFN V5 (LeapfrogIntegrator, etc.).
            dt: Paso de tiempo fijo para la simulación.
        """
        self.geometry = geometry
        self.integrator = integrator
        self.dt = dt

    def solve(
        self,
        x0: torch.Tensor,
        v0: torch.Tensor,
        steps: int,
        force: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Genera una trayectorias Hamiltoniana de N pasos.

        Args:
            x0: Posición inicial [B, ...] o [B, H, D]
            v0: Velocidad inicial — misma forma que x0
            steps: Número de pasos de integración
            force: Fuerza externa opcional

        Returns:
            (x_history, v_history): tensores de forma [steps+1, B, ...]
        """
        x_history = [x0]
        v_history = [v0]

        curr_x, curr_v = x0, v0

        for _ in range(steps):
            result = self.integrator.step(
                curr_x, curr_v, force=force, dt=self.dt, **kwargs
            )
            curr_x = result["x"]
            curr_v = result["v"]
            x_history.append(curr_x)
            v_history.append(curr_v)

        return torch.stack(x_history), torch.stack(v_history)

    def compute_hamiltonian(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Calcula la energía total del sistema: H = T + V.

        T (cinética): usa `geometry.compute_kinetic_energy` si existe,
                      si no usa la norma L2 estándar.
        V (potencial): usa `geometry.compute_potential_energy` si existe.
        """
        if hasattr(self.geometry, 'compute_kinetic_energy'):
            kinetic = self.geometry.compute_kinetic_energy(x, v)
        else:
            kinetic = 0.5 * torch.sum(v * v, dim=-1)

        potential_fn = getattr(
            self.geometry, 'compute_potential_energy',
            lambda x_: torch.zeros_like(kinetic)
        )
        potential = potential_fn(x)

        return kinetic + potential

    def energy_drift(self, x_hist: torch.Tensor, v_hist: torch.Tensor) -> torch.Tensor:
        """
        Calcula el drift de energía relativo a lo largo de la trayectoria.

        Un drift bajo indica que el integrador es simplécticamente preciso.
        Útil para métricas de debugging y verificación de integradores.

        Returns:
            drift (escalar): drift medio normalizado por la energía inicial.
        """
        H_t = torch.stack([
            self.compute_hamiltonian(x_hist[i], v_hist[i])
            for i in range(x_hist.shape[0])
        ])
        H0 = H_t[0].unsqueeze(0)
        drift = torch.abs(H_t - H0) / (torch.abs(H0) + 1e-8)
        return drift.mean()


__all__ = ['HamiltonianTrajectorySolver']
