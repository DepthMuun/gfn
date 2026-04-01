"""
gfn/physics/hamiltonian.py — GFN V5
Ported from: gfn_old/nn/physics/dynamics/hamiltonian.py

HamiltonianTrajectorySolver: generates physically consistent trajectories
by calculating the total system energy H = T + V.
"""
import torch
from typing import Optional, Tuple, Any


class HamiltonianTrajectorySolver:
    """
    Hamiltonian dynamics solver in phase space (x, p).

    Integrates the equations of motion to generate physically
    consistent trajectories with energy conservation (subject to the
    numerical drift of the chosen integrator).

    Compatible with any GFN V5 integrator that accepts
    `step(x, v, force, dt)` → Dict["x", "v"].
    """

    def __init__(self, geometry: Any, integrator: Any, dt: float = 0.01):
        """
        Args:
            geometry: Geometry object (should have `compute_kinetic_energy` if possible).
            integrator: GFN V5 integrator (LeapfrogIntegrator, etc.).
            dt: Fixed time step for the simulation.
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
        Generates a Hamiltonian trajectory of N steps.

        Args:
            x0: Initial position [B, ...] or [B, H, D]
            v0: Initial velocity — same shape as x0
            steps: Number of integration steps
            force: Optional external force

        Returns:
            (x_history, v_history): tensors of shape [steps+1, B, ...]
        """
        x_history = [x0]
        v_history = [v0]

        curr_x, curr_v = x0, v0

        for _ in range(steps):
            out = self.integrator.step(
                curr_x, curr_v, force=force, dt=self.dt, **kwargs
            )
            curr_x, curr_v = out["x"], out["v"]
            x_history.append(curr_x)
            v_history.append(curr_v)

        return torch.stack(x_history), torch.stack(v_history)

    def compute_hamiltonian(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Calculates the total energy of the system: H = T + V.

        T (kinetic): uses `geometry.compute_kinetic_energy` if available,
                     otherwise uses the standard L2 norm.
        V (potential): uses `geometry.compute_potential_energy` if available.
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
        Calculates the relative energy drift along the trajectory.

        A low drift indicates that the integrator is symplectically accurate.
        Useful for debugging metrics and integrator verification.

        Returns:
            drift (scalar): mean drift normalized by the initial energy.
        """
        H_t = torch.stack([
            self.compute_hamiltonian(x_hist[i], v_hist[i])
            for i in range(x_hist.shape[0])
        ])
        H0 = H_t[0].unsqueeze(0)
        drift = torch.abs(H_t - H0) / (torch.abs(H0) + 1e-8)
        return drift.mean()


__all__ = ['HamiltonianTrajectorySolver']
