"""
Energy and Hamiltonian Analysis — GSSM Research Tests
=======================================================
Deep analysis of energy conservation and Hamiltonian dynamics.
Tests actual GSSM model behavior, not mocks.

Research Questions:
- Is energy conserved during forward pass?
- What's the Hamiltonian trajectory behavior?
- Phase space volume preservation?
"""

import torch
import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from test_reporter import ResultsReporter
import gfn


class TestEnergyHamiltonian:
    """Analyze energy and Hamiltonian dynamics."""

    def test_energy_trajectory_conservation(self):
        """
        Track energy (kinetic + potential) through trajectory.
        Tests: Is energy conserved in the manifold dynamics?
        """
        reporter = ResultsReporter("energy_conservation", "energy_hamiltonian")
        
        try:
            model = gfn.create(
                'gssm',
                vocab_size=128,
                dim=64,
                heads=4,
                topology_type='torus',
                holographic=False,
                initial_spread=0.0
            )
            model.eval()
            
            x = torch.randint(0, 128, (8, 16))
            
            with torch.no_grad():
                _, (xf, vf), info = model(x)
                x_seq = info['x_seq']
                v_seq = info['v_seq']
            
            energies = []
            for step in range(x_seq.shape[1]):
                x_step = x_seq[:, step]
                v_step = v_seq[:, step]
                
                kinetic = 0.5 * torch.sum(v_step ** 2, dim=-1)
                potential = 0.5 * torch.sum(x_step ** 2, dim=-1)
                total_energy = kinetic + potential
                
                energies.append(total_energy.mean().item())
            
            energies = np.array(energies)
            energy_variance = np.var(energies)
            energy_drift = abs(energies[-1] - energies[0])
            
            reporter.log_metric("initial_energy", energies[0])
            reporter.log_metric("final_energy", energies[-1])
            reporter.log_metric("energy_variance", energy_variance)
            reporter.log_metric("energy_drift", energy_drift)
            reporter.log_metric("energy_drift_ratio", energy_drift / (abs(energies[0]) + 1e-8))
            
            if energy_variance < 100.0:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Large energy variance: {energy_variance}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_phase_space_volume(self):
        """
        Measure phase space volume preservation.
        Tests: Is the symplectic structure preserved?
        """
        reporter = ResultsReporter("phase_volume", "energy_hamiltonian")
        
        try:
            model = gfn.create(
                'gssm',
                vocab_size=128,
                dim=64,
                heads=4,
                topology_type='torus',
                holographic=False,
                initial_spread=0.0
            )
            model.eval()
            
            num_samples = 100
            x = torch.randint(0, 128, (num_samples, 8)).float()
            
            with torch.no_grad():
                _, (xf, vf), info = model(x.long())
                x_final = info['x_final']
                v_final = info['v_final']
            
            initial_spread_x = x.std().item()
            initial_spread_v = vf.reshape(vf.shape[0], -1).std().item()
            final_spread_x = x_final.reshape(x_final.shape[0], -1).std().item()
            final_spread_v = v_final.reshape(v_final.shape[0], -1).std().item()
            
            volume_ratio = (final_spread_x * final_spread_v) / (initial_spread_x * initial_spread_v + 1e-8)
            
            reporter.log_metric("initial_x_spread", initial_spread_x)
            reporter.log_metric("final_x_spread", final_spread_x)
            reporter.log_metric("initial_v_spread", initial_spread_v)
            reporter.log_metric("final_v_spread", final_spread_v)
            reporter.log_metric("volume_ratio", volume_ratio)
            
            if 0.5 < volume_ratio < 2.0:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Volume not preserved: {volume_ratio}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_hamiltonian_trajectory_stability(self):
        """
        Analyze stability of Hamiltonian trajectories.
        Tests: Do trajectories remain stable or become chaotic?
        """
        reporter = ResultsReporter("hamiltonian_stability", "energy_hamiltonian")
        
        try:
            model = gfn.create(
                'gssm',
                vocab_size=128,
                dim=32,
                heads=2,
                topology_type='torus',
                holographic=False,
                initial_spread=0.0
            )
            model.eval()
            
            x1 = torch.randint(0, 128, (4, 8)).float()
            x2 = x1 + torch.randn_like(x1) * 0.01
            
            with torch.no_grad():
                _, (xf1, vf1), _ = model(x1.long())
                _, (xf2, vf2), _ = model(x2.long())
            
            dist_initial = torch.norm(x1 - x2).item()
            dist_final = torch.norm(xf1 - xf2).item()
            
            lyapunov_approx = np.log(dist_final / (dist_initial + 1e-8))
            
            reporter.log_metric("initial_distance", dist_initial)
            reporter.log_metric("final_distance", dist_final)
            reporter.log_metric("lyapunov_exponent_approx", lyapunov_approx)
            
            if lyapunov_approx < 2.0:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Trajectories diverge: {lyapunov_approx}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_momentum_conservation(self):
        """
        Check if momentum is conserved in the dynamics.
        Tests: Does total momentum stay constant?
        """
        reporter = ResultsReporter("momentum_conservation", "energy_hamiltonian")
        
        try:
            model = gfn.create(
                'gssm',
                vocab_size=128,
                dim=64,
                heads=4,
                topology_type='torus',
                holographic=False,
                initial_spread=0.0
            )
            model.eval()
            
            x = torch.randint(0, 128, (8, 16))
            
            with torch.no_grad():
                _, (xf, vf), info = model(x)
                v_seq = info['v_seq']
            
            total_momentum_initial = v_seq[:, 0].sum(dim=-1).mean().item()
            total_momentum_final = vf.sum(dim=-1).mean().item()
            
            momentum_change = abs(total_momentum_final - total_momentum_initial)
            
            reporter.log_metric("initial_total_momentum", total_momentum_initial)
            reporter.log_metric("final_total_momentum", total_momentum_final)
            reporter.log_metric("momentum_change", momentum_change)
            
            if momentum_change < 10.0:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Momentum not conserved: {momentum_change}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
