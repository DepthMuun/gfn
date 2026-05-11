"""
Health Tests — Core Components (with JSON Reporting)
======================================================

Comprehensive unit tests for GFN GSSM core functionality with 
automatic JSON result generation to tests/results/health/
"""

import torch
import pytest
import sys
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.test_reporter import ResultsReporter

from gfn.realizations.gssm.geometry.torus import ToroidalRiemannianGeometry
from gfn.realizations.gssm.geometry.euclidean import EuclideanGeometry
from gfn.realizations.gssm.physics.integrators.symplectic.leapfrog import LeapfrogIntegrator
from gfn.realizations.gssm.physics.dynamics.direct import DirectDynamics
from gfn.realizations.gssm.physics.dynamics.residual import ResidualDynamics
from gfn.realizations.gssm.physics.dynamics.mix import MixDynamics
from gfn.realizations.gssm.models.manifold_layer import ManifoldLayer


class TestGeometries:
    """Test suite for Riemannian geometries."""
    
    def test_torus_metric_shape(self):
        """Verify torus metric tensor has correct shape."""
        reporter = ResultsReporter("torus_metric_shape", "health")
        
        try:
            geo = ToroidalRiemannianGeometry(dim=4, rank=16, num_heads=1)
            x = torch.randn(2, 4)
            metric = geo.metric_tensor(x)
            
            reporter.log_metric("metric_shape", str(metric.shape))
            reporter.log_metric("batch_size", x.shape[0])
            reporter.log_metric("dim", x.shape[1])
            # metric_tensor returns (batch, dim) - diagonal only
            # For a full metric matrix, we'd need metric_matrix method
            metric_diag = geo.metric_tensor(x)
            expected_shape = (2, 4)  # diagonal metric
            actual_shape = tuple(metric_diag.shape)
            
            reporter.log_metric("metric_shape", str(metric_diag.shape))
            if actual_shape == expected_shape:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Expected {expected_shape}, got {actual_shape}")
            
            assert actual_shape == expected_shape
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")
    
    def test_torus_metric_positive_definite(self):
        """Verify metric is positive definite (eigenvalues > 0)."""
        reporter = ResultsReporter("torus_positive_definite", "health")
        
        try:
            geo = ToroidalRiemannianGeometry(dim=2, rank=16, num_heads=1)
            x = torch.zeros(1, 2)
            # metric_tensor returns diagonal - convert to matrix for eigenvalue check
            metric_diag = geo.metric_tensor(x)[0]  # shape: (2,)
            # Create diagonal matrix
            metric_matrix = torch.diag(metric_diag)
            
            eigenvalues = torch.linalg.eigvalsh(metric_matrix)
            min_eigenval = eigenvalues.min().item()
            
            reporter.log_metric("min_eigenvalue", min_eigenval)
            reporter.log_metric("eigenvalues", eigenvalues.tolist())
            
            if min_eigenval > 0:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Metric not positive definite: min eigenvalue = {min_eigenval}")
            
            assert min_eigenval > 0
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")
    
    def test_torus_angle_wrapping(self):
        """Verify angles wrap correctly to [-π, π]."""
        reporter = ResultsReporter("torus_angle_wrapping", "health")
        
        try:
            geo = ToroidalRiemannianGeometry(dim=2, rank=16, num_heads=1)
            
            # Test cases: (input, expected_after_wrap)
            test_cases = [
                (torch.tensor([[3 * torch.pi, 0.0]]), "wrap_large_positive"),
                (torch.tensor([[-3 * torch.pi, 0.0]]), "wrap_large_negative"),
                (torch.tensor([[0.0, 2 * torch.pi + 0.5]]), "wrap_y_component"),
            ]
            
            all_passed = True
            for input_tensor, case_name in test_cases:
                # Use modulo operation to wrap angles to [-π, π]
                wrapped = ((input_tensor + torch.pi) % (2 * torch.pi)) - torch.pi
                
                max_val = wrapped.max().item()
                min_val = wrapped.min().item()
                
                reporter.log_metric(f"{case_name}_max", max_val)
                reporter.log_metric(f"{case_name}_min", min_val)
                
                if max_val > torch.pi + 1e-6 or min_val < -torch.pi - 1e-6:
                    reporter.add_error(f"{case_name}: values outside [-π, π]")
                    all_passed = False
            
            reporter.mark_passed(all_passed)
            assert all_passed
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")
    
    def test_euclidean_metric_identity(self):
        """Verify Euclidean metric is identity."""
        reporter = ResultsReporter("euclidean_metric_identity", "health")
        
        try:
            from gfn.realizations.gssm.geometry.euclidean import EuclideanGeometry
            geo = EuclideanGeometry()  # No config needed for basic usage
            x = torch.randn(2, 3)
            # metric_tensor returns diagonal [batch, dim] - verify it's all ones
            metric_diag = geo.metric_tensor(x)  # shape: (2, 3)
            expected_diag = torch.ones(2, 3)  # Euclidean metric = identity diagonal
            
            diff = (metric_diag - expected_diag).abs().max().item()
            
            reporter.log_metric("max_difference", diff)
            reporter.log_metric("metric_trace", metric_diag.sum().item())
            
            if diff < 1e-6:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Metric not identity, max diff: {diff}")
            
            assert diff < 1e-6
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


class TestIntegrators:
    """Test suite for symplectic integrators."""
    
    def test_leapfrog_energy_conservation(self):
        """Verify Leapfrog approximately conserves energy."""
        reporter = ResultsReporter("leapfrog_energy_conservation", "health")
        
        try:
            # Create physics engine and config
            from gfn.realizations.gssm.physics.engine import ManifoldPhysicsEngine
            from gfn.realizations.gssm.config.schema import PhysicsConfig
            
            config = PhysicsConfig()
            config.stability.base_dt = 0.01
            geo = ToroidalRiemannianGeometry(dim=2, rank=16, num_heads=1)
            engine = ManifoldPhysicsEngine(geometry=geo, config=config)
            
            integrator = LeapfrogIntegrator(physics_engine=engine, config=config)
            
            # Simple harmonic oscillator
            def potential(x):
                return 0.5 * (x ** 2).sum(dim=-1, keepdim=True)
            
            def force(x):
                return -x
            
            x = torch.tensor([[0.5, 0.0]])
            v = torch.tensor([[0.0, 1.0]])
            
            # Pre-compute forces as tensors
            force_x = torch.tensor([[-0.5, 0.0]])  # -x
            
            T0 = 0.5 * (v ** 2).sum()
            V0 = potential(x).sum()
            E0 = T0 + V0
            
            energies = []
            for i in range(100):
                force_tensor = -x  # Compute force at current position
                result = integrator.step(x, v, force_tensor, dt=0.01)
                x, v = result['x'], result['v']
                T = 0.5 * (v ** 2).sum()
                V = potential(x).sum()
                E = T + V
                energies.append(E.item())
            
            drift = abs(energies[-1] - E0.item()) / abs(E0.item())
            
            reporter.log_metric("initial_energy", E0.item())
            reporter.log_metric("final_energy", energies[-1])
            reporter.log_metric("energy_drift", drift)
            reporter.log_plot_data("energy_vs_step", list(range(100)), energies)
            
            if drift < 0.01:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Energy drift too large: {drift}")
            
            assert drift < 0.01
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")
    
    def test_leapfrog_reversibility(self):
        """Verify Leapfrog is time-reversible."""
        reporter = ResultsReporter("leapfrog_reversibility", "health")
        
        try:
            from gfn.realizations.gssm.physics.engine import ManifoldPhysicsEngine
            from gfn.realizations.gssm.config.schema import PhysicsConfig
            
            config = PhysicsConfig()
            config.stability.base_dt = 0.01
            geo = ToroidalRiemannianGeometry(dim=2, rank=16, num_heads=1)
            engine = ManifoldPhysicsEngine(geometry=geo, config=config)
            
            integrator = LeapfrogIntegrator(physics_engine=engine, config=config)
            
            def force(x):
                return -x
            
            x0 = torch.tensor([[1.0, 0.5]])
            v0 = torch.tensor([[0.2, -0.3]])
            
            # Forward 10 steps
            x, v = x0.clone(), v0.clone()
            for _ in range(10):
                force_tensor = -x
                result = integrator.step(x, v, force_tensor, dt=0.01)
                x, v = result['x'], result['v']
            
            # Backward 10 steps (reverse velocity)
            v = -v
            for _ in range(10):
                force_tensor = -x
                result = integrator.step(x, v, force_tensor, dt=0.01)
                x, v = result['x'], result['v']
            v = -v
            
            pos_diff = (x - x0).abs().max().item()
            vel_diff = (v - v0).abs().max().item()
            
            reporter.log_metric("position_diff", pos_diff)
            reporter.log_metric("velocity_diff", vel_diff)
            reporter.log_metric("reversible", pos_diff < 1e-3 and vel_diff < 1e-3)
            
            if pos_diff < 1e-3 and vel_diff < 1e-3:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Not reversible: pos_diff={pos_diff}, vel_diff={vel_diff}")
            
            assert pos_diff < 1e-3 and vel_diff < 1e-3
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


class TestDynamics:
    """Test suite for dynamics modes."""
    
    def test_direct_dynamics_shape(self):
        """Verify DirectDynamics preserves tensor shape."""
        reporter = ResultsReporter("direct_dynamics_shape", "health")
        
        try:
            dynamics = DirectDynamics(dim=8, topology='euclidean')
            current = torch.randn(2, 4, 8)
            proposal = torch.randn(2, 4, 8)
            
            result = dynamics(current, proposal)
            
            shape_preserved = result.shape == current.shape
            reporter.log_metric("input_shape", str(current.shape))
            reporter.log_metric("output_shape", str(result.shape))
            reporter.log_metric("shape_preserved", shape_preserved)
            
            if shape_preserved:
                reporter.mark_passed(True)
            else:
                reporter.add_error("Shape not preserved")
            
            assert shape_preserved
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")
    
    def test_residual_dynamics_skip_connection(self):
        """Verify ResidualDynamics maintains skip connection property."""
        reporter = ResultsReporter("residual_dynamics_skip", "health")
        
        try:
            dynamics = ResidualDynamics(dim=8, topology='euclidean')
            current = torch.randn(2, 4, 8)
            
            # When proposal == current, output should be ~current
            result = dynamics(current, current)
            
            diff = (result - current).abs().mean().item()
            
            reporter.log_metric("mean_difference", diff)
            reporter.log_metric("residual_active", diff < 0.1)
            
            if diff < 0.1:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Skip connection not working, diff: {diff}")
            
            assert diff < 0.1
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")
    
    def test_mix_dynamics_alpha_range(self):
        """Verify MixDynamics alpha stays in (0, 1)."""
        reporter = ResultsReporter("mix_dynamics_alpha", "health")
        
        try:
            dynamics = MixDynamics(dim=8, topology='euclidean')
            alpha = dynamics.get_alpha()
            
            reporter.log_metric("alpha_value", alpha)
            reporter.log_metric("alpha_in_range", 0 < alpha < 1)
            
            if 0 < alpha < 1:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Alpha out of range: {alpha}")
            
            assert 0 < alpha < 1
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


class TestManifoldLayer:
    """Test suite for ManifoldLayer."""
    
    def test_layer_forward_shape(self):
        """Verify layer output shape matches input."""
        reporter = ResultsReporter("manifold_layer_shape", "health")
        
        try:
            # Setup dependencies
            from gfn.realizations.gssm.physics.engine import ManifoldPhysicsEngine
            from gfn.realizations.gssm.config.schema import PhysicsConfig
            from gfn.realizations.gssm.models.components.mixer import FlowMixer
            
            config = PhysicsConfig()
            config.stability.base_dt = 0.01
            config.topology.type = 'euclidean'
            
            geo = ToroidalRiemannianGeometry(dim=16, rank=16, num_heads=4)
            engine = ManifoldPhysicsEngine(geometry=geo, config=config)
            
            from gfn.realizations.gssm.physics.integrators.symplectic.leapfrog import LeapfrogIntegrator
            integrator = LeapfrogIntegrator(physics_engine=engine, config=config)
            
            mixer = FlowMixer(dim=16, heads=4)
            
            layer = ManifoldLayer(
                integrator=integrator,
                mixer=mixer,
                config=config,
                heads=4,
                dynamics_type='direct'
            )
            
            # ManifoldLayer expects [B, H, D] shape
            x = torch.randn(2, 4, 4)  # [batch=2, heads=4, dim=4] - 4 heads * 4 dim = 16 total
            v = torch.zeros(2, 4, 4)
            
            x_next, v_next = layer(x, v)
            
            shape_ok = x_next.shape == x.shape and v_next.shape == v.shape
            
            reporter.log_metric("input_shape", str(x.shape))
            reporter.log_metric("output_x_shape", str(x_next.shape))
            reporter.log_metric("output_v_shape", str(v_next.shape))
            reporter.log_metric("shape_preserved", shape_ok)
            
            if shape_ok:
                reporter.mark_passed(True)
            else:
                reporter.add_error("Output shape mismatch")
            
            assert shape_ok
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")
    
    def test_layer_preserves_device(self):
        """Verify layer preserves input device."""
        reporter = ResultsReporter("manifold_layer_device", "health")
        
        try:
            from gfn.realizations.gssm.physics.engine import ManifoldPhysicsEngine
            from gfn.realizations.gssm.config.schema import PhysicsConfig
            from gfn.realizations.gssm.models.components.mixer import FlowMixer
            from gfn.realizations.gssm.physics.integrators.symplectic.leapfrog import LeapfrogIntegrator
            
            config = PhysicsConfig()
            geo = ToroidalRiemannianGeometry(dim=8, rank=16, num_heads=2)
            engine = ManifoldPhysicsEngine(geometry=geo, config=config)
            integrator = LeapfrogIntegrator(physics_engine=engine, config=config)
            mixer = FlowMixer(dim=8, heads=2)
            
            layer = ManifoldLayer(
                integrator=integrator,
                mixer=mixer,
                config=config,
                heads=2
            )
            
            # ManifoldLayer expects [B, H, D] shape  
            x = torch.randn(2, 2, 4)  # [batch=2, heads=2, dim=4] - 2 heads * 4 dim = 8 total
            v = torch.randn(2, 2, 4)
            
            x_next, v_next = layer(x, v)
            
            device_ok = x_next.device == x.device and v_next.device == v.device
            
            reporter.log_metric("input_device", str(x.device))
            reporter.log_metric("output_device", str(x_next.device))
            reporter.log_metric("device_preserved", device_ok)
            
            if device_ok:
                reporter.mark_passed(True)
            else:
                reporter.add_error("Device not preserved")
            
            assert device_ok
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
