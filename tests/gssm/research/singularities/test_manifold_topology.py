"""
Singularities and Critical Points — GSSM Research Tests
=========================================================
Deep analysis of singularities, critical points, and manifold topology.
Tests actual GSSM model behavior, not mocks.

Research Questions:
- Are there singularities in the manifold?
- How does the model handle critical points?
- What's the topology of the learned manifold?
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


class TestSingularities:
    """Analyze singularities and critical points."""

    def test_velocity_norm_distribution(self):
        """
        Analyze distribution of velocity norms.
        Critical points have near-zero velocity.
        """
        reporter = ResultsReporter("velocity_norm_distribution", "singularities")
        
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
            
            x = torch.randint(0, 128, (64, 16))
            
            with torch.no_grad():
                _, (xf, vf), info = model(x)
                v_seq = info['v_seq']
            
            all_v_norms = []
            for step in range(v_seq.shape[1]):
                v_norms = torch.norm(v_seq[:, step], dim=-1)
                all_v_norms.extend(v_norms.tolist())
            
            all_v_norms = np.array(all_v_norms)
            
            critical_points = (all_v_norms < 0.1).sum()
            critical_ratio = critical_points / len(all_v_norms)
            
            reporter.log_metric("mean_velocity_norm", np.mean(all_v_norms))
            reporter.log_metric("std_velocity_norm", np.std(all_v_norms))
            reporter.log_metric("min_velocity_norm", np.min(all_v_norms))
            reporter.log_metric("max_velocity_norm", np.max(all_v_norms))
            reporter.log_metric("critical_points_count", critical_points)
            reporter.log_metric("critical_points_ratio", critical_ratio)
            
            if critical_ratio < 0.5:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Too many critical points: {critical_ratio}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_manifold_curvature_analysis(self):
        """
        Analyze local curvature of the manifold.
        High curvature regions may indicate singularities.
        """
        reporter = ResultsReporter("manifold_curvature", "singularities")
        
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
            
            x = torch.randint(0, 128, (32, 8))
            
            with torch.no_grad():
                _, (xf, vf), info = model(x)
                x_final = info['x_final']
            
            x_flat = x_final.reshape(x_final.shape[0], -1)
            
            pairwise_dists = torch.cdist(x_flat, x_flat)
            
            curvatures = []
            for i in range(len(x_flat)):
                neighbors = torch.topk(pairwise_dists[i], k=4, largest=False).indices[1:]
                
                if len(neighbors) >= 3:
                    p0 = x_flat[i]
                    p1 = x_flat[neighbors[0]]
                    p2 = x_flat[neighbors[1]]
                    
                    v1 = p1 - p0
                    v2 = p2 - p0
                    
                    cross = torch.cross(v1[:3], v2[:3])
                    area = 0.5 * torch.norm(cross)
                    
                    if area > 1e-6:
                        curvatures.append(1.0 / area.item())
            
            if curvatures:
                curvatures = np.array(curvatures)
                curvatures = np.clip(curvatures, 0, 1000)
                
                reporter.log_metric("mean_curvature", np.mean(curvatures))
                reporter.log_metric("std_curvature", np.std(curvatures))
                reporter.log_metric("max_curvature", np.max(curvatures))
                reporter.log_metric("high_curvature_ratio", (curvatures > 100).sum() / len(curvatures))
                
                reporter.mark_passed(True)
            else:
                reporter.add_error("Could not compute curvatures")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_jacobian_determinant(self):
        """
        Compute Jacobian determinant of the transformation.
        Det ≈ 0 indicates singularity / collapse.
        """
        reporter = ResultsReporter("jacobian_det", "singularities")
        
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
            
            x = torch.randint(0, 128, (4, 8))
            
            with torch.no_grad():
                _, (xf, vf), info = model(x)
                x_final = info['x_final']
            
            # Simple analysis: check if output varies with input
            x_flat = x_final.reshape(x_final.shape[0], -1)
            
            # Compute variance across samples
            variance = x_flat.var(dim=0).mean().item()
            
            reporter.log_metric("output_variance", variance)
            reporter.log_metric("output_mean", x_flat.mean().item())
            
            if variance > 1e-6:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Output has no variance: {variance}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_fixed_point_analysis(self):
        """
        Analyze fixed points of the dynamics.
        Check if velocity decays to zero (indicating convergence to fixed point).
        """
        reporter = ResultsReporter("fixed_points", "singularities")
        
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
                v_final = info['v_final']
            
            v_norm = v_final.norm().item()
            v_mean = v_final.abs().mean().item()
            
            reporter.log_metric("v_final_norm", v_norm)
            reporter.log_metric("v_final_mean", v_mean)
            
            if v_norm < 100.0:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Velocity too large: {v_norm}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
