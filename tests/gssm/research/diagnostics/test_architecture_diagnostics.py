"""
Diagnostic Tests for GSSM Architecture Issues
===============================================
Tests specifically designed to diagnose and fix:
1. Condition number / rank deficiency
2. Vanishing gradients
3. Integrator effects
4. dt effects

These tests help identify root causes and optimal configurations.
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


class TestConditionNumberDiagnosis:
    """Diagnose the rank deficiency / condition number issue."""

    def test_activation_rank_analysis(self):
        """
        Detailed analysis of activation matrix rank.
        Goal: Understand why condition numbers are in the millions.
        """
        reporter = ResultsReporter("condition_number_diagnosis", "diagnostics")
        
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
            
            x = torch.randint(0, 128, (32, 16))
            
            with torch.no_grad():
                _, (xf, vf), info = model(x)
                x_seq = info['x_seq']
            
            for layer_idx in [0, 4, 8, 15]:
                if layer_idx >= x_seq.shape[1]:
                    continue
                    
                x_layer = x_seq[:, layer_idx].reshape(x_seq.shape[0], -1)
                
                # Full SVD
                _, s, _ = torch.svd(x_layer)
                
                # Compute rank with different thresholds
                rank_tol_1e3 = (s > 1e-3).sum().item()
                rank_tol_1e4 = (s > 1e-4).sum().item()
                rank_tol_1e5 = (s > 1e-5).sum().item()
                rank_tol_1e6 = (s > 1e-6).sum().item()
                
                # Variance explained
                var_explained_90 = (torch.cumsum(s**2, dim=0) / (s**2).sum() < 0.90).sum().item() + 1
                var_explained_99 = (torch.cumsum(s**2, dim=0) / (s**2).sum() < 0.99).sum().item() + 1
                var_explained_999 = (torch.cumsum(s**2, dim=0) / (s**2).sum() < 0.999).sum().item() + 1
                
                # Singular value distribution
                s_normalized = s / s[0]
                entropy_s = -(s_normalized * torch.log(s_normalized + 1e-10)).sum()
                
                reporter.log_metric(f"layer_{layer_idx}_singular_0", s[0].item())
                reporter.log_metric(f"layer_{layer_idx}_singular_last", s[-1].item())
                reporter.log_metric(f"layer_{layer_idx}_rank_1e-3", rank_tol_1e3)
                reporter.log_metric(f"layer_{layer_idx}_rank_1e-4", rank_tol_1e4)
                reporter.log_metric(f"layer_{layer_idx}_rank_1e-5", rank_tol_1e5)
                reporter.log_metric(f"layer_{layer_idx}_rank_1e-6", rank_tol_1e6)
                reporter.log_metric(f"layer_{layer_idx}_var_90", var_explained_90)
                reporter.log_metric(f"layer_{layer_idx}_var_99", var_explained_99)
                reporter.log_metric(f"layer_{layer_idx}_var_999", var_explained_999)
                reporter.log_metric(f"layer_{layer_idx}_singular_entropy", entropy_s.item())
            
            reporter.mark_passed(True)
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_embedding_diversity_analysis(self):
        """
        Check if embeddings have sufficient diversity.
        Low diversity = rank deficiency.
        """
        reporter = ResultsReporter("embedding_diversity", "diagnostics")
        
        try:
            model = gfn.create(
                'gssm',
                vocab_size=128,
                dim=64,
                heads=4,
                topology_type='torus',
                holographic=False,
                initial_spread=0.1
            )
            model.eval()
            
            # Test with different token distributions
            test_cases = [
                ("uniform", torch.randint(0, 128, (64, 8))),
                ("sequential", torch.arange(128).unsqueeze(0).expand(64, -1)[:, :8] % 128),
                ("repeated", torch.ones(64, 8, dtype=torch.long) * 42),
                ("random_normal", torch.randn(64, 8) * 10 + 64),
            ]
            
            for name, x in test_cases:
                with torch.no_grad():
                    _, (xf, vf), info = model(x)
                    x_final = info['x_final']
                
                x_flat = x_final.reshape(x_final.shape[0], -1)
                
                # Compute covariance matrix
                x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
                cov = x_centered.T @ x_centered / x_flat.shape[0]
                
                eigenvalues = torch.linalg.eigvalsh(cov)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                
                if len(eigenvalues) > 0:
                    condition = (eigenvalues[-1] / eigenvalues[0]).item()
                    effective_rank = len(eigenvalues) * (1 - (eigenvalues / eigenvalues.sum()).mean()).item()
                else:
                    condition = 0.0
                    effective_rank = 0.0
                
                reporter.log_metric(f"{name}_condition", condition)
                reporter.log_metric(f"{name}_effective_rank", effective_rank)
                reporter.log_metric(f"{name}_num_nonzero_eigenvalues", len(eigenvalues))
            
            reporter.mark_passed(True)
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


class TestGradientDiagnosis:
    """Diagnose vanishing gradient issue."""

    def test_gradient_by_layer_detailed(self):
        """
        Detailed gradient analysis per layer.
        Goal: Identify exactly which layers have vanishing gradients.
        """
        reporter = ResultsReporter("gradient_by_layer", "diagnostics")
        
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
            model.train()
            
            x = torch.randint(0, 128, (16, 8))
            target = torch.randint(0, 128, (16, 8))
            
            # Multiple forward-backward passes to get statistics
            grad_stats = {f"layer_{i}": [] for i in range(4)}
            
            for trial in range(10):
                model.zero_grad()
                x_trial = torch.randint(0, 128, (16, 8))
                
                logits, (xf, vf), _ = model(x_trial)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 128), target.reshape(-1)
                )
                loss.backward()
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        for i in range(4):
                            if f"layer.{i}." in name:
                                grad_stats[f"layer_{i}"].append(param.grad.norm().item())
            
            for layer_name, grads in grad_stats.items():
                if grads:
                    reporter.log_metric(f"{layer_name}_mean_grad", np.mean(grads))
                    reporter.log_metric(f"{layer_name}_std_grad", np.std(grads))
                    reporter.log_metric(f"{layer_name}_min_grad", np.min(grads))
                    reporter.log_metric(f"{layer_name}_max_grad", np.max(grads))
            
            # Check for vanishing
            all_means = [np.mean(grad_stats[f"layer_{i}"]) for i in range(4) if grad_stats[f"layer_{i}"]]
            if all_means:
                min_mean = min(all_means)
                max_mean = max(all_means)
                ratio = min_mean / (max_mean + 1e-10)
                reporter.log_metric("min_max_ratio", ratio)
                
                if ratio > 0.01:
                    reporter.mark_passed(True)
                else:
                    reporter.add_error(f"Gradient ratio too low: {ratio}")
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_gradient_flow_comparison(self):
        """
        Compare gradient flow with vs without skip connections.
        """
        reporter = ResultsReporter("gradient_flow_comparison", "diagnostics")
        
        try:
            # Standard model
            model_std = gfn.create(
                'gssm',
                vocab_size=128,
                dim=64,
                heads=4,
                topology_type='torus',
                holographic=False,
                initial_spread=0.0
            )
            model_std.train()
            
            x = torch.randint(0, 128, (8, 8))
            target = torch.randint(0, 128, (8, 8))
            
            model_std.zero_grad()
            logits, _, _ = model_std(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), target.reshape(-1))
            loss.backward()
            
            first_layer_grad_std = []
            last_layer_grad_std = []
            for name, param in model_std.named_parameters():
                if param.grad is not None:
                    if "layer.0" in name:
                        first_layer_grad_std.append(param.grad.norm().item())
                    if "layer.3" in name:
                        last_layer_grad_std.append(param.grad.norm().item())
            
            reporter.log_metric("first_layer_grad_std", np.mean(first_layer_grad_std) if first_layer_grad_std else 0)
            reporter.log_metric("last_layer_grad_std", np.mean(last_layer_grad_std) if last_layer_grad_std else 0)
            
            # With larger initial_spread (more exploration)
            model_spread = gfn.create(
                'gssm',
                vocab_size=128,
                dim=64,
                heads=4,
                topology_type='torus',
                holographic=False,
                initial_spread=1.0
            )
            model_spread.train()
            
            model_spread.zero_grad()
            logits, _, _ = model_spread(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), target.reshape(-1))
            loss.backward()
            
            first_layer_grad_spread = []
            last_layer_grad_spread = []
            for name, param in model_spread.named_parameters():
                if param.grad is not None:
                    if "layer.0" in name:
                        first_layer_grad_spread.append(param.grad.norm().item())
                    if "layer.3" in name:
                        last_layer_grad_spread.append(param.grad.norm().item())
            
            reporter.log_metric("first_layer_grad_spread", np.mean(first_layer_grad_spread) if first_layer_grad_spread else 0)
            reporter.log_metric("last_layer_grad_spread", np.mean(last_layer_grad_spread) if last_layer_grad_spread else 0)
            
            reporter.mark_passed(True)
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


class TestIntegratorDiagnosis:
    """Diagnose integrator effects on stability."""

    def test_integrator_comparison(self):
        """
        Compare different integrators for stability.
        """
        reporter = ResultsReporter("integrator_comparison", "diagnostics")
        
        integrators_to_test = ['leapfrog', 'yoshida', 'heun']
        
        for integrator_name in integrators_to_test:
            try:
                model = gfn.create(
                    'gssm',
                    vocab_size=128,
                    dim=32,
                    heads=2,
                    topology_type='torus',
                    holographic=False,
                    initial_spread=0.0,
                    physics={'stability': {'integrator_type': integrator_name, 'base_dt': 0.1}}
                )
                model.eval()
                
                # Run multiple forward passes
                x = torch.randint(0, 128, (16, 8))
                
                x_norms = []
                v_norms = []
                has_nan = False
                
                for _ in range(5):
                    with torch.no_grad():
                        _, (xf, vf), info = model(x)
                        x_norms.append(torch.norm(xf).item())
                        v_norms.append(torch.norm(vf).item())
                        if torch.isnan(xf).any() or torch.isnan(vf).any():
                            has_nan = True
                
                reporter.log_metric(f"{integrator_name}_x_norm_mean", np.mean(x_norms))
                reporter.log_metric(f"{integrator_name}_v_norm_mean", np.mean(v_norms))
                reporter.log_metric(f"{integrator_name}_has_nan", int(has_nan))
                reporter.log_metric(f"{integrator_name}_x_norm_std", np.std(x_norms))
                
            except Exception as e:
                reporter.log_metric(f"{integrator_name}_error", 1)
                reporter.add_error(f"{integrator_name}: {str(e)}")
        
        reporter.mark_passed(True)
        
        filepath = reporter.save()
        reporter.print_summary()
        print(f"  Results saved to: {filepath}")

    def test_dt_sensitivity(self):
        """
        Test sensitivity to different dt values.
        """
        reporter = ResultsReporter("dt_sensitivity", "diagnostics")
        
        dt_values = [0.01, 0.05, 0.1, 0.2, 0.5]
        
        for dt in dt_values:
            try:
                model = gfn.create(
                    'gssm',
                    vocab_size=128,
                    dim=32,
                    heads=2,
                    topology_type='torus',
                    holographic=False,
                    initial_spread=0.0,
                    physics={'stability': {'base_dt': dt}}
                )
                model.eval()
                
                x = torch.randint(0, 128, (16, 8))
                
                with torch.no_grad():
                    _, (xf, vf), info = model(x)
                
                x_norm = torch.norm(xf).item()
                v_norm = torch.norm(vf).item()
                has_nan = int(torch.isnan(xf).any() or torch.isnan(vf).any())
                
                reporter.log_metric(f"dt_{dt}_x_norm", x_norm)
                reporter.log_metric(f"dt_{dt}_v_norm", v_norm)
                reporter.log_metric(f"dt_{dt}_has_nan", has_nan)
                
            except Exception as e:
                reporter.add_error(f"dt={dt}: {str(e)}")
        
        reporter.mark_passed(True)
        
        filepath = reporter.save()
        reporter.print_summary()
        print(f"  Results saved to: {filepath}")


class TestTopologyDiagnosis:
    """Test different topology configurations."""

    def test_topology_comparison(self):
        """
        Compare torus vs euclidean topology.
        """
        reporter = ResultsReporter("topology_comparison", "diagnostics")
        
        topologies = ['torus', 'euclidean']
        
        for topo in topologies:
            try:
                model = gfn.create(
                    'gssm',
                    vocab_size=128,
                    dim=64,
                    heads=4,
                    topology_type=topo,
                    holographic=False,
                    initial_spread=0.1
                )
                model.eval()
                
                x = torch.randint(0, 128, (16, 8))
                
                with torch.no_grad():
                    _, (xf, vf), info = model(x)
                    x_seq = info['x_seq']
                
                x_norms = [torch.norm(x_seq[:, i]).item() for i in range(x_seq.shape[1])]
                v_norms = [torch.norm(vf).item()]
                
                # SVD analysis
                x_flat = xf.reshape(xf.shape[0], -1)
                _, s, _ = torch.svd(x_flat)
                condition = (s[0] / s[-1]).item() if s[-1] > 1e-10 else float('inf')
                
                reporter.log_metric(f"{topo}_x_norm_final", x_norms[-1])
                reporter.log_metric(f"{topo}_v_norm_final", v_norms[0])
                reporter.log_metric(f"{topo}_condition_number", condition)
                reporter.log_metric(f"{topo}_x_norm_change", abs(x_norms[-1] - x_norms[0]))
                
            except Exception as e:
                reporter.add_error(f"{topo}: {str(e)}")
        
        reporter.mark_passed(True)
        
        filepath = reporter.save()
        reporter.print_summary()
        print(f"  Results saved to: {filepath}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
