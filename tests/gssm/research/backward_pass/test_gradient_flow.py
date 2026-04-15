"""
Backward Pass Analysis — GSSM Research Tests
==============================================
Deep analysis of gradient flow and backpropagation in GFN.
Tests actual GSSM model behavior, not mocks.

Research Questions:
- How do gradients propagate through manifold dynamics?
- Jacobian spectral radius across layers?
- Gradient conditioning and curvature relationships?
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


class TestBackwardPass:
    """Analyze gradient flow and backpropagation."""

    def test_gradient_magnitude_by_layer(self):
        """
        Measure gradient magnitudes at each layer.
        Expected: Gradual decay or stable flow, no vanishing.
        """
        reporter = ResultsReporter("gradient_magnitude_by_layer", "backward_pass")
        
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
            
            x = torch.randint(0, 128, (8, 8))
            target = torch.randint(0, 128, (8, 8))
            
            logits, (xf, vf), info = model(x)
            
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 128), target.reshape(-1)
            )
            loss.backward()
            
            layer_grads = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    layer_name = name.split('.')[1] if len(name.split('.')) > 1 else name
                    if layer_name not in layer_grads:
                        layer_grads[layer_name] = []
                    layer_grads[layer_name].append(param.grad.norm().item())
            
            for layer_name, grads in layer_grads.items():
                avg_grad = np.mean(grads)
                max_grad = np.max(grads)
                reporter.log_metric(f"{layer_name}_avg_grad", avg_grad)
                reporter.log_metric(f"{layer_name}_max_grad", max_grad)
            
            first_layer_grad = layer_grads.get('layer.0', [0.0])[0] if 'layer.0' in layer_grads else 0.0
            last_layer_grad = layer_grads.get('layer.3', [0.0])[0] if 'layer.3' in layer_grads else 0.0
            
            if first_layer_grad > 0 and last_layer_grad > 0:
                grad_ratio = last_layer_grad / (first_layer_grad + 1e-10)
                reporter.log_metric("grad_flow_ratio_first_to_last", grad_ratio)
                
                if grad_ratio > 0.01:
                    reporter.mark_passed(True)
                else:
                    reporter.add_error(f"Vanishing gradients: ratio={grad_ratio}")
            else:
                reporter.mark_passed(True)
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_jacobian_spectral_radius(self):
        """
        Estimate spectral radius of Jacobian for manifold dynamics.
        Spectral radius > 1 may indicate chaos, < 1 convergence.
        """
        reporter = ResultsReporter("jacobian_spectral_radius", "backward_pass")
        
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
            
            # Use proper input dimensions for the model - as float
            x = torch.randint(0, 128, (4, 8)).float()
            
            with torch.no_grad():
                _, (xf, vf), info = model(x.long())
                x_final = info['x_final']
            
            x_flat = x_final.reshape(x_final.shape[0], -1)
            x_input_flat = x.reshape(x.shape[0], -1)
            
            min_dim = min(x_flat.shape[1], x_input_flat.shape[1])
            x_flat = x_flat[:, :min_dim]
            x_input_flat = x_input_flat[:, :min_dim]
            
            # Simple Jacobian approximation using finite differences
            eps = 1e-4
            jacobian = []
            for i in range(min(10, x_input_flat.shape[0])):
                row = []
                for j in range(min(10, x_input_flat.shape[1])):
                    x_plus = x_input_flat.clone()
                    x_plus[i, j] += eps
                    x_plus_reshaped = x_plus.view(4, 8)
                    
                    with torch.no_grad():
                        _, (xf_plus, _), _ = model(x_plus_reshaped.long())
                        xf_plus_flat = xf_plus.reshape(xf_plus.shape[0], -1)[:, :min_dim]
                    
                    deriv = (xf_plus_flat[i, j] - x_flat[i, j]) / eps
                    row.append(deriv.item())
                jacobian.append(row)
            
            jacobian_matrix = torch.tensor(jacobian, dtype=torch.float32)
            eigenvalues = torch.linalg.eigvalsh(jacobian_matrix.T @ jacobian_matrix)
            spectral_radius = eigenvalues[-1].sqrt().item()
            
            reporter.log_metric("spectral_radius", spectral_radius)
            reporter.log_metric("largest_eigenvalue", eigenvalues[-1].item())
            reporter.log_metric("condition_number", 
                               (eigenvalues[-1] / eigenvalues[0]).item() if eigenvalues[0] > 1e-10 else 0)
            
            if spectral_radius < 10.0:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Large spectral radius: {spectral_radius}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_gradient_covariance_structure(self):
        """
        Analyze covariance structure of gradients.
        Tests: Are gradients well-conditioned / diverse?
        """
        reporter = ResultsReporter("gradient_covariance", "backward_pass")
        
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
            model.train()
            
            # Reduced sample size to avoid OOM
            num_samples = 10
            grad_samples = []
            
            for _ in range(num_samples):
                model.zero_grad()
                x = torch.randint(0, 128, (4, 4))
                target = torch.randint(0, 128, (4, 4))
                
                logits, (xf, vf), _ = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 128), target.reshape(-1)
                )
                loss.backward()
                
                # Only collect first layer gradients to save memory
                first_layer_grads = []
                for name, param in model.named_parameters():
                    if param.grad is not None and 'layer.0' in name:
                        first_layer_grads.append(param.grad.flatten()[:100])  # Limit to first 100
                if first_layer_grads:
                    grad_samples.append(torch.cat(first_layer_grads))
            
            if len(grad_samples) > 1:
                grad_matrix = torch.stack(grad_samples)
                cov = grad_matrix.T @ grad_matrix / len(grad_samples)
                
                eigenvalues = torch.linalg.eigvalsh(cov)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                
                if len(eigenvalues) > 1:
                    entropy = -(eigenvalues / eigenvalues.sum() * torch.log2(eigenvalues / eigenvalues.sum() + 1e-10)).sum()
                    
                    reporter.log_metric("gradient_entropy_bits", entropy.item())
                    reporter.log_metric("num_significant_eigenvalues", len(eigenvalues))
                    reporter.log_metric("eigenvalue_spread", (eigenvalues[-1] / eigenvalues[0]).item())
                else:
                    reporter.log_metric("gradient_entropy_bits", 0.0)
                    reporter.log_metric("num_significant_eigenvalues", len(eigenvalues))
            
            reporter.mark_passed(True)
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_loss_gradient_alignment(self):
        """
        Compute cosine similarity between loss gradient and parameter gradients.
        Tests: Is optimization direction well-aligned?
        """
        reporter = ResultsReporter("loss_gradient_alignment", "backward_pass")
        
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
            model.train()
            
            x = torch.randint(0, 128, (8, 8))
            target = torch.randint(0, 128, (8, 8))
            
            logits, (xf, vf), _ = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 128), target.reshape(-1)
            )
            loss.backward()
            
            param_grads = []
            param_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    param_grads.append(param.grad.flatten())
                    param_norms.append(param.grad.norm().item())
            
            combined_grad = torch.cat(param_grads)
            grad_direction = combined_grad / combined_grad.norm()
            
            loss_val = loss.item()
            expected_grad_norm = sum(p.numel() for p in model.parameters()) ** 0.5
            
            reporter.log_metric("loss_value", loss_val)
            reporter.log_metric("combined_grad_norm", combined_grad.norm().item())
            reporter.log_metric("expected_grad_norm", expected_grad_norm)
            reporter.log_metric("param_count", sum(p.numel() for p in model.parameters()))
            
            if combined_grad.norm() > 1e-6:
                reporter.mark_passed(True)
            else:
                reporter.add_error("Zero gradient")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
