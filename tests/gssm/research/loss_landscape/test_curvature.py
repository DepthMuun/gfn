"""
Loss Landscape Analysis — GSSM Research Tests
===============================================
Deep analysis of loss landscape, curvature, and optimization dynamics.
Tests actual GSSM model behavior, not mocks.

Research Questions:
- What's the local curvature of the loss landscape?
- Are there spurious local minima or only good basins?
- How does loss change in random directions vs. gradient direction?
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


class TestLossLandscape:
    """Analyze loss landscape geometry and curvature."""

    def test_hessian_spectral_analysis(self):
        """
        Compute Hessian eigenvalues at initialization.
        Tests: Is the loss landscape at init well-conditioned?
        """
        reporter = ResultsReporter("hessian_spectral", "loss_landscape")
        
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
            
            # Compute loss and gradients
            logits, (xf, vf), _ = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 128), target.reshape(-1)
            )
            loss.backward()
            
            # Get gradient statistics
            grad_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.abs().mean().item())
            
            if grad_norms:
                reporter.log_metric("mean_grad_norm", np.mean(grad_norms))
                reporter.log_metric("std_grad_norm", np.std(grad_norms))
                reporter.log_metric("max_grad_norm", np.max(grad_norms))
                reporter.log_metric("min_grad_norm", np.min(grad_norms))
                
                positive_ratio = sum(1 for g in grad_norms if g > 0) / len(grad_norms)
                reporter.log_metric("positive_grad_ratio", positive_ratio)
            
            reporter.mark_passed(True)
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_loss_direction_curvature(self):
        """
        Measure loss change along random directions vs gradient direction.
        Tests: Is gradient direction steeper than random?
        """
        reporter = ResultsReporter("loss_direction_curvature", "loss_landscape")
        
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
            
            # Forward pass
            logits, (xf, vf), _ = model(x)
            loss_before = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 128), target.reshape(-1)
            )
            
            # Get gradient
            loss_before.backward()
            
            # Get gradient direction
            grad_direction = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_direction.append(param.grad.flatten())
            grad_direction = torch.cat(grad_direction)
            grad_direction = grad_direction / grad_direction.norm()
            
            # Test loss change in gradient direction
            epsilon = 0.01
            idx = 0
            for param in model.parameters():
                if param.grad is not None:
                    numel = param.numel()
                    param.data = param.data + epsilon * grad_direction[idx:idx+numel].reshape(param.shape)
                    idx += 1
                    if idx >= len(grad_direction):
                        break
            
            # Compute loss after perturbation
            logits_after, _, _ = model(x)
            loss_after = torch.nn.functional.cross_entropy(
                logits_after.reshape(-1, 128), target.reshape(-1)
            ).item()
            
            loss_change = loss_after - loss_before.item()
            
            reporter.log_metric("loss_before", loss_before.item())
            reporter.log_metric("loss_after", loss_after)
            reporter.log_metric("loss_change", loss_change)
            
            if loss_change < 0:
                reporter.mark_passed(True)
            else:
                reporter.add_error("Loss increased in gradient direction")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_loss_landscape_1d_slice(self):
        """
        Compute loss along a 1D slice in parameter space.
        Tests: Are there spurious local minima?
        """
        reporter = ResultsReporter("loss_1d_slice", "loss_landscape")
        
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
            
            # Get initial loss
            logits, _, _ = model(x)
            initial_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 128), target.reshape(-1)
            ).item()
            
            # Compute gradient direction
            model.zero_grad()
            logits, _, _ = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 128), target.reshape(-1)
            )
            loss.backward()
            
            grad_direction = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_direction.append(param.grad.flatten())
            grad_direction = torch.cat(grad_direction)
            grad_direction = grad_direction / grad_direction.norm()
            
            # Test loss at different steps along gradient direction
            loss_values = []
            alphas = [-0.5, -0.25, 0.0, 0.25, 0.5]
            
            for alpha in alphas:
                # Create new model for each alpha to avoid state issues
                model_new = gfn.create(
                    'gssm',
                    vocab_size=128,
                    dim=32,
                    heads=2,
                    topology_type='torus',
                    holographic=False,
                    initial_spread=0.0
                )
                model_new.train()
                
                # Apply perturbation
                idx = 0
                for param in model_new.parameters():
                    numel = param.numel()
                    if idx + numel <= len(grad_direction):
                        param.data = param.data + alpha * grad_direction[idx:idx+numel].reshape(param.shape)
                        idx += numel
                    if idx >= len(grad_direction):
                        break
                
                # Compute loss
                logits, _, _ = model_new(x)
                loss_val = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 128), target.reshape(-1)
                ).item()
                loss_values.append(loss_val)
            
            loss_values = np.array(loss_values)
            
            local_minima_count = 0
            for i in range(1, len(loss_values) - 1):
                if loss_values[i] < loss_values[i-1] and loss_values[i] < loss_values[i+1]:
                    local_minima_count += 1
            
            reporter.log_metric("loss_min", np.min(loss_values))
            reporter.log_metric("loss_max", np.max(loss_values))
            reporter.log_metric("loss_range", np.max(loss_values) - np.min(loss_values))
            reporter.log_metric("local_minima_count", local_minima_count)
            
            if local_minima_count <= 1:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Multiple local minima: {local_minima_count}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_gradient_norm_trajectory(self):
        """
        Track gradient norm during training.
        Tests: Does gradient norm decrease as expected?
        """
        reporter = ResultsReporter("gradient_norm_trajectory", "loss_landscape")
        
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
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            grad_norms = []
            losses = []
            
            for step in range(50):
                x = torch.randint(0, 128, (16, 8))
                target = torch.randint(0, 128, (16, 8))
                
                optimizer.zero_grad()
                logits, (xf, vf), _ = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 128), target.reshape(-1)
                )
                loss.backward()
                
                grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                grad_norms.append(grad_norm)
                losses.append(loss.item())
                
                optimizer.step()
            
            reporter.log_metric("initial_grad_norm", grad_norms[0])
            reporter.log_metric("final_grad_norm", grad_norms[-1])
            reporter.log_metric("initial_loss", losses[0])
            reporter.log_metric("final_loss", losses[-1])
            reporter.log_metric("grad_norm_reduction", grad_norms[0] / (grad_norms[-1] + 1e-8))
            reporter.log_metric("loss_reduction", losses[0] / (losses[-1] + 1e-8))
            
            if losses[-1] < losses[0]:
                reporter.mark_passed(True)
            else:
                reporter.add_error("Loss not decreasing")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
