"""
Forward Pass Analysis — GSSM Research Tests
=============================================
Deep analysis of layer-by-layer activations in GFN forward pass.
Tests actual GSSM model behavior, not mocks.

Research Questions:
- How do activations evolve through manifold layers?
- What's the spectral properties of activation matrices?
- Is there gradient flow saturation?
- Activation entropy across layers?
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


class TestActivationDynamics:
    """Analyze activation patterns through manifold layers."""

    def test_layer_activation_magnitudes(self):
        """
        Measure L2 norm of (x, v) states through each layer.
        Expected: Gradual transformation, not sudden jumps.
        """
        reporter = ResultsReporter("layer_activation_magnitudes", "forward_pass")
        
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
            
            batch_size = 16
            seq_len = 8
            x = torch.randint(0, 128, (batch_size, seq_len))
            
            with torch.no_grad():
                logits, (xf, vf), info = model(x)
                x_seq = info['x_seq']  # [B, L, H, D]
                v_seq = info['v_seq']
            
            layer_norms_x = []
            layer_norms_v = []
            
            for layer_idx in range(seq_len):
                x_layer = x_seq[:, layer_idx]
                v_layer = v_seq[:, layer_idx]
                
                norm_x = torch.norm(x_layer, dim=-1).mean().item()
                norm_v = torch.norm(v_layer, dim=-1).mean().item()
                
                layer_norms_x.append(norm_x)
                layer_norms_v.append(norm_v)
                
                reporter.log_metric(f"layer_{layer_idx}_x_norm", norm_x)
                reporter.log_metric(f"layer_{layer_idx}_v_norm", norm_v)
            
            reporter.log_metric("x_norm_ratio_first_last", 
                               layer_norms_x[-1] / (layer_norms_x[0] + 1e-8))
            reporter.log_metric("v_norm_ratio_first_last",
                               layer_norms_v[-1] / (layer_norms_v[0] + 1e-8))
            
            max_norm_change = max(abs(layer_norms_x[i+1] - layer_norms_x[i]) 
                                  for i in range(len(layer_norms_x)-1))
            reporter.log_metric("max_layer_norm_change", max_norm_change)
            
            if max_norm_change < 10.0:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Large norm jump detected: {max_norm_change}")
            
            assert max_norm_change < 10.0
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_activation_singular_values(self):
        """
        Compute singular values of activation matrices per layer.
        Tests: Is the manifold learning low-rank or full-rank?
        """
        reporter = ResultsReporter("activation_singular_values", "forward_pass")
        
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
                
                _, s, _ = torch.svd(x_layer)
                
                rank_90 = (torch.cumsum(s**2, dim=0) / (s**2).sum() < 0.90).sum().item() + 1
                rank_99 = (torch.cumsum(s**2, dim=0) / (s**2).sum() < 0.99).sum().item() + 1
                
                reporter.log_metric(f"layer_{layer_idx}_top_singular", s[0].item())
                reporter.log_metric(f"layer_{layer_idx}_rank_90", rank_90)
                reporter.log_metric(f"layer_{layer_idx}_rank_99", rank_99)
                reporter.log_metric(f"layer_{layer_idx}_condition_number", 
                                   (s[0] / s[-1]).item() if s[-1] > 1e-8 else float('inf'))
            
            reporter.mark_passed(True)
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_activation_entropy(self):
        """
        Compute entropy of activation distributions.
        High entropy = diverse activations, low = saturated/deterministic.
        """
        reporter = ResultsReporter("activation_entropy", "forward_pass")
        
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
            x = torch.randint(0, 128, (num_samples, 8))
            
            with torch.no_grad():
                _, (xf, vf), info = model(x)
                x_seq = info['x_seq']
            
            for layer_idx in range(min(4, x_seq.shape[1])):
                x_layer = x_seq[:, layer_idx].flatten()
                
                hist = torch.histc(x_layer, bins=50)
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                
                entropy = -(hist * torch.log2(hist + 1e-10)).sum().item()
                
                reporter.log_metric(f"layer_{layer_idx}_entropy_bits", entropy)
                reporter.log_metric(f"layer_{layer_idx}_std", x_layer.std().item())
                reporter.log_metric(f"layer_{layer_idx}_mean", x_layer.mean().item())
            
            reporter.mark_passed(True)
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_gradient_flow_through_layers(self):
        """
        Measure backprop gradient magnitudes through each layer.
        Tests: Vanishing/exploding gradients in manifold layers?
        """
        reporter = ResultsReporter("gradient_flow", "forward_pass")
        
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
            
            layer_grad_norms = []
            for name, param in model.named_parameters():
                if 'layer' in name.lower() and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    layer_grad_norms.append(grad_norm)
            
            if layer_grad_norms:
                avg_grad = np.mean(layer_grad_norms)
                max_grad = np.max(layer_grad_norms)
                min_grad = np.min(layer_grad_norms)
                
                reporter.log_metric("avg_gradient_norm", avg_grad)
                reporter.log_metric("max_gradient_norm", max_grad)
                reporter.log_metric("min_gradient_norm", min_grad)
                reporter.log_metric("gradient_range", max_grad - min_grad)
                
                if min_grad > 1e-8:
                    reporter.mark_passed(True)
                else:
                    reporter.add_error(f"Vanishing gradients detected: min={min_grad}")
                
                assert min_grad > 1e-8, "Vanishing gradients detected"
            else:
                reporter.add_error("No gradients found")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
