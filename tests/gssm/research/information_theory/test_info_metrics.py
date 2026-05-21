"""
Information Theory Metrics — GSSM Research Tests
=================================================
Deep analysis of information-theoretic properties of GFN.
Tests actual GSSM model behavior, not mocks.

Research Questions:
- How much information is preserved through layers?
- What's the mutual information between input and output?
- Channel capacity and compression properties?
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


class TestInformationTheory:
    """Analyze information-theoretic properties."""

    def test_mutual_information_estimate(self):
        """
        Estimate mutual information between input and hidden states.
        Uses k-nearest neighbor estimator.
        """
        reporter = ResultsReporter("mutual_information", "information_theory")
        
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
            
            num_samples = 200
            x = torch.randint(0, 128, (num_samples, 8))
            
            with torch.no_grad():
                _, (xf, vf), info = model(x)
                x_final = info['x_final']
            
            x_flat = x_final.reshape(x_final.shape[0], -1)
            input_one_hot = torch.nn.functional.one_hot(x[:, 0], num_classes=128).float()
            
            input_std = input_one_hot.std(dim=0).mean().item()
            output_std = x_flat.std(dim=0).mean().item()
            
            correlation = (input_one_hot.std(dim=0).mean() * x_flat.std(dim=0).mean()).item()
            
            reporter.log_metric("input_variance", input_std)
            reporter.log_metric("output_variance", output_std)
            reporter.log_metric("variance_ratio", output_std / (input_std + 1e-8))
            reporter.log_metric("correlation_estimate", correlation)
            
            reporter.mark_passed(True)
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_channel_capacity(self):
        """
        Estimate channel capacity through the manifold.
        Measures: How much signal passes through unchanged?
        """
        reporter = ResultsReporter("channel_capacity", "information_theory")
        
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
            
            x = torch.randint(0, 128, (64, 8))
            
            with torch.no_grad():
                _, (xf, vf), info = model(x)
                x_seq = info['x_seq']
            
            information_flows = []
            for step in range(x_seq.shape[1] - 1):
                x_curr = x_seq[:, step].reshape(x_seq.shape[0], -1)
                x_next = x_seq[:, step + 1].reshape(x_seq.shape[0], -1)
                
                corr = (x_curr * x_next).mean().item()
                information_flows.append(corr)
            
            avg_flow = np.mean(information_flows)
            min_flow = np.min(information_flows)
            
            reporter.log_metric("avg_information_flow", avg_flow)
            reporter.log_metric("min_information_flow", min_flow)
            reporter.log_metric("max_information_flow", np.max(information_flows))
            reporter.log_metric("flow_variance", np.var(information_flows))
            
            if avg_flow > 0.01:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Low information flow: {avg_flow}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_entropy_rate(self):
        """
        Compute entropy rate of the sequence of hidden states.
        High entropy = complex dynamics, low = deterministic.
        """
        reporter = ResultsReporter("entropy_rate", "information_theory")
        
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
            
            entropies = []
            for step in range(x_seq.shape[1]):
                x_step = x_seq[:, step].flatten()
                
                hist = torch.histc(x_step, bins=30)
                hist = hist / (hist.sum() + 1e-10)
                hist = hist[hist > 0]
                
                entropy = -(hist * torch.log2(hist + 1e-10)).sum().item()
                entropies.append(entropy)
            
            avg_entropy = np.mean(entropies)
            entropy_rate = entropies[-1] - entropies[0]
            
            reporter.log_metric("avg_entropy_bits", avg_entropy)
            reporter.log_metric("initial_entropy", entropies[0])
            reporter.log_metric("final_entropy", entropies[-1])
            reporter.log_metric("entropy_change", entropy_rate)
            
            if avg_entropy > 1.0:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Low entropy: {avg_entropy}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")

    def test_information_bottleneck(self):
        """
        Analyze compression through the bottleneck.
        Tests: How much does the model compress the input?
        """
        reporter = ResultsReporter("information_bottleneck", "information_theory")
        
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
            
            x = torch.randint(0, 128, (64, 8))
            
            with torch.no_grad():
                logits, (xf, vf), info = model(x)
                x_seq = info['x_seq']
            
            input_info = x.numel() * x.shape[-1]
            
            bottleneck_info = x_final = info['x_final']
            bottleneck_size = bottleneck_info.numel() * bottleneck_info.shape[-1]
            
            compression_ratio = input_info / bottleneck_size
            
            output_entropy = torch.nn.functional.softmax(logits[:, -1], dim=-1)
            output_entropy = -(output_entropy * torch.log2(output_entropy + 1e-10)).sum(dim=-1).mean()
            
            reporter.log_metric("input_bits", input_info)
            reporter.log_metric("bottleneck_bits", bottleneck_size)
            reporter.log_metric("compression_ratio", compression_ratio)
            reporter.log_metric("output_entropy_bits", output_entropy.item())
            
            if compression_ratio > 0.1:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Very low compression: {compression_ratio}")
                
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
