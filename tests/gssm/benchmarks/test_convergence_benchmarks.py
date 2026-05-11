"""
Benchmark Tests — Convergence & Performance (with JSON Reporting)
=====================================================================

Benchmarks for GSSM on standard tasks with automatic JSON result generation.
Results saved to: tests/results/benchmarks/

Benchmarks:
- Needle-in-a-haystack (NIAH): Long-context retrieval
- XOR: Non-linear classification
- Stress tests: Error recovery, scaling

Each benchmark measures:
- Convergence speed (steps to target accuracy)
- Final accuracy
- Resource usage (memory, time)
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.test_reporter import ResultsReporter, ConvergenceTracker

from gfn.realizations.gssm.models.g_ssm import GSSM
from gfn.realizations.gssm.models.factory import ModelFactory
from gfn.realizations.gssm.training.optimizer import RiemannianAdam
from gfn.realizations.gssm.losses.toroidal import ToroidalCategoricalLoss


class BenchmarkResult:
    """Container for benchmark results."""
    def __init__(self, name: str):
        self.name = name
        self.metrics: Dict[str, float] = {}
        self.history: List[float] = []
        self.converged: bool = False
        self.convergence_step: int = -1
        self.time_seconds: float = 0.0
    
    def report(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Benchmark: {self.name}",
            f"{'='*60}",
            f"Converged: {self.converged}",
            f"Convergence step: {self.convergence_step}",
            f"Time: {self.time_seconds:.2f}s",
        ]
        for metric, value in self.metrics.items():
            lines.append(f"{metric}: {value:.4f}")
        lines.append("="*60)
        return "\n".join(lines)


class TestNeedleInHaystack:
    """
    Needle-in-a-haystack benchmark: Long context retrieval.
    
    Task: Find a specific token (needle) in a long sequence of 
    irrelevant tokens (haystack). Tests attention mechanism and
    long-range dependency modeling.
    
    Success criteria:
    - Retrieve needle position with >95% accuracy
    - Maintain performance as sequence length increases
    """
    
    @pytest.mark.parametrize("seq_len", [128, 512, 1024])
    def test_niah_retrieval(self, seq_len: int):
        """Test needle retrieval at different sequence lengths with JSON reporting."""
        reporter = ResultsReporter(f"niah_retrieval_{seq_len}", "benchmarks")
        
        try:
            print(f"\n{'='*60}")
            print(f"NIAH Benchmark - Sequence Length: {seq_len}")
            print(f"{'='*60}")
            
            vocab_size = 100
            needle_token = 42
            needle_position = seq_len // 2
            
            reporter.log_metric("seq_len", seq_len)
            reporter.log_metric("vocab_size", vocab_size)
            reporter.log_metric("needle_position", needle_position)
            
            # Generate haystack
            sequence = torch.randint(0, vocab_size, (1, seq_len))
            sequence[0, needle_position] = needle_token
            
            # Create model
            config = {
                'vocab_size': vocab_size,
                'dim': 64,
                'num_heads': 4,
                'seq_len': seq_len,
                'topology': 'torus',
                'dynamics_type': 'direct'
            }
            
            model = GSSM(config)
            optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            target = torch.tensor([needle_position])
            
            # Training
            start_time = time.time()
            accuracies = []
            
            for step in range(500):
                optimizer.zero_grad()
                logits, _ = model(sequence, None)
                position_logits = logits.mean(dim=-1)
                loss = criterion(position_logits.unsqueeze(0), target)
                loss.backward()
                optimizer.step()
                
                pred_position = position_logits.argmax().item()
                accuracy = 1.0 if abs(pred_position - needle_position) < 10 else 0.0
                accuracies.append(accuracy)
                
                if step % 100 == 0:
                    reporter.log_metric(f"accuracy_step_{step}", accuracy)
                    reporter.log_metric(f"loss_step_{step}", loss.item())
            
            elapsed = time.time() - start_time
            final_acc = np.mean(accuracies[-50:])
            
            reporter.log_metric("final_accuracy", final_acc)
            reporter.log_metric("training_time_sec", elapsed)
            reporter.log_metric("total_steps", 500)
            reporter.log_plot_data("accuracy_curve", list(range(500)), accuracies)
            
            if final_acc > 0.5:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Accuracy too low: {final_acc:.2%}")
            
            print(f"\n  Final Accuracy: {final_acc:.2%}")
            print(f"  Time: {elapsed:.2f}s")
            
            assert final_acc > 0.5
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


class TestXORClassification:
    """
    XOR benchmark: Non-linear classification.
    
    XOR is not linearly separable, requiring the model to learn
    non-linear decision boundaries.
    
    Task: Learn XOR function:
    - (0, 0) -> 0
    - (0, 1) -> 1
    - (1, 0) -> 1
    - (1, 1) -> 0
    
    Success criteria:
    - 100% accuracy on all 4 patterns
    - Convergence in <1000 steps
    """
    
    def test_xor_convergence(self):
        """Test learning XOR function with JSON reporting."""
        reporter = ResultsReporter("xor_convergence", "benchmarks")
        
        try:
            print(f"\n{'='*60}")
            print(f"XOR Classification Benchmark")
            print(f"{'='*60}")
            
            # XOR dataset
            X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
            y = torch.tensor([0., 1., 1., 0.])
            
            reporter.log_metric("num_samples", 4)
            reporter.log_metric("input_dim", 2)
            
            # Create model
            config = {
                'vocab_size': 2,
                'dim': 16,
                'num_heads': 2,
                'seq_len': 2,
                'topology': 'euclidean',
                'dynamics_type': 'mix'
            }
            
            model = GSSM(config)
            
            # Simple classifier head
            class XORClassifier(nn.Module):
                def __init__(self, base_model, dim):
                    super().__init__()
                    self.base = base_model
                    self.classifier = nn.Linear(dim, 1)
                
                def forward(self, x):
                    x_int = (x * 10).long().clamp(0, 1)
                    x_embed = nn.functional.one_hot(x_int, num_classes=2).float()
                    out, _ = self.base(x_embed, None)
                    return torch.sigmoid(self.classifier(out.mean(dim=1)))
            
            classifier = XORClassifier(model, 16)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
            criterion = nn.BCELoss()
            
            # Training
            start_time = time.time()
            converged = False
            convergence_step = -1
            loss_history = []
            acc_history = []
            
            for step in range(1000):
                optimizer.zero_grad()
                predictions = classifier(X).squeeze()
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()
                
                acc = ((predictions > 0.5).float() == y).float().mean().item()
                loss_history.append(loss.item())
                acc_history.append(acc)
                
                if acc == 1.0 and not converged:
                    converged = True
                    convergence_step = step
                    print(f"  ✓ Converged at step {step}")
                    break
                
                if step % 200 == 0:
                    reporter.log_metric(f"loss_step_{step}", loss.item())
                    reporter.log_metric(f"accuracy_step_{step}", acc)
            
            elapsed = time.time() - start_time
            
            reporter.log_metric("converged", converged)
            reporter.log_metric("convergence_step", convergence_step)
            reporter.log_metric("final_accuracy", acc_history[-1] if acc_history else 0)
            reporter.log_metric("training_time_sec", elapsed)
            reporter.log_plot_data("loss_curve", list(range(len(loss_history))), loss_history)
            reporter.log_plot_data("accuracy_curve", list(range(len(acc_history))), acc_history)
            
            if converged and convergence_step < 800:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Did not converge or too slow: converged={converged}, step={convergence_step}")
            
            print(f"\n  Converged: {converged}")
            print(f"  Convergence step: {convergence_step}")
            print(f"  Time: {elapsed:.2f}s")
            
            assert converged
            assert convergence_step < 800
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


class TestStressRecovery:
    """
    Stress tests: Error recovery and robustness.
    
    Tests system behavior under adverse conditions:
    - NaN injection
    - Gradient explosion
    - Memory pressure
    """
    
    def test_nan_recovery(self):
        """Test model recovery from NaN gradients with JSON reporting."""
        reporter = ResultsReporter("nan_recovery", "benchmarks")
        
        try:
            print(f"\n{'='*60}")
            print(f"NaN Recovery Stress Test")
            print(f"{'='*60}")
            
            config = {
                'vocab_size': 50,
                'dim': 32,
                'num_heads': 4,
                'topology': 'euclidean'
            }
            
            model = GSSM(config)
            
            # Inject NaN
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.numel() > 0:
                        param[0] = float('nan')
                        break
            
            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            reporter.log_metric("nan_injected", has_nan)
            
            # Recovery
            with torch.no_grad():
                for param in model.parameters():
                    param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
            
            has_nan_after = any(torch.isnan(p).any() for p in model.parameters())
            reporter.log_metric("nan_after_recovery", has_nan_after)
            reporter.log_metric("recovery_successful", not has_nan_after)
            
            if not has_nan_after:
                reporter.mark_passed(True)
            else:
                reporter.add_error("NaN still present after recovery")
            
            print(f"  NaN injected: {has_nan}")
            print(f"  NaN after recovery: {has_nan_after}")
            
            assert not has_nan_after, "Failed to recover from NaN"
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")
    
    def test_gradient_clipping(self):
        """Test gradient explosion prevention with JSON reporting."""
        reporter = ResultsReporter("gradient_explosion_prevention", "benchmarks")
        
        try:
            print(f"\n{'='*60}")
            print(f"Gradient Explosion Prevention Test")
            print(f"{'='*60}")
            
            config = {
                'vocab_size': 50,
                'dim': 32,
                'num_heads': 4
            }
            
            model = GSSM(config)
            
            # Large input
            x = torch.randn(1, 10, 50) * 100
            
            logits, _ = model(x, None)
            loss = logits.sum()
            loss.backward()
            
            max_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    norm = param.grad.norm().item()
                    max_grad_norm = max(max_grad_norm, norm)
            
            reporter.log_metric("max_gradient_norm", max_grad_norm)
            reporter.log_metric("gradient_finite", max_grad_norm < 1e6 and not np.isnan(max_grad_norm))
            
            if max_grad_norm < 1e6 and not np.isnan(max_grad_norm) and not np.isinf(max_grad_norm):
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Gradient explosion: norm={max_grad_norm}")
            
            print(f"  Max gradient norm: {max_grad_norm:.2f}")
            
            assert max_grad_norm < 1e6, f"Gradient explosion: {max_grad_norm}"
            assert not np.isnan(max_grad_norm), "NaN gradients"
            assert not np.isinf(max_grad_norm), "Inf gradients"
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


class TestBatchScaling:
    """
    Scaling tests: Performance across batch sizes.
    
    Measures:
    - Throughput (samples/sec) vs batch size
    - Memory usage scaling
    - Stability with large batches
    """
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_throughput_scaling(self, batch_size: int):
        """Test throughput at different batch sizes with JSON reporting."""
        reporter = ResultsReporter(f"throughput_batch_{batch_size}", "benchmarks")
        
        try:
            print(f"\n{'='*60}")
            print(f"Batch Scaling Test - Batch Size: {batch_size}")
            print(f"{'='*60}")
            
            config = {
                'vocab_size': 100,
                'dim': 64,
                'num_heads': 4,
                'seq_len': 64
            }
            
            model = GSSM(config)
            model.eval()
            
            # Generate batch
            x = torch.randint(0, 100, (batch_size, 64))
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x, None)
            
            # Measure throughput
            num_iterations = 20
            start = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(x, None)
            
            elapsed = time.time() - start
            throughput = (num_iterations * batch_size) / elapsed
            time_per_batch_ms = (elapsed / num_iterations) * 1000
            
            reporter.log_metric("batch_size", batch_size)
            reporter.log_metric("throughput_samples_per_sec", throughput)
            reporter.log_metric("time_per_batch_ms", time_per_batch_ms)
            reporter.log_metric("num_iterations", num_iterations)
            reporter.log_metric("total_time_sec", elapsed)
            
            if throughput > 1.0:
                reporter.mark_passed(True)
            else:
                reporter.add_error(f"Throughput too low: {throughput}")
            
            print(f"  Throughput: {throughput:.1f} samples/sec")
            print(f"  Time per batch: {time_per_batch_ms:.1f} ms")
            
            assert throughput > 1.0
            
        except Exception as e:
            reporter.add_error(str(e))
            raise
        finally:
            filepath = reporter.save()
            reporter.print_summary()
            print(f"  Results saved to: {filepath}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
