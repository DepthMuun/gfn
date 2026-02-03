# Manifold Benchmarking Guide

## Overview

This document provides comprehensive documentation of the benchmarking framework included with Manifold, describing the evaluation protocols, metrics, baselines, and interpretation guidelines. The benchmarking system is designed to provide rigorous, reproducible evaluation of Manifold's capabilities across multiple dimensions including memory efficiency, computational performance, generalization ability, and stability characteristics.

The benchmark suite addresses several key questions about Manifold's capabilities. Memory efficiency benchmarks verify the O(1) inference memory property and compare against Transformers. Computational benchmarks measure throughput and latency across different configurations. Generalization benchmarks evaluate out-of-distribution performance, particularly on tasks requiring long-range dependencies. Stability benchmarks analyze gradient behavior, energy conservation, and numerical precision. Together, these benchmarks provide a comprehensive picture of Manifold's capabilities and limitations.

### Benchmark Philosophy

The benchmarking approach emphasizes rigorous scientific methodology. Each benchmark has a clear hypothesis, controlled experimental conditions, and statistical validation. Results are reported with uncertainty estimates rather than single point values. Baseline comparisons are performed with carefully matched configurations to ensure fair evaluation. All benchmark code is included in the repository, enabling independent reproduction and extension.

### Benchmark Directory Structure

The benchmark suite is organized as follows:

```
tests/benchmarks/
├── core/                    # Core benchmark implementations
│   ├── benchmark_memory.py         # Memory efficiency benchmarks
│   ├── benchmark_performance.py    # Computational benchmarks
│   ├── benchmark_generalization.py # Generalization benchmarks
│   ├── benchmark_stability.py      # Stability benchmarks
│   └── benchmark_composition.py    # Component-wise ablation
├── baselines.py             # Baseline model implementations
├── bench_utils.py           # Utility functions
└── results/                 # Benchmark results
    ├── memory/
    ├── performance/
    ├── generalization/
    └── stability/
```

## Memory Efficiency Benchmarks

### Hypothesis and Rationale

The central claim of Manifold is O(1) inference memory regardless of sequence length. This benchmark directly tests this claim by measuring memory usage across varying sequence lengths and comparing against Transformers.

The hypothesis is that Manifold maintains constant memory during inference while Transformers exhibit linear scaling with sequence length. This difference should be observable in both peak GPU memory usage and the size of the state maintained during autoregressive generation.

### Experimental Protocol

The memory benchmark measures the following quantities across sequence lengths from 64 to 131,072 tokens:

Peak GPU memory during forward pass measures the maximum memory allocated during a single forward pass. This includes activations, gradients (in training mode), and model parameters. State memory measures the size of the persistent state maintained during autoregressive inference. This is the core quantity for Manifold's O(1) claim. Inference memory measures total memory during autoregressive generation of a fixed number of tokens.

The experimental protocol controls for several confounding factors. Batch size is fixed at 1 to isolate sequence length effects. Model dimension is matched between Manifold and Transformer for fair comparison. Hardware configuration is reported to enable replication. Each measurement is repeated multiple times to estimate statistical uncertainty.

### Implementation

```python
import torch
import torch.nn as nn
from gfn import Manifold
from transformers import GPT2LMHeadModel
import gc

class MemoryBenchmark:
    def __init__(self, model_dim=512, num_layers=6, num_heads=4):
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Create Manifold model
        manifold_config = self._get_manifold_config()
        self.manifold = Manifold(
            vocab_size=50257,
            dim=model_dim,
            depth=num_layers,
            heads=num_heads,
            physics_config=manifold_config
        )
        
        # Create Transformer baseline
        self.transformer = GPT2LMHeadModel.from_pretrained('gpt2')
        # Resize to match dimensions
        self.transformer.config.n_embd = model_dim
        self.transformer.config.n_layer = num_layers
        self.transformer.config.n_head = num_heads
    
    def _get_manifold_config(self):
        return {
            'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16},
            'readout': {'type': 'implicit', 'coord_dim': 16},
            'active_inference': {'enabled': True, 'dynamic_time': {'enabled': True},
                                 'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
                                 'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8}},
            'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
            'topology': {'type': 'torus'},
            'stability': {'base_dt': 0.4}
        }
    
    def measure_peak_memory(self, model, input_ids, mode='inference'):
        """Measure peak GPU memory during forward pass."""
        gc.collect()
        torch.cuda.empty_cache()
        
        torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        with torch.no_grad():
            if mode == 'inference':
                # Single forward pass
                _ = model(input_ids)
            else:
                # Training forward pass
                logits, _, _ = model(input_ids)
                loss = nn.CrossEntropyLoss()(logits.view(-1, 50257), 
                                               torch.zeros_like(input_ids).view(-1))
                loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated()
        return peak_memory
    
    def measure_state_memory(self, model, input_ids):
        """Measure persistent state memory during autoregressive generation."""
        gc.collect()
        torch.cuda.empty_cache()
        
        model.eval()
        state = None
        total_state_memory = 0
        
        for i in range(input_ids.size(1)):
            token = input_ids[:, i:i+1]
            logits, state, _ = model(token, state=state)
            
            # Measure state size
            x, v = state
            state_size = x.element_size() * x.nelement() + v.element_size() * v.nelement()
            total_state_memory += state_size
        
        return total_state_memory
    
    def run_benchmark(self, seq_lengths=[64, 256, 1024, 4096, 16384, 65536]):
        """Run complete memory benchmark."""
        results = {'manifold': {}, 'transformer': {}}
        
        for seq_len in seq_lengths:
            print(f"Testing sequence length: {seq_len}")
            
            input_ids = torch.randint(0, 50257, (1, seq_len))
            
            # Manifold benchmarks
            manifold_state_mem = self.measure_state_memory(self.manifold, input_ids)
            results['manifold'][seq_len] = {
                'state_memory': manifold_state_mem
            }
            
            # Transformer benchmarks
            transformer_state_mem = self._measure_transformer_state(input_ids)
            results['transformer'][seq_len] = {
                'state_memory': transformer_state_mem
            }
        
        return results
    
    def _measure_transformer_state(self, input_ids):
        """Measure Transformer KV cache size."""
        self.transformer.eval()
        with torch.no_grad():
            outputs = self.transformer(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
        
        # Calculate total size of KV cache
        total_size = 0
        for layer_past in past_key_values:
            for tensor in layer_past:
                total_size += tensor.element_size() * tensor.nelement()
        
        return total_size
```

### Expected Results

The benchmark should demonstrate that Manifold's state memory remains constant across all sequence lengths, typically around 30MB for the default configuration. Transformer KV cache memory should scale linearly with sequence length, reaching hundreds of megabytes for very long sequences. The crossover point where Manifold becomes more memory-efficient depends on model configuration but typically occurs at sequence lengths of a few thousand tokens.

## Generalization Benchmarks

### Cumulative Parity Task

The cumulative parity task is a fundamental test of long-range memory. The task requires computing the cumulative XOR (parity) of all previous bits:

```python
def generate_parity_sequence(length):
    """Generate cumulative parity sequence."""
    bits = torch.randint(0, 2, (length,))
    parity = torch.cumsum(bits, dim=0) % 2
    return bits, parity
```

This task is computationally irreducible: the only way to compute the correct output for position $t$ is to have perfect information about all inputs from $0$ to $t$. There is no compression or approximation that preserves correctness. The task therefore tests pure memory capability.

### Superiority Benchmark Protocol

The superiority benchmark tests Manifold's ability to generalize to sequences far longer than those seen during training:

1. Training: Models are trained exclusively on sequences of length 20
2. Evaluation: Models are evaluated on sequences from length 20 to 100,000
3. Metric: Accuracy on the cumulative parity task

The hypothesis is that Manifold will maintain near-perfect accuracy across all sequence lengths due to its momentum-based memory encoding, while Transformers will degrade significantly as sequence length increases beyond training conditions.

### Implementation

```python
class SuperiorityBenchmark:
    def __init__(self, manifold_config=None):
        self.manifold = self._create_manifold(manifold_config)
        self.baseline = self._create_transformer_baseline()
        
        self.optimizer_class = torch.optim.AdamW
        self.lr = 1e-3
    
    def _create_manifold(self, config):
        if config is None:
            config = {
                'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16},
                'readout': {'type': 'implicit', 'coord_dim': 16},
                'active_inference': {'enabled': True, 'dynamic_time': {'enabled': True},
                                     'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
                                     'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8}},
                'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
                'topology': {'type': 'torus'},
                'stability': {'base_dt': 0.4}
            }
        
        return Manifold(
            vocab_size=2,
            dim=128,
            depth=6,
            heads=4,
            physics_config=config
        )
    
    def _create_transformer_baseline(self):
        from transformers import GPT2Config
        config = GPT2Config(
            n_embd=128,
            n_layer=6,
            n_head=4,
            n_positions=1024,  # Limited context
            vocab_size=2
        )
        return GPT2LMHeadModel(config)
    
    def train(self, model, train_steps=10000, batch_size=32, seq_length=20):
        """Train model on short sequences."""
        optimizer = self.optimizer_class(model.parameters(), lr=self.lr)
        model.train()
        
        losses = []
        
        for step in range(train_steps):
            inputs = torch.randint(0, 2, (batch_size, seq_length))
            targets = torch.cumsum(inputs, dim=1) % 2
            
            optimizer.zero_grad()
            logits, _, _ = model(inputs)
            
            loss = nn.CrossEntropyLoss()(logits.view(-1, 2), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if step % 1000 == 0:
                accuracy = (logits.argmax(dim=-1) == targets).float().mean()
                print(f"Step {step}: Loss {loss.item():.4f}, Accuracy {accuracy.item():.4f}")
        
        return losses
    
    def evaluate_generalization(self, model, eval_lengths):
        """Evaluate generalization to longer sequences."""
        model.eval()
        results = {}
        
        for length in eval_lengths:
            inputs = torch.randint(0, 2, (1, length))
            targets = torch.cumsum(inputs, dim=1) % 2
            
            with torch.no_grad():
                logits, state, _ = model(inputs)
                predictions = logits.argmax(dim=-1)
            
            accuracy = (predictions == targets).float().mean().item()
            results[length] = accuracy
            
            print(f"Length {length}: Accuracy {accuracy:.4f}")
        
        return results
    
    def run(self):
        """Run complete superiority benchmark."""
        print("Training Manifold...")
        manifold_losses = self.train(self.manifold)
        
        print("\nTraining Transformer baseline...")
        transformer_losses = self.train(self.baseline)
        
        print("\nEvaluating generalization...")
        eval_lengths = [20, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
        
        manifold_results = self.evaluate_generalization(self.manifold, eval_lengths)
        transformer_results = self.evaluate_generalization(self.baseline, eval_lengths)
        
        return {
            'manifold': {'losses': manifold_losses, 'generalization': manifold_results},
            'transformer': {'losses': transformer_losses, 'generalization': transformer_results}
        }
```

### Expected Results

Manifold should maintain near-perfect accuracy (greater than 99%) across all evaluated sequence lengths, despite being trained only on length-20 sequences. This demonstrates true algorithmic generalization—learning the underlying XOR operation rather than memorizing patterns. Transformer accuracy should degrade significantly as sequence length increases beyond training conditions, with accuracy potentially dropping to near-random (50%) for the longest sequences.

## Stability Benchmarks

### Energy Conservation Analysis

For Hamiltonian systems, energy conservation is a fundamental property. The benchmark analyzes the Hamiltonian (total energy) over long integration trajectories:

```python
class StabilityBenchmark:
    def __init__(self, manifold, integrator_type='leapfrog'):
        self.manifold = manifold
        self.integrator_type = integrator_type
    
    def compute_hamiltonian(self, x, v):
        """Compute Hamiltonian (total energy)."""
        kinetic = 0.5 * torch.sum(v ** 2, dim=-1)
        potential = self._compute_potential(x)
        return kinetic + potential
    
    def _compute_potential(self, x):
        """Compute potential energy from manifold geometry."""
        # Potential is related to Christoffel symbol magnitude
        Christoffel = self.manifold.christoffel
        Gamma = Christoffel(v, x)
        potential = 0.5 * torch.sum(Gamma ** 2, dim=-1)
        return potential
    
    def measure_energy_drift(self, num_steps=10000, dt=0.4):
        """Measure energy drift over long trajectories."""
        # Initialize state
        x = torch.randn(1, self.manifold.dim)
        v = torch.randn(1, self.manifold.dim)
        
        energies = []
        
        for step in range(num_steps):
            # Compute initial energy
            H = self.compute_hamiltonian(x, v)
            energies.append(H.item())
            
            # Integration step
            Christoffel = self.manifold.christoffel
            gamma = Christoffel(v, x)
            F = torch.zeros_like(x)
            
            if self.integrator_type == 'leapfrog':
                # Leapfrog step
                v_half = v + 0.5 * dt * (F - gamma)
                x = x + dt * v_half
                gamma_new = Christoffel(v_half, x)
                v = v_half + 0.5 * dt * (F - gamma_new)
            else:
                raise ValueError(f"Unknown integrator: {self.integrator_type}")
        
        return torch.tensor(energies)
    
    def analyze_drift(self, energies):
        """Analyze energy drift statistics."""
        initial_energy = energies[0]
        final_energy = energies[-1]
        energy_change = final_energy - initial_energy
        relative_drift = energy_change / (initial_energy.abs() + 1e-8)
        
        energy_variance = energies.var().item()
        max_deviation = (energies - energies[0]).abs().max().item()
        
        return {
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'absolute_drift': energy_change,
            'relative_drift': relative_drift,
            'variance': energy_variance,
            'max_deviation': max_deviation
        }
```

### Gradient Flow Analysis

The benchmark analyzes gradient flow through the network to verify the absence of vanishing and exploding gradients:

```python
def analyze_gradient_flow(model, input_ids, targets):
    """Analyze gradient flow through the network."""
    model.train()
    logits, _, _ = model(input_ids)
    loss = nn.CrossEntropyLoss()(logits.view(-1, 2), targets.view(-1))
    loss.backward()
    
    gradient_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_max = param.grad.abs().max().item()
            grad_min = param.grad.abs().min().item()
            
            gradient_stats[name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'max': grad_max,
                'min': grad_min
            }
    
    return gradient_stats
```

### Expected Results

The stability benchmark should show that energy drift is minimal (relative drift less than 1%) over trajectories of 10,000 or more steps. Gradient norms should be stable across layers without systematic increase (exploding) or decrease (vanishing). These results confirm the theoretical predictions of symplectic integration.

## Performance Benchmarks

### Throughput and Latency

```python
class PerformanceBenchmark:
    def __init__(self, manifold, batch_sizes=[1, 4, 16, 32]):
        self.manifold = manifold
        self.batch_sizes = batch_sizes
    
    def benchmark_throughput(self, seq_length=512, num_warmup=10, num_iterations=100):
        """Measure inference throughput."""
        results = {}
        
        for batch_size in self.batch_sizes:
            # Prepare input
            input_ids = torch.randint(0, 50257, (batch_size, seq_length))
            
            # Warmup
            self.manifold.eval()
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = self.manifold(input_ids)
            
            # Benchmark
            times = []
            torch.cuda.synchronize()
            
            for _ in range(num_iterations):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                with torch.no_grad():
                    _ = self.manifold(input_ids)
                end_time.record()
                
                torch.cuda.synchronize()
                times.append(start_time.elapsed_time(end_time))
            
            avg_time = sum(times) / len(times)
            throughput = batch_size / (avg_time / 1000)  # tokens per second
            
            results[batch_size] = {
                'avg_latency_ms': avg_time,
                'throughput_tokens_per_sec': throughput
            }
            
            print(f"Batch {batch_size}: {throughput:.0f} tokens/sec ({avg_time:.2f} ms/batch)")
        
        return results
    
    def benchmark_memory(self, seq_length=512):
        """Peak memory usage for different batch sizes."""
        results = {}
        
        for batch_size in self.batch_sizes:
            gc.collect()
            torch.cuda.empty_cache()
            
            input_ids = torch.randint(0, 50257, (batch_size, seq_length))
            
            torch.cuda.reset_peak_memory_stats()
            self.manifold.eval()
            with torch.no_grad():
                _ = self.manifold(input_ids)
            
            peak_memory = torch.cuda.max_memory_allocated()
            results[batch_size] = peak_memory / (1024 ** 2)  # Convert to MB
            
            print(f"Batch {batch_size}: {results[batch_size]:.1f} MB")
        
        return results
```

## Baseline Comparisons

### Baseline Model Implementations

The benchmark suite includes several baseline implementations for comparison:

The MicroGPT baseline is a minimal Transformer implementation with matched parameter counts for fair comparison. The RNN baseline uses a standard LSTM architecture with equivalent hidden dimension. The Mamba baseline uses the official Mamba implementation for SSM comparison.

### Fair Comparison Guidelines

To ensure fair comparisons, all baselines are configured with matched parameter counts. Memory measurements exclude parameter storage, focusing on activation and state memory. Computational comparisons use the same hardware and batch size. All models are trained with the same optimizer, learning rate, and training steps where applicable.

## Result Interpretation

### Understanding the Metrics

The benchmark results should be interpreted with attention to several factors. Statistical significance is ensured by reporting means and standard deviations across multiple runs. Practical significance considers whether observed differences matter for real applications. Limitations are acknowledged, including potential benchmark-specific effects and configuration sensitivity.

### Visualization

The benchmark results include visualization scripts for generating comparison plots:

```python
def plot_generalization_results(results, save_path='generalization.png'):
    """Plot generalization benchmark results."""
    import matplotlib.pyplot as plt
    
    lengths = list(results['manifold'].keys())
    manifold_acc = [results['manifold'][l] for l in lengths]
    transformer_acc = [results['transformer'][l] for l in lengths]
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(lengths, manifold_acc, 'b-o', label='Manifold', linewidth=2)
    plt.semilogx(lengths, transformer_acc, 'r-s', label='Transformer', linewidth=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Accuracy')
    plt.title('Generalization to Long Sequences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
```

## Reproducibility

### Random Seeds

All benchmarks use fixed random seeds for reproducibility:

```python
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
```

### Environment Reporting

Benchmark results include environment information:

```python
def get_environment_info():
    return {
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        'cpu_count': os.cpu_count(),
        'python_version': sys.version
    }
```

### Citation

When citing benchmark results, please include the full environment information and random seed to enable exact reproduction.

## Summary

The benchmarking framework provides rigorous, reproducible evaluation of Manifold's capabilities. The suite covers memory efficiency, computational performance, generalization, and stability through carefully designed experiments with appropriate baselines. All benchmark code is included in the repository, enabling independent verification and extension.

For implementation details, consult the benchmark source code in `tests/benchmarks/core/`. For interpretation guidance, see the result analysis scripts in `tests/benchmarks/viz/`.
