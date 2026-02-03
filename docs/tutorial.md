# Manifold Tutorial: From Basics to Advanced Applications

## Introduction

This tutorial provides a comprehensive guide to using Manifold for sequence modeling tasks, progressing from basic concepts through advanced applications. The tutorial assumes familiarity with PyTorch and deep learning fundamentals but does not require prior knowledge of the mathematical concepts underlying Manifold. Each section builds upon previous sections, with practical examples and exercises to reinforce understanding.

The tutorial is organized into five major sections. The first section introduces the basic workflow of creating, training, and using Manifold models. The second section explores the configuration system in depth, showing how to customize every aspect of the architecture. The third section demonstrates complete applications including text generation, sequence classification, and multimodal tasks. The fourth section covers advanced topics including custom integrators, distributed training, and optimization techniques. The fifth section provides troubleshooting guidance and best practices for production deployment.

### Learning Objectives

After completing this tutorial, you will understand how Manifold models are structured and why they differ from conventional architectures. You will be able to create and configure models for various sequence modeling tasks. You will be capable of implementing custom components including integrators, loss functions, and active inference mechanisms. You will be prepared to debug issues, optimize performance, and deploy models in production environments.

## Section 1: Getting Started with Manifold

### Your First Manifold Model

The simplest possible Manifold model demonstrates the core concepts with minimal configuration:

```python
import torch
import torch.nn as nn
from gfn import Manifold

# Create a minimal model for binary sequence modeling
model = Manifold(
    vocab_size=2,      # Binary vocabulary (0, 1)
    dim=64,            # Hidden dimension
    depth=2,           # Number of M-layers
    heads=2,           # Number of attention heads
    integrator_type='leapfrog'  # Symplectic integrator
)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
```

This model can be trained on simple sequence tasks like the cumulative parity task, where it demonstrates remarkable generalization capabilities. The model maintains constant memory during inference regardless of sequence length, a property that distinguishes Manifold from Transformers.

### Running a Forward Pass

Understanding the forward pass is essential for effective use of Manifold:

```python
# Prepare input: batch of sequences
batch_size = 4
seq_length = 16
input_ids = torch.randint(0, 2, (batch_size, seq_length))

# Forward pass
logits, state, trajectory = model(input_ids)

print(f"Input shape: {input_ids.shape}")
print(f"Logits shape: {logits.shape}")  # [batch, seq_len, vocab_size]
print(f"Final position shape: {state[0].shape}")  # [batch, dim]
print(f"Final velocity shape: {state[1].shape}")  # [batch, dim]
print(f"Trajectory length: {len(trajectory)}")  # Number of timesteps
```

The forward pass returns three objects: the output logits for prediction, the final state $(x, v)$ for stateful inference, and the full trajectory for diagnostic purposes. The state can be passed to subsequent forward passes to maintain context across multiple inference steps.

### Training on a Simple Task

The cumulative parity task demonstrates Manifold's memory capabilities:

```python
import torch.nn.functional as F

def generate_parity_batch(batch_size, seq_length):
    """Generate a batch of cumulative parity sequences."""
    inputs = torch.randint(0, 2, (batch_size, seq_length))
    targets = torch.cumsum(inputs, dim=1) % 2  # Cumulative XOR
    return inputs, targets

# Training loop
model = Manifold(vocab_size=2, dim=128, depth=4, heads=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    inputs, targets = generate_parity_batch(32, 20)
    
    optimizer.zero_grad()
    logits, _, _ = model(inputs)
    
    loss = criterion(logits.view(-1, 2), targets.view(-1))
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()
        print(f"Epoch {epoch}: Loss {loss.item():.4f}, Accuracy {accuracy.item():.4f}")
```

Training on sequences of length 20 and testing on sequences of length 100,000 demonstrates Manifold's ability to generalize beyond its training distribution through momentum-based memory encoding.

### Exercise 1.1: Basic Training

Create and train a Manifold model on the copy task, where the model must reproduce an input sequence after a delay. Observe how the model's performance changes with sequence length. Compare the memory usage of Manifold against a Transformer on the same task.

## Section 2: Configuration Deep Dive

### Understanding the Configuration Structure

Manifold's configuration system uses nested dictionaries that mirror the internal architecture:

```python
# Complete configuration structure
config = {
    # Embedding configuration
    'embedding': {
        'type': 'functional',     # functional, implicit, standard
        'mode': 'linear',         # linear, binary
        'coord_dim': 16
    },
    
    # Readout configuration
    'readout': {
        'type': 'implicit',       # implicit, explicit
        'coord_dim': 16
    },
    
    # Active inference configuration
    'active_inference': {
        'enabled': True,
        'dynamic_time': {'enabled': True},
        'reactive_curvature': {
            'enabled': True,
            'plasticity': 0.2
        },
        'singularities': {
            'enabled': True,
            'strength': 20.0,
            'threshold': 0.8
        }
    },
    
    # Fractal configuration
    'fractal': {
        'enabled': True,
        'threshold': 0.5,
        'alpha': 0.2
    },
    
    # Topology configuration
    'topology': {
        'type': 'torus'           # torus, sphere, plane
    },
    
    # Stability configuration
    'stability': {
        'base_dt': 0.4
    }
}

model = Manifold(
    vocab_size=50257,
    dim=512,
    depth=6,
    heads=4,
    physics_config=config
)
```

### Embedding Options

The embedding configuration determines how input tokens are transformed into manifold coordinates:

```python
# Option 1: Functional embedding (recommended)
# O(1) vocabulary scaling, best generalization
functional_config = {
    'type': 'functional',
    'mode': 'linear',     # Smooth interpolation
    'coord_dim': 16
}

# Option 2: Implicit embedding
# Learnable coordinate table, moderate vocabulary scaling
implicit_config = {
    'type': 'implicit',
    'coord_dim': 16
}

# Option 3: Standard embedding
# Fixed lookup table, simplest implementation
standard_config = {
    'type': 'standard'
}
```

The functional embedding with linear mode is recommended for most applications due to its superior out-of-distribution generalization. The mode parameter determines how token IDs are mapped to coordinates: linear mode provides smooth interpolation, while binary mode uses discrete binary representations.

### Integrator Options

Different integrators provide different tradeoffs between accuracy and speed:

```python
# Leapfrog integrator (recommended)
# Second-order accurate, symplectic, good speed
leapfrog_config = {
    'type': 'leapfrog',
    'base_dt': 0.4
}

# Forest-Ruth integrator
# Fourth-order accurate, symplectic, slower
forest_ruth_config = {
    'type': 'forest_ruth',
    'base_dt': 0.4
}

# Heun integrator
# Second-order accurate, not symplectic, good balance
heun_config = {
    'type': 'heun',
    'base_dt': 0.4
}

# RK4 integrator
# Fourth-order accurate, not symplectic, highest precision
rk4_config = {
    'type': 'rk4',
    'base_dt': 0.4
}
```

The leapfrog integrator is recommended for most applications due to its symplectic structure, which ensures volume preservation and gradient stability. Higher-order integrators provide greater accuracy at increased computational cost.

### Active Inference Tuning

The active inference parameters can be tuned for different tasks:

```python
# Conservative configuration (stable, less adaptive)
conservative_config = {
    'enabled': True,
    'dynamic_time': {'enabled': False},
    'reactive_curvature': {
        'enabled': True,
        'plasticity': 0.1  # Slow adaptation
    },
    'singularities': {
        'enabled': True,
        'strength': 10.0,   # Mild singularities
        'threshold': 0.9    # High threshold
    }
}

# Aggressive configuration (adaptive, potentially unstable)
aggressive_config = {
    'enabled': True,
    'dynamic_time': {'enabled': True},
    'reactive_curvature': {
        'enabled': True,
        'plasticity': 0.5  # Fast adaptation
    },
    'singularities': {
        'enabled': True,
        'strength': 30.0,   # Strong singularities
        'threshold': 0.7    # Low threshold
    }
}
```

The default parameters provide a balance between adaptation and stability. For tasks requiring rapid context switching, the aggressive configuration may be appropriate. For tasks requiring stable long-term memory, the conservative configuration is recommended.

### Exercise 2.1: Configuration Comparison

Train identical Manifold models with different embedding configurations on the parity task. Compare the generalization to sequences 10x and 100x longer than the training sequences. Analyze how the embedding type affects out-of-distribution performance.

## Section 3: Complete Applications

### Text Generation

A complete text generation pipeline using Manifold:

```python
import torch
from gfn import Manifold, RiemannianAdam

class ManifoldLM:
    def __init__(self, vocab_size, dim=512, depth=6, heads=4):
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
        
        self.model = Manifold(
            vocab_size=vocab_size,
            dim=dim,
            depth=depth,
            heads=heads,
            physics_config=config
        )
        self.tokenizer = None  # Set externally
    
    def train(self, dataloader, epochs=10, lr=1e-4):
        optimizer = RiemannianAdam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids, labels = batch
                
                optimizer.zero_grad()
                logits, _, _ = self.model(input_ids)
                
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.05)
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}: Avg Loss {total_loss/len(dataloader):.4f}")
    
    @torch.no_grad()
    def generate(self, prompt, max_tokens=100, temperature=1.0, top_k=40):
        self.model.eval()
        generated = list(prompt)
        state = None
        
        for _ in range(max_tokens):
            input_ids = torch.tensor([generated[-1]]).unsqueeze(0)
            
            logits, state, _ = self.model(input_ids, state=state)
            logits = logits / temperature
            logits = top_k_filtering(logits, top_k)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            if next_token == EOS_TOKEN:
                break
        
        return generated

def top_k_filtering(logits, top_k=40):
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits
```

This implementation provides a complete language model pipeline with training and autoregressive generation. The state persistence during generation demonstrates Manifold's O(1) inference memory.

### Sequence Classification

Sequence classification with Manifold:

```python
class ManifoldClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, dim=256, depth=4, heads=4):
        super().__init__()
        
        config = {
            'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16},
            'readout': {'type': 'explicit', 'coord_dim': 16},
            'active_inference': {'enabled': True, 'dynamic_time': {'enabled': True},
                                 'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
                                 'singularities': {'enabled': False}},
            'fractal': {'enabled': False},
            'topology': {'type': 'torus'},
            'stability': {'base_dt': 0.4}
        }
        
        self.manifold = Manifold(
            vocab_size=vocab_size,
            dim=dim,
            depth=depth,
            heads=heads,
            physics_config=config
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask=None):
        logits, (x_final, v_final), _ = self.manifold(input_ids)
        
        # Use final state for classification
        if attention_mask is not None:
            # Mask out padding positions
            mask = attention_mask.unsqueeze(-1).expand_as(x_final)
            x_final = x_final * mask
            lengths = attention_mask.sum(dim=1, keepdim=True)
            x_final = x_final.sum(dim=1) / lengths
        else:
            x_final = x_final.mean(dim=1)
        
        return self.classifier(x_final)
```

This classifier uses the final manifold state for prediction. The attention mask handling allows processing of variable-length sequences with padding.

### Exercise 3.1: Build a Complete Application

Implement a sequence-to-sequence model using Manifold for machine translation. Use a Manifold encoder to process the source sequence and a separate Manifold decoder (with cross-attention to encoder states) to generate the target sequence.

## Section 4: Advanced Topics

### Custom Integrators

Implementing a custom integrator requires subclassing the SymplecticIntegrator base class:

```python
from gfn.integrators import SymplecticIntegrator

class CustomIntegrator(SymplecticIntegrator):
    def __init__(self, dt=0.4, order=2, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.order = order
    
    def step(self, x, v, F, christoffel_fn, **kwargs):
        """Custom integration step."""
        if self.order == 2:
            # Second-order scheme (Leapfrog-like)
            gamma = christoffel_fn(v, x)
            v_half = v + 0.5 * self.dt * (F - gamma)
            x_new = x + self.dt * v_half
            gamma_new = christoffel_fn(v_half, x_new)
            v_new = v_half + 0.5 * self.dt * (F - gamma_new)
        else:
            # Higher-order scheme would go here
            raise NotImplementedError("Higher-order schemes not implemented")
        
        # Apply stability mechanisms
        v_new = self._normalize_velocity(v_new)
        
        return x_new, v_new
    
    def _normalize_velocity(self, v):
        """Normalize velocity to prevent explosion."""
        norm = v.norm(dim=-1, keepdim=True)
        return v / (norm + 1e-6)
```

The custom integrator must preserve the symplectic structure to ensure gradient stability. The velocity normalization is applied as a post-processing step to maintain numerical stability.

### Distributed Training

Manifold supports distributed training across multiple GPUs:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_distributed(rank, world_size, model, train_loader, epochs):
    setup_distributed(rank, world_size)
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    optimizer = RiemannianAdam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(rank)
            targets = targets.to(rank)
            
            optimizer.zero_grad()
            logits, _, _ = model(inputs)
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
        
        if rank == 0:
            print(f"Epoch {epoch+1} completed")
    
    dist.destroy_process_group()

# Launch with torchrun
# torchrun --nproc_per_node=4 train_distributed.py
```

The distributed training implementation uses PyTorch's native DDP for gradient synchronization. The RiemannianAdam optimizer is compatible with distributed training and maintains geometric constraints across all replicas.

### Memory Optimization

For memory-constrained environments, several techniques reduce memory requirements:

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientManifold(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Standard initialization
        self.manifold = Manifold(**config)
    
    def forward(self, x, use_checkpointing=True):
        if use_checkpointing:
            # Use gradient checkpointing for memory efficiency
            return self._checkpointed_forward(x)
        else:
            return self.manifold(x)
    
    def _checkpointed_forward(self, x):
        """Forward pass with gradient checkpointing."""
        def run_layer(layer, input):
            return layer(input)
        
        logits, state, traj = x, None, []
        for i in range(len(self.manifold.layers)):
            if i == 0:
                logits = checkpoint(run_layer, self.manifold.layers[i], logits)
            else:
                logits = checkpoint(run_layer, self.manifold.layers[i], logits, state)
                state = self.manifold.layers[i].state
            
            traj.append(logits)
        
        return self.manifold.readout(logits), state, traj
```

Gradient checkpointing trades compute for memory by recomputing intermediate activations during the backward pass instead of storing them during the forward pass. This technique can significantly reduce memory requirements for large models.

### Exercise 4.1: Implement and Benchmark

Implement the Forest-Ruth integrator (fourth-order symplectic integrator) and benchmark its accuracy versus computational cost compared to the leapfrog integrator. Analyze at what sequence lengths the accuracy improvement justifies the computational overhead.

## Section 5: Troubleshooting and Best Practices

### Common Issues and Solutions

**Gradient NaN or Inf**: This typically indicates numerical instability. Reduce the learning rate, enable stricter gradient clipping, and check that curvature clamping is active. Verify that the integration timestep is not too large for the dynamics being modeled.

**Poor Convergence**: Ensure that the configuration uses recommended settings. The functional embedding with linear mode should be used unless there is a specific reason to change it. Verify that RiemannianAdam is being used instead of standard optimizers. Check that gradient clipping threshold is appropriate (0.05 is recommended).

**Memory Issues During Training**: Reduce batch size, enable gradient checkpointing, and consider reducing sequence length during training. The model can be trained on shorter sequences and still generalize to longer sequences due to its momentum-based memory.

**Slow Inference**: Ensure that CUDA kernels are compiled for maximum performance. Use the appropriate batch size for your hardware. Consider quantization for deployment in resource-constrained environments.

### Performance Optimization Checklist

When optimizing for production deployment, consider the following optimizations:

The first category involves model optimization. Model quantization reduces precision from 32-bit to 16-bit or 8-bit integers, significantly reducing memory and improving inference speed. Operator fusion combines sequential operations into single optimized kernels, reducing memory bandwidth requirements. Layer pruning removes less important heads or layers to reduce computation while maintaining accuracy.

The second category involves inference optimization. Batch processing aggregates multiple sequences into a single forward pass, improving hardware utilization. Dynamic batching adaptively groups sequences based on length, maximizing throughput. Caching compiled models prevents recompilation overhead in production systems.

The third category involves hardware optimization. Mixed precision training uses 16-bit precision where possible, reducing memory and increasing throughput. Tensor core utilization ensures efficient use of NVIDIA GPU hardware. Memory binding optimization places data on appropriate memory types (HBM, VRAM, RAM) based on access patterns.

### Best Practices Summary

The following practices have been found to improve success with Manifold:

Configuration best practices include starting with the optimal configuration and deviating only when necessary. The functional embedding with linear mode should be the default choice. Active inference should be enabled unless there is a specific reason to disable it. The leapfrog integrator provides the best balance of stability and speed for most applications.

Training best practices include using RiemannianAdam for optimization. Gradient clipping should be set to 0.05. Learning rates should be similar to those used with Transformers. Longer training with smaller models often outperforms shorter training with larger models.

Inference best practices include leveraging state persistence for O(1) memory. Temperature and top-k filtering should be tuned for specific applications. State can be saved and restored for checkpointing across long sequences.

## Conclusion

This tutorial has provided a comprehensive introduction to Manifold, covering basic usage, configuration options, complete applications, advanced topics, and troubleshooting guidance. The geometric framework of Manifold provides principled approaches to sequence modeling that complement purely empirical methods. Users who invest in understanding the mathematical foundations will be better equipped to leverage Manifold's full capabilities and adapt the framework to novel applications.

For further exploration, consult the Architecture document for detailed component descriptions, the Mathematical Foundations document for deeper treatment of the theory, and the API Reference for complete parameter documentation. The demos directory contains additional examples covering various applications and use cases.
