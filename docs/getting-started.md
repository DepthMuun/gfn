# Getting Started with Manifold

## Introduction

Manifold is a Geometric Flow Network (GFN) implementation that reformulates sequence modeling through the lens of Geometric Mechanics. Instead of storing explicit token histories like traditional Transformers, Manifold encodes context into the momentum of a dynamic particle traversing a curved semantic manifold. This approach yields a physically-structured State Space Model (SSM) with O(1) inference memory, infinite context horizons, and symplectic stability guarantees.

This guide provides a comprehensive introduction to Manifold, covering installation, basic usage patterns, and core concepts necessary for effective utilization of the framework. The document is structured to accommodate users with varying backgrounds, from machine learning practitioners familiar with standard architectures to researchers interested in the mathematical foundations of geometric deep learning.

### Core Capabilities

Manifold implements several fundamental capabilities that distinguish it from conventional sequence modeling approaches. The system provides constant-time inference memory regardless of sequence length, enabling processing of arbitrarily long contexts without memory scaling issues. The symplectic integration framework ensures gradient stability over infinite horizons, eliminating the vanishing and exploding gradient problems that plague standard recurrent architectures. The active inference mechanism enables adaptive dynamics that respond to uncertainty signals, providing a principled approach to the stability-plasticity dilemma. The functional embedding system supports vocabulary scalability without parameter growth, a critical property for applications with large or growing token vocabularies.

### Design Philosophy

The design of Manifold follows a physics-first philosophy where all operations are grounded in differential geometry and Hamiltonian mechanics. This approach provides theoretical guarantees that purely empirical architectures cannot offer. The geodesic equation governs state evolution, symplectic integration ensures phase-space volume preservation, and Liouville's theorem provides the foundation for understanding information flow. This mathematical rigor translates into practical benefits: guaranteed stability, interpretable dynamics, and principled mechanisms for handling memory and uncertainty.

## Installation

### System Requirements

Manifold requires Python 3.8 or higher and depends on several scientific computing libraries. The core dependencies include PyTorch 2.0 or higher for tensor operations and automatic differentiation, NumPy for array manipulations, and SciPy for scientific functions including special mathematical operations related to Riemannian geometry. Additional packages including Matplotlib and Seaborn are required for visualization modules, while TQDM provides progress bars for training loops.

For GPU acceleration, NVIDIA GPUs with compute capability 7.0 or higher (Volta architecture and newer) are recommended. The implementation supports CUDA 11.8 and 12.x for GPU acceleration. Apple Silicon Macs are supported through the MPS backend, though some features may have reduced functionality compared to CUDA implementations. A minimum of 16GB GPU memory is recommended for development work with standard sequence lengths, while production deployments should provision at least 40GB of GPU memory per GPU for larger configurations.

### Basic Installation

The simplest installation method uses pip to install the package directly from PyPI:

```bash
pip install gfn
```

This installation provides the core functionality of Manifold with pure PyTorch implementations of all major components. The installation is suitable for initial experimentation and development work where maximum performance is not critical.

### Development Installation

For users who wish to modify the source code or access the latest development features, cloning the repository and installing in editable mode is recommended:

```bash
git clone https://github.com/Manifold-Laboratory/manifold.git
cd manifold
pip install -e "."
```

This installation mode links the package to the local source directory, allowing changes to the code to take effect without reinstalling. Users who modify the source code can immediately test their changes by running Python scripts that import the package.

### CUDA Installation

For users requiring maximum performance, the CUDA kernels can be compiled to provide significant speedups on NVIDIA GPUs. The compilation process requires CUDA Toolkit and a compatible compiler:

```bash
# For CUDA 12.x (recommended)
cd manifold/gfn/cuda
python compile_cuda_kernels.py --cuda-version 12.9

# For CUDA 11.8
cd manifold/gfn/cuda
python compile_cuda_kernels.py --cuda-version 11.8
```

The compiled kernels provide 5-10x speedup over pure PyTorch implementations for the most computationally intensive operations. The build process automatically detects the installed CUDA version and configures the appropriate compiler flags.

### Verification

After installation, verify that the package is correctly installed and accessible:

```python
import torch
from gfn import Manifold

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create a minimal model for verification
model = Manifold(
    vocab_size=2,
    dim=64,
    depth=2,
    heads=2
)
print(f"Model created successfully: {sum(p.numel() for p in model.parameters())} parameters")
```

This verification script confirms that all dependencies are correctly installed and that the Manifold package can be imported without errors. The model creation test exercises the core components of the system.

## Quick Start

### Basic Model Instantiation

The following example demonstrates how to create and use a Manifold model for a simple sequence modeling task. This example uses the optimal configuration derived from extensive benchmarking:

```python
import torch
import torch.nn as nn
from gfn import Manifold, RiemannianAdam

# Configure the model with optimal settings
config = {
    'embedding': {
        'type': 'functional',
        'mode': 'linear',
        'coord_dim': 16
    },
    'readout': {
        'type': 'implicit',
        'coord_dim': 16
    },
    'active_inference': {
        'enabled': True,
        'dynamic_time': {'enabled': True},
        'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
        'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8}
    },
    'fractal': {
        'enabled': True,
        'threshold': 0.5,
        'alpha': 0.2
    },
    'topology': {
        'type': 'torus'
    },
    'stability': {
        'base_dt': 0.4
    }
}

# Initialize the model
model = Manifold(
    vocab_size=50257,  # Standard GPT-2 vocabulary size
    dim=512,
    depth=6,
    heads=4,
    integrator_type='leapfrog',
    physics_config=config,
    impulse_scale=80.0,
    holographic=True
)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

This configuration represents the optimal settings validated through the superiority benchmark suite. The functional embedding with linear mode provides superior out-of-distribution generalization compared to binary mode. The implicit readout enables the model to learn task-specific output transformations. Active inference mechanisms are fully enabled with carefully tuned parameters that balance adaptation speed against stability.

### Training Loop

Training a Manifold model follows patterns familiar to practitioners of deep learning, with several important considerations specific to the geometric framework:

```python
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device='cuda'):
    """Train a Manifold model with physics-aware optimization."""
    
    # Move model to device
    model = model.to(device)
    
    # Use RiemannianAdam for manifold-aware optimization
    optimizer = RiemannianAdam(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        retraction_mode='normalize'
    )
    
    # Learning rate schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, (x_final, v_final), trajectory = model(inputs)
            
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping is essential for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                logits, _, _ = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return model
```

Several aspects of this training loop warrant specific attention. The RiemannianAdam optimizer is required for proper optimization on the manifold parameter space. Standard Adam optimization assumes a Euclidean flat space, which is suboptimal for Riemannian models and can lead to instability. The gradient clipping threshold of 0.05 is stricter than the typical values of 0.1-1.0 used with standard architectures, reflecting the different gradient dynamics of geometric systems.

### Inference Pattern

Inference with Manifold follows a recurrent pattern that leverages the O(1) memory property:

```python
@torch.no_grad()
def generate(model, prompt, max_length=100, temperature=1.0, top_k=50, device='cuda'):
    """Generate text using autoregressive inference with O(1) memory."""
    
    model.eval()
    model = model.to(device)
    
    # Initialize state
    batch_size = 1
    state = None
    
    generated = list(prompt)
    
    for _ in range(max_length):
        # Prepare input
        input_ids = torch.tensor([generated[-1]]).unsqueeze(0).to(device)
        
        # Forward pass with state persistence
        logits, state, _ = model(input_ids, state=state)
        
        # Apply temperature and top-k filtering
        logits = logits / temperature
        logits = top_k_filtering(logits, top_k)
        
        # Sample next token
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        generated.append(next_token)
        
        # Early stopping on EOS token
        if next_token == EOS_TOKEN_ID:
            break
    
    return generated

def top_k_filtering(logits, top_k=50):
    """Filter logits to keep only top-k most likely tokens."""
    values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
    mask = torch.zeros_like(logits).scatter_(1, indices, 1.0)
    masked_logits = logits.masked_fill(mask == 0, float('-inf'))
    return masked_logits
```

This inference pattern demonstrates the key advantage of Manifold: the state variable maintains constant size regardless of how many tokens have been processed. The state encodes the entire context through geometric and momentum-based representations rather than explicit token storage.

## Configuration Guide

### Configuration Structure

Manifold uses a hierarchical configuration system that mirrors the internal architecture of the system. Understanding this structure is essential for effective customization and tuning:

```python
# Configuration hierarchy
config = {
    'embedding': {...},           # Input transformation
    'readout': {...},             # Output transformation
    'active_inference': {...},    # Adaptive dynamics
    'fractal': {...},             # Hierarchical structure
    'topology': {...},            # Global structure
    'stability': {...},           # Numerical parameters
}
```

Each top-level key corresponds to a major subsystem, with nested dictionaries providing detailed configuration options. This structure ensures consistency between components while allowing fine-grained control.

### Embedding Configuration

The embedding configuration controls how input sequences are transformed into manifold coordinates:

```python
embedding_config = {
    'type': 'functional',         # 'functional', 'implicit', or 'standard'
    'mode': 'linear',             # 'linear' or 'binary' (for functional type)
    'coord_dim': 16               # Coordinate dimensionality
}
```

The functional type uses neural fields (SIREN) to parameterize the embedding, providing O(1) vocabulary scalability. The linear mode provides smooth interpolation between token representations and has been empirically shown to outperform binary mode for out-of-distribution generalization. The coordinate dimension must be consistent across all components for proper geometric coherence.

### Active Inference Configuration

The active inference configuration enables uncertainty-aware learning and adaptive dynamics:

```python
active_inference_config = {
    'enabled': True,
    'dynamic_time': {
        'enabled': True           # Adaptive timestep based on uncertainty
    },
    'reactive_curvature': {
        'enabled': True,
        'plasticity': 0.2         # Rate of curvature adaptation
    },
    'singularities': {
        'enabled': True,
        'strength': 20.0,         # Intensity of induced singularities
        'threshold': 0.8          # Uncertainty threshold for activation
    }
}
```

These mechanisms allow the system to modulate its dynamics based on internal state estimates, providing a principled approach to balancing exploration and exploitation during learning.

### Integrator Configuration

The integrator configuration controls the numerical integration scheme:

```python
integrator_config = {
    'type': 'leapfrog',           # 'leapfrog', 'forest_ruth', 'heun', 'rk4'
    'base_dt': 0.4,               # Base timestep
}
```

The leapfrog integrator is recommended for most applications, providing a good balance of accuracy and computational efficiency. Higher-order integrators like forest_ruth and rk4 provide greater accuracy at increased computational cost.

## Next Steps

With the installation complete and basic usage understood, users should proceed to the following resources for deeper understanding:

The Architecture document provides detailed coverage of all Manifold components, their interactions, and design decisions. The Mathematical Foundations document explains the physics and geometry underlying the system in an accessible manner. The Implementation Guide provides advanced configuration options, optimization strategies, and troubleshooting advice. The API Reference provides complete documentation of all classes, functions, and parameters.

For hands-on experimentation, the demos directory contains example scripts covering various tasks including copy operations, sorting, and text generation. These examples provide starting points for custom applications.
