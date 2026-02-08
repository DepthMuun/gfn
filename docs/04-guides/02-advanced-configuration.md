# Advanced Configuration

## Philosophy

Advanced configuration allows you to adjust system behavior beyond default values. This section documents parameter interactions and optimization strategies.

## Configuration Hierarchy

Configuration follows this hierarchy:

1. Code constants (gfn/constants.py) - lowest priority
2. YAML configuration files - medium priority
3. Command-line arguments - higher priority
4. Environment variables - highest priority

For debugging, use the highest priority level. For production, use YAML files.

## Model Parameters

### Dimension and Depth

Dimension (dim) and depth control model capacity.

```yaml
model:
  dim: 512        # Embedding dimension
  depth: 6        # Number of processing layers
  rank: 64        # Rank for low-rank approximation
  heads: 8        # Attention heads
```

**Guidelines:**

- dim >= 4 * heads for good distribution
- depth between 4 and 12 for most tasks
- rank between dim/8 and dim/4 for accuracy/speed balance

For large models:

```yaml
model:
  dim: 1024
  depth: 12
  rank: 128
  heads: 16
```

For small models (embedded, edge devices):

```yaml
model:
  dim: 256
  depth: 4
  rank: 32
  heads: 4
```

### Integrator Type

The integrator affects accuracy and stability.

```yaml
model:
  integrator_type: "leapfrog"  # Default, balanced
  # integrator_type: "yoshida"  # More precise, slower
  # integrator_type: "heun"     # More stable, less precise
  # integrator_type: "forest_ruth"  # Alternative to Yoshida
```

## Physics Parameters

### Friction

Friction damps dynamics and prevents infinite oscillations.

```yaml
physics:
  friction_scale: 0.02      # Main friction coefficient
  default_friction: 0.002   # Base friction
  velocity_friction_scale: 0.02  # Velocity-dependent friction
```

**Interactions:**

- High friction + high LR = divergence
- Low friction + low LR = slow convergence
- Very low friction = persistent oscillations

**Tuning:**

For fast convergence: friction_scale = 0.05, LR = 1e-3

For maximum stability: friction_scale = 0.1, LR = 1e-4

For maximum exploration: friction_scale = 0.005, LR = 5e-4

### Timestep

The timestep (dt) controls integration granularity.

```yaml
physics:
  dt: 0.05                  # Default timestep
  leapfrog_substeps: 3      # Substeps per token
```

**Interactions:**

- High dt + low substeps = unstable
- Low dt + high substeps = stable but slow
- dt * substeps = effective time per token

**Typical combinations:**

```yaml
# Fast and unstable
physics:
  dt: 0.1
  leapfrog_substeps: 1

# Balanced (default)
physics:
  dt: 0.05
  leapfrog_substeps: 3

# Precise and slow
physics:
  dt: 0.02
  leapfrog_substeps: 5
```

### Physical Regularization

Loss weights control how much the model respects physics.

```yaml
physics:
  lambda_h: 0.0      # Energy conservation
  lambda_g: 0.00005  # Geodesic regularization
  lambda_k: 0.0001   # Kinetic energy
  lambda_n: 0.0      # Noether symmetries
```

**Effects:**

- lambda_h > 0: The model conserves energy better but may underfit
- lambda_g > 0: Trajectories are more geodesic
- lambda_k > 0: Additional damping

**Guidelines:**

- For simple tasks: lambda_g = 0
- For tasks that require reasoning: lambda_g = 0.0001
- For maximum stability: lambda_h = 0.001, lambda_k = 0.001

## Optimizer Parameters

### Learning Rate

The learning rate is the most important hyperparameter.

```yaml
training:
  learning_rate: 0.0001  # Default
  warmup_steps: 100      # Warmup steps
  warmup_ratio: 0.1      # Alternative warmup ratio
```

**Interactions:**

- High LR + small batch = divergence
- Low LR + large batch = slow convergence
- Optimal LR depends on batch size and model

**Schedule:**

The system uses linear warmup + cosine decay:

```python
lr = base_lr * warmup_ratio (warmup)
lr = base_lr * 0.5 * (1 + cos(pi * progress)) (decay)
```

**Tuning for different scales:**

| Model Scale | Recommended LR |
|------------------|----------------|
| small (dim=256) | 5e-4 |
| medium (dim=512) | 1e-4 |
| large (dim=1024) | 5e-5 |

### Adam Parameters

```yaml
training:
  adam_beta1: 0.9       # Momentum
  adam_beta2: 0.99      # RMSProp
  adam_epsilon: 1e-7    # Stability
  weight_decay: 0.001   # L2 regularization
```

**Guidelines:**

- beta2 = 0.99 for noisy data
- beta2 = 0.999 for clean data
- weight_decay = 0.01 for severe overfitting

### Gradient Clipping

```yaml
training:
  grad_clip_norm: 1.0    # Norm clipping
  grad_clip_value: 1.0   # Absolute value clipping
```

Enable when:

- Gradients explode during training
- Loss oscillates strongly
- After large LR changes

## Initialization Parameters

### Initialization Scales

```yaml
initialization:
  init_std: 0.01        # Standard deviation
  init_x0_scale: 0.01   # Position scale
  init_v0_scale: 0.005  # Velocity scale
  gate_bias_open: 1.0   # Open gate bias
  gate_bias_closed: -3.0 # Closed gate bias
```

**Guidelines:**

- Lower init_std for deep models
- init_v0_scale < init_x0_scale to avoid initial oscillations
- Low gate_bias_open for conservative gates

## Preset Configurations

### Stable Configuration

For maximum production stability:

```yaml
model:
  dim: 512
  depth: 6
  rank: 64
  integrator_type: "leapfrog"

physics:
  friction_scale: 0.05
  dt: 0.02
  leapfrog_substeps: 5
  lambda_g: 0.0001

training:
  learning_rate: 5e-5
  adam_beta2: 0.99
  grad_clip_norm: 0.5
  warmup_steps: 200
```

### Fast Configuration

For fast iterative experiments:

```yaml
model:
  dim: 256
  depth: 4
  rank: 32
  integrator_type: "heun"

physics:
  friction_scale: 0.02
  dt: 0.1
  leapfrog_substeps: 1
  lambda_g: 0.0

training:
  learning_rate: 1e-3
  warmup_steps: 50
```

### High-Precision Configuration

For maximum trajectory quality:

```yaml
model:
  dim: 512
  depth: 8
  rank: 128
  integrator_type: "yoshida"

physics:
  friction_scale: 0.01
  dt: 0.02
  leapfrog_substeps: 5
  lambda_g: 0.0002
  lambda_h: 0.001

training:
  learning_rate: 5e-5
  adam_beta2: 0.999
  grad_clip_norm: 1.0
```

## GPU Parameters

### Memory Configuration

```yaml
gpu:
  max_batch_size: 16    # Batch limit per GPU
  precision: "float32"  # float32 or float16
  cuda_kernel: true     # Use CUDA kernels
```

float16 can cut memory usage in half but may cause numerical instability.

### Multi-GPU

```bash
# Distributed training
torchrun --nproc_per_node=4 train.py --config ...
```

The system automatically parallelizes across multiple GPUs with DataParallel.

## Configuration Debugging

### Environment Variables

```bash
# Force CPU
export GFN_DEVICE=cpu

# Disable CUDA
export GFN_USE_CUDA=0

# Verbose logging
export GFN_LOG_LEVEL=debug

# Save all checks
export GFN_SAFETY_CHECKS=all
```

### Gradient Logging

```python
# In your script
from gfn.utils import GradientMonitor

monitor = GradientMonitor(model)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    monitor.log()
```

### Performance Profiling

```bash
# Basic profiler
python -m cProfile -s cumulative train.py | head -50

# Detailed profiler with PyTorch
python -m torch.utils.bottleneck train.py
```

## Common Configuration Errors

**high lr + high friction = divergence**

If the loss diverges, lower LR first.

**high dt + low substeps = inaccurate trajectories**

If validation metrics are poor, increase dt or substeps.

**low rank = underfitting**

If the model does not learn, increase rank.

**high lambda_g = underfitting**

If the loss does not decrease, reduce lambda_g to 0.

## Version Notes

This documentation assumes the current development version. Some options may not be available in earlier versions (v2.6.5 or earlier).

---

**Manifold Labs (Joaquín Stürtz)**
