# MANIFOLD Training Guide


**Last Updated:** January 27, 2026

Complete guide for training MANIFOLD models, from quick start to advanced optimization.



## 1. Quick Start

### 1.1 Minimal Example

```python
import torch
from gfn.model import Manifold
from gfn.optim import RiemannianAdam

# Model
model = Manifold(vocab_size=1000, dim=256, depth=6, heads=4).cuda()

# CRITICAL: Use RiemannianAdam (standard Adam will fail)
optimizer = RiemannianAdam(
    model.parameters(),
    lr=1e-4,
    retraction='normalize',
    max_norm=10.0
)

# Training loop
criterion = torch.nn.CrossEntropyLoss()
for x, y in dataloader:
    x, y = x.cuda(), y.cuda()
    optimizer.zero_grad()
    
    logits, _, _ = model(x)
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    
    # Gradient clipping (REQUIRED)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
    
    optimizer.step()
```

### 1.2 Installation Verification

```python
# Verify RiemannianAdam is available
from gfn.optim import RiemannianAdam
print("RiemannianAdam available ✓")

# Verify physics is configured
model = Manifold(vocab_size=2, dim=128, depth=6, heads=4)
assert model.physics_config is not None
print("Physics configuration verified ✓")
```



## 2. Configuration

### 2.1 Basic Configuration (config.yaml)

```yaml
model:
  vocab_size: 50257
  dim: 512
  depth: 12
  heads: 8
  rank: 32
  integrator_type: leapfrog
  impulse_scale: 80.0
  holographic: true

optimizer:
  type: RiemannianAdam
  lr: 1e-4
  weight_decay: 1e-4
  retraction: normalize
  max_norm: 10.0

training:
  batch_size: 32
  max_epochs: 100
  gradient_clip: 0.05

scheduler:
  type: OneCycleLR
  max_lr: 1e-4
  pct_start: 0.2
```

### 2.2 Advanced Physics Configuration

The following is the optimal configuration validated in the superiority benchmark:

```yaml
model:
  physics_config:
    # Embedding (LINEAR MODE - superior for generalization)
    embedding:
      type: functional    # O(1) with respect to vocabulary size
      mode: linear        # Linear mode - critical for O(1) generalization
      coord_dim: 16
    
    # Readout (IMPLICIT MODE - holographic alignment)
    readout:
      type: implicit      # Implicit readout - the latent state IS the answer
    
    # Active Inference
    active_inference:
      enabled: true
      dynamic_time:
        enabled: true
      reactive_curvature:
        enabled: true
        plasticity: 0.2
      singularities:
        enabled: true
        strength: 20.0
        threshold: 0.8
    
    # Fractal Resolution
    fractal:
      enabled: true
      threshold: 0.5
      alpha: 0.2
    
    # Stability
    stability:
      base_dt: 0.4        # Optimal timestep (0.3-0.5 recommended)
      curvature_clamp: 5.0
```

### 2.3 Task-Specific Configuration

| Task | dim | depth | LR (main) | LR (gates) | Batch |
|------|-----|-------|-----------|------------|-------|
| Binary Parity | 128 | 6 | 1e-3 | 1e-2 | 128 |
| Language | 512 | 12 | 3e-5 | 1e-4 | 16 |
| Fine-tuning | 256 | 6 | 1e-5 | 1e-4 | 32 |



## 3. Critical Hyperparameters

### 3.1 Learning Rate

**Recommended Values**:

| Task | LR (weights) | LR (gates) |
|------|--------------|------------|
| Binary tasks | 1e-3 | 1e-2 |
| Language modeling | 3e-5 | 1e-4 |
| Fine-tuning | 1e-5 | 1e-4 |

**Sensitivity**: MANIFOLD models are **highly sensitive** to learning rate.

| LR | Behavior |
|-----|----------------|
| >3e-4 | Diverges or oscillates |
| 1e-4 | Optimal for binary tasks |
| 3e-5 | Optimal for language |
| <1e-5 | Slow convergence (>10K steps) |

**Finding Optimal LR**:

```python
from torch.optim.lr_scheduler import LRFinder

# Run LR range test
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1e-2, num_iter=100)
lr_finder.plot()  # Look for steepest descent
```

### 3.2 Gradient Clipping

**Required**: `max_norm=0.05` (stricter than standard 0.1-1.0)

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
```

**Why**: Geometric updates amplify gradient magnitudes near convergence.

### 3.3 Integration Timestep (dt)

**Default value**: `0.4` (recommended for most tasks)

**Adjustment**:

| dt | Behavior |
|-----|----------------|
| 0.1-0.2 | More stable, slower convergence |
| 0.3-0.4 | **Optimal** - balanced |
| 0.5-0.6 | Risk of instability |
| >0.7 | Complete divergence |

Set via `physics_config['stability']['base_dt`.



## 4. Optimizer: RiemannianAdam

### 4.1 Why It Is Required

Standard Adam performs Euclidean updates that violate manifold constraints, causing "Euclidean drift":

| Symptom | Cause | Solution |
|---------|-------|----------|
| Loss oscillates chaotically | Standard Adam | Use RiemannianAdam |
| Diverges after ~100 steps | Geometric drift | RiemannianAdam with retraction |

### 4.2 Recommended Configuration

```python
from gfn.optim import RiemannianAdam

optimizer = RiemannianAdam(
    model.parameters(),
    lr=1e-4,              # Lower than standard models
    betas=(0.9, 0.999),   # Standard Adam betas
    eps=1e-8,
    weight_decay=1e-4,    # Decoupled weight decay
    retraction='normalize',  # 'normalize', 'cayley', 'euclidean'
    max_norm=10.0         # Manifold bound
)
```

### 4.3 Retraction Types

| Type | Description | Use Case |
|------|-------------|-------------|
| `'normalize'` | Project weights to bounded sphere | **Recommended** |
| `'cayley'` | Orthogonal retraction (experimental) | Square matrices |
| `'euclidean'` | Disable retraction (not recommended) | Debugging |



## 5. Learning Rate Schedules

### 5.1 OneCycleLR (Recommended)

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    total_steps=total_steps,  # epochs * steps_per_epoch
    pct_start=0.2,            # 20% warm-up
    anneal_strategy='cos'
)

# Call after each step
for batch in dataloader:
    train_step(batch)
    scheduler.step()
```

**Benefits**:

- Fast warm-up prevents early instability
- Cosine annealing helps convergence

### 5.2 Alternative: ReduceLROnPlateau

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,      # Halve LR
    patience=10,     # Wait 10 epochs
    threshold=1e-4
)

# Call after each epoch
scheduler.step(val_loss)
```



## 6. Loss Functions

### 6.1 ToroidalDistanceLoss (for toroidal tasks)

```python
from gfn.losses import ToroidalDistanceLoss

criterion = ToroidalDistanceLoss()

# For parity task
logits, _, _ = model(x)  # [batch, seq, coord_dim]
loss = criterion(logits, targets_angle.unsqueeze(-1).expand_as(logits))
```

**Use when**: `physics_config['readout']['type'] == 'implicit'`

### 6.2 Cross-Entropy (Standard Readout)

```python
criterion = torch.nn.CrossEntropyLoss()

# Compute loss
logits, _, _ = model(x)  # [batch, seq, vocab]
loss = criterion(logits.view(-1, vocab_size), y.view(-1))
```

**Use when**: `physics_config['readout']['type'] == 'standard'`

### 6.3 Hamiltonian Regularization (Experimental)

```python
from gfn.losses import hamiltonian_loss, geodesic_regularization

# Energy loss
loss_ham = hamiltonian_loss(
    v_sequence,
    states=x_sequence,
    metric_fn=metric_fn,
    lambda_h=0.01
)

# Geodesic regularization
loss_geo = geodesic_regularization(
    christoffels,
    lambda_g=0.001
)

total_loss = task_loss + loss_ham + loss_geo
```



## 7. Complete Training Loop

```python
import torch
from torch.utils.data import DataLoader
from gfn.model import Manifold
from gfn.optim import RiemannianAdam
from gfn.losses import ToroidalDistanceLoss
from torch.optim.lr_scheduler import OneCycleLR

# Optimal configuration
physics_config = {
    'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16},
    'readout': {'type': 'implicit', 'coord_dim': 16},
    'active_inference': {
        'enabled': True,
        'dynamic_time': {'enabled': True},
        'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
        'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8}
    },
    'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
    'topology': {'type': 'torus'},
    'stability': {'base_dt': 0.4}
}

# Setup
model = Manifold(
    vocab_size=2,
    dim=128,
    depth=6,
    heads=4,
    integrator_type='leapfrog',
    physics_config=physics_config,
    impulse_scale=80.0,
    holographic=True
).cuda()

# Optimizer with differentiated rates
optimizer = RiemannianAdam([
    {'params': [p for n, p in model.named_parameters()
                if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
     'lr': 1e-3, 'weight_decay': 1e-4},
    {'params': [p for n, p in model.named_parameters()
                if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
     'lr': 1e-2, 'weight_decay': 0}
])

criterion = ToroidalDistanceLoss()

max_epochs = 100
total_steps = max_epochs * len(train_loader)
scheduler = OneCycleLR(optimizer, max_lr=2e-3, total_steps=total_steps)

# Training
for epoch in range(max_epochs):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        x, y = batch
        x, y = x.cuda(), y.cuda()
        
        # Forward
        optimizer.zero_grad()
        output = model(x, collect_christ=False)
        x_pred = output[0] if isinstance(output, tuple) else output
        
        loss = criterion(x_pred, y.float().unsqueeze(-1).expand_as(x_pred))
        
        # Physics losses (optional)
        loss_phy = 0.0
        if isinstance(output, tuple) and len(output) >= 6:
            christoffels = output[2]
            if christoffels and len(christoffels) > 0:
                loss_phy = geodesic_regularization(None, christoffels, lambda_g=0.001)
        
        total_loss = loss + loss_phy
        
        # Backward
        if not torch.isnan(total_loss):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Validation
    model.eval()
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch}, Val Acc: {val_acc:.2%}")
    
    # Checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, f'checkpoint_epoch_{epoch}.pth')
```



## 8. Convergence Monitoring

### 8.1 Key Metrics

| Metric | Expected Behavior |
|--------|------------------------|
| Task Loss | Monotonically decreasing |
| Gradient Norm | Stable (not collapsing to 0) |
| Velocity Norm | Bounded (not exploding) |
| Accuracy | Constantly increasing |

### 8.2 Diagnostic Plots

```python
import matplotlib.pyplot as plt

# Track during training
losses = []
grad_norms = []
velocity_norms = []
accuracies = []

# In training loop:
grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
grad_norms.append(grad_norm.item())

_, (_, v), _ = model(x)
velocity_norms.append(v.norm(dim=-1).mean().item())

# Plot
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1); plt.plot(losses); plt.title('Loss')
plt.subplot(1, 4, 2); plt.plot(grad_norms); plt.title('Gradient Norm')
plt.subplot(1, 4, 3); plt.plot(velocity_norms); plt.title('Velocity Norm')
plt.subplot(1, 4, 4); plt.plot(accuracies); plt.title('Accuracy')
plt.tight_layout()
plt.show()
```

### 8.3 Pathologies

| Symptom | Cause | Solution |
|---------|-------|----------|
| Loss oscillates wildly | Use standard Adam | Switch to RiemannianAdam |
| Diverges after ~100 steps | Missing velocity normalization or excessive dt | Version v2.6.0+ has automatic normalization |
| Plateau at ~80% accuracy | Learning rate too low or insufficient capacity | Increase LR or increase dim/depth |
| NaN gradients | Numerical overflow | Reduce learning rate |



## 9. Distributed Training

### 9.1 Data Parallel (Single Node, Multi-GPU)

```python
model = Manifold(...).cuda()
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Training proceeds normally
```

### 9.2 Distributed Data Parallel (Multi-Node)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])

# Model
model = Manifold(...).cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# DataLoader with DistributedSampler
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler, ...)

# Training
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # Shuffle between ranks
    for batch in loader:
        train_step(batch)
```



## 10. Memory Optimization

### 10.1 Mixed Precision (FP16/BF16)

**Status**: Experimental (stability being tested)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        logits, _, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
    scaler.step(optimizer)
    scaler.update()
```

### 10.2 Gradient Accumulation

```python
accumulation_steps = 4

for i, (x, y) in enumerate(dataloader):
    logits, _, _ = model(x)
    loss = criterion(...) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
        optimizer.step()
        optimizer.zero_grad()
```



## 11. Task-Specific Guides

### 11.1 Binary Reasoning Tasks (Parity, XOR)

```yaml
model:
  dim: 128
  depth: 6
  physics_config:
    embedding:
      type: functional
      mode: linear      # Critical for generalization
    readout:
      type: implicit   # Holographic alignment

optimizer:
  lr: 1e-3

training:
  batch_size: 128
  gradient_clip: 1.0
```

### 11.2 Language Modeling

```yaml
model:
  dim: 512
  depth: 12
  physics_config:
    embedding:
      type: functional  # O(1) vocabulary scaling
      mode: linear
    readout:
      type: standard

optimizer:
  lr: 3e-5  # Lower for language

training:
  batch_size: 16
  gradient_accumulation: 4
```



## 12. Checkpointing

### 12.1 Save Checkpoint

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    'config': config,  # Save configuration
}, 'checkpoint.pth')
```

### 12.2 Load Checkpoint

```python
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epoch = checkpoint['epoch']
```



## 13. Troubleshooting

| Problem | Solution |
|----------|----------|
| Loss oscillates | Use RiemannianAdam |
| Model diverges | Reduce LR or dt |
| Slow convergence | Increase LR or verify data |
| OOM | Reduce batch size, enable gradient accumulation |
| Poor generalization | Train longer or increase depth |
| NaN gradients | Reduce learning rate, verify input data |



## 14. Best Practices Summary

| Practice | Description |
|----------|-------------|
| ✓ Always use RiemannianAdam | Prevents Euclidean drift |
| ✓ Clip gradients to 0.05-1.0 | Numerical stability |
| ✓ Start with lr=1e-4 (binary) | Tune from there |
| ✓ Use OneCycleLR scheduler | Warm-up and annealing |
| ✓ Monitor velocity norms | Detect instability |
| ✓ Save checkpoints frequently | Recovery from failures |
| ✓ Validate on held-out data | Detect overfitting |
| ✓ Use linear mode in embedding | Superior generalization |



**Document Version**: 2.6.2  
**Last Update**: January 27, 2026  
**Status**: Production Ready

For API reference, see [API.md](API.md).  
For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).  
For benchmarks, see [BENCHMARKS.md](BENCHMARKS.md).  
For theoretical foundations, see [PHYSICS.md](PHYSICS.md).
