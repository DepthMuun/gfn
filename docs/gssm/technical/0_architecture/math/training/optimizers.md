# Optimizers - Riemannian Adam

## What is Riemannian Adam?

Riemannian Adam is an extension of the standard Adam optimizer that respects manifold constraints. While standard Adam works in Euclidean space, Riemannian Adam accounts for curved geometries like the torus.

Think of it as: "Adam that understands the manifold's shape."

---

## Standard Adam Review

Adam combines momentum and adaptive learning rates:

### Update Rule

**Momentum** (first moment):
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

**Adaptive rate** (second moment):
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

**Bias correction**:
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**Parameter update**:
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where:
- $g_t$ = gradient at step $t$
- $\beta_1$ = 0.9 (momentum decay)
- $\beta_2$ = 0.999 (second moment decay)
- $\eta$ = learning rate
- $\epsilon$ = 1e-8 (numerical stability)

---

## Riemannian Extension

### For Euclidean Geometry

Identical to standard Adam. No special handling needed.

### For Torus Geometry

After parameter update, **wrap position parameters**:

$$\theta_t = \arctan_2(\sin(\theta_t), \cos(\theta_t))$$

This ensures positions stay in $[-\pi, \pi]$.

**Why?**
- Gradient descent moves in tangent space (flat)
- But parameters live on curved manifold
- Wrapping projects back to valid manifold coordinates

---

## Dual-Parameter Group Optimization

### The Problem

GSSM has two types of parameters:

**Group 1: Network Parameters**
- Embedding weights
- Mixer weights
- Readout weights
- Normal learning rate needed

**Group 2: Physics Parameters**
- Initial state $x_0, v_0$
- Impulse scale
- Gating parameters
- Need higher learning rate

**Why different rates?**
- Physics params are few but crucial
- Their gradients are typically smaller
- They need faster adaptation

### Solution: Parameter Groups

```
param_groups = [
    {
        'params': network_params,
        'lr': lr,              # Normal rate
        'weight_decay': 1e-4,
    },
    {
        'params': physics_params,
        'lr': lr * 10,         # 10x faster
        'weight_decay': 0.0,   # No decay
    }
]
```

**Typical values**:
- Base LR: 1e-3
- Physics LR: 1e-2 (10x base)
- Weight decay: 1e-4 (network only)

---

## Riemannian SGD

Simpler alternative to Adam:

$$\theta_t = \theta_{t-1} - \eta g_t$$

With torus wrapping for position parameters.

**When to use**:
- Simplicity preferred
- Large batch sizes
- Less memory (no momentum buffers)

---

## Optimizer Selection Guide

| Optimizer | Use Case | Notes |
|-----------|----------|-------|
| **RiemannianAdam** | Default | Best for most cases |
| **RiemannianSGD** | Large scale | Simpler, less memory |
| **AdamW** | Standard | If no manifold constraints |
| **Standard SGD** | Baseline | Without Riemannian features |

---

## Configuration

```python
# Simple usage
optimizer = RiemannianAdam(
    model.parameters(),
    lr=1e-3,
    geometry_type='torus'  # or 'euclidean'
)

# Dual-group (recommended)
optimizer = make_gfn_optimizer(
    model,
    lr=1e-3,
    physics_lr_scale=10.0,
    weight_decay=1e-4
)
```

---

## Physical Interpretation

### Gradient Descent on Manifolds

Standard gradient descent:
$$\theta_{new} = \theta - \eta \nabla L$$

This moves in the tangent space, which is flat.

But the manifold is curved! After moving, we must **retract** back to the manifold:
$$\theta_{new} = \text{retract}(\theta - \eta \nabla L)$$

For torus: $\text{retract}(x) = \arctan_2(\sin(x), \cos(x))$

### Why Different LR for Physics Params?

Think of the system as:
- **Network weights**: Fine-tuning the machinery
- **Physics params**: Fundamental constants

Physics params need larger updates because:
1. They affect the entire dynamics
2. They're initialized with uncertainty
3. Small gradients need amplification

---

## When to Use Riemannian Optimizers

**Always use RiemannianAdam for:**
- Torus topology
- Any manifold-constrained parameters
- Production training

**Can use standard Adam for:**
- Euclidean topology only
- Quick experiments
- When simplicity matters

---

*File: technical/0_architecture/math/training/optimizers.md*
*Last Updated: 2026-04-02*
