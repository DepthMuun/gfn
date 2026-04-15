# Hysteresis - Mathematical Foundation

## Overview

Hysteresis provides a "memory" mechanism through ghost forces that persist across timesteps, allowing the model to remember previous states.

---

## 1. Physical Analogy

In physics, hysteresis occurs when the output of a system depends on its history. A classic example is magnetic hysteresis:

```
B = μ(H + M)
```

Where the magnetization M depends on the history of the magnetic field H.

---

## 2. Hysteresis in GSSM

### Purpose

- Remember previous states without explicit memory
- Create "ghost forces" that guide current dynamics
- Enable path-dependent behavior

### Configuration

```python
physics = {
    'hysteresis': {
        'enabled': True,
        'ghost_force': True,
        'hyst_decay': 0.1,
        'hyst_update_w': 1.0,
        'hyst_update_b': 0.0,
        'hyst_readout_w': 1.0,
        'hyst_readout_b': 0.0
    }
}
```

---

## 3. State Update Equation

### Hidden State

The hysteresis maintains a hidden state $h_t$ that evolves:

$$h_t = (1 - \alpha) \cdot h_{t-1} + \alpha \cdot v_t$$

Where:
- $h_t$ is the hysteresis state at time t
- $v_t$ is the velocity at time t
- $\alpha$ is the decay rate (`hyst_decay`)

### In Code

```python
# From HysteresisModule
def forward(self, x, v, topo_id):
    # Update hysteresis state
    self.h_state = (1 - self.decay) * self.h_state + self.decay * v
    
    # Compute ghost force
    ghost = self.readout(self.h_state)
    return ghost
```

---

## 4. Ghost Force Computation

### Formula

$$F_{ghost} = W \cdot \tanh(b + h_t)$$

Where:
- $W$ is the weight matrix (`hyst_readout_w`)
- $b$ is the bias (`hyst_readout_b`)
- $h_t$ is the hysteresis state
- $\tanh$ provides saturation (prevents explosion)

### Components

```python
# Linear projection
h_proj = self.h_state @ self.weight + self.bias

# Non-linear activation
ghost = self.weight_scale * torch.tanh(h_proj)
```

---

## 5. Update Rule

### Weight Update

The hysteresis module can also update its own weights based on velocity:

$$W_{new} = W_{old} + \eta \cdot v \cdot \text{error}$$

Where:
- $\eta$ is `hyst_update_w`
- error is a learning signal

### Bias Update

$$b_{new} = b_{old} + \eta_b \cdot \text{error}$$

With $\eta_b$ = `hyst_update_b`

---

## 6. Topology Awareness

### Topo ID

The hysteresis can be topology-aware:

```python
# Different behavior for different topologies
if topo_id == 1:  # Torus
    # Use angular dynamics
else:  # Euclidean
    # Use linear dynamics
```

---

## 7. Complete Forward Pass

```python
class HysteresisModule(nn.Module):
    def __init__(self, config, dim, heads):
        self.decay = config.hyst_decay
        self.update_w = config.hyst_update_w
        self.update_b = config.hyst_update_b
        self.readout_w = config.hyst_readout_w
        self.readout_b = config.hyst_readout_b
        
        # Hidden state
        self.h_state = None
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x, v, topo_id):
        # Initialize state if needed
        if self.h_state is None:
            self.h_state = torch.zeros_like(v)
        
        # Update hysteresis state
        self.h_state = (1 - self.decay) * self.h_state + self.decay * v
        
        # Compute ghost force
        ghost = torch.tanh(self.h_state @ self.weight + self.bias)
        ghost = ghost * self.readout_w + self.readout_b
        
        return ghost
```

---

## 8. Parameter Summary

| Parameter | Symbol | Default | Effect |
|-----------|--------|---------|--------|
| `enabled` | - | False | Enable hysteresis |
| `ghost_force` | - | True | Use ghost force |
| `hyst_decay` | $\alpha$ | 0.1 | Memory decay rate |
| `hyst_update_w` | $\eta_W$ | 1.0 | Weight update rate |
| `hyst_update_b` | $\eta_b$ | 0.0 | Bias update rate |
| `hyst_readout_w` | $W_{scale}$ | 1.0 | Ghost force scale |
| `hyst_readout_b` | $b_{scale}$ | 0.0 | Ghost force bias |

---

## 9. Behavior Examples

### High Decay (α = 0.9)

- Remembers recent states strongly
- Short-term memory
- Fast adaptation

### Low Decay (α = 0.01)

- Remembers distant states
- Long-term memory
- Slow adaptation

### Decay = 0

- No memory (state doesn't update)
- Static ghost force

---

## 10. Use Cases

1. **Sequence modeling**: Remember previous tokens
2. **Autoregressive generation**: Maintain context
3. **Temporal patterns**: Capture long-range dependencies
4. **Path-dependent logic**: Tasks where order matters

---

*File: technical/0_architecture/math/04_hysteresis.md*
*Last Updated: 2026-04-02*
