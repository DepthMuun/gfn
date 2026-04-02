# Dynamics - Mathematical Foundation

## Overview

Dynamics modules determine how the state updates from the current state and the integrator's proposal. They act as "routing" mechanisms for state evolution.

---

## 1. Direct Dynamics

### Formula

$$x_{next} = \text{norm}(x_{proposal})$$

The integrator proposal directly becomes the next state.

### Implementation

```python
class DirectDynamics(BaseDynamics):
    def forward(self, current_state, absolute_proposal):
        return self._apply_norm(absolute_proposal)
```

### Properties

- Simplest dynamics
- No residual connection
- Full replacement of state
- **Default behavior**

---

## 2. Residual Dynamics

### Formula

$$x_{next} = x_{current} + \sigma(s) \cdot \text{norm}(x_{proposal} - x_{current})$$

Where:
- $s$ is a learnable residual scale parameter
- $\sigma(s)$ is the sigmoid function
- The difference uses geodesic distance for torus

### Toroidal Geodesic Difference

$$\Delta x_{torus} = \arctan_2(\sin(x_{proposal} - x_{current}), \cos(x_{proposal} - x_{current}))$$

### Implementation

```python
class ResidualDynamics(BaseDynamics):
    def __init__(self, ..., residual_scale=0.1):
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale))
    
    def forward(self, current_state, absolute_proposal):
        if self.topology == 'torus':
            residual = torch.atan2(
                torch.sin(absolute_proposal - current_state),
                torch.cos(absolute_proposal - current_state)
            )
        else:
            residual = absolute_proposal - current_state
        
        scale = torch.sigmoid(self.residual_scale)
        next_state = current_state + scale * normalized_residual
        
        if self.topology == 'torus':
            next_state = torch.atan2(torch.sin(next_state), torch.cos(next_state))
        
        return next_state
```

### Properties

- Learnable correction magnitude
- Smooth interpolation between states
- Better gradient flow

---

## 3. Gated Dynamics

### Formula

$$g = \sigma(W_g \cdot [x_{current}; x_{proposal}])$$

$$x_{next} = \text{norm}(g \cdot x_{proposal} + (1-g) \cdot x_{current})$$

Where:
- $[;]$ denotes concatenation
- $\sigma$ is sigmoid
- $W_g$ is a learnable linear layer

### Implementation

```python
class GatedDynamics(BaseDynamics):
    def __init__(self, dim, ...):
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
    
    def forward(self, current_state, absolute_proposal):
        gate_input = torch.cat([current_state, absolute_proposal], dim=-1)
        g = self.gate(gate_input)
        mixed = g * absolute_proposal + (1.0 - g) * current_state
        return self._apply_norm(mixed)
```

### Properties

- State-dependent gate
- Can learn complex update rules
- More expressive than mix
- Requires more parameters

---

## 4. Comparison

| Dynamics | Formula | Parameters | Use Case |
|----------|---------|------------|----------|
| `direct` | $x_{next} = x_{proposal}$ | None | Default, simple |
| `residual` | $x_{next} = x + \sigma(s) \cdot \Delta x$ | 1 scalar | Smooth updates |
| `gated` | $x_{next} = g \cdot x_{proposal} + (1-g) \cdot x$ | Linear layer | Complex routing |

---

## 5. Configuration

```python
physics = {
    'dynamics': {
        'type': 'direct'  # 'direct', 'residual', 'gated'
    }
}
```

---

*File: technical/0_architecture/math/08_dynamics.md*
*Last Updated: 2026-04-02*
