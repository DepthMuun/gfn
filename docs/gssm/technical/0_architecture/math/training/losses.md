# Losses - Physics Informed

## What are Physics-Informed Losses?

Standard losses (like CrossEntropy) only care about prediction accuracy. Physics-informed losses add constraints from the underlying physics, ensuring the model learns physically meaningful representations.

Think of it as: "Not just predicting correctly, but evolving in a physically sensible way."

---

## Loss Components

### 1. Geodesic Regularization

**What it penalizes**: Excessive curvature in trajectories.

**Physical meaning**: A trajectory should follow the "straightest possible line" on the manifold (geodesic). High deviation means the path is physically inefficient.

**Formula**:
$$L_{geo} = \frac{1}{T} \sum_t \|\Gamma(x_t, v_t)\|^2$$

Where:
- $\Gamma$ = Christoffel symbols (geometric force)
- $x_t$ = position at time $t$
- $v_t$ = velocity at time $t$

**Intuition**:
- Small $\Gamma$ = following natural manifold curvature ✓
- Large $\Gamma$ = fighting against manifold structure ✗

**When to use**: Always (small weight ~0.001)

---

### 2. Hamiltonian Conservation

**What it penalizes**: Energy drift over time.

**Physical meaning**: In a closed system, total energy should be conserved. For our Hamiltonian system:
$$H = \frac{1}{2}\|v\|^2 + V(x)$$

**Formula**:
$$L_{ham} = \text{Var}_t[H_t] = \frac{1}{T}\sum_t (H_t - \bar{H})^2$$

Where:
- $H_t$ = Hamiltonian at time $t$
- $\bar{H}$ = average energy over trajectory

**Intuition**:
- Symplectic integrators preserve energy oscillations
- This loss penalizes systematic drift
- Helps maintain long-term stability

**When to use**: Long sequences, energy-sensitive tasks

---

### 3. Kinetic Regularization

**What it penalizes**: Excessive velocity/kinetic energy.

**Physical meaning**: Prevents runaway acceleration and numerical instability.

**Formula**:
$$KE = \frac{1}{2}\|v\|^2$$
$$L_{kin} = \frac{1}{T}\sum_t \text{ReLU}(KE_t - KE_{max})$$

Where:
- $KE_t$ = kinetic energy at time $t$
- $KE_{max}$ = maximum allowed kinetic energy (default 10.0)

**Intuition**:
- Allows normal velocities
- Penalizes only when exceeding threshold
- Prevents explosion

**When to use**: When training shows velocity explosion

---

## Combined Physics Loss

### Formula

$$L_{physics} = \lambda_{geo} L_{geo} + \lambda_{ham} L_{ham} + \lambda_{kin} L_{kin}$$

### Typical Weights

| Component | Weight | Use Case |
|-----------|--------|----------|
| Geodesic | 0.001 | Always on |
| Hamiltonian | 0.0-0.01 | Long sequences |
| Kinetic | 0.0-0.01 | Instability issues |

---

## Physics-Informed Generative Loss

### Complete Loss

The main training loss combines standard prediction with physics regularization:

$$L_{total} = L_{CE} + \lambda_{physics} \cdot L_{physics}$$

Where:
- $L_{CE}$ = CrossEntropy (predicts next token)
- $L_{physics}$ = Combined physics loss
- $\lambda_{physics}$ = weight (typically 0.01)

### Entropy Bonus (Optional)

$$L_{entropy} = -\sum_i p_i \log p_i$$

Added to encourage exploration:
$$L_{total} = L_{CE} - \alpha_{ent} L_{entropy} + \lambda_{physics} L_{physics}$$

---

## Physical Interpretation

### Trajectory Quality

The physics loss acts as a "trajectory regularizer":

**Good trajectory**:
- Follows geodesics (low $L_{geo}$)
- Conserves energy (low $L_{ham}$)
- Moderate velocity (low $L_{kin}$)

**Bad trajectory**:
- Wanders randomly (high $L_{geo}$)
- Energy explodes or collapses (high $L_{ham}$)
- Runaway velocity (high $L_{kin}$)

### Analogy: Driving a Car

- **Geodesic**: Staying in lane vs swerving
- **Hamiltonian**: Maintaining speed vs accelerating randomly
- **Kinetic**: Speed limit vs racing

---

## When to Use Physics Losses

**Always use**:
- Geodesic (small weight) - helps stability

**Use when needed**:
- Hamiltonian - for long trajectories
- Kinetic - if seeing velocity explosion

**Weights**:
- Start with $\lambda_{geo} = 0.001$
- Add others only if problems occur
- Don't make physics loss dominate (keep < 10% of CE loss)

---

## Configuration

```python
physics = {
    'losses': {
        'lambda_geo': 0.001,    # Geodesic weight
        'lambda_ham': 0.0,      # Hamiltonian weight
        'lambda_kin': 0.0,      # Kinetic weight
        'max_kinetic': 10.0,    # Kinetic energy threshold
    }
}
```

---

*File: technical/0_architecture/math/training/losses.md*
*Last Updated: 2026-04-02*
