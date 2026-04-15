# Stochasticity - Mathematical Foundation

## Overview

Stochasticity introduces random forces into the dynamics, simulating Langevin dynamics and providing exploration/noise injection for robust training.

---

## 1. Langevin Dynamics

### Physical Background

The Langevin equation describes Brownian motion with friction:

$$m \frac{dv}{dt} = -\gamma v + F_{ext} + \xi(t)$$

Where:
- $m$ is mass
- $\gamma$ is friction coefficient
- $F_{ext}$ is external force
- $\xi(t)$ is random noise with $\langle \xi(t) \rangle = 0$ and $\langle \xi(t)\xi(t') \rangle = 2\gamma k_B T \delta(t-t')$

In GSSM, we simplify to:

$$\frac{dv}{dt} = a_{net} + F_{stochastic}$$

---

## 2. Brownian Force

### Formula

From `BrownianForce.forward()`:

$$F_{brownian} = \sigma \cdot \frac{1}{\sqrt{dt}} \cdot \mathcal{N}(0, 1)$$

Where:
- $\sigma$ is the noise magnitude parameter
- $dt$ is the integration timestep
- $\mathcal{N}(0, 1)$ is standard normal random noise

### Scaling

The $\frac{1}{\sqrt{dt}}$ scaling ensures that when integrated:

$$\int_0^{dt} F_{brownian} \, dt = \sigma \cdot \sqrt{dt} \cdot \mathcal{N}(0, 1)$$

This follows the Itô calculus scaling where Brownian increments scale as $\sqrt{dt}$.

### Variance

$$\text{Var}(F_{brownian}) = \frac{\sigma^2}{dt}$$

After integration:

$$\text{Var}(\Delta x) = \sigma^2 \cdot dt$$

---

## 3. Ornstein-Uhlenbeck (OU) Force

### Purpose

Adds mean-reverting noise for smooth, correlated exploration (colored noise instead of white noise).

### Discrete Update

From `OUDynamicsForce.forward()`:

$$n_{t+1} = n_t + \theta(\mu - n_t)dt + \sigma \frac{1}{\sqrt{dt}} \mathcal{N}(0, 1)$$

Where:
- $n_t$ is the OU state (stored as `_prev_noise`)
- $\theta$ is mean reversion speed
- $\mu$ is mean reversion level
- $\sigma$ is noise magnitude

### Continuous Form

The OU process solves:

$$dn = -\theta(n - \mu)dt + \sigma dW$$

With solution:

$$n(t) = n_0 e^{-\theta t} + \mu(1 - e^{-\theta t}) + \sigma \int_0^t e^{-\theta(t-s)} dW(s)$$

### Mean and Variance

$$\mathbb{E}[n(t)] = n_0 e^{-\theta t} + \mu(1 - e^{-\theta t})$$

$$\text{Var}[n(t)] = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})$$

As $t \to \infty$:
- Mean → $\mu$
- Variance → $\frac{\sigma^2}{2\theta}$

---

## 4. Configuration

```python
physics = {
    'active_inference': {
        'stochasticity': {
            'enabled': True,
            'type': 'brownian',    # or 'ou'
            'sigma': 0.01,       # Noise magnitude
            'theta': 0.15,       # OU: mean reversion speed
            'mu': 0.0            # OU: mean reversion level
        }
    }
}
```

### Parameter Effects

| Parameter | Brownian | OU | Effect |
|-----------|----------|-----|--------|
| `sigma` | Noise mag | Noise mag | Overall noise scale |
| `theta` | N/A | Reversion | Higher = faster mean return |
| `mu` | N/A | Target | Target mean for OU |

---

## 5. Temperature Analogy

### Effective Temperature

The noise magnitude relates to an effective temperature:

$$T_{eff} \propto \frac{\sigma^2}{2\gamma}$$

Where $\gamma$ is the friction coefficient.

Higher $\sigma$ = Higher temperature = More exploration

### Use in Training

- **Low temperature** (small $\sigma$): Fine-tuning, exploitation
- **High temperature** (large $\sigma$): Exploration, escaping local minima
- **Annealing**: Start high, decrease over time

---

## 6. Integration with Physics Engine

```python
# In PhysicsEngine.compute_acceleration()

if self.stochasticity_module is not None and dt is not None:
    stoch_force = self.stochasticity_module(x, v, dt)
    net_accel = net_accel + stoch_force
```

The stochastic force is added to the net acceleration before integration.

---

## 7. Comparison: Brownian vs OU

| Property | Brownian | OU |
|----------|----------|-----|
| Correlation | Independent | Autocorrelated |
| Mean | 0 | $\mu$ |
| Variance | $\sigma^2/dt$ | $\sigma^2/(2\theta)$ |
| Smoothness | Noisy | Smooth |
| Use case | Exploration | Smooth dynamics |

---

## 8. Numerical Safety

### dt Clamping

```python
safe_dt = max(dt, 1e-8)
```

Prevents division by zero in the $\frac{1}{\sqrt{dt}}$ term.

### Invalid dt Handling

```python
if dt <= 0:
    return zeros_like(v)
```

Returns zero force if dt is invalid.

---

*File: technical/0_architecture/math/06_stochasticity.md*
*Last Updated: 2026-04-02*
