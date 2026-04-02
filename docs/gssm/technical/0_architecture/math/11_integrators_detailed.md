# Integrators - Detailed Implementation

**Source Files Analyzed:**
- `physics/integrators/symplectic/leapfrog.py`
- `physics/integrators/symplectic/yoshida.py`
- `physics/integrators/symplectic/verlet.py`
- `physics/integrators/symplectic/forest_ruth.py`
- `physics/integrators/symplectic/omelyan.py`
- `physics/integrators/runge_kutta/rk4.py`
- `physics/integrators/runge_kutta/heun.py`

---

## 1. Common Interface

All integrators inherit from `BaseIntegrator` and implement:

```python
def step(self, x, v, force=None, dt=None, steps=1, **kwargs) -> Dict[str, Tensor]:
    """Returns {'x': x_next, 'v': v_next}"""
```

**Inherited from Base**:
- `_clamp_velocity()`: Hard clamp to MAX_VELOCITY
- `_resolve_topology()`: Wrap position for torus
- `_get_acceleration()`: Call physics engine
- `_resolve_friction_mu()`: Get friction coefficient

---

## 2. Leapfrog (Störmer-Verlet)

**File**: `physics/integrators/symplectic/leapfrog.py`
**Order**: 2nd order
**Symplectic**: Yes

### Algorithm

```python
for step in range(steps):
    # 1. Half-step velocity (Kick)
    mu1 = self._resolve_friction_mu(curr_x, curr_v, force)
    a1 = self._get_acceleration(curr_x, curr_v, force, dt)
    a1_nf = a1 + mu1 * curr_v
    v_half = (curr_v + 0.5 * dt * a1_nf) / (1.0 + 0.5 * dt * mu1 + EPS)
    v_half = self._clamp_velocity(v_half)
    
    # 2. Full-step position (Drift)
    curr_x = self._resolve_topology(curr_x + dt * v_half)
    
    # 3. Re-evaluate at new position
    mu2 = self._resolve_friction_mu(curr_x, v_half, force)
    a2 = self._get_acceleration(curr_x, v_half, force, dt)
    a2_nf = a2 + mu2 * v_half
    
    # 4. Final half-step velocity (Kick)
    a_avg = (a1_nf + a2_nf) / 2
    mu_avg = (mu1 + mu2) / 2
    curr_v = (curr_v + dt * a_avg) / (1.0 + dt * mu_avg + EPS)
    curr_v = self._clamp_velocity(curr_v)
```

### Mathematical Form

$$v_{n+1/2} = \frac{v_n + \frac{\Delta t}{2}a_n}{1 + \frac{\Delta t}{2}\mu_n}$$

$$x_{n+1} = x_n + \Delta t \cdot v_{n+1/2}$$

$$v_{n+1} = \frac{v_n + \Delta t \cdot a_{avg}}{1 + \Delta t \cdot \mu_{avg}}$$

### Properties

- **Force evaluations**: 2 per step
- **Time-reversible**: Yes
- **Energy drift**: Minimal (symplectic)
- **Default**: Yes (most stable)

---

## 3. Yoshida

**File**: `physics/integrators/symplectic/yoshida.py`
**Order**: 4th order
**Symplectic**: Yes

### Coefficients

```python
w1 = 1.3512071919596576   # 1/(2 - 2^(1/3))
w0 = -1.7024143839193153  # -2^(1/3)/(2 - 2^(1/3))

c1 = w1 / 2.0
c2 = (w0 + w1) / 2.0
c3 = c2
c4 = c1

d1 = w1
d2 = w0
d3 = w1
```

### Algorithm

```python
for step in range(steps):
    # Sub-step 1
    curr_x = self._resolve_topology(curr_x + self.c1 * dt * curr_v)
    a1 = self._get_acceleration(curr_x, curr_v, force, dt)
    curr_v = self._clamp_velocity(curr_v + self.d1 * dt * a1)
    
    # Sub-step 2
    curr_x = self._resolve_topology(curr_x + self.c2 * dt * curr_v)
    a2 = self._get_acceleration(curr_x, curr_v, force, dt)
    curr_v = self._clamp_velocity(curr_v + self.d2 * dt * a2)
    
    # Sub-step 3
    curr_x = self._resolve_topology(curr_x + self.c3 * dt * curr_v)
    a3 = self._get_acceleration(curr_x, curr_v, force, dt)
    curr_v = self._clamp_velocity(curr_v + self.d3 * dt * a3)
    
    # Final drift
    curr_x = self._resolve_topology(curr_x + self.c4 * dt * curr_v)
```

### Properties

- **Force evaluations**: 3 per step
- **Accuracy**: Higher (4th order vs 2nd)
- **Cost**: ~3× leapfrog
- **Use case**: Long simulations requiring precision

---

## 4. Verlet (Velocity Verlet)

**File**: `physics/integrators/symplectic/verlet.py`
**Order**: 2nd order
**Symplectic**: Yes

### Algorithm

```python
for step in range(steps):
    # Initial acceleration
    a0 = self._get_acceleration(curr_x, curr_v, force, dt)
    
    # Position: x' = x + v·dt + 0.5·a·dt²
    curr_x = self._resolve_topology(
        curr_x + curr_v * dt + 0.5 * a0 * dt ** 2
    )
    
    # Velocity average for stability
    v_avg = curr_v + 0.5 * a0 * dt
    
    # New acceleration
    a1 = self._get_acceleration(curr_x, v_avg, force, dt)
    
    # Velocity: v' = v + 0.5·(a0 + a1)·dt
    curr_v = self._clamp_velocity(curr_v + 0.5 * (a0 + a1) * dt)
```

### Mathematical Form

$$x_{n+1} = x_n + v_n \Delta t + \frac{1}{2} a_n \Delta t^2$$

$$v_{n+1} = v_n + \frac{1}{2}(a_n + a_{n+1}) \Delta t$$

### Properties

- **Force evaluations**: 2 per step
- **Similar to**: Leapfrog (mathematically equivalent)
- **Use case**: When position needs explicit acceleration term

---

## 5. Forest-Ruth

**File**: `physics/integrators/symplectic/forest_ruth.py`
**Order**: 4th order
**Symplectic**: Yes

### Coefficients

```python
θ = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))  # ~1.3512

c1 = θ / 2.0
c2 = (1.0 - θ) / 2.0
c3 = c2
c4 = θ / 2.0

d1 = θ
d2 = 1.0 - 2.0 * θ
d3 = θ
```

### Algorithm

```python
for step in range(steps):
    # Position drift
    curr_x = self._resolve_topology(curr_x + self.c1 * dt * curr_v)
    
    # Velocity kick
    a = self._get_acceleration(curr_x, curr_v, force, dt)
    curr_v = self._clamp_velocity(curr_v + self.d1 * dt * a)
    
    # Position drift
    curr_x = self._resolve_topology(curr_x + self.c2 * dt * curr_v)
    
    # Velocity kick
    a = self._get_acceleration(curr_x, curr_v, force, dt)
    curr_v = self._clamp_velocity(curr_v + self.d2 * dt * a)
    
    # Position drift
    curr_x = self._resolve_topology(curr_x + self.c3 * dt * curr_v)
    
    # Velocity kick
    a = self._get_acceleration(curr_x, curr_v, force, dt)
    curr_v = self._clamp_velocity(curr_v + self.d3 * dt * a)
    
    # Final drift
    curr_x = self._resolve_topology(curr_x + self.c4 * dt * curr_v)
```

### Properties

- **Force evaluations**: 3 per step
- **Accuracy**: 4th order
- **Alternative**: To Yoshida (slightly different coefficients)

---

## 6. Omelyan

**File**: `physics/integrators/symplectic/omelyan.py`
**Order**: 2nd order (optimized)
**Symplectic**: Yes

### Coefficients

```python
# Optimized parameter
ζ = 0.1931833275037836  # ~0.1932

c1 = ζ
c2 = 0.5 - ζ
c3 = c2
c4 = ζ

d1 = 0.5
d2 = ζ
d3 = 0.5 - ζ
d4 = 0.5 - ζ
d5 = ζ
d6 = 0.5
```

### Algorithm

6 force evaluations per step for optimized 2nd order.

### Properties

- **Force evaluations**: 6 per step
- **Accuracy**: Optimized 2nd order
- **Use case**: When 2nd order is sufficient but needs optimization

---

## 7. RK4 (Runge-Kutta 4th Order)

**File**: `physics/integrators/runge_kutta/rk4.py`
**Order**: 4th order
**Symplectic**: No

### Algorithm

```python
for step in range(steps):
    # k1 = f(x, v)
    k1_v = curr_v
    k1_a = self._get_acceleration(curr_x, curr_v, force, dt)
    
    # k2 = f(x + h/2 * k1_v, v + h/2 * k1_a)
    k2_v_val = curr_v + (h / 2.0) * k1_a
    k2_x_val = self._resolve_topology(curr_x + (h / 2.0) * k1_v)
    k2_a = self._get_acceleration(k2_x_val, clamp(k2_v_val), force, dt)
    k2_v = k2_v_val
    
    # k3 = f(x + h/2 * k2_v, v + h/2 * k2_a)
    k3_v_val = curr_v + (h / 2.0) * k2_a
    k3_x_val = self._resolve_topology(curr_x + (h / 2.0) * k2_v)
    k3_a = self._get_acceleration(k3_x_val, clamp(k3_v_val), force, dt)
    k3_v = k3_v_val
    
    # k4 = f(x + h * k3_v, v + h * k3_a)
    k4_v_val = curr_v + h * k3_a
    k4_x_val = self._resolve_topology(curr_x + h * k3_v)
    k4_a = self._get_acceleration(k4_x_val, clamp(k4_v_val), force, dt)
    k4_v = k4_v_val
    
    # Update: weighted average
    curr_x = self._resolve_topology(
        curr_x + (h / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    )
    curr_v = self._clamp_velocity(
        curr_v + (h / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a)
    )
```

### Mathematical Form

$$x_{n+1} = x_n + \frac{\Delta t}{6}(v_1 + 2v_2 + 2v_3 + v_4)$$

$$v_{n+1} = v_n + \frac{\Delta t}{6}(a_1 + 2a_2 + 2a_3 + a_4)$$

Where:
- $k_1 = (v_n, a(x_n, v_n))$
- $k_2 = (v_n + \frac{h}{2}a_1, a(x_n + \frac{h}{2}v_n, v_n + \frac{h}{2}a_1))$
- $k_3 = (v_n + \frac{h}{2}a_2, a(x_n + \frac{h}{2}v_2, v_n + \frac{h}{2}a_2))$
- $k_4 = (v_n + h a_3, a(x_n + h v_3, v_n + h a_3))$

### Properties

- **Force evaluations**: 4 per step
- **Accuracy**: 4th order (high)
- **Symplectic**: No (energy drifts)
- **Use case**: When accuracy matters more than energy conservation

---

## 8. Heun (Improved Euler)

**File**: `physics/integrators/runge_kutta/heun.py`
**Order**: 2nd order
**Symplectic**: No

### Algorithm

```python
# Predictor (Euler step)
x_pred = curr_x + dt * curr_v
v_pred = curr_v + dt * a(curr_x, curr_v)

# Corrector (average)
curr_x = curr_x + dt/2 * (curr_v + v_pred)
curr_v = curr_v + dt/2 * (a(curr_x, curr_v) + a(x_pred, v_pred))
```

### Mathematical Form

$$\tilde{x} = x_n + \Delta t \cdot v_n$$
$$\tilde{v} = v_n + \Delta t \cdot a(x_n, v_n)$$

$$x_{n+1} = x_n + \frac{\Delta t}{2}(v_n + \tilde{v})$$
$$v_{n+1} = v_n + \frac{\Delta t}{2}(a(x_n, v_n) + a(\tilde{x}, \tilde{v}))$$

### Properties

- **Force evaluations**: 2 per step
- **Accuracy**: 2nd order
- **Alternative**: To Leapfrog (not symplectic)

---

## 9. Comparison Table

| Integrator | Order | Symplectic | Force Evals/Step | Accuracy | Cost |
|------------|-------|------------|------------------|----------|------|
| Leapfrog | 2 | ✅ | 2 | Good | 1× |
| Verlet | 2 | ✅ | 2 | Good | 1× |
| Yoshida | 4 | ✅ | 3 | High | 3× |
| Forest-Ruth | 4 | ✅ | 3 | High | 3× |
| Omelyan | 2 | ✅ | 6 | Optimized | 6× |
| RK4 | 4 | ❌ | 4 | Very High | 4× |
| Heun | 2 | ❌ | 2 | Good | 1× |

---

## 10. Selection Guide

### For Training (Stability Priority)
- **Leapfrog** (default): Most stable, good accuracy
- **Verlet**: Alternative, similar to leapfrog

### For Long Sequences (Accuracy Priority)
- **Yoshida**: 4th order symplectic
- **Forest-Ruth**: Alternative 4th order

### For Quick Experiments
- **Heun**: Simple, not symplectic
- **RK4**: High accuracy, not symplectic

### Avoid
- **Omelyan**: 6 force evals is expensive
- **RK4 for long training**: Energy drift accumulates

---

## 11. CUDA Fast Path

Some integrators have optimized C++ CUDA kernels:

```python
if CUDA_AVAILABLE and leapfrog_fused is not None and is_low_rank:
    x_next, v_next = leapfrog_fused(
        x, v, U, W, force,
        dt, steps, clamp_val, friction, vel_scale, v_sat,
        gate_w, gate_b, sing_thresh, sing_strength,
        trace_norm, is_paper
    )
```

**Conditions for fast path**:
- CUDA available
- Geometry is LowRank or PaperLowRank
- Force is provided
- Tensors are on GPU

---

*File: technical/0_architecture/math/11_integrators_detailed.md*
*Last Updated: 2026-04-02*
