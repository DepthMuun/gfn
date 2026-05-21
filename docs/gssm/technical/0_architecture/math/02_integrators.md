# Integrators - Mathematical Foundation

## Overview

Integrators solve the Hamiltonian system to compute the next state (x, v) from the current state and acceleration. They are numerical methods for solving ODEs.

---

## 1. Hamiltonian System

The state evolution follows:

$$\frac{dx}{dt} = v$$
$$\frac{dv}{dt} = a_{net}(x, v, F)$$

This is a system of first-order ODEs that can be written as:

$$\dot{z} = f(z)$$

Where $z = (x, v)$ is the phase space state.

---

## 2. Symplectic Integrators

Symplectic integrators preserve the symplectic structure of phase space, leading to better energy conservation over long simulations.

### 2.1 Leapfrog (Velocity Verlet)

**Order**: 2nd order
**File**: `physics/integrators/symplectic/leapfrog.py`

**Algorithm**:

```
# Step 1: Half-step velocity
v_half = v + 0.5 * dt * a(x, v, F)

# Step 2: Full-step position  
x_new = x + dt * v_half

# Step 3: Compute new acceleration
a_new = compute_acceleration(x_new, v_half, F)

# Step 4: Full-step velocity
v_new = v_half + 0.5 * dt * a_new
```

**Mathematical Form**:

$$v_{n+1/2} = v_n + \frac{\Delta t}{2} a(x_n, v_n)$$

$$x_{n+1} = x_n + \Delta t \cdot v_{n+1/2}$$

$$v_{n+1} = v_{n+1/2} + \frac{\Delta t}{2} a(x_{n+1}, v_{n+1/2})$$

**Properties**:
- Time-reversible
- Energy-preserving (symplectic)
- Simple and fast
- **Recommended for training**

### 2.2 Verlet Integrator

**Order**: 2nd order
**File**: `physics/integrators/symplectic/verlet.py`

**Algorithm**:

```
x_new = 2*x - x_prev + dt^2 * a(x, v, F)
v_new = (x_new - x_prev) / (2*dt)
```

**Mathematical Form**:

$$x_{n+1} = 2x_n - x_{n-1} + \Delta t^2 \cdot a(x_n)$$

$$v_n = \frac{x_{n+1} - x_{n-1}}{2\Delta t}$$

### 2.3 Yoshida Integrator

**Order**: 4th order
**File**: `physics/integrators/symplectic/yoshida.py`

**Algorithm**:

Uses optimized coefficients for higher accuracy:

```
c1 = c4 =  1/(2 - 2^(1/3))
c2 = c3 = (1 - 2^(1/3))/(2 - 2^(1/3))
d1 = d3 =  1/(2 - 2^(1/3))
d2 = -(2^(1/3))/(2 - 2^(1/3))

# Sequence of half-steps
v += c1 * d1 * dt * a(x, v)
x += c1 * d1 * dt * v
v += c2 * d2 * dt * a(x, v)
x += c2 * d2 * dt * v
v += c3 * d2 * dt * a(x, v)
x += c3 * d2 * dt * v
v += c4 * d1 * dt * a(x, v)
```

**Properties**:
- 4th order accuracy (smaller error)
- More expensive (3× leapfrog)
- Good for long simulations

### 2.4 Forest-Ruth Integrator

**Order**: 4th order
**File**: `physics/integrators/symplectic/forest_ruth.py`

**Algorithm**:

```
θ = 1/(2 - 2^(1/3))

v += θ * dt/2 * a(x, v)
x += θ * dt * v
v += (1-θ) * dt/2 * a(x, v)
x += (1-2θ) * dt * v
v += (1-θ) * dt/2 * a(x, v)
x += θ * dt * v
v += θ * dt/2 * a(x, v)
```

### 2.5 Omelyan Integrator

**Order**: 2nd order (optimized)
**File**: `physics/integrators/symplectic/omelyan.py`

**Algorithm**:

```
λ = 0.1932...  # Optimized parameter

v += (1-2λ) * dt/2 * a(x, v)
x += λ * dt * v
v += λ * dt * a(x, v)
x += (1-2λ) * dt * v
v += λ * dt * a(x, v)
x += λ * dt * v
v += (1-2λ) * dt/2 * a(x, v)
```

---

## 3. Runge-Kutta Methods

Standard ODE solvers (not symplectic).

### 3.1 RK4 (Runge-Kutta 4th Order)

**Order**: 4th order
**File**: `physics/integrators/runge_kutta/rk4.py`

**Algorithm**:

```
k1 = a(x, v, F)

k2 = a(x + dt/2 * v, v + dt/2 * k1, F)

k3 = a(x + dt/2 * v + dt^2/4 * k1, v + dt/2 * k2, F)

k4 = a(x + dt * v + dt^2 * k2, v + dt * k3, F)

v_new = v + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
x_new = x + dt * v + dt^2/6 * (k1 + k2 + k3)
```

**Mathematical Form**:

$$v_{n+1} = v_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

Where:
- $k_1 = a(x_n, v_n)$
- $k_2 = a(x_n + \frac{\Delta t}{2}v_n, v_n + \frac{\Delta t}{2}k_1)$
- $k_3 = a(x_n + \frac{\Delta t}{2}v_n + \frac{\Delta t^2}{4}k_1, v_n + \frac{\Delta t}{2}k_2)$
- $k_4 = a(x_n + \Delta t v_n + \Delta t^2 k_2, v_n + \Delta t k_3)$

**Properties**:
- High accuracy
- Not symplectic (energy drifts)
- More expensive than leapfrog

### 3.2 Heun Integrator

**Order**: 2nd order
**File**: `physics/integrators/runge_kutta/heun.py`

**Algorithm**:

```
# Predictor
x_pred = x + dt * v
v_pred = v + dt * a(x, v, F)

# Corrector
v_new = v + dt/2 * (a(x, v, F) + a(x_pred, v_pred, F))
x_new = x + dt/2 * (v + v_pred)
```

---

## 4. Adaptive Integrator

**File**: `physics/integrators/adaptive.py`

Automatically adjusts timestep based on error estimate.

**Algorithm**:

```
dt_eff = dt
for step in range(max_attempts):
    # Take step with dt_eff
    x1, v1 = step(x, v, dt_eff)
    
    # Take two half-steps
    x2, v2 = step(x, v, dt_eff/2)
    x2, v2 = step(x2, v2, dt_eff/2)
    
    # Estimate error
    error = ||x1 - x2|| + ||v1 - v2||
    
    if error < tolerance:
        break
    
    dt_eff = dt_eff * min(max_factor, safety * (tolerance/error)^(1/order))

return x2, v2
```

---

## 5. Comparison Summary

| Integrator | Order | Symplectic | Cost | Stability |
|------------|-------|------------|------|-----------|
| leapfrog | 2nd | ✅ | 1× | Best |
| verlet | 2nd | ✅ | 1× | Good |
| omelyan | 2nd | ✅ | ~1.5× | Good |
| heun | 2nd | ❌ | ~1.5× | Moderate |
| rk4 | 4th | ❌ | ~4× | Good |
| yoshida | 4th | ✅ | ~3× | Excellent |
| forest_ruth | 4th | ✅ | ~3× | Excellent |
| adaptive | Variable | ❌ | Variable | Good |

---

## 6. Selection Guidelines

- **Training**: Use `leapfrog` (most stable)
- **Long sequences**: Use `yoshida` or `forest_ruth`
- **Quick experiments**: Use `heun` or `rk4`
- **Variable precision**: Use `adaptive`

---

*File: technical/0_architecture/math/02_integrators.md*
*Last Updated: 2026-04-02*
