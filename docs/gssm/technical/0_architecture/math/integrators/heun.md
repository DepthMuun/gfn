# Heun Integrator

## What is it?

The Heun method (also called Improved Euler or explicit trapezoidal rule) is a second-order predictor-corrector method. It uses an initial prediction step followed by a correction step that averages the slopes.

Named after Karl Heun (1859-1929), a German mathematician who contributed to differential equation theory.

---

## The Algorithm

Heun uses a predictor-corrector approach:

### Step 1: Predictor (Euler Step)

Make an initial prediction using standard Euler method:

$$\tilde{x} = x_n + \Delta t \cdot v_n$$
$$\tilde{v} = v_n + \Delta t \cdot a(x_n, v_n)$$

### Step 2: Corrector (Average)

Correct using the average of initial and predicted slopes:

$$x_{n+1} = x_n + \frac{\Delta t}{2} \cdot (v_n + \tilde{v})$$
$$v_{n+1} = v_n + \frac{\Delta t}{2} \cdot (a(x_n, v_n) + a(\tilde{x}, \tilde{v}))$$

---

## Geometric Interpretation

Heun approximates the integral using the trapezoidal rule:

$$\int_{t_n}^{t_{n+1}} f(t) dt \approx \frac{\Delta t}{2} [f(t_n) + f(t_{n+1})]$$

This is more accurate than simple rectangle rule (Euler) because it accounts for the slope change over the interval.

---

## Properties

| Property | Value |
|----------|-------|
| **Order** | 2nd order |
| **Symplectic** | No |
| **Force evaluations** | 2 per step |
| **Type** | Predictor-corrector |
| **Stability** | Better than Euler |

---

## Error Analysis

Local truncation error: $O(\Delta t^3)$

Global error: $O(\Delta t^2)$

Compared to standard Euler:
- Euler: error $\propto \Delta t$
- Heun: error $\propto \Delta t^2$

Heun is significantly more accurate than Euler for the same step size.

---

## Comparison with Leapfrog

| Aspect | Leapfrog | Heun |
|--------|----------|------|
| Order | 2nd | 2nd |
| Symplectic | Yes | No |
| Structure | Kick-drift-kick | Predictor-corrector |
| Energy | Conserved | Drifts |
| Accuracy | Similar | Similar |
| Use case | Hamiltonian | General ODEs |

---

## When to Use

**Use Heun for:**
- General ODEs (not necessarily Hamiltonian)
- Quick prototyping
- Educational purposes
- When symplectic property not needed

**Don't use for:**
- Long Hamiltonian simulations (use Leapfrog)
- Production training (use Leapfrog)

---

*File: technical/0_architecture/math/integrators/heun.md*
*Last Updated: 2026-04-02*
