# RK4 (Runge-Kutta 4th Order)

## What is it?

The Runge-Kutta 4th order method (RK4) is a classic numerical integration technique developed by Carl Runge and Martin Kutta around 1900. Unlike symplectic integrators, RK4 prioritizes accuracy over energy conservation.

RK4 is the most widely used general-purpose ODE solver due to its excellent accuracy-to-cost ratio.

---

## The Algorithm

RK4 uses a weighted average of four intermediate estimates to compute the next state.

### Four Intermediate Steps

Given the current state $(x_n, v_n)$:

#### k1: Initial slope
$$k_{1x} = v_n$$
$$k_{1v} = a(x_n, v_n)$$

#### k2: Midpoint using k1
$$k_{2x} = v_n + \frac{\Delta t}{2} \cdot k_{1v}$$
$$x_{mid} = x_n + \frac{\Delta t}{2} \cdot k_{1x}$$
$$k_{2v} = a(x_{mid}, v_n + \frac{\Delta t}{2} \cdot k_{1v})$$

#### k3: Midpoint using k2
$$k_{3x} = v_n + \frac{\Delta t}{2} \cdot k_{2v}$$
$$x_{mid2} = x_n + \frac{\Delta t}{2} \cdot k_{2x}$$
$$k_{3v} = a(x_{mid2}, v_n + \frac{\Delta t}{2} \cdot k_{2v})$$

#### k4: Endpoint using k3
$$k_{4x} = v_n + \Delta t \cdot k_{3v}$$
$$x_{end} = x_n + \Delta t \cdot k_{3x}$$
$$k_{4v} = a(x_{end}, v_n + \Delta t \cdot k_{3v})$$

### Final Update

$$x_{n+1} = x_n + \frac{\Delta t}{6} \cdot (k_{1x} + 2k_{2x} + 2k_{3x} + k_{4x})$$

$$v_{n+1} = v_n + \frac{\Delta t}{6} \cdot (k_{1v} + 2k_{2v} + 2k_{3v} + k_{4v})$$

---

## The Butcher Tableau

RK4 can be represented as:

$$
\begin{array}{c|cccc}
0 & & & & \\
1/2 & 1/2 & & & \\
1/2 & 0 & 1/2 & & \\
1 & 0 & 0 & 1 & \\
\hline
& 1/6 & 1/3 & 1/3 & 1/6
\end{array}
$$

This compactly encodes the coefficient structure.

---

## Properties

| Property | Value |
|----------|-------|
| **Order** | 4th order (error ~ $O(\Delta t^5)$) |
| **Symplectic** | No |
| **Force evaluations** | 4 per step |
| **Energy conservation** | Drifts over time |
| **Accuracy** | Very high |

---

## Why it Works

### Weighted Average

The coefficients (1/6, 1/3, 1/3, 1/6) are carefully chosen to:
- Match Taylor series expansion up to 4th order
- Minimize truncation error
- Provide optimal accuracy for 4 evaluations

### Simpson's Rule Connection

The weights approximate Simpson's rule for integration:

$$\int_0^{\Delta t} f(t) dt \approx \frac{\Delta t}{6}[f(0) + 4f(\Delta t/2) + f(\Delta t)]$$

RK4 extends this to handle state-dependent forces.

---

## Error Analysis

Local truncation error: $O(\Delta t^5)$

Global error: $O(\Delta t^4)$

The error constant is smaller than symplectic methods of the same order.

---

## Comparison with Symplectic Methods

| Aspect | RK4 | Yoshida/Forest-Ruth |
|--------|-----|---------------------|
| Order | 4th | 4th |
| Symplectic | No | Yes |
| Energy drift | Linear in time | Bounded/oscillatory |
| Accuracy per step | Higher | Lower |
| Long-term behavior | Energy error grows | Energy oscillates |
| Force evaluations | 4 | 3 |

---

## When to Use

**Use RK4 for:**
- Short trajectories where accuracy matters
- Non-Hamiltonian systems
- Validation and testing
- When symplectic properties don't matter

**Don't use for:**
- Long training runs (energy drift)
- Production systems (accumulating errors)
- When stability matters more than accuracy

---

## Energy Drift

Unlike symplectic methods, RK4 has systematic energy drift:

$$\frac{\Delta E}{E} \propto t \cdot \Delta t^4$$

Over long times, this can cause significant deviation from the true trajectory.

---

*File: technical/0_architecture/math/integrators/rk4.md*
*Last Updated: 2026-04-02*
