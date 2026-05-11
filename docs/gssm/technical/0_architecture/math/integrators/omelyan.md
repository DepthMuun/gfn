# Omelyan Integrator

## What is it?

The Omelyan integrator is a second-order symplectic method with optimized coefficients. Unlike standard leapfrog, it uses a more complex sequence of substeps to minimize the error constant for 2nd order methods.

Developed by Igor Omelyan in 1997 as an optimized symplectic integrator for molecular dynamics.

---

## The Algorithm

Omelyan uses 6 force evaluations per step to achieve optimized 2nd order accuracy.

### Optimized Parameter

$$\zeta \approx 0.1932$$

This value is numerically optimized to minimize the truncation error coefficient.

### Step Sequence

#### First Kick (Partial)
$$v_{1/2} = v_n + \frac{1-2\zeta}{2} \cdot \Delta t \cdot a(x_n)$$

#### First Drift
$$x_{1/6} = x_n + \zeta \cdot \Delta t \cdot v_{1/2}$$

#### Second Kick
$$v_{1/3} = v_{1/2} + \zeta \cdot \Delta t \cdot a(x_{1/6})$$

#### Second Drift
$$x_{2/3} = x_{1/6} + (1-2\zeta) \cdot \Delta t \cdot v_{1/3}$$

#### Third Kick
$$v_{2/3} = v_{1/3} + \zeta \cdot \Delta t \cdot a(x_{2/3})$$

#### Third Drift
$$x_{5/6} = x_{2/3} + \zeta \cdot \Delta t \cdot v_{2/3}$$

#### Final Kick
$$v_{n+1} = v_{2/3} + \frac{1-2\zeta}{2} \cdot \Delta t \cdot a(x_{5/6})$$

$$x_{n+1} = x_{5/6}$$

---

## Properties

| Property | Value |
|----------|-------|
| **Order** | 2nd order |
| **Symplectic** | Yes |
| **Force evaluations** | 6 per step |
| **Error constant** | Optimized (smaller than leapfrog) |
| **Cost** | 6× Leapfrog |

---

## Optimization Principle

The parameter $\zeta \approx 0.1932$ is chosen to minimize:

$$C(\zeta) = \sum |\text{error\_coefficients}|$$

This gives Omelyan a smaller error constant than standard leapfrog for the same order, making it more accurate step-for-step (though much more expensive).

---

## Comparison with Leapfrog

| Aspect | Leapfrog | Omelyan |
|--------|----------|---------|
| Order | 2nd | 2nd |
| Force evals | 2 | 6 |
| Accuracy | Good | Better (optimized) |
| Cost | 1× | 3× |
| Use case | General | High-precision 2nd order |

---

## When to Use

**Use Omelyan for:**
- When you need 2nd order accuracy but want smaller error constant
- High-precision short simulations
- Validation and testing

**Don't use for:**
- Training (too expensive)
- Long sequences (cost prohibitive)
- Production (use leapfrog instead)

---

## Error Analysis

Both methods are 2nd order:

$$\text{Error} \propto \Delta t^3$$

But Omelyan has a smaller proportionality constant:

$$\frac{\text{Omelyan\_error}}{\text{Leapfrog\_error}} \approx 0.4$$

So for the same $\Delta t$, Omelyan is ~2.5× more accurate (but 3× more expensive).

---

*File: technical/0_architecture/math/integrators/omelyan.md*
*Last Updated: 2026-04-02*
