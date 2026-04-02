# Yoshida Integrator

## What is it?

The Yoshida integrator is a fourth-order symplectic integrator. It achieves higher accuracy than Leapfrog by using a carefully constructed sequence of substeps with optimized coefficients.

Named after Haruo Yoshida, who developed the coefficient optimization scheme in 1990.

---

## The Algorithm

Yoshida constructs a 4th order method by composing three 2nd order leapfrog steps with different step sizes.

### Coefficients

The Yoshida coefficients are derived from:

$$w_1 = \frac{1}{2 - 2^{1/3}} \approx 1.3512$$
$$w_0 = \frac{-2^{1/3}}{2 - 2^{1/3}} \approx -1.7024$$

From these, the position and velocity coefficients are:

$$c_1 = c_4 = \frac{w_1}{2}$$
$$c_2 = c_3 = \frac{w_0 + w_1}{2}$$

$$d_1 = d_3 = w_1$$
$$d_2 = w_0$$

### Step Sequence

The algorithm performs three force evaluations per full step:

#### Sub-step 1
$$x_1 = x_n + c_1 \cdot \Delta t \cdot v_n$$
$$v_1 = v_n + d_1 \cdot \Delta t \cdot a(x_1)$$

#### Sub-step 2
$$x_2 = x_1 + c_2 \cdot \Delta t \cdot v_1$$
$$v_2 = v_1 + d_2 \cdot \Delta t \cdot a(x_2)$$

#### Sub-step 3
$$x_3 = x_2 + c_3 \cdot \Delta t \cdot v_2$$
$$v_3 = v_2 + d_3 \cdot \Delta t \cdot a(x_3)$$

#### Final Drift
$$x_{n+1} = x_3 + c_4 \cdot \Delta t \cdot v_3$$
$$v_{n+1} = v_3$$

---

## Properties

| Property | Value |
|----------|-------|
| **Order** | 4th order (error ~ $O(\Delta t^5)$ per step) |
| **Symplectic** | Yes |
| **Force evaluations** | 3 per step |
| **Cost** | ~3× Leapfrog |
| **Accuracy** | Higher than Leapfrog |

---

## Why it Works

### Composition Method

Yoshida is a composition of three leapfrog steps:

$$\Phi_{\Delta t} = \Phi_{\alpha_3 \Delta t} \circ \Phi_{\alpha_2 \Delta t} \circ \Phi_{\alpha_1 \Delta t}$$

Where $\Phi$ represents a leapfrog step and the coefficients $\alpha_i$ are chosen to cancel out 3rd and 4th order error terms.

### Error Cancellation

The specific values of $w_0$ and $w_1$ are chosen such that:

1. The sum of coefficients equals 1: $2w_1 + w_0 = 1$
2. Higher-order error terms cancel out
3. The method remains symmetric (time-reversible)

### Symplectic Property

Since each sub-step is symplectic and the composition of symplectic maps is symplectic, the full Yoshida step preserves the symplectic structure.

---

## Error Analysis

Local truncation error per step: $O(\Delta t^5)$

Global error after $N$ steps: $O(\Delta t^4)$

Compared to Leapfrog:
- Yoshida: error $\propto \Delta t^4$
- Leapfrog: error $\propto \Delta t^2$

For the same $\Delta t$, Yoshida is more accurate by a factor of $\Delta t^2$.

---

## When to Use

**Use Yoshida for:**
- Long simulations where accuracy matters
- When you need to minimize energy drift over many steps
- Scientific computing requiring high precision
- Validation against analytical solutions

**Don't use when:**
- Training speed is critical (3× cost)
- Stability matters more than accuracy (use Leapfrog)
- Short trajectories (error advantage not worth cost)

---

## Comparison with Leapfrog

| Aspect | Leapfrog | Yoshida |
|--------|----------|---------|
| Order | 2nd | 4th |
| Force evals/step | 2 | 3 |
| Accuracy | Good | Excellent |
| Speed | Fast | 1.5× slower |
| Energy drift | Low | Very low |

---

## Coefficient Derivation

The coefficients come from solving:

$$\sum \alpha_i = 1$$
$$\sum \alpha_i^3 = 0$$

The solution gives:
- $\alpha_1 = \alpha_3 = w_1$
- $\alpha_2 = w_0$

These ensure 4th order accuracy while maintaining symmetry.

---

*File: technical/0_architecture/math/integrators/yoshida.md*
*Last Updated: 2026-04-02*
