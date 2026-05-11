# Forest-Ruth Integrator

## What is it?

The Forest-Ruth integrator is a fourth-order symplectic method developed by Etienne Forest and Ronald Ruth in 1990. Like Yoshida, it achieves higher accuracy through composition of lower-order steps, but uses different coefficient values optimized for specific properties.

---

## The Algorithm

Forest-Ruth uses a symmetric composition of three drift-kick pairs.

### Coefficients

$$\theta = \frac{1}{2 - 2^{1/3}} \approx 1.3512$$

Position coefficients:
$$c_1 = c_4 = \frac{\theta}{2}$$
$$c_2 = c_3 = \frac{1 - \theta}{2}$$

Velocity coefficients:
$$d_1 = d_3 = \theta$$
$$d_2 = 1 - 2\theta$$

### Step Sequence

#### First Drift
$$x_1 = x_n + c_1 \cdot \Delta t \cdot v_n$$

#### First Kick
$$v_1 = v_n + d_1 \cdot \Delta t \cdot a(x_1)$$

#### Second Drift
$$x_2 = x_1 + c_2 \cdot \Delta t \cdot v_1$$

#### Second Kick
$$v_2 = v_1 + d_2 \cdot \Delta t \cdot a(x_2)$$

#### Third Drift
$$x_3 = x_2 + c_3 \cdot \Delta t \cdot v_2$$

#### Third Kick
$$v_3 = v_2 + d_3 \cdot \Delta t \cdot a(x_3)$$

#### Final Drift
$$x_{n+1} = x_3 + c_4 \cdot \Delta t \cdot v_3$$
$$v_{n+1} = v_3$$

---

## Properties

| Property | Value |
|----------|-------|
| **Order** | 4th order |
| **Symplectic** | Yes |
| **Force evaluations** | 3 per step |
| **Symmetric** | Yes (time-reversible) |
| **Cost** | ~3× Leapfrog |

---

## Comparison with Yoshida

| Aspect | Yoshida | Forest-Ruth |
|--------|---------|-------------|
| Coefficients | Different derivation | Different derivation |
| Accuracy | Same order (4th) | Same order (4th) |
| Performance | Equivalent | Equivalent |
| Choice | Preference | Preference |

Both are fourth-order symplectic integrators with 3 force evaluations per step. They differ only in the specific coefficient values.

---

## When to Use

**Use Forest-Ruth when:**
- You need 4th order accuracy
- You want an alternative to Yoshida coefficients
- Long-term energy conservation is critical

**Equivalent to Yoshida:**
- Both achieve same accuracy
- Choose based on preference or testing

---

*File: technical/0_architecture/math/integrators/forest_ruth.md*
*Last Updated: 2026-04-02*
