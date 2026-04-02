# Leapfrog Integrator

## What is it?

The Leapfrog (also called Störmer-Verlet or Velocity Verlet) is a second-order symplectic integrator. It is the default integrator in GSSM because it provides the best balance of stability, accuracy, and computational cost.

The name "leapfrog" comes from the way position and velocity "leap over" each other in time: velocity is computed at half-time steps while position is computed at full time steps.

---

## The Algorithm

Leapfrog uses a "kick-drift-kick" pattern:

### Step 1: Half-Step Velocity (Kick)

Update velocity using the acceleration at the current position:

$$v_{n+1/2} = v_n + \frac{\Delta t}{2} \cdot a(x_n, v_n)$$

Where:
- $v_n$ is velocity at time step $n$
- $a(x_n, v_n)$ is the acceleration computed by the PhysicsEngine
- $\Delta t$ is the time step

### Step 2: Full-Step Position (Drift)

Update position using the half-step velocity:

$$x_{n+1} = x_n + \Delta t \cdot v_{n+1/2}$$

### Step 3: Re-evaluate Acceleration

Compute acceleration at the new position:

$$a_{n+1} = a(x_{n+1}, v_{n+1/2})$$

### Step 4: Half-Step Velocity (Kick)

Complete the velocity update:

$$v_{n+1} = v_{n+1/2} + \frac{\Delta t}{2} \cdot a_{n+1}$$

---

## With Friction

When friction is present, the algorithm becomes:

$$v_{n+1/2} = \frac{v_n + \frac{\Delta t}{2} \cdot a_n}{1 + \frac{\Delta t}{2} \cdot \mu}$$

$$x_{n+1} = x_n + \Delta t \cdot v_{n+1/2}$$

$$v_{n+1} = \frac{v_{n+1/2} + \frac{\Delta t}{2} \cdot a_{n+1}}{1 + \frac{\Delta t}{2} \cdot \mu}$$

Where $\mu$ is the friction coefficient.

---

## Properties

| Property | Value |
|----------|-------|
| **Order** | 2nd order (error ~ $O(\Delta t^3)$ per step) |
| **Symplectic** | Yes (preserves phase space volume) |
| **Time-reversible** | Yes |
| **Force evaluations** | 2 per step |
| **Energy drift** | Minimal over long times |

---

## Why it Works

### Symplectic Property

Leapfrog preserves the symplectic 2-form $\omega = dp \wedge dq$ in phase space. This means:

- Energy oscillates around the true value but doesn't drift systematically
- Long-term behavior remains qualitatively correct
- Good for Hamiltonian systems

### Time Reversibility

If you reverse time (flip $v \to -v$ and integrate backwards), you return to the original state exactly. This implies:

$$x_{-n} = x_0 \quad \text{and} \quad v_{-n} = -v_0$$

when integrated backwards from $(x_n, -v_n)$.

### Error Analysis

Local truncation error per step: $O(\Delta t^3)$

Global error after $N$ steps: $O(\Delta t^2)$

The error doesn't accumulate catastrophically due to the symplectic property.

---

## When to Use

**Always use Leapfrog for:**
- Training (most stable)
- Long sequences (energy conservation)
- Production systems (reliable)
- When stability matters more than extreme accuracy

**Don't use when:**
- You need 4th order accuracy (use Yoshida instead)
- Non-Hamiltonian dynamics dominate

---

## Relationship to Verlet

Leapfrog and Verlet are mathematically equivalent but implemented differently:

- **Leapfrog**: Computes velocity at half-integer time steps
- **Verlet**: Computes position using $x_{n+1} = 2x_n - x_{n-1} + a_n \Delta t^2$

They produce identical trajectories given the same initial conditions.

---

*File: technical/0_architecture/math/integrators/leapfrog.md*
*Last Updated: 2026-04-02*
