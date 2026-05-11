# Velocity Verlet Integrator

## What is it?

The Velocity Verlet (or simply Verlet) integrator is a second-order symplectic method that explicitly includes velocity in the calculation. It is mathematically equivalent to Leapfrog but organized differently.

Originally developed by Loup Verlet in 1967 for molecular dynamics simulations.

---

## The Algorithm

Verlet computes position using both velocity and acceleration explicitly.

### Step 1: Initial Acceleration

Compute acceleration at current state:

$$a_n = a(x_n, v_n)$$

### Step 2: Position Update

Update position using current velocity and acceleration:

$$x_{n+1} = x_n + v_n \cdot \Delta t + \frac{1}{2} a_n \cdot \Delta t^2$$

### Step 3: Velocity Average

Compute intermediate velocity:

$$v_{avg} = v_n + \frac{1}{2} a_n \cdot \Delta t$$

### Step 4: New Acceleration

Compute acceleration at new position:

$$a_{n+1} = a(x_{n+1}, v_{avg})$$

### Step 5: Velocity Update

Complete velocity update using average acceleration:

$$v_{n+1} = v_n + \frac{1}{2} (a_n + a_{n+1}) \cdot \Delta t$$

---

## Comparison with Leapfrog

| Aspect | Leapfrog | Verlet |
|--------|----------|--------|
| Velocity storage | Half-integer steps | Integer steps |
| Position update | Uses $v_{n+1/2}$ | Uses $v_n$ and $a_n$ |
| Implementation | Simpler | More explicit |
| Result | Same trajectory | Same trajectory |

Both methods produce identical trajectories for the same initial conditions and time step.

---

## Properties

| Property | Value |
|----------|-------|
| **Order** | 2nd order |
| **Symplectic** | Yes |
| **Force evaluations** | 2 per step |
| **Time-reversible** | Yes |
| **Energy conservation** | Good |

---

## Error Analysis

Local truncation error: $O(\Delta t^3)$

Global error: $O(\Delta t^2)$

The position update includes the $\frac{1}{2} a \Delta t^2$ term which gives better accuracy than first-order methods.

---

## When to Use

**Use Verlet when:**
- You need position at every integer time step
- Velocity must be synchronized with position
- You want explicit velocity in the algorithm
- Molecular dynamics simulations

**Equivalent to Leapfrog:**
- Either choice produces same results
- Choose based on implementation preference

---

## Historical Note

The Verlet algorithm was originally developed for simulating Lennard-Jones fluids in molecular dynamics. Its simplicity and symplectic properties made it the standard for MD simulations for decades.

The equivalence with Leapfrog was recognized later, showing both are different implementations of the same underlying symplectic map.

---

*File: technical/0_architecture/math/integrators/verlet.md*
*Last Updated: 2026-04-02*
