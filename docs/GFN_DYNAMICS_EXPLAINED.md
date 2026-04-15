# GFN Dynamics: G-SSM vs ISN Architectures

This document explains the two GFN realizations in our codebase: **G-SSM** (Geometric State Space Model) and **ISN** (Intelligent Simulation Network). Both implement the GFN paradigm but with different architectural choices.

## 1. The Core GFN Paradigm

In both realizations, the internal state is not a statistical memory but a **particle** evolving on a differentiable manifold $M$.

## 2. Two Realizations: G-SSM (2SSM) vs ISN (1SSM)

### G-SSM: Second-Order State Space Model (2SSM)

The G-SSM realization maintains a **phase-space state** consisting of position and velocity:

$$\text{State: } z = [x, v] \in M \times TM$$

**Equations of Motion:**
$$\begin{aligned}
\frac{dx}{dt} &= v \\
\frac{dv}{dt} &= -\Gamma(x, v) + F_{\text{ext}} + F_{\text{friction}} + F_{\text{ghost}}
\end{aligned}$$

Where:
- $x \in M$ is the position on the manifold
- $v \in T_xM$ is the velocity (tangent vector)
- $\Gamma(x,v)$ = Christoffel-induced force (geometric curvature)
- $F_{\text{ext}}$ = External input force from token embeddings
- $F_{\text{friction}}$ = $-\mu \cdot v$ from geometry's friction gate
- $F_{\text{ghost}}$ = Ghost force from hysteresis module (if enabled)

**Key Properties:**
- **2nd Order**: Evolution depends on both position and velocity
- **Hamiltonian Structure**: Implicitly preserves phase-space volume
- **Physical**: Models actual particle dynamics on curved manifolds

> **Note on ODE Order**: G-SSM is classified as a 2nd Order ODE system because the position evolution `dx/dt = v` combined with velocity evolution `dv/dt = a(x,v)` is mathematically equivalent to `d²x/dt² = a(x, dx/dt)` — the second derivative of position. While written as two coupled 1st-order equations for numerical integration, the system describes 2nd-order dynamics in the position variable.

### ISN: First-Order State Space Model (1SSM)

The ISN realization maintains only a **world state** (single vector):

$$\text{State: } x \in M$$

**Equation of Motion:**
$$\frac{dx}{dt} = \text{Drift}(x) + \text{Coupling}(u)$$

Where:
- $x \in M$ is the state of the world
- $u$ is the external input (token processed by Scanner)
- **Drift$(x)$**: Internal dynamics that ensures continuity when $u=0$ (non-linear, typically $\tanh(W \cdot x + b)$)
- **Coupling$(u)$**: External signal coupling (typically $W \cdot u$)

**Key Properties:**
- **1st Order**: State evolution depends only on current state
- **Simpler**: No velocity component, reduced state size
- **Abstract**: Does not explicitly model physical particle dynamics

## 3. Key Differences

| Characteristic | G-SSM (2SSM) | ISN (1SSM) |
| :--- | :--- | :--- |
| **State Space** | $(x, v)$ - Position + Velocity | $x$ - Position only |
| **Order** | 2nd Order ODE | 1st Order ODE |
| **Dynamics** | Physical particle on manifold | Abstract flow on manifold |
| **Invariants** | Implicit Hamiltonian structure | Explicit energy-like quantities |
| **Complexity** | Higher (2x state size) | Lower (compact state) |
| **Use Case** | Physics simulation, geometric deep learning | Language modeling, sequence processing |
| **Geometry** | Riemannian with Christoffel symbols | Riemannian with drift fields |

## 4. Architectural Relationship

**G-SSM came first** as the original implementation exploring geometric deep learning concepts. **ISN evolved from G-SSM** as a simplified realization focused on sequence modeling.

Both share:
- Geometric manifold structure
- Gating mechanisms for stability
- Invariant-preserving dynamics
- O(1) memory complexity

They differ in:
- State representation (phase-space vs position-only)
- Physical interpretability (explicit particle vs abstract flow)
- Computational trade-offs (accuracy vs efficiency)

## 5. When to Use Which?

**Use G-SSM when:**
- You need physical interpretability
- Modeling systems with momentum/inertia
- Simulating actual particle dynamics
- Working with physics-informed applications

**Use ISN when:**
- You want maximum computational efficiency
- Sequence modeling is the primary goal
- Physical interpretability is less critical
- Working with language or time-series data

---

"The particle does not remember the past; it simply flows along the curvature the past left on the manifold."
