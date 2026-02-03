# MANIFOLD Physics and Mathematics

**Last Updated:** January 27, 2026

This document presents the mathematical foundations of the MANIFOLD architecture, derived from differential geometry and Hamiltonian mechanics. The formulation establishes a bridge between symbolic reasoning (discrete logic) and geometric dynamics (continuous flow).



## 1. Geodesic Equation

### 1.1 Core Principle

The central principle of MANIFOLD states that state evolution follows geodesics (shortest paths) on a Riemannian manifold. The fundamental equation is:

$$\frac{d^2 x^k}{d\tau^2} + \Gamma^k_{ij}(x) \frac{dx^i}{d\tau} \frac{dx^j}{d\tau} = F^k_{\text{ext}}$$

Where notation follows physics mathematics conventions:

| Symbol | Meaning | Neural Analogue |
|--------|---------|-----------------|
| $x^k$ | Position component k | Latent state |
| $\Gamma^k_{ij}$ | Christoffel symbols (curvature) | Non-linear interactions |
| $F^k_{\text{ext}}$ | External force | Token embedding |
| $\tau$ | Abstract time | Processing step |

### 1.2 Vector Form

In simplified index notation, the geodesic equation is expressed as:

$$\ddot{x} + \Gamma(x)(\dot{x}, \dot{x}) = F$$

This form emphasizes that acceleration includes both space curvature (first term) and external forces (second term). Absence of external forces ($\ddot{x} = -\Gamma(x)(\dot{x}, \dot{x})$) produces geodesics, the shortest paths between points on a curved manifold.



## 2. Christoffel Symbols

### 2.1 Standard Definition

Christoffel symbols describe how the affine connection (covariant derivative) acts on vectors. The classic definition derived from the metric tensor $g$ is:

$$\Gamma^k_{ij} = \frac{1}{2} g^{k\ell} \left( \frac{\partial g_{j\ell}}{\partial x^i} + \frac{\partial g_{i\ell}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^\ell} \right)$$

### 2.2 Low-Rank Formulation (Implementation)

Instead of deriving $\Gamma$ from a metric tensor $g$ (which requires solving partial differential equations), MANIFOLD parameterizes $\Gamma$ directly as a low-rank quadratic operator:

$$\Gamma(v, x) \approx W \cdot \left[ (U^T v)^2 \odot \sigma\left(\|U^T v\|\right) \right]$$

This formulation provides important computational features:

| Component | Dimension | Function |
|-----------|-----------|----------|
| $U \in \mathbb{R}^{d \times r}$ | Basis matrix | Velocity projection to rank space |
| $W \in \mathbb{R}^{d \times r}$ | Weight matrix | Curvature composition |
| $(U^T v)^2$ | Quadratic operation | Second-order interactions |
| $\sigma(\cdot)$ | Soft saturation | Numerical stability |
| $r$ (rank) | Integer 16-64 | Low-rank compression |

### 2.3 Advantages of Low-Rank Formulation

The low-rank approximation offers three fundamental advantages:

1. **Parameter efficiency**: Only $2 \cdot d \cdot r$ parameters are learned instead of $d^2$ for a full curvature matrix.

2. **Implicit regularization**: The low-rank structure prevents overfitting by limiting the range of possible interactions.

3. **Interpretability**: Columns of $U$ and rows of $W$ can be interpreted as independent "modes" of curvature.



## 3. Hamiltonian Structure

### 3.1 Pseudo-Energy Function

Although the system is forced (non-conservative), MANIFOLD maintains a Hamiltonian structure that ensures stability. The pseudo-energy is defined as:

$$\mathcal{H}(x, v) = \frac{1}{2} \|v\|^2 + V(x)$$

Where the first term represents kinetic energy and the second potential. For MANIFOLD, the potential $V(x)$ is implicit in the Christoffel geometry.

### 3.2 Complete Dynamics

Complete dynamics including thermodynamic friction is:

$$\frac{dx}{dt} = v$$
$$\frac{dv}{dt} = F_{\text{token}} - \Gamma(v, x) - F_{\text{friction}}(v, x)$$

### 3.3 Thermodynamic Friction (The Clutch)

The friction term enables switching between memory and computation modes:

$$F_{\text{friction}} = -\mu(x, u) \cdot v$$

Where the friction coefficient is learned as:

$$\mu(x, u) = \text{sigmoid}(W_{\text{gate}} \cdot x) \cdot \mu_{\text{max}}$$

**Operational Regimes**:

| Regime | μ | Behavior |
|--------|---|----------|
| Superfluid | μ ≈ 0 | Perfect memory conservation |
| Dissipative | μ >> 0 | Clean information update |



## 4. Symplectic Integration

### 4.1 Leapfrog Scheme (Velocity Verlet)

The main MANIFOLD integrator implements the second-order Leapfrog scheme:

$$v_{n+\frac{1}{2}} = v_n + \frac{1}{2} \Delta t \cdot a_n$$
$$x_{n+1} = x_n + \Delta t \cdot v_{n+\frac{1}{2}}$$
$$v_{n+1} = v_{n+\frac{1}{2}} + \frac{1}{2} \Delta t \cdot a_{n+1}$$

### 4.2 Complete Implementation with Friction

```python
def leapfrog_step(x, v, F, christoffel, dt, gate_activ):
    # Learned friction coefficient
    mu = torch.sigmoid(gate_activ) * 5.0
    
    # Acceleration at time t
    gamma = christoffel(v, x)
    friction = mu * v
    a_t = F - gamma - friction
    
    # Half-step velocity
    v_half = v + 0.5 * dt * a_t
    
    # Full-step position
    x_next = x + dt * v_half
    
    # Acceleration at time t+1
    gamma_next = christoffel(v_half, x_next)
    friction_next = mu * v_half
    a_next = F - gamma_next - friction_next
    
    # Half-step velocity final
    v_next = v_half + 0.5 * dt * a_next
    
    # Velocity normalization (critical for stability)
    v_next = v_next / (||v_next|| + \epsilon)
    
    return x_next, v_next
```

### 4.3 Integrator Properties

| Property | Description | Implication |
|----------|-------------|-------------|
| Time-reversible | Symmetric under t → -t | Numerical stability |
| Volume-preserving | $\det\left(\frac{\partial(x', v')}{\partial(x, v)}\right) = 1$ | Stable gradients |
| Local error | $O(\Delta t^3)$ | Step precision |
| Global error | $O(\Delta t^2)$ | Accumulated precision |



## 5. High-Order Integrators

### 5.1 Forest-Ruth (4th Order Symplectic)

The Forest-Ruth integrator provides superior precision for complex reasoning tasks:

$$\theta = \frac{1}{2 - 2^{1/3}}$$
$$\lambda = 1 - 2 \cdot \theta$$

The scheme applies multiple sub-steps with optimized coefficients to minimize energy error.

### 5.2 Integrator Comparison

| Integrator | Order | Symplectic | Energy Error | Speed |
|------------|-------|------------|--------------|-------|
| Euler | 1 | No | High | Fast |
| Heun | 2 | No | Medium | Fast |
| Leapfrog | 2 | Yes | Low | Fast |
| Forest-Ruth | 4 | Yes | **Very Low** | Medium |
| Yoshida | 4 | Yes | Very Low | Medium |
| RK4 | 4 | No | Low (may diverge) | Slow |

### 5.3 The Runge-Kutta Paradox

A key benchmark discovery is that high-order Runge-Kutta (RK4) diverges instantly in MANIFOLD, while lower-order symplectic methods remain stable.

**Explanation (Singularity Aliasing)**:

1. **Discontinuous Topology**: Features like logical singularities create a non-smooth ($C^2$ or less) force field.

2. **Multi-Stage Error**: RK4 evaluates 4 stages per step. If an intermediate stage evaluates a position inside a high-curvature singularity, the 4th-order polynomial over-extrapolates the force.

3. **Stability Criterion**: Lower-order methods (Heun, Euler, Leapfrog) are more "local" and don't attempt to model high-order derivatives of a field that isn't smooth.



## 6. Gradient Flow Stability

### 6.1 Liouville's Theorem

Liouville's theorem states that phase-space volume is preserved under Hamiltonian evolution:

$$\frac{d}{dt} \det\left(\frac{\partial(x, v)}{\partial(x_0, v_0)}\right) = 0$$

### 6.2 Implications for Gradients

Volume preservation has direct consequences for gradient propagation:

$$\frac{\partial \mathcal{L}}{\partial x_0} = J^T \cdot \frac{\partial \mathcal{L}}{\partial x_T}$$
$$\left\|\frac{\partial \mathcal{L}}{\partial x_0}\right\| \approx \left\|\frac{\partial \mathcal{L}}{\partial x_T}\right\|$$

Where $J$ is the flow Jacobian. As $\det(J) = 1$, gradients neither vanish nor explode.



## 7. Reactive Curvature

### 7.1 Energy-Based Plasticity

Reactive curvature modulates geometry based on model uncertainty, measured by thought kinetic energy:

$$K = \frac{1}{2} \|v\|^2$$

The plasticity scalar is defined as:

$$\lambda(K) = \alpha \cdot \tanh(K)$$

### 7.2 Effective Connection

The effective Christoffel connection includes plasticity modulation:

$$\Gamma_{\text{eff}} = \Gamma_{\text{base}} \cdot (1 + \lambda(K))$$

### 7.3 Interpretation

When the model is "confused" (high oscillation/velocity):
- The space becomes viscous
- Curvature increases
- The reasoning process "brakes" automatically
- More information is integrated before proceeding



## 8. Logical Singularities

### 8.1 Concept

Singularities represent discrete logical decisions as topological attractors in continuous space. This allows a purely continuous system to represent discrete logical operations.

### 8.2 Formulation

When local curvature exceeds a critical threshold, the system "opens" a sub-manifold:

$$x_{\text{macro}} \xrightarrow{\mathcal{R} > \tau} x_{\text{micro}}$$

### 8.3 Configuration

```python
singularities = {
    'enabled': True,
    'strength': 20.0,    # Attraction strength
    'threshold': 0.8     # Activation
}
```



## 9. Fractal Manifolds

### 9.1 Motivation

Continuous integration has inherent temporal precision limitations. High-frequency dynamics risk being "aliased" or completely lost.

### 9.2 Recursive Solution

Fractal manifolds implement adaptive resolution:

```python
fractal = {
    'enabled': True,
    'threshold': 0.5,    # Critical curvature
    'alpha': 0.2         # Refinement factor
}
```

When $\mathcal{R} > \tau$, the manifold recursively opens a sub-manifold with timestep $dt' = \alpha \cdot dt$, resolving high-frequency dynamics that would be lost by the macro integrator.

### 9.3 Geometric Interpretation

This mechanism is analogous to renormalization in theoretical physics: different scales of phenomena require different temporal resolutions to be captured correctly.



## 10. Riemannian Manifold Optimization

### 10.1 The Problem

Standard Euclidean updates violate learnable manifold constraints:

$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \nabla \mathcal{L}$$

can produce matrices that violate properties necessary to maintain a valid Christoffel structure.

### 10.2 Solution: Retractions

Retractions map Euclidean updates back to the manifold:

$$W_{\text{new}} = \text{Retract}_M(W_{\text{old}} - \eta \cdot \nabla \mathcal{L})$$

### 10.3 Retraction Types

| Type | Formula | Use Case |
|------|---------|----------|
| Normalize | $W \cdot \min(1, \text{max\_norm}/\|W\|)$ | **Recommended** |
| Cayley | $(I - \frac{1}{2} A)^{-1}(I + \frac{1}{2} A)$ | Orthogonal matrices |
| Exponential | $\exp(A)$ | Matrix manifolds |



## 11. Formulation Summary

### 11.1 Principal Equations

| Concept | Equation | Description |
|---------|----------|-------------|
| Geodesic | $\ddot{x} + \Gamma(\dot{x}, \dot{x}) = F$ | State evolution |
| Christoffel (LR) | $\Gamma \approx W[(U^T v)^2 \odot \sigma]$ | Low-rank curvature |
| Friction | $\mu = \text{sigmoid}(W_g x) \cdot 5.0$ | Thermodynamic gating |
| Plasticity | $\lambda = \alpha \cdot \tanh(\|v\|^2/2)$ | Reactive curvature |
| Leapfrog | See Section 4.1 | Numerical integration |
| Hamiltonian | $\mathcal{H} = \frac{1}{2}\|v\|^2 + V(x)$ | Pseudo-energy |

### 11.2 Physical-Neural Interpretation

| Physical Concept | Neural Analogue |
|-----------------|-----------------|
| Particle | Latent state |
| Trajectory | Reasoning process |
| Geodesic | Optimal inference |
| Curvature | Non-linear interactions |
| Singularity | Logical decision |
| Friction | Forgetting/Update |
| Phase space | Memory |



**Document Version**: 2.6.2  
**Last Updated**: January 27, 2026

For practical applications, see [API.md](API.md).  
For system architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).  
For benchmarks, see [BENCHMARKS.md](BENCHMARKS.md).  
For complete derivations, see [SCIENTIFIC_PAPER.md](SCIENTIFIC_PAPER.md).
