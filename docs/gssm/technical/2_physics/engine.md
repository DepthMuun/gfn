# Physics Engine & Dynamics

The `gfn/physics` module handles the temporal evolution of the manifold state. It defines how particles (data) move through the space under the influence of neural "forces".

## 1. Physics Engine (`engine.py`)
The `ManifoldPhysicsEngine` is the central orchestrator. It bridges Geometry and Integrators.
- **Function**: Translates abstract coordinates into accelerations using the geodesic equation:
  $$\frac{d^2x^\sigma}{dt^2} + \Gamma^\sigma_{\mu\nu} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt} = F^\sigma$$
- **Metric Normalization**: Optionally scales velocities by the local metric trace $(\text{Tr}(g))$ to prevent numerical explosion.

---

## 2. Integrators (`gfn/physics/integrators/`)
Solvers that compute the next state $(x_{t+1}, v_{t+1})$ given $(x_t, v_t)$ and $Force_t$.

### Symplectic Integrators (Conservation Focused)
- **Leapfrog**: 2nd-order, volume-preserving.
- **Yoshida**: 4th-order, high-precision symplectic solver.
- **Forest-Ruth**: Specialized 4th-order solver for Hamiltonian systems.

### Runge-Kutta Integrators (Accuracy Focused)
- **RK4**: Classic 4th-order solver. Better for non-conservative fields but prone to energy drift in long sequences.

---

## 3. Dynamics Framework (`gfn/physics/dynamics/`)
Defines the "Update Law" for the manifold state.

- **Direct**: $x_{next} = \text{Integrator}(x, v, f)$.
- **Residual**: The model predicts the *acceleration* $\Delta v$ instead of the total force.
- **Gated**: A learned gate (Hamiltonian Gate) modulates how much of the force is applied based on the current state.

---

## 4. Adaptive Gating (`gating.py`)
GFN uses **Riemannian Gating** to adjust the time-step $dt$ based on the manifold's complexity.

- **Intuition**: If the space is "very curved" (high Christoffel norm), the model automatically slows down (smaller $dt$) to preserve numerical stability.
- **Thermodynamic Layer**: An alternative gating mechanism that uses "entropy-energy" balance to modulate flow speed.

---

## 5. Metric Normalization (`normalization.py`)
Ensures that the magnitude of $x$ and $v$ remains consistent across different topologies.
- **Toroidal Norm**: Projects velocities into the tangent space of the torus.
- **Trace Norm**: Scales vectors by $\frac{1}{\sqrt{\text{Tr}(g)}}$ for Riemannian consistency.
