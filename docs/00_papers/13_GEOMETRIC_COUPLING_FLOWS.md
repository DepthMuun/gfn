# Symplectic Coupling Flows: Hybridizing Hamiltonian Dynamics and Normalizing Flows

**Author:** Joaquin Stürtz  
**Date:** January 26, 2026

**Abstract**  
Standard symplectic integrators, while stable, are typically restricted to separable Hamiltonian systems where the kinetic energy is a simple quadratic form of momentum. We introduce **Symplectic Coupling Flows**, a hybrid architecture that reformulates the numerical integration of geodesic flows as a sequence of volume-preserving coupling transformations. By borrowing the triangular structure of Normalizing Flows (e.g., NICE/RealNVP), we decouple the evolution of position and velocity into a series of shear mappings. This formulation guarantees a strictly unit Jacobian determinant ($\det J = 1$) regardless of the complexity of the learned force fields or the non-linear "drift" functions. We demonstrate that this approach enables the learning of **Intrinsic Kinematics**, where the relationship between velocity and coordinate updates is no longer fixed by Newtonian physics but is instead an optimized neural mapping, providing a flexible yet conservative foundation for Geodesic Flow Networks (GFN).



## 1. The Coupling Transformation in Phase Space

### 1.1 Non-Linear Shear Invariance
A coupling flow maps an input $(x, v)$ to an output $(x', v')$ via a sequence of triangular transformations. For a state divided into two partitions, a shear transformation takes the form:
$$ y_1 = x_1 $$
$$ y_2 = x_2 + f(x_1) $$
The Jacobian of this mapping is lower-triangular with ones on the diagonal, ensuring that the volume in phase space is preserved exactly ($\det J = 1$). In the context of GFN, we apply this principle to the joint evolution of position and velocity.

### 1.2 Symplectic Splitting as Coupling
We decompose a single integration step $\Delta t$ into a symmetric splitting of "Kick" (velocity update) and "Drift" (position update) operators. Unlike traditional integrators that assume a fixed identity for the drift, our formulation allows for a learnable **Neural Drift**:

1.  **Half-Kick (Velocity):** 
    $$ \mathbf{v}_{t+1/2} = \mathbf{v}_t + \frac{\Delta t}{2} \cdot \mathbf{a}(\mathbf{x}_t, \mathbf{v}=0, \mathbf{F}_t) $$
2.  **Neural Drift (Position):**
    $$ \mathbf{x}_{t+1} = \mathbf{x}_t + \Delta t \cdot \left( \mathbf{v}_{t+1/2} + \mathcal{G}_\theta(\mathbf{v}_{t+1/2}) \right) $$
3.  **Full-Kick (Velocity):**
    $$ \mathbf{v}_{t+1} = \mathbf{v}_{t+1/2} + \frac{\Delta t}{2} \cdot \mathbf{a}(\mathbf{x}_{t+1}, \mathbf{v}=0, \mathbf{F}_t) $$

where $\mathbf{a}$ represents the acceleration (including learned Christoffel forces) and $\mathcal{G}_\theta$ is a learned MLP that warps the kinematic relationship.

## 2. Learnable Kinematics and the Neural Drift

### 2.1 Beyond Newtonian Drift
In standard Euclidean physics, the drift is simply $\Delta x = v \Delta t$. However, on complex semantic manifolds, the "effective mass" or "inertial resistance" may vary depending on the direction of thought. By introducing the **Drift Network** $\mathcal{G}_\theta(\mathbf{v})$, we allow the model to learn a non-linear velocity-to-position mapping.

### 2.2 Preserving the Symplectic Structure
Even with a complex neural network $\mathcal{G}_\theta$, the volume preservation is maintained because the update to $\mathbf{x}$ depends only on the current (half-stepped) $\mathbf{v}$. This is a **triangular coupling**: the change in position is a function of velocity, and the change in velocity is a function of position. As long as the partitions are updated sequentially, the Jacobian remains unit-valued.

## 3. Implementation and Discretization

### 3.1 Separable Approximation of Christoffel Forces
To ensure exact coupling, the acceleration $\mathbf{a}(\mathbf{x})$ should ideally be independent of $\mathbf{v}$. However, Riemannian Christoffel symbols $\Gamma(v, v)$ are inherently quadratic in velocity. We resolve this by evaluating the geometric force at a base velocity state (e.g., $\mathbf{v}=0$) during the "Kick" phase, treating the velocity-dependent terms as a separate coupling or an auxiliary force.

### 3.2 Toroidal Topology and Periodic Boundaries
The coupling flow is designed to respect the topological constraints of the manifold. When operating on a torus $\mathbb{T}^n$, the position update is followed by a modular wrapping operator:
$$ \mathbf{x}_{t+1} = (\mathbf{x}_t + \Delta t \cdot \text{Drift}(\mathbf{v}_{t+1/2})) \pmod L $$
Because the wrapping is a local isometry (except at the boundary which is a null set in measure), it preserves the volume-preserving property of the coupling flow.

## 4. Comparative Advantages

| Feature | Standard Symplectic | Normalizing Flows | Symplectic Coupling Flow |
| : | : | : | : |
| **Physics** | Fixed Hamiltonian | None (General) | **Learnable Hamiltonian** |
| **Volume** | Preserved ($O(\Delta t^n)$) | Exact ($\det J = 1$) | **Exact ($\det J = 1$)** |
| **Invertibility** | Semi-analytical | Analytical | **Analytical** |
| **Kinematics** | Linear ($\dot{x}=v$) | N/A | **Neural ($\dot{x}=v+G(v)$)** |

## 5. Empirical Observations: Semantic Inertia

By training the Drift Network, we observe the emergence of **Semantic Inertia**: the model learns to "heavy" certain regions of the latent space where high-precision reasoning is required, effectively slowing down the flow ($\Delta x \approx 0$) to allow for more integration steps per unit of coordinate shift. Conversely, in "shallow" regions, the flow is accelerated, mimicking a form of adaptive time-stepping within a fixed-step integrator framework.

## 6. Conclusion

Symplectic Coupling Flows provide a mathematically rigorous way to inject learnable neural components into the heart of a geometric integrator. By framing the update as a series of shear transformations, we gain the flexibility of deep neural networks while retaining the exact conservation laws required for stable, long-horizon sequence modeling in Geodesic Flow Networks.



**References**

[1] Dinh, L., Krueger, D., & Bengio, Y. (2014). *NICE: Non-linear Independent Components Estimation*. arXiv:1410.8516.  
[2] Dinh, L., Sohl-Dickstein, J., & Samy, B. (2017). *Density estimation using Real NVP*. ICLR.  
[3] Hairer, E., et al. (2006). *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*. Springer.  
[4] Rezende, D. J., & Mohamed, S. (2015). *Variational Inference with Normalizing Flows*. ICML.  
[5] Clemente, A. V., et al. (2021). *Symplectic Hamiltonian Neural Networks*. arXiv.  
[6] Marsden, J. E., & West, M. (2001). *Discrete Mechanics and Variational Integrators*. Acta Numerica.  
