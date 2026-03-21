# Symplectic Coupling Flows: Hybridizing Hamiltonian Dynamics and Normalizing Flows

**Author:** Joaquin Stürtz  
**Date:** January 26, 2026

**Abstract**  
Standard symplectic integrators, while stable, are typically restricted to separable Hamiltonian systems where the kinetic energy is a simple quadratic form of momentum. We introduce **Symplectic Coupling Flows**, a hybrid architecture that reformulates the numerical integration of geodesic flows as a sequence of volume-preserving coupling transformations. By borrowing the triangular structure of Normalizing Flows (e.g., NICE/RealNVP), we decouple the evolution of position and velocity into a series of shear mappings. This formulation guarantees a strictly unit Jacobian determinant ($\det J = 1$) regardless of the complexity of the learned force fields or the non-linear "drift" functions. We demonstrate that this approach enables the learning of **Intrinsic Kinematics**, where the relationship between velocity and coordinate updates is no longer fixed by Newtonian physics but is instead an optimized neural mapping, providing a flexible yet conservative foundation for Geodesic Flow Networks (GFN).



## 1. The Coupling Transformation in Phase Space

### 1.1 Non-Linear Shear Invariance
A coupling flow maps an input $(x, v)$ to an output $(x', v')$ via a sequence of triangular transformations. For a state divided into two partitions $(x_1, x_2)$ and $(v_1, v_2)$, a shear transformation takes the form:

$$ y_1^i = x_1^i $$
$$ y_2^k = x_2^k + f^k(x_1) $$

The Jacobian of this mapping is block-lower-triangular with ones on the diagonal blocks, ensuring that the volume in phase space is preserved exactly ($\det J = 1$). The determinant of the Jacobian matrix $J$ is the product of the diagonal elements, which are all unity. In the context of GFN, we apply this principle to the joint evolution of position $x^i$ and velocity $v^k$ on the phase space $T\mathcal{M}$.

### 1.2 Symplectic Splitting as Coupling
We decompose a single integration step $\Delta t$ into a symmetric splitting of "Kick" (velocity update) and "Drift" (position update) operators. Unlike traditional integrators that assume a fixed identity for the drift, our formulation allows for a learnable **Neural Drift**:

1.  **Half-Kick (Velocity):** 
    $$ v_{t+1/2}^k = v_t^k + \frac{\Delta t}{2} \cdot a^k(x_t, v_t) $$
    where the acceleration $a^k(x, v) = F^k - \Gamma^k_{ij}(x) v^i v^j$ incorporates both the external force $F^k$ and the geometric acceleration from Christoffel symbols.

2.  **Neural Drift (Position):**
    $$ x_{t+1}^i = x_t^i + \Delta t \cdot \left( v_{t+1/2}^i + \mathcal{G}^i_\theta(v_{t+1/2}) \right) $$
    where $\mathcal{G}^i_\theta(v)$ is a learned MLP that warps the kinematic relationship, providing intrinsic learnable kinematics beyond the Newtonian $\dot{x}^i = v^i$.

3.  **Full-Kick (Velocity):**
    $$ v_{t+1}^k = v_{t+1/2}^k + \frac{\Delta t}{2} \cdot a^k(x_{t+1}, v_{t+1/2}) $$
    The acceleration at the new position $x_{t+1}$ uses the half-stepped velocity $v_{t+1/2}$ to maintain the symplectic structure.

where $a^k(x, v)$ represents the acceleration (including learned Christoffel forces $\Gamma^k_{ij}(x) v^i v^j$) and $\mathcal{G}^i_\theta(v)$ is a learned MLP that warps the kinematic relationship, enabling the model to learn non-Euclidean inertial properties.

## 2. Learnable Kinematics and the Neural Drift

### 2.1 Beyond Newtonian Drift
In standard Euclidean physics, the drift is simply $\dot{x}^i = v^i$ or equivalently $\Delta x^i = v^i \Delta t$. However, on complex semantic manifolds, the "effective mass" or "inertial resistance" may vary depending on the direction of thought encoded in the velocity vector $v^i$. By introducing the **Drift Network** $\mathcal{G}^i_\theta(v)$, we allow the model to learn a non-linear velocity-to-position mapping:

$$ \dot{x}^i = v^i + \mathcal{G}^i_\theta(v) $$

This neural drift modifies the effective kinematics while preserving the symplectic structure through the triangular coupling. The Christoffel symbols $\Gamma^k_{ij}(x)$ that characterize the base manifold geometry remain separate from this learned kinematic modification.

### 2.2 Preserving the Symplectic Structure
Even with a complex neural network $\mathcal{G}^i_\theta$, the volume preservation is maintained because the update to $x^i$ depends only on the current (half-stepped) $v^k$. This is a **triangular coupling**: the change in position $\Delta x^i = \Delta t \cdot (v^i + \mathcal{G}^i_\theta(v))$ is a function of velocity, and the change in velocity $\Delta v^k = \frac{\Delta t}{2} a^k(x, v)$ is a function of position. As long as the partitions are updated sequentially, the Jacobian remains unit-valued:

$$ \det\left(\frac{\partial(x_{t+1}, v_{t+1})}{\partial(x_t, v_t)}\right) = 1 $$

This ensures that the phase space volume is preserved exactly, maintaining the geometric integrity of the flow over arbitrarily long sequences.

## 3. Implementation and Discretization

### 3.1 Separable Approximation of Christoffel Forces
To ensure exact coupling in the triangular structure, the acceleration $a^k(x, v)$ must be evaluated in a way that respects the partition structure. The Christoffel symbols $\Gamma^k_{ij}(x) v^i v^j$ are inherently quadratic in velocity. We resolve this by decomposing the velocity into the two partitions and treating the cross-terms appropriately:

$$ a^k(x, v) = F^k - \Gamma^k_{ij}(x) v^i v^j $$

where the quadratic form $\Gamma^k_{ij}(x) v^i v^j$ is computed explicitly from the learned metric tensor. During implementation, we ensure that the velocity dependence is properly evaluated at each stage of the integration, avoiding the approximation of setting $v=0$ which would discard the geometric acceleration entirely.

### 3.2 Toroidal Topology and Periodic Boundaries
The coupling flow is designed to respect the topological constraints of the manifold. When operating on a torus $\mathbb{T}^n$ with coordinates $x^i \in [0, L)$, the position update is followed by a modular wrapping operator:

$$ x_{t+1}^i = \left( x_t^i + \Delta t \cdot \left( v_{t+1/2}^i + \mathcal{G}^i_\theta(v_{t+1/2}) \right) \right) \mod L $$

Because the wrapping is a local isometry (except at the boundary which is a null set in measure), it preserves the volume-preserving property of the coupling flow. The Christoffel symbols on a flat torus are identically zero, $\Gamma^k_{ij}(x) = 0$, simplifying the dynamics to pure Newtonian motion with neural drift on the compact topology.

### 3.3 Jacobian Computation for Neural Drift
The neural drift $\mathcal{G}^i_\theta(v)$ introduces a modification to the Jacobian that must be tracked for proper inverse computation. The full Jacobian of the coupling transformation is:

$$ J = \begin{pmatrix} I & 0 \\ \frac{\partial \text{Kick}}{\partial x} & I \end{pmatrix} \begin{pmatrix} I & \frac{\partial \text{Drift}}{\partial v} \\ 0 & I \end{pmatrix} $$

where $\frac{\partial \text{Drift}}{\partial v}$ includes both the identity (from $v^i$) and the Jacobian of the neural network $\frac{\partial \mathcal{G}^i_\theta}{\partial v^j}$. The determinant remains unity because $J$ is a product of shear transformations, each with unit determinant.

## 4. Comparative Advantages

| Feature | Standard Symplectic | Normalizing Flows | Symplectic Coupling Flow |
| : | : | : | : |
| **Physics** | Fixed Hamiltonian | None (General) | **Learnable Hamiltonian** |
| **Volume** | Preserved ($O(\Delta t^n)$) | Exact ($\det J = 1$) | **Exact ($\det J = 1$)** |
| **Invertibility** | Semi-analytical | Analytical | **Analytical** |
| **Kinematics** | Linear ($\dot{x}^i = v^i$) | N/A | **Neural ($\dot{x}^i = v^i + \mathcal{G}^i_\theta(v)$)** |
| **Christoffel Terms** | $\Gamma^k_{ij}(x) v^i v^j$ | N/A | **Explicit** |

The Symplectic Coupling Flow uniquely combines the exact volume preservation of normalizing flows with the geometric structure of Hamiltonian systems. The Christoffel symbols $\Gamma^k_{ij}(x)$ are explicitly incorporated into the acceleration computation, maintaining the geometric character of geodesic flows while enabling learnable kinematics through the neural drift $\mathcal{G}^i_\theta(v)$.

## 5. Empirical Observations: Semantic Inertia

By training the Drift Network $\mathcal{G}^i_\theta(v)$, we observe the emergence of **Semantic Inertia**: the model learns to assign "heavy" effective mass to certain regions of the latent space where high-precision reasoning is required. This manifests as $\mathcal{G}^i_\theta(v) \approx -v^i$ in these regions, effectively slowing down the flow ($\Delta x^i \approx 0$) to allow for more integration steps per unit of coordinate shift. Conversely, in "shallow" regions, the drift network learns $\mathcal{G}^i_\theta(v) \approx 0$, accelerating the flow and mimicking a form of adaptive time-stepping within a fixed-step integrator framework. The Christoffel symbols $\Gamma^k_{ij}(x)$ complement this by encoding the geometric complexity of the semantic manifold independently of the learned kinematics.

## 6. Conclusion

Symplectic Coupling Flows provide a mathematically rigorous way to inject learnable neural components into the heart of a geometric integrator. By framing the update as a series of shear transformations with triangular Jacobians, we gain the flexibility of deep neural networks while retaining the exact conservation laws required for stable, long-horizon sequence modeling. The explicit treatment of Christoffel symbols $\Gamma^k_{ij}(x)$ ensures that the base geometric structure is preserved, while the neural drift $\mathcal{G}^i_\theta(v)$ enables the learning of intrinsic kinematic properties that go beyond Newtonian physics. This hybrid architecture represents a significant advance in the integration of differential geometry with modern deep learning techniques for sequence modeling.



**References**

[1] Dinh, L., Krueger, D., & Bengio, Y. (2014). *NICE: Non-linear Independent Components Estimation*. arXiv:1410.8516.  
[2] Dinh, L., Sohl-Dickstein, J., & Samy, B. (2017). *Density estimation using Real NVP*. ICLR.  
[3] Hairer, E., et al. (2006). *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*. Springer.  
[4] Rezende, D. J., & Mohamed, S. (2015). *Variational Inference with Normalizing Flows*. ICML.  
[5] Clemente, A. V., et al. (2021). *Symplectic Hamiltonian Neural Networks*. arXiv.  
[6] Marsden, J. E., & West, M. (2001). *Discrete Mechanics and Variational Integrators*. Acta Numerica.
