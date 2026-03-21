# The Runge-Kutta Paradox: Reasoning in Piecewise Riemannian Varieties

**Author:** Joaquin Stürtz  
**Date:** January 26, 2026

**Abstract**  
In computational physics and numerical analysis, high-order integrators such as the 4th-order Runge-Kutta (RK4) method are regarded as the gold standard for accuracy and convergence. However, we identify a counter-intuitive phenomenon—termed the **Runge-Kutta Paradox**—where these sophisticated methods exhibit catastrophic divergence when applied to neural sequence modeling within Geodesic Flow Networks (GFN). We demonstrate that the latent "thought space" of a GFN is not a smooth Riemannian manifold, but a **Piecewise Riemannian Variety** characterized by logical singularities and sharp metric transitions. In these non-smooth environments, high-order polynomial extrapolation leads to "Singularity Aliasing," while lower-order, structurally "local" integrators (e.g., Heun, Leapfrog) provide superior stability and logical reliability. This paper explores the fundamental link between numerical stability, geometric discontinuity, and the emergence of symbolic reasoning in continuous neural flows.



## 1. Introduction: The Smoothness Assumption

The prevailing paradigm in Neural Ordinary Differential Equations (Neural ODEs) and continuous-depth networks assumes that the underlying vector field is sufficiently smooth (typically $C^1$ or $C^2$) to justify the use of adaptive, high-order numerical solvers. This assumption implies that the model's intelligence resides in the smooth interpolation of statistical correlations.

We argue that true symbolic reasoning requires the opposite: the ability to represent sharp transitions, binary logic, and "points of no return." When a neural network learns to represent such discrete logic, the geometry of its latent space naturally evolves into a **Piecewise Riemannian Variety**. In this regime, the classic tools of numerical integration fail in a predictable yet paradoxical manner. The Christoffel symbols $\Gamma^k_{ij}(x)$ that characterize the smooth parts of the manifold become discontinuous at the boundaries between pieces, creating challenges for numerical methods that assume smoothness.

## 2. Theory of Piecewise Riemannian Varieties

We define the semantic space of a reasoning agent as a collection of smooth manifold "pieces" or charts, $\mathcal{M} = \bigcup_i \mathcal{V}_i$, where the transition between any two pieces $\mathcal{V}_i$ and $\mathcal{V}_j$ involves a discontinuity in the metric tensor $g_{\mu\nu}(x)$. At the boundaries, the metric transitions abruptly, causing the Christoffel symbols to be undefined or infinite.

### 2.1 The Geometry of Logical Certainty
In a Geodesic Flow Network, the motion of a "thought" is governed by the geodesic equation expressed in covariant form:

$$ \frac{D v^k}{dt} = \dot{v}^k + \Gamma^k_{ij}(x) v^i v^j = 0 $$

where $v^k = \dot{x}^k$ are the velocity components and $\Gamma^k_{ij}(x)$ are the Christoffel symbols of the Levi-Civita connection derived from the metric tensor $g_{ij}(x)$. To implement discrete logic, we introduce a **Singularity Potential** $V(x)$ that modifies the local curvature. The effective metric is defined as:

$$ g_{eff,ij}(x, v) = g_{base,ij}(x) \cdot \Phi(v, x) $$

where $\Phi(v, x)$ is a modulation factor that accounts for two critical phenomena: **Reactive Curvature** and **Logical Singularities**. The effective Christoffel symbols are then computed from this modulated metric:

$$ \Gamma_{eff}^k_{ij}(x, v) = \frac{1}{2} g_{eff}^{kl}(x, v) \left( \partial_i g_{eff,jl} + \partial_j g_{eff,il} - \partial_l g_{eff,ij} \right) $$

This formulation ensures that the connection properties (torsion-free, metric-compatible) are preserved while enabling dynamic geometric adaptation.

### 2.2 Reactive Curvature and Plasticity
To prevent the divergence of thoughts during high-uncertainty states, we implement a plasticity mechanism where the metric deforms based on the local kinetic energy $E = \frac{1}{2} g_{ij}(x) v^i v^j$:

$$ \Phi_{\text{plasticity}}(v) = 1 + \alpha \cdot \tanh\left(\gamma \cdot \frac{1}{d} \sum_{i=1}^d v_i^2\right) $$

This ensures that as the "velocity" of reasoning increases, the manifold becomes "heavier," effectively braking the trajectory through the increased Christoffel symbols $\Gamma_{eff}^k_{ij}(x, v)$ and forcing the model to integrate more information before making a decision. The acceleration $a^k = -\Gamma_{eff}^k_{ij}(x, v) v^i v^j$ scales with the plasticity, providing a geometric mechanism for uncertainty management.

### 2.3 Logical Singularities (Semantic Black Holes)
A symbolic decision (e.g., a bit-flip or a classification) is modeled as a region of infinite curvature that "traps" the trajectory. We use a potential function $V(x)$ and a sigmoid-based trigger to modulate the metric:

$$ S(x) = \sigma\left(\kappa \cdot (V(x) - \tau)\right) $$
$$ \Phi_{\text{singularity}}(x) = 1 + S(x) \cdot (\beta - 1) $$

where $\tau$ is the logical threshold and $\beta$ is the singularity strength. As $V(x) \to \tau$, the manifold undergoes a phase transition from a smooth Euclidean-like space to a high-curvature "bottleneck" that enforces symbolic certainty. The effective Christoffel symbols become extremely large near the singularity, creating an impenetrable barrier that stabilizes the trajectory in the logical state.

## 3. The Runge-Kutta Paradox

### 3.1 High-Order Failure: Singularity Aliasing
The 4th-order Runge-Kutta (RK4) method evaluates the vector field at four points: the current state ($k_1$), two mid-points ($k_2, k_3$), and a predicted end-point ($k_4$). The final update is a weighted average:

$$ x_{n+1}^i = x_n^i + \frac{\Delta t}{6}\left(k_1^i + 2k_2^i + 2k_3^i + k_4^i\right) $$

where each $k_p$ represents the velocity at a different evaluation point. This polynomial extrapolation assumes the field is locally linear or quadratic. However, if any of the intermediate stages ($k_2, k_3, k_4$) lands within the high-curvature region of a logical singularity, the resulting gradient is extreme. The Christoffel symbols $\Gamma^k_{ij}(x)$ are discontinuous at the singularity, violating the smoothness assumptions of RK4. The integrator, attempting to maintain 4th-order precision, over-fits this local spike and projects the state to infinity. We term this **Singularity Aliasing**.

### 3.2 The "Local Realism" of Low-Order Schemes
Lower-order methods exhibit what we call **Local Realism**. Consider the Heun method (RK2), which only evaluates the geometric acceleration at the boundaries of the step:
1. **Predictor:** $\tilde{x}_{n+1}^i = x_n^i + \Delta t \cdot v_n^i$
2. **Corrector:** $x_{n+1}^i = x_n^i + \frac{\Delta t}{2}\left(v_n^i + \tilde{v}_{n+1}^i\right)$

where the velocity update is given by $v_{n+1}^k = v_n^k + \Delta t \cdot a^k(x_n, v_n)$. By evaluating the gradient only at the boundaries of the step, Heun is "agnostic" to the chaotic higher-order derivatives inside the step interval. It treats the singularity as a local impulse rather than a smooth landscape to be modeled, making it robust to the discontinuities in $\Gamma^k_{ij}(x)$.

## 4. Symplectic Stability in Non-Smooth Varieties

For long-horizon reasoning, we utilize the **Leapfrog Integrator**, a second-order symplectic scheme that alternates between position and velocity updates while properly accounting for the geometric acceleration:

$$ v_{n+\frac{1}{2}}^k = v_n^k + \frac{\Delta t}{2} a^k(x_n, v_n) $$
$$ x_{n+1}^i = x_n^i + \Delta t \cdot v_{n+\frac{1}{2}}^i $$
$$ v_{n+1}^k = v_{n+\frac{1}{2}}^k + \frac{\Delta t}{2} a^k(x_{n+1}, v_{n+\frac{1}{2}}) $$

where the acceleration is $a^k(x, v) = -\Gamma^k_{ij}(x) v^i v^j$. The Leapfrog method is particularly robust in Piecewise Riemannian Varieties because it preserves the phase-space volume even when the metric is non-smooth. The symplectic structure ensures that $\det\left(\frac{\partial(x_{n+1}, v_{n+1})}{\partial(x_n, v_n)}\right) = 1$, preventing the energy drift that would otherwise cause trajectories to escape from logical states. It allows the model to "tunnel" through sharp transitions without the energy divergence typical of non-symplectic methods, as the geometric coupling encoded in $\Gamma^k_{ij}(x)$ is respected by the symmetric update pattern.

## 5. Discussion: The Efficiency of Roughness

The Runge-Kutta Paradox suggests that for machine reasoning, **less is more**. High-order integration is not merely computationally expensive; it is fundamentally incompatible with the discrete nature of logic, as it requires the smoothness of Christoffel symbols $\Gamma^k_{ij}(x)$ that logical singularities do not provide.

A perfectly smooth manifold represents a world of "maybes" and "probabilities." A piecewise variety, with its "rough" edges and singularities where $\Gamma^k_{ij}(x)$ becomes unbounded, represents a world of "ifs" and "thens." By embracing low-order integration schemes like Leapfrog that respect the geometric structure without requiring smoothness, we enable neural networks to maintain stable symbolic states within a continuous, differentiable flow. The covariant derivative $\frac{D v^k}{dt}$ provides the mathematical framework for understanding how discrete reasoning emerges from continuous dynamics on non-smooth manifolds.

## 6. Conclusion

We have identified the Runge-Kutta Paradox as a fundamental limitation of high-order numerical methods in the context of neural reasoning. By formalizing the concept of **Piecewise Riemannian Varieties**, we provide a geometric framework for understanding how continuous models can perform discrete logic. The transition from smooth interpolation to piecewise reasoning is not a matter of model scale, but of geometric topology and the numerical "honesty" of the integrator. The Christoffel symbols $\Gamma^k_{ij}(x)$ that characterize the smooth regions of the manifold become the key to understanding why certain integrators succeed while others fail in the presence of logical singularities.



## References

[1] Gromov, M. (1999). *Metric Structures for Riemannian and Non-Riemannian Spaces*. Birkhäuser.  
[2] Clarke, F. H. (1990). *Optimization and Nonsmooth Analysis*. SIAM.  
[3] Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical Integration*. Springer.  
[4] Butcher, J. C. (2016). *Numerical Methods for Ordinary Differential Equations*. Wiley.  
[5] Arnold, V. I. (1992). *Catastrophe Theory*. Springer-Verlag.  
[6] Chen, R. T. Q., et al. (2018). *Neural Ordinary Differential Equations*. NeurIPS.  
[7] Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*. CRC Press.
