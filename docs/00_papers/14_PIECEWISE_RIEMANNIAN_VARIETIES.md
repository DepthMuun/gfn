# The Runge-Kutta Paradox: Reasoning in Piecewise Riemannian Varieties

**Author:** Joaquin Stürtz  
**Date:** January 26, 2026

**Abstract**  
In computational physics and numerical analysis, high-order integrators such as the 4th-order Runge-Kutta (RK4) method are regarded as the gold standard for accuracy and convergence. However, we identify a counter-intuitive phenomenon—termed the **Runge-Kutta Paradox**—where these sophisticated methods exhibit catastrophic divergence when applied to neural sequence modeling within Geodesic Flow Networks (GFN). We demonstrate that the latent "thought space" of a GFN is not a smooth Riemannian manifold, but a **Piecewise Riemannian Variety** characterized by logical singularities and sharp metric transitions. In these non-smooth environments, high-order polynomial extrapolation leads to "Singularity Aliasing," while lower-order, structurally "local" integrators (e.g., Heun, Leapfrog) provide superior stability and logical reliability. This paper explores the fundamental link between numerical stability, geometric discontinuity, and the emergence of symbolic reasoning in continuous neural flows.



## 1. Introduction: The Smoothness Assumption

The prevailing paradigm in Neural Ordinary Differential Equations (Neural ODEs) and continuous-depth networks assumes that the underlying vector field is sufficiently smooth (typically $C^1$ or $C^2$) to justify the use of adaptive, high-order numerical solvers. This assumption implies that the model's intelligence resides in the smooth interpolation of statistical correlations. 

We argue that true symbolic reasoning requires the opposite: the ability to represent sharp transitions, binary logic, and "points of no return." When a neural network learns to represent such discrete logic, the geometry of its latent space naturally evolves into a **Piecewise Riemannian Variety**. In this regime, the classic tools of numerical integration fail in a predictable yet paradoxical manner.

## 2. Theory of Piecewise Riemannian Varieties

We define the semantic space of a reasoning agent as a collection of smooth manifold "pieces" or charts, $\mathcal{M} = \bigcup_i \mathcal{V}_i$, where the transition between any two pieces $\mathcal{V}_i$ and $\mathcal{V}_j$ involves a discontinuity in the metric tensor $g_{\mu\nu}$.

### 2.1 The Geometry of Logical Certainty
In a Geodesic Flow Network, the motion of a "thought" is governed by the geodesic equation:
$$\frac{d^2 x^k}{dt^2} + \Gamma^k_{ij} \frac{dx^i}{dt} \frac{dx^j}{dt} = 0$$
where $\Gamma^k_{ij}$ are the Christoffel symbols. To implement discrete logic, we introduce a **Singularity Potential** $V(x)$ that modifies the local curvature. The effective Christoffel symbols are defined as:
$$\Gamma_{\text{eff}} = \Gamma_{\text{base}} \cdot \Phi(v, x)$$
where $\Phi(v, x)$ is a modulation factor that accounts for two critical phenomena: **Reactive Curvature** and **Logical Singularities**.

### 2.2 Reactive Curvature and Plasticity
To prevent the divergence of thoughts during high-uncertainty states, we implement a plasticity mechanism where the metric deforms based on the local kinetic energy $E \approx |v|^2$:
$$\Phi_{\text{plasticity}}(v) = 1 + \alpha \cdot \tanh(\gamma |v|^2)$$
This ensures that as the "velocity" of reasoning increases, the manifold becomes "heavier," effectively braking the trajectory and forcing the model to integrate more information.

### 2.3 Logical Singularities (Semantic Black Holes)
A symbolic decision (e.g., a bit-flip or a classification) is modeled as a region of infinite curvature that "traps" the trajectory. We use a potential function $V(x)$ and a sigmoid-based trigger:
$$S(x) = \sigma\left(\kappa \cdot (V(x) - \tau)\right)$$
$$\Phi_{\text{singularity}}(x) = 1 + S(x) \cdot (\beta - 1)$$
where $\tau$ is the logical threshold and $\beta$ is the singularity strength. As $V(x) \to \tau$, the manifold undergoes a phase transition from a smooth Euclidean-like space to a high-curvature "bottleneck" that enforces symbolic certainty.

## 3. The Runge-Kutta Paradox

### 3.1 High-Order Failure: Singularity Aliasing
The 4th-order Runge-Kutta (RK4) method evaluates the vector field at four points: the current state ($k_1$), two mid-points ($k_2, k_3$), and a predicted end-point ($k_4$). The final update is a weighted average:
$$x_{n+1} = x_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$
This polynomial extrapolation assumes the field is locally linear or quadratic. However, if any of the intermediate stages ($k_2, k_3, k_4$) lands within the high-curvature region of a logical singularity, the resulting gradient is extreme. The integrator, attempting to maintain 4th-order precision, over-fits this local spike and projects the state to infinity. We term this **Singularity Aliasing**.

### 3.2 The "Local Realism" of Low-Order Schemes
Lower-order methods exhibit what we call **Local Realism**. Consider the Heun method (RK2):
1. **Predictor:** $\tilde{x}_{n+1} = x_n + \Delta t \cdot f(x_n)$
2. **Corrector:** $x_{n+1} = x_n + \frac{\Delta t}{2}[f(x_n) + f(\tilde{x}_{n+1})]$

By evaluating the gradient only at the boundaries of the step, Heun is "agnostic" to the chaotic higher-order derivatives inside the step interval. It treats the singularity as a local impulse rather than a smooth landscape to be modeled.

## 4. Symplectic Stability in Non-Smooth Varieties

For long-horizon reasoning, we utilize the **Leapfrog Integrator**, a second-order symplectic scheme that alternates between position and velocity updates:
$$v_{n+1/2} = v_n + \frac{\Delta t}{2} a(x_n)$$
$$x_{n+1} = x_n + \Delta t v_{n+1/2}$$
$$v_{n+1} = v_{n+1/2} + \frac{\Delta t}{2} a(x_{n+1})$$
The Leapfrog method is particularly robust in Piecewise Riemannian Varieties because it preserves the phase-space volume even when the metric is non-smooth. It allows the model to "tunnel" through sharp transitions without the energy divergence typical of non-symplectic methods.

## 5. Discussion: The Efficiency of Roughness

The Runge-Kutta Paradox suggests that for machine reasoning, **less is more**. High-order integration is not merely computationally expensive; it is fundamentally incompatible with the discrete nature of logic. 

A perfectly smooth manifold represents a world of "maybes" and "probabilities." A piecewise variety, with its "rough" edges and singularities, represents a world of "ifs" and "thens." By embracing low-order integration, we enable neural networks to maintain stable symbolic states within a continuous, differentiable flow.

## 6. Conclusion

We have identified the Runge-Kutta Paradox as a fundamental limitation of high-order numerical methods in the context of neural reasoning. By formalizing the concept of **Piecewise Riemannian Varieties**, we provide a geometric framework for understanding how continuous models can perform discrete logic. The transition from smooth interpolation to piecewise reasoning is not a matter of model scale, but of geometric topology and the numerical "honesty" of the integrator.



## References

[1] Gromov, M. (1999). *Metric Structures for Riemannian and Non-Riemannian Spaces*. Birkhäuser.  
[2] Clarke, F. H. (1990). *Optimization and Nonsmooth Analysis*. SIAM.  
[3] Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical Integration*. Springer.  
[4] Butcher, J. C. (2016). *Numerical Methods for Ordinary Differential Equations*. Wiley.  
[5] Arnold, V. I. (1992). *Catastrophe Theory*. Springer-Verlag.  
[6] Chen, R. T. Q., et al. (2018). *Neural Ordinary Differential Equations*. NeurIPS.  
[7] Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*. CRC Press.  
