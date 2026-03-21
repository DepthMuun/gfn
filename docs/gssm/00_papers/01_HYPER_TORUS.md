# The Hyper‑Torus: Topology‑Aligned Geometry for Constant‑Memory Sequence Modeling

**Author:** Joaquin Stürtz  
**Date:** January 24, 2026

**Abstract**  
Cyclic reasoning tasks (parity, modular arithmetic, periodic phase tracking) are poorly expressed in Euclidean latent spaces. We present the **Hyper‑Torus**, a topology‑aligned manifold design that embeds the latent state into a product of circles $T^n = (S^1)^n$ with a physically meaningful Riemannian metric. Token processing becomes forced geodesic motion with symplectic discretization and periodic boundary conditions. The resulting architecture preserves information as phase and momentum, achieving constant‑memory inference and long‑horizon stability. We formalize the geometry, derive the geodesic equations (including Christoffel interactions and friction gates), describe the discretization via energy‑preserving integrators, and detail training losses for periodic targets. Empirical evaluations on algorithmic tasks demonstrate robust extrapolation and reduced drift relative to non‑geometric baselines.



## 1. Motivation and Overview

Modern sequence models often store an explicit history (e.g., KV caches), causing inference memory to grow with sequence length. Many algorithmic tasks, however, are naturally expressed as phase evolution (e.g., XOR parity as a half‑rotation; counters as winding numbers). In such settings:
- Euclidean embeddings require continuous "effort" to remain on cyclic trajectories.
- Periodicity is brittle near boundaries, leading to drift and aliasing.

The Hyper‑Torus addresses these issues by aligning the manifold with the task: states are represented as phases on $S^1$, coupled into $T^n$, and evolved by physically structured updates. Symplectic discretization preserves qualitative dynamics; periodic boundary conditions remove edge discontinuities; friction gates implement controlled forgetting for regime switching.



## 2. Geometric Preliminaries

### 2.1 The Torus
We define the latent configuration space as a product of circles:
$$
\mathcal{M} \cong T^n = \underbrace{S^1 \times \cdots \times S^1}_{n\ \text{times}} ,
$$
with local coordinates $\mathbf{x} = (\theta_1, \phi_1, \theta_2, \phi_2, \dots)$ in paired blocks. Each pair $(\theta, \phi)$ encodes a two‑phase subsystem (inner/outer angles). The embedding into $\mathbb{R}^{2n}$ is given by:
$$
\mathbf{x} \mapsto \big( (R + r\cos\theta_1)\cos\phi_1,\ (R + r\cos\theta_1)\sin\phi_1,\ \dots \big)
$$

### 2.2 Riemannian Metric
Following a standard torus of revolution, we adopt a diagonal metric tensor $g_{ij}$ in angular coordinates:
$$
g(\theta,\phi) = \begin{pmatrix}
g_{\theta\theta} & g_{\theta\phi} \\
g_{\phi\theta} & g_{\phi\phi}
\end{pmatrix} = \begin{pmatrix}
r^2 & 0 \\
0 & (R + r\cos\theta)^2
\end{pmatrix}.
$$
Tiling this block across all $n$ coordinate pairs yields a block‑diagonal metric $g(\mathbf{x})$ for $T^n$. Here $R$ is the major radius (global scale) and $r$ the minor radius (local scale). This geometry produces curvature terms that act as stabilizing "geometric forces."

The determinant of the metric is:
$$
\det(g) = r^2 (R + r\cos\theta)^2,
$$
and the inverse metric is:
$$
g^{\theta\theta} = \frac{1}{r^2}, \quad g^{\phi\phi} = \frac{1}{(R + r\cos\theta)^2}, \quad g^{\theta\phi} = g^{\phi\theta} = 0.
$$

### 2.3 Geodesic Equations with Forcing and Dissipation
Let $\mathbf{x} \in \mathcal{M}$ and velocity $\mathbf{v} = \dot{\mathbf{x}} \in T_{\mathbf{x}}\mathcal{M}$. The forced geodesic dynamics in covariant form are:
$$
\frac{Dv^k}{dt} = \frac{dv^k}{dt} + \Gamma^k_{ij}(\mathbf{x}) \, v^i v^j = F^k(\mathbf{x},u) - \mu(\mathbf{x},u) \, v^k,
$$
which expands to the explicit form:
$$
\ddot{x}^k + \Gamma^k_{ij}(\mathbf{x}) \, v^i v^j = F^k(\mathbf{x},u) - \mu(\mathbf{x},u) \, v^k.
$$
Here:
- $\frac{Dv^k}{dt} = \frac{dv^k}{dt} + \Gamma^k_{ij} v^i v^j$ is the covariant derivative (acceleration along the manifold)
- $\frac{dv^k}{dt} = \ddot{x}^k$ is the ordinary second derivative of position with respect to time
- $\Gamma^k_{ij}(\mathbf{x})$ are Christoffel symbols induced by the metric $g$
- $F^k(\mathbf{x},u)$ is a token‑driven force from the input embedding
- $\mu(\mathbf{x},u) \ge 0$ is a dissipation coefficient ("clutch") that gates between conservative memory and rapid rewriting

For the diagonal torus metric, the non-zero Christoffel symbols for each $(\theta, \phi)$ block are:
$$
\Gamma^\theta_{\phi\phi} = \frac{(R + r\cos\theta) \, (-r\sin\theta)}{r^2} = +\frac{(R + r\cos\theta)\sin\theta}{r},
$$
$$
\Gamma^\phi_{\theta\phi} = \Gamma^\phi_{\phi\theta} = \frac{r\sin\theta}{R + r\cos\theta}.
$$
These produce centrifugal and Coriolis‑like effects that naturally regulate phase motion. All other Christoffel symbols vanish due to the block‑diagonal structure of the metric.



## 3. Periodic Boundary Conditions and Distances

### 3.1 Boundary Wrapping
Periodic coordinates are wrapped modulo $2\pi$ component‑wise:
$$
\theta \mapsto (\theta + 2\pi) \mod 2\pi, \quad \phi \mapsto (\phi + 2\pi) \mod 2\pi.
$$
In vector form, this is:
$$
\mathbf{x} \leftarrow \mathbf{x} \pmod{2\pi}.
$$
This removes discontinuities at chart boundaries and prevents unbounded drift.

### 3.2 Toroidal Distance
Prediction targets that live on $T^n$ use the shortest angular distance on each circle:
$$
d_{S^1}(x_1, x_2) = \min\!\big(|\Delta|,\ 2\pi - |\Delta|\big), \quad \Delta = x_1 - x_2 \pmod{2\pi}.
$$
The toroidal distance is the Euclidean distance in the embedded space, or equivalently the sum of circular distances across all dimensions:
$$
d_{\text{torus}}(\mathbf{x}_1, \mathbf{x}_2) = \sqrt{\sum_{k=1}^n d_{S^1}^2\big(x^k_1, x^k_2\big)}.
$$
Loss functions for periodic targets include:
- Circular MSE: $L_{\text{circular}} = \sum_{k=1}^n d_{S^1}^2\big(x^k_{\text{pred}}, x^k_{\text{target}}\big)$
- Phase loss for smooth gradients: $L_{\text{phase}} = 1 - \cos\big(\mathbf{x}_{\text{pred}} - \mathbf{x}_{\text{target}}\big)$



## 4. Token‑to‑Force Mapping and Multi‑Head Geometry

Tokens are embedded into per‑step forces $\mathbf{F}_\theta(\mathbf{u}_t)$ that act on $\mathbf{v}$ and, via the Christoffel connection, deform trajectories. The latent state is split into $H$ heads. Each head owns a curvature module and integrator step; heads are mixed back linearly:
$$
\mathbf{F} = \sum_{h=1}^H w_h \, \mathbf{F}^{(h)}, \quad \sum_{h=1}^H w_h = 1.
$$
This parallel decomposition yields:
- Independent sub‑geometries for specialized reasoning channels
- Learned per‑head time scales to adapt resolution to local complexity
- Emergent specialization through differentiable competition



## 5. Friction Gates and Reactive Curvature

### 5.1 Thermodynamic Gating ("The Clutch")
We parameterize dissipation using periodic features to respect the torus topology:
$$
\mu(\mathbf{x}, u) = \sigma\big(W_{\text{state}} \, [\sin\mathbf{x}, \cos\mathbf{x}] + W_{\text{input}} \, \mathbf{u}\big),
$$
where $[\sin\mathbf{x}, \cos\mathbf{x}]$ concatenates sine and cosine features to avoid coordinate singularities at the wrap point. Small $\mu$ keeps memory conservative; large $\mu$ rapidly damps momentum to overwrite state with new information.

### 5.2 Reactive Curvature
Curvature can be modulated by kinetic energy $K = \frac{1}{2} g_{ij}(\mathbf{x}) v^i v^j$ via an effective metric:
$$
g_{\text{eff},ij}(\mathbf{x}, v) = g_{ij}(\mathbf{x}) \cdot \big(1 + \alpha \, \tanh K\big),
$$
where $\alpha$ is a learnable scaling parameter. The Christoffel symbols derived from $g_{\text{eff}}$ encode increased geometric resistance in high‑energy regimes, improving stability through difficult transitions. In tensor form, the modulated Christoffel contraction is:
$$
\left[\Gamma_{\text{eff}}(\mathbf{v}, \mathbf{v})\right]^k = \Gamma^k_{ij}(\mathbf{x}) \, v^i v^j \cdot \big(1 + \alpha \, \tanh K\big).
$$
The acceleration then becomes:
$$
a^k = -\Gamma^k_{ij}(\mathbf{x}) v^i v^j \cdot \big(1 + \alpha \, \tanh K\big) - \mu(\mathbf{x}, \mathbf{u}) v^k.
$$

### 5.3 Singularity Potentials (Optional)
Logical bottlenecks may be modeled as localized potentials that strengthen curvature near thresholds. The effective potential adds a force term:
$$
F^k_{\text{sing}} = -g^{kj} \frac{\partial V_{\text{sing}}}{\partial x^j},
$$
where $V_{\text{sing}}$ is a learnable potential well centered at discrete logical states, imitating "event horizons" that stabilize discrete flips.



## 6. Discretization: Energy‑Preserving Integrators

We discretize the dynamics with structure‑preserving schemes:
- Symplectic Verlet and Leapfrog (kick‑drift‑kick)
- Higher‑order symplectic compositions (Yoshida, Forest‑Ruth, Omelyan) for smoother regimes
- Heun/RK2 as a robust non‑symplectic baseline

A typical Leapfrog step for $(\mathbf{x}, \mathbf{v})$ with torus boundary wrapping is:
$$
\mathbf{v}_{\tfrac{1}{2}} = \mathbf{v} + \tfrac{1}{2}\,\Delta t \, \mathbf{a}(\mathbf{x}, \mathbf{v}, \mathbf{u}),
$$
$$
\mathbf{x}' = \operatorname{wrap}\big(\mathbf{x} + \Delta t \, \mathbf{v}_{\tfrac{1}{2}}\big),
$$
$$
\mathbf{v}' = \mathbf{v}_{\tfrac{1}{2}} + \tfrac{1}{2}\,\Delta t \, \mathbf{a}(\mathbf{x}', \mathbf{v}_{\tfrac{1}{2}}, \mathbf{u}),
$$
where the acceleration is:
$$
a^k(\mathbf{x}, \mathbf{v}, \mathbf{u}) = F^k(\mathbf{x}, \mathbf{u}) - \Gamma^k_{ij}(\mathbf{x}) \, v^i v^j - \mu(\mathbf{x}, \mathbf{u}) \, v^k,
$$
with $\Gamma^k_{ij}(\mathbf{x}) \, v^i v^j$ denoting the Christoffel contraction. The wrap function applies $2\pi$ periodic boundary conditions to each coordinate. This preserves qualitative phase‑space structure and prevents long‑horizon drift.



## 7. Parallelization (Optional)

For training throughput, first‑order affine approximations of the recurrence:
$$
\mathbf{v}_t = A_t \, \mathbf{v}_{t-1} + \mathbf{b}_t, \qquad
\mathbf{x}_t = \mathbf{x}_{t-1} + \Delta t_t \, \mathbf{v}_t,
$$
can be evaluated via associative parallel scans on GPUs. The parallel scan computes all timesteps simultaneously in $\mathcal{O}(\log L)$ depth, reducing serial dependency while maintaining functional agreement with sequential updates for locally linear systems.



## 8. Training Objectives for Periodic Targets

We combine task losses with physics‑informed regularization:
- Cross‑entropy or toroidal/circular distance losses for outputs on $T^n$
- Hamiltonian loss $L_H$ to penalize spurious energy creation:
$$
L_H = \lambda_H \, \mathbb{E}\big[\,|E_{t+1} - E_t|\,\big], \quad
E_t = \frac{1}{2} \, g_{ij}(\mathbf{x}_t) \, v_t^i \, v_t^j
$$
- Geodesic curvature regularization to temper excessive curvature excursions
- Optional soft‑symmetry (Noether) terms to align isomeric heads

The Hamiltonian (total energy) includes kinetic energy plus any potential from input forces:
$$
H(\mathbf{x}_t, \mathbf{v}_t) = E_t + V(\mathbf{x}_t),
$$
where $V(\mathbf{x}_t)$ is derived from integrating the input forces.



## 9. Complexity and Memory

Hyper‑Torus maintains a fixed‑size state $(\mathbf{x}_t, \mathbf{v}_t)$ independent of sequence length $L$:
$$
\text{Memory} = O(1),\qquad
\text{Compute} \approx O(L \cdot d^2)\ \text{(dense Christoffel computation)} .
$$
Symplectic discretization reduces gradient pathologies; periodic wrapping eliminates boundary artifacts, supporting stable infinite‑horizon reasoning. The computational bottleneck is the evaluation of Christoffel symbols, which for a diagonal metric reduces to $\mathcal{O}(d)$ operations.



## 10. Empirical Behavior

On cyclic algorithmic tasks (e.g., cumulative parity), Hyper‑Torus exhibits:
- Stable phase tracking visualized as limit cycles on periodic coordinates
- Strong extrapolation far beyond training sequence lengths
- Reduced drift versus non‑geometric baselines through boundary‑aware updates and energy‑preserving integration

The winding number $\mathbf{w} = \big\lfloor \frac{\mathbf{x} + \pi}{2\pi} \big\rfloor$ provides a discrete topological invariant that encodes the cumulative phase evolution, enabling robust counting over arbitrarily long sequences.



## 11. Discussion and Limitations

While torus topology aligns well with cyclic reasoning, non‑smooth dynamics (sharp logical transitions) can challenge high‑order explicit schemes; lower‑order symplectic or robust RK2 steps often perform best. Learned friction and reactive curvature provide practical stabilization but introduce task‑dependent hyperparameters. The assumption of constant metric parameters during integration is an approximation that may need refinement for highly dynamic applications. Formal convergence proofs for learned curvature remain open research.



## 12. Conclusion

Hyper‑Torus reframes sequence modeling as phase‑space evolution on $T^n$. By matching topology to task and respecting geometry in discretization, the architecture preserves information as momentum and winding rather than as explicit memory buffers. This yields constant‑memory inference, robust long‑horizon stability, and a principled path to physically grounded neural reasoning where the Riemannian structure provides natural regularization and topological protection for cyclic computations.



**References**  

[1]  Riemann, B. (1854). Über die Hypothesen, welche der Geometrie zu Grunde liegen.  
[2]  Arnold, V. I. (1989). Mathematical Methods of Classical Mechanics.  
[3]  Hairer, E., Lubich, C., & Wanner, G. (2006). Geometric Numerical Integration.  
[4]  Chen, R. T. Q., et al. (2018). Neural Ordinary Differential Equations.  
[5]  Bronstein, M. M., et al. (2021). Geometric Deep Learning.  
[6]  MacKay, D. J. C. (2003). Information Theory, Inference, and Learning Algorithms.  
[7]  Yoshida, H. (1990). Construction of higher order symplectic integrators.  
[8]  Omelyan, I. P., et al. (2002). Symplectic algorithms for molecular dynamics equations.  
[9]  Dinh, L., et al. (2014). NICE: Non‑linear Independent Components Estimation.  
[10]  Rezende, D. J., & Mohamed, S. (2015). Variational Inference with Normalizing Flows.
