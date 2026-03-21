# Reactive Geometry: Energy‑Modulated Curvature for Stable Neural Geodesic Flow

**Author:** Joaquin Stürtz  
**Date:** January 24, 2026

**Abstract**  
We introduce **Reactive Geometry**, a principled extension of Riemannian modeling in which the latent manifold stiffens in response to the instantaneous kinetic energy of the state. The metric acts as a self‑regulating substrate: high energy (semantic uncertainty) increases curvature and friction to brake chaotic motion; low energy (semantic certainty) allows inertial geodesic reasoning. Concretely, reactive curvature appears as a bounded multiplicative scaling of Christoffel interactions computed from a low‑rank geometric operator, combined with state‑ and input‑dependent dissipation gates and optional singularity potentials that stabilize discrete logical transitions. We present the mathematical formulation, derive the effective acceleration, detail structure‑preserving discretizations, and outline training objectives aligned with periodic targets and energy conservation.



## 1. Introduction

Riemannian manifolds provide structure for latent computation, yet fixed geometries cannot capture the subjective "effort" of reasoning. In Reactive Geometry, the manifold responds to the state co‑vector's energy, producing a closed‑loop control: energy → curvature → braking → reduced energy. This establishes a physically grounded path to uncertainty handling and long‑horizon stability in neural geodesic flow.

The fundamental insight is that the metric tensor $g_{ij}(\mathbf{x}, \mathbf{v})$ can depend not only on position but also on the kinetic energy of the state, creating a Finsler-like structure while maintaining the computational efficiency of Riemannian methods. This reactivity provides intrinsic stability without explicit gradient clipping or architectural constraints.



## 2. Formalism

### 2.1 Low‑Rank Curvature Operator
Let $\mathbf{x} \in \mathbb{R}^d$ be coordinates and $\mathbf{v} = \dot{\mathbf{x}} \in T_{\mathbf{x}}\mathcal{M}$ the velocity. The Christoffel symbols of the Levi-Civita connection are defined from the metric tensor:
$$
\Gamma^k_{ij}(\mathbf{x}) = \frac{1}{2} g^{kl}(\mathbf{x}) \left( \frac{\partial g_{jl}}{\partial x^i} + \frac{\partial g_{il}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^l} \right).
$$
For computational efficiency, we parameterize the metric via a low‑rank factorization:
$$
g_{ij}(\mathbf{x}) = \delta_{ij} + \sum_{r=1}^{R} U_{ir}(\mathbf{x}) \, U_{jr}(\mathbf{x}),
$$
where $U_{ir}$ are learnable functions of position (typically linear projections with activation functions). The inverse metric is computed as:
$$
g^{ij}(\mathbf{x}) = (\delta + U U^T)^{-1}_{ij},
$$
which can be approximated using the Woodbury identity for numerical stability:
$$
g^{ij}(\mathbf{x}) \approx \delta^{ij} - U^{i}_r (\delta^{rs} + U^{s}_t U^{t}_r)^{-1} U^{j}_s.
$$

In practice, the Christoffel action on velocity is computed efficiently via:
$$
\left[\Gamma(\mathbf{v}, \mathbf{v})\right]^k = \Gamma^k_{ij}(\mathbf{x}) \, v^i \, v^j,
$$
where the projection $\mathbf{p} = U^T \mathbf{v}$ is used to scale curvature, ensuring numerical robustness through saturation:
$$
\text{saturation}(\mathbf{p}) = \frac{\mathbf{p}}{1 + \|\mathbf{p}\|}, \quad \text{or} \quad \tanh(\mathbf{p}).
$$

### 2.2 Reactive Curvature (Plasticity)
The effective connection multiplies base curvature by a bounded function of kinetic energy. Rather than tensor multiplication, we define an effective metric scaling:
$$
g_{\text{eff},ij}(\mathbf{x}, \mathbf{v}) = g_{ij}(\mathbf{x}) \cdot \big(1 + \alpha \, \tanh K\big),
$$
where:
- $\alpha \ge 0$ is a plasticity coefficient (learned)
- $K = \frac{1}{2} g_{ij}(\mathbf{x}) v^i v^j$ is the kinetic energy

The Christoffel symbols derived from $g_{\text{eff}}$ give the reactive curvature. For practical implementation, we apply the scaling directly to the Christoffel action:
$$
\left[\Gamma_{\text{eff}}(\mathbf{v}, \mathbf{v})\right]^k = \big(1 + \alpha \, \tanh K\big) \cdot \Gamma^k_{ij}(\mathbf{x}) \, v^i \, v^j.
$$
As $K$ increases, curvature stiffens and turns motion more strongly, acting as a self‑braking mechanism that reduces oscillations and runaway drift.

### 2.3 Dissipation Gates ("The Clutch")
Reactive Geometry uses a conformal symplectic dissipation term to regulate state rewriting:
$$
\mu(\mathbf{x}, u) = \sigma\big(W_{\text{state}} \, [\sin\mathbf{x}, \cos\mathbf{x}] + W_{\text{input}} \, \mathbf{u}\big) \cdot \mu_{\text{max}},
$$
where:
- $[\sin\mathbf{x}, \cos\mathbf{x}]$ concatenates periodic features for toroidal topologies
- $\mathbf{u}$ is the input embedding (token representation)
- $\sigma(\cdot)$ is the sigmoid activation ensuring $\mu \in [0, \mu_{\text{max}}]$
- $\mu_{\text{max}}$ is a fixed maximum dissipation coefficient

Small $\mu$ keeps memory conservative (inertial flow); large $\mu$ damps momentum to allow rapid overwriting of the latent state with new information.

### 2.4 Singularity Potentials (Optional)
Discrete logical "flips" can be stabilized by localized potentials that create regions of high curvature. The singularity potential is:
$$
S(\mathbf{x}) = \sigma\big(V(\mathbf{x})\big),
$$
where $V(\mathbf{x})$ is a learned position-gate. The effective metric modification is:
$$
g_{\text{sing},ij}(\mathbf{x}) = g_{\text{eff},ij}(\mathbf{x}) \cdot \big(1 + (S(\mathbf{x}) - S_0) \cdot (\beta - 1)\big),
$$
where:
- $S_0$ is a threshold (typically $0.5$)
- $\beta > 1$ is the singularity strength
- The effect is a soft, differentiable intensification of curvature near high‑certainty regions, forming "event horizons" that trap the state after decisive transitions

The resulting Christoffel symbols encode this enhanced curvature, stabilizing discrete logical transitions while maintaining full differentiability.

### 2.5 Effective Acceleration
The latent dynamics combine forcing, curvature, and dissipation in covariant form:
$$
\frac{Dv^k}{dt} = \frac{dv^k}{dt} + \Gamma^k_{ij}(\mathbf{x}) \, v^i v^j = F^k(\mathbf{x}, \mathbf{u}) - \mu(\mathbf{x}, \mathbf{u}) \, v^k,
$$
which expands to the explicit acceleration:
$$
a^k(\mathbf{x}, \mathbf{v}, \mathbf{u}) = F^k(\mathbf{x}, \mathbf{u}) - \Gamma^k_{ij}(\mathbf{x}) \, v^i \, v^j - \mu(\mathbf{x}, \mathbf{u}) \, v^k.
$$
In vector notation, this is:
$$
\mathbf{a}(\mathbf{x}, \mathbf{v}, \mathbf{u}) = \mathbf{F}(\mathbf{x}, \mathbf{u}) - \Gamma(\mathbf{x})[\mathbf{v}, \mathbf{v}] - \mu(\mathbf{x}, \mathbf{u}) \, \mathbf{v}.
$$
This form yields a conservative memory mode ($\mu \approx 0$) and a dissipative rewrite mode ($\mu \gg 0$), both modulated by reactive curvature from $K$.



## 3. Geometry–Topology Interplay

Reactive Geometry is compatible with non‑Euclidean topologies. In toroidal settings:
- Coordinates are periodic ($\mathbf{x} \bmod 2\pi$), preventing edge discontinuities
- Periodic features $[\sin\mathbf{x}, \cos\mathbf{x}]$ feed friction and potential gates
- The diagonal torus metric blocks (inner/outer angles) produce centrifugal/Coriolis‑like terms that naturally regulate phase motion

For a torus $T^n$ with metric $g(\mathbf{x})$, the Christoffel symbols vanish identically in angular coordinates, and the reactive curvature manifests through the modulation $\alpha \, \tanh K$.



## 4. Discretization and Stability

### 4.1 Structure‑Preserving Integrators
Reactive dynamics are discretized via energy‑preserving schemes:
- Leapfrog/Verlet (kick‑drift‑kick) for robust qualitative stability
- Higher‑order symplectic compositions (Yoshida, Forest‑Ruth, Omelyan) in smooth regimes
- Heun/RK2 as a non‑symplectic but stable baseline

A typical Leapfrog step with periodic wrapping is:
$$
\mathbf{v}_{\tfrac{1}{2}} = \mathbf{v} + \tfrac{1}{2} \, \Delta t \, \mathbf{a}(\mathbf{x}, \mathbf{v}, \mathbf{u}),
$$
$$
\mathbf{x}' = \operatorname{wrap}\!\big(\mathbf{x} + \Delta t \, \mathbf{v}_{\tfrac{1}{2}}\big),
$$
$$
\mathbf{v}' = \mathbf{v}_{\tfrac{1}{2}} + \tfrac{1}{2} \, \Delta t \, \mathbf{a}(\mathbf{x}', \mathbf{v}_{\tfrac{1}{2}}, \mathbf{u}).
$$

### 4.2 Saturation and Clamps
To avoid numerical explosion under non‑smooth forces, curvature outputs are softly saturated (e.g., via $\tanh$) and clamped within bounded ranges. Dissipation coefficients use sigmoid activation and a fixed scale to keep updates within safe limits. These measures, together with reactive modulation, reduce oscillations and help maintain energy budgets:
$$
\Gamma^k_{ij} \leftarrow \tanh\!\big(\Gamma^k_{ij} / \Gamma_{\text{max}}\big) \cdot \Gamma_{\text{max}}.
$$

### 4.3 Time‑Scale Adaptation
Reactive Geometry can be paired with learned per‑head time scales and optional dynamic time gating:
$$
\Delta t_{\text{eff}} = \Delta t \cdot \sigma(W_t \, [\mathbf{x}, \mathbf{v}] + b_t),
$$
reducing $\Delta t$ in complex regions (high curvature) and increasing it in near‑flat geometry, improving stability without extra memory.



## 5. Implementation Overview

- **Curvature operator:** Low‑rank symmetric parameterization of the metric with projection‑based scaling and smooth saturation.
- **Reactive curvature:** Multiplicative plasticity using a bounded function of kinetic energy, applied to the Christoffel action on velocity.
- **Friction gates:** State‑dependent via periodic features and input‑dependent via the force; both combined through sigmoid activation.
- **Singularity potentials:** Differentiable intensification of curvature near high‑certainty regions using a learned position gate and soft thresholding.
- **Topology:** Periodic wrapping for toroidal coordinates; periodic features in gates and potentials; diagonal torus metric blocks for stabilizing interactions.
- **Execution paths:** Reliable Python loops for correctness; optional fused GPU kernels; optional scan‑based parallelization for training throughput, preserving functional behavior with bounded numerical differences.



## 6. Training Objectives

Reactive Geometry integrates task losses with physics‑informed regularization:
- **Task loss:** Cross‑entropy for discrete prediction, or periodic targets using toroidal or bounded phase losses.
- **Hamiltonian penalty:** Discourages spurious energy creation by penalizing $|E_{t+1} - E_t|$, with energy computed under the metric:
$$
E_t = \frac{1}{2} g_{ij}(\mathbf{x}_t) \, v_t^i \, v_t^j.
$$
- **Geodesic regularization:** Tempers curvature excursions via a mean‑squared curvature term:
$$
L_{\text{curv}} = \mathbb{E}\big[\|\Gamma(\mathbf{x})\|_F^2\big].
$$
- **Symmetry regularization (optional):** Encourages consistent geometric responses across isomeric heads through Noether symmetry terms.



## 7. Empirical Behavior

Reactive Geometry consistently reduces drift by braking high‑energy trajectories and stabilizing discrete transitions with optional singularity potentials. On cyclic algorithmic tasks, it supports long‑horizon phase tracking and robust extrapolation, especially in combination with toroidal topologies and structure‑preserving integrators. In non‑smooth regimes, lower‑order symplectic or RK2 steps often outperform high‑order explicit schemes.

The kinetic energy $K$ serves as an intrinsic uncertainty measure: high $K$ indicates the state is traversing uncertain regions, triggering reactive braking; low $K$ indicates confident, stable reasoning.



## 8. Discussion and Limitations

Reactive manifolds relate to Finslerian extensions by allowing velocity‑dependent geometry, yet remain practical via simple bounded scaling of curvature and gates. The formulation $g_{\text{eff},ij}(\mathbf{x}, \mathbf{v})$ is technically a Randers-type Finsler metric, but the simplification to Riemannian with reactive scaling maintains computational tractability.

Singularities provide useful stabilization for discrete logic but must be softly gated to preserve differentiability. CUDA fused paths and parallel scans yield throughput gains, though strict floating‑point parity with sequential updates is not guaranteed.

Formal convergence proofs for learned curvature and reactive flows are an open direction, though empirical evidence suggests stable training dynamics across a range of tasks.



## 9. Conclusion

Reactive Geometry equips neural geodesic flow with an internal feedback loop: energy modulates curvature and friction, which in turn regulates energy. This yields stable long‑horizon behavior, principled uncertainty handling, and improved alignment between geometry and computation. The covariant formulation ensures geometric consistency, while the reactive mechanisms provide adaptive stability without explicit architectural constraints.

Combined with topology‑aware modeling and structure‑preserving discretization, Reactive Geometry forms a foundation for constant‑memory, physically grounded sequence reasoning where the manifold geometry actively participates in the computation.



**References**  

[1]  Friston, K. (2010). The free‑energy principle: a unified brain theory?  
[2]  Amari, S. (2016). Information Geometry and Its Applications.  
[3]  Einstein, A. (1916). The Foundation of the General Theory of Relativity.  
[4]  Bao, D., Chern, S. S., & Shen, Z. (2000). An Introduction to Riemann‑Finsler Geometry.  
[5]  Sitzmann, V., et al. (2020). Implicit Neural Representations with Periodic Activation Functions.  
[6]  Nicolis, G., & Prigogine, I. (1977). Self‑Organization in Non‑Equilibrium Systems.  
[7]  Bejan, A. (2000). Shape and Structure, from Engineering to Nature.
