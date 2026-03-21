# Thermodynamic Gating: Dissipation-Controlled Memory in Geodesic Flow Networks

**Author:** Joaquin Stürtz  
**Date:** January 24, 2026

**Abstract**  
Hamiltonian latent dynamics preserve phase-space volume, enabling long-horizon persistence of information. Yet strict conservation impedes contextual switching: a purely conservative system cannot relax into new semantic attractors without oscillation. We introduce **Thermodynamic Gating**, a mechanism that couples Hamiltonian geodesic flow to a learned dissipative field and a dynamic time gate. A state- and input-dependent friction coefficient $\mu(\mathbf{x}, u)$ implements selective irreversibility (the "clutch"), while a curvature-informed gate scales the effective time step $\Delta t_{\text{eff}} = g(\mathbf{x}) \cdot \Delta t$. The result is a conformal-symplectic update that supports both persistent memory (coasting) and decisive rewriting (damping), verified in the reference implementation by an implicit friction Leapfrog scheme and toroidal topology features for periodic computation.



## 1. Motivation: Conservation, Updating, and Selective Irreversibility

- Conservation preserves gradients and memory but resists settling; naive damping destroys long-horizon stability.
- Intelligent behavior requires switching between an **Isolated Regime** (conservative flow) and an **Open Regime** (entropy-producing update).
- Thermodynamic Gating provides a physical basis for the forget gate: dissipation is produced only where and when information must be rewritten.

The key insight is that information preservation requires energy conservation (Hamiltonian dynamics), while information updating requires energy dissipation (Landauer's principle). By learning when to apply dissipation, the network can maintain memory over long horizons while allowing decisive updates when semantic context shifts.



## 2. Continuous-Time Model

We augment Hamiltonian dynamics with a dissipative force proportional to velocity. In covariant form, the equations of motion are:

$$
\dot{x}^i = v^i, \quad \frac{Dv^k}{dt} = \frac{dv^k}{dt} + \Gamma^k_{ij}(\mathbf{x}) \, v^i v^j = F^k(\mathbf{x}, \mathbf{u}) - \mu(\mathbf{x}, \mathbf{u}) \, v^k.
$$

Expanding the covariant derivative, this becomes:

$$
\ddot{x}^k + \Gamma^k_{ij}(\mathbf{x}) \, v^i v^j = F^k(\mathbf{x}, \mathbf{u}) - \mu(\mathbf{x}, \mathbf{u}) \, v^k.
$$

In vector notation, the Christoffel term is:

$$
\Gamma(\mathbf{x})[\mathbf{v}, \mathbf{v}] = \big(\Gamma^k_{ij}(\mathbf{x}) \, v^i v^j\big)_{k=1}^d.
$$

Here:
- $\mathbf{x} \in \mathcal{M}$ is the position on the manifold
- $\mathbf{v} \in T_{\mathbf{x}}\mathcal{M}$ is the velocity
- $\frac{dv^k}{dt} = \ddot{x}^k$ is the ordinary acceleration in local coordinates
- $F^k(\mathbf{x}, \mathbf{u})$ is an external force embedding from the input token
- $\Gamma^k_{ij}(\mathbf{x})$ are Christoffel symbols of the Levi-Civita connection
- $\mu(\mathbf{x}, \mathbf{u}) \ge 0$ is a learned dissipation coefficient

Two regimes arise:
- **Memory mode** ($\mu \approx 0$): Near-Hamiltonian coasting, enabling information persistence across long time horizons.
- **Update mode** ($\mu \gg 0$): Rapid damping, enabling information rewriting at semantic transitions.

Additionally, a learned gate $g(\mathbf{x}) \in (0, 1]$ scales the effective step size:

$$
\Delta t_{\text{eff}} = g(\mathbf{x}) \cdot \Delta t,
$$

shrinking time in "hard" regions (high curvature) and expanding it in "flat" regions (skip-like behavior for efficient computation).



## 3. Discrete-Time: Conformal Symplectic Leapfrog with Implicit Friction

Let $\Delta t$ be the base step and $g(\mathbf{x}) \in (0, 1]$ the dynamic scale. Define:

$$
\Delta t_{\text{eff}} = g(\mathbf{x}) \cdot \Delta t, \qquad h = \frac{1}{2} \Delta t_{\text{eff}}.
$$

A single Leapfrog step with implicit friction is:

1. **Kick (half step, implicit damping):**
$$
v^{k}_{\;n+\tfrac{1}{2}} = \frac{v^k_{\;n} + h \left[F^k(\mathbf{x}_n, \mathbf{u}_n) - \Gamma^k_{ij}(\mathbf{x}_n) \, v^i_{\;n} v^j_{\;n}\right]}{1 + h \, \mu(\mathbf{x}_n, \mathbf{u}_n)}.
$$

2. **Drift (full step position):**
$$
\mathbf{x}_{n+1} = \operatorname{wrap}\!\left(\mathbf{x}_n + \Delta t_{\text{eff}} \; \mathbf{v}_{\;n+\tfrac{1}{2}}\right).
$$

3. **Kick (half step at new position):**
$$
v^{k}_{\;n+1} = \frac{v^{k}_{\;n+\tfrac{1}{2}} + h \left[F^k(\mathbf{x}_{n+1}, \mathbf{u}_n) - \Gamma^k_{ij}(\mathbf{x}_{n+1}) \, v^{i}_{\;n+\tfrac{1}{2}} v^{j}_{\;n+\tfrac{1}{2}}\right]}{1 + h \, \mu(\mathbf{x}_{n+1}, \mathbf{u}_n)}.
$$

The wrap(·) function enforces the manifold topology (e.g., periodic wrapping on a torus). This implicit form is stable under large $\mu$ and matches the fused CUDA/Python reference update. In dimensions, $\mu$ may be per-coordinate; the gate is bounded by a sigmoid scale (maximum friction).

**Properties:**
- Time-reversibility is conformally broken only by $\mu$; when $\mu = 0$, the scheme is symplectic and preserves phase-space volume.
- Energy production/consumption is localized to transitions, enabling "dash-and-stop" behavior where the system coasts then rapidly updates.
- The dynamic time gate $g(\mathbf{x})$ preserves qualitative trajectories by shrinking steps near strong curvature, preventing numerical aliasing.

The implicit friction formulation $\frac{v + h \cdot a}{1 + h\mu}$ ensures stability even when $\mu$ is large, avoiding the numerical instability of explicit damping schemes.



## 4. Implementation Summary (Architecture-Level)

- **Friction Gate (Thermodynamic Clutch):** A linear mapping over state features (and optionally input force) produces:
  $$
  \mu(\mathbf{x}, \mathbf{u}) = \alpha \cdot \sigma\big(W_{\text{state}} \phi(\mathbf{x}) + W_{\text{input}} \mathbf{u} + \mathbf{b}\big),
  $$
  where $\alpha > 0$ is a learnable scale, $\sigma(\cdot)$ is the sigmoid, and $\phi(\mathbf{x}) = [\sin\mathbf{x}, \cos\mathbf{x}]$ provides periodic features for toroidal manifolds.

- **Curvature Gate (Dynamic Time):** A small MLP over $\mathbf{x}$ yields $g(\mathbf{x}) \in (0, 1]$, scaling $\Delta t$ per head:
  $$
  g(\mathbf{x}) = \sigma\big(W_g \phi(\mathbf{x}) + b_g\big).
  $$
  On torus, inputs use $\phi(\mathbf{x}) = [\sin\mathbf{x}, \cos\mathbf{x}]$ to respect periodicity.

- **Curvature Interaction:** A low-rank Christoffel operator computes $\Gamma^k_{ij}(\mathbf{x})$ with symmetric structure to approximate torsion-free geometry:
  $$
  \Gamma^k_{ij}(\mathbf{x}) \approx \sum_{r=1}^{R} U^k_r(\mathbf{x}) \, U_{ij,r}(\mathbf{x}),
  $$
  where outputs are softly clamped for numerical safety: $\Gamma^k_{ij} \leftarrow \tanh(\Gamma^k_{ij} / \Gamma_{\text{max}}) \cdot \Gamma_{\text{max}}$.

- **Integrators:** Leapfrog/Verlet/Yoshida/Forest–Ruth implement the geodesic step; Leapfrog uses the implicit friction update above and respects topology via periodic wrapping.

- **Multi-Head Manifolds:** The latent state is split across $H$ heads; each head applies its own geometry, gate, and integrator, then results are mixed back:
  $$
  \mathbf{F} = \sum_{h=1}^{H} w_h \, \mathbf{F}^{(h)}, \quad \sum_{h=1}^{H} w_h = 1.
  $$

These elements compose a geodesic layer that performs learned physics-informed computation without violating manifold constraints.



## 5. Topology and Periodic Features

- **Toroidal topology** $T^n$ bounds coordinates and models cyclic computation naturally (parity, phase, modular arithmetic).
- **Periodic features** $\phi(\mathbf{x}) = [\sin\mathbf{x}, \cos\mathbf{x}]$ are used both in friction gating and dynamic time gating to preserve continuity across $2\pi$ boundaries.
- **Position updates** apply wrap($\cdot$) to maintain coordinates on the manifold:
  $$
  x^k \leftarrow (x^k + 2\pi) \mod 2\pi, \quad \forall k.
  $$
  Training losses include toroidal distance terms to avoid boundary artifacts:
  $$
  L_{\text{torus}} = \sum_{k=1}^{n} d_{S^1}^2\big(x^k_{\text{pred}}, x^k_{\text{target}}\big),
  $$
  where $d_{S^1}$ is the circular distance on $S^1$.



## 6. Training and Regularization

- **Task losses** (e.g., cross-entropy or toroidal distance) drive semantic objectives.
- **Hamiltonian stability terms** penalize spurious energy creation during coasting segments:
  $$
  L_H = \lambda_H \, \mathbb{E}\big[\,|E_{t+1} - E_t|\,\big], \quad E_t = \frac{1}{2} g_{ij}(\mathbf{x}_t) \, v_t^i \, v_t^j.
  $$
- **Geodesic regularization** encourages curvature smoothness and reduces instability under strong gates:
  $$
  L_{\text{geo}} = \mathbb{E}\big[\|\Gamma(\mathbf{x})\|_F^2\big].
  $$
- **Velocity saturation** limits runaway speeds while keeping gradients well-behaved:
  $$
  \mathbf{v} \leftarrow \tanh(\mathbf{v} / v_{\text{max}}) \cdot v_{\text{max}}.
  $$

The energy conservation penalty $L_H$ is crucial: it prevents the dissipation gate from being used as a "hack" to reduce all dynamics, instead encouraging the model to use $\mu$ only where truly needed for semantic transitions.



## 7. Information-Theoretic Perspective

Thermodynamic Gating realizes **Landauer's Principle**: erasure requires dissipating energy. In practice, trained models exhibit:

- **Low dissipation** during long stationary contexts (memory conservation), where $\mu \approx 0$ and the system preserves information via Hamiltonian flow.
- **Sharp, localized spikes** of $\mu$ at semantic transitions (entropy production where rewriting is required), implementing the thermodynamic cost of information processing.
- **Dynamic time contraction** near complex geometric regions to avoid aliasing and numerical instability, naturally implementing variable-time computation.

This yields precise state changes without sacrificing long-horizon persistence, creating a principled balance between information preservation and information updating based on the Second Law of Thermodynamics.



## 8. Practical Notes

- **Maximum friction scale** is bounded to maintain stability under implicit updates: $\mu_{\text{max}}$ is typically set to $1.0-5.0$.
- **Gates are per-head and per-batch**, enabling heterogeneous regimes across manifold subspaces (some heads may be in memory mode while others update).
- **CUDA fused paths** accelerate integration and gating; Python fallbacks preserve correctness where fused kernels are unavailable.
- **Toroidal wrapping** is applied after position updates; gating features remain periodic to avoid discontinuities in the learned functions.
- **Implicit friction** is solved analytically at each step, requiring no iterative solver for the division by $(1 + h\mu)$.



## 9. Conclusion

Thermodynamic Gating unifies conservative memory and decisive updating within a single geometric computation layer. By combining a conformal-symplectic integrator with learned friction and dynamic time gating, Geodesic Flow Networks achieve stability, persistence, and controllability on compact manifolds—enabling symbolic trajectory formation and robust reasoning over long horizons.

The key contribution is the physical grounding: dissipation is not an ad-hoc regularization but a learned thermodynamic process that implements the fundamental tradeoff between memory persistence and information updating. This framework provides both theoretical understanding and practical tools for building neural networks with intrinsic long-horizon stability.



**References**

[1] Prigogine, I. (1955). Introduction to Thermodynamics of Irreversible Processes. Thomas.  
[2] Landauer, R. (1961). Irreversibility and Heat Generation in the Computing Process. IBM Journal of Research and Development.  
[3] Greydanus, S., et al. (2019). Hamiltonian Neural Networks. NeurIPS.  
[4] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.  
[5] Ottinger, H. C. (2005). Beyond Equilibrium Thermodynamics. Wiley-Interscience.  
[6] Cranmer, M., et al. (2020). Lagrangian Neural Networks. ICLR.  
[7] Schlögl, F. (1971). Thermodynamic stability of non-equilibrium states. Zeitschrift für Physik.
