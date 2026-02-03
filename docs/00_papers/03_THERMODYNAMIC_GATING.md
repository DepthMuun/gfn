# Thermodynamic Gating: Dissipation-Controlled Memory in Geodesic Flow Networks

**Author:** Joaquin Stürtz  
**Date:** January 24, 2026

**Abstract**  
Hamiltonian latent dynamics preserve phase-space volume, enabling long-horizon persistence of information. Yet strict conservation impedes contextual switching: a purely conservative system cannot relax into new semantic attractors without oscillation. We introduce **Thermodynamic Gating**, a mechanism that couples Hamiltonian geodesic flow to a learned dissipative field and a dynamic time gate. A state- and input-dependent friction coefficient $\mu(x,u)$ implements selective irreversibility (the “clutch”), while a curvature-informed gate scales the effective time step $\Delta t_{\text{eff}} = g(x)\,\Delta t$. The result is a conformal-symplectic update that supports both persistent memory (coasting) and decisive rewriting (damping), verified in the reference implementation by an implicit friction Leapfrog scheme and toroidal topology features for periodic computation.



## 1. Motivation: Conservation, Updating, and Selective Irreversibility

- Conservation preserves gradients and memory but resists settling; naive damping destroys long-horizon stability.  
- Intelligent behavior requires switching between an **Isolated Regime** (conservative flow) and an **Open Regime** (entropy-producing update).  
- Thermodynamic Gating provides a physical basis for the forget gate: dissipation is produced only where and when information must be rewritten.



## 2. Continuous-Time Model

We augment Hamiltonian dynamics with a dissipative co-vector proportional to velocity:

$$ \dot{x} = v, \quad \dot{v} = F(x,u) - \Gamma(x,v) - \mu(x,u)\odot v $$

where $F$ is an external force embedding (token-driven), $\Gamma$ is the Christoffel-like curvature interaction, and $\mu(x,u) \ge 0$ is a learned dissipation coefficient. Two regimes arise:

- Memory mode ($\mu \approx 0$): near-Hamiltonian coasting (information persistence).  
- Update mode ($\mu \gg 0$): rapid damping (information rewriting at transitions).

Additionally, a learned gate $g(x)\in(0,1]$ scales the effective step size, shrinking time in “hard” regions (high curvature) and expanding it in “flat” regions (skip-like behavior).



## 3. Discrete-Time: Conformal Symplectic Leapfrog with Implicit Friction

Let $\Delta t$ be the base step and $g(x)\in(0,1]$ the dynamic scale. Define $\Delta t_{\text{eff}} = g(x)\,\Delta t$ and $h = \tfrac{1}{2}\Delta t_{\text{eff}}$. A single Leapfrog step with implicit friction is:

1) Kick (half step, implicit damping):
$$ v_{n+\tfrac{1}{2}} = \frac{v_n + h\,[F(x_n,u_n) - \Gamma(x_n,v_n)]}{1 + h\,\mu(x_n,u_n)} $$

2) Drift (full step position):
$$ x_{n+1} = \operatorname{wrap}\!\left(x_n + \Delta t_{\text{eff}}\; v_{n+\tfrac{1}{2}}\right) $$

3) Kick (half step at new position):
$$ v_{n+1} = \frac{v_{n+\tfrac{1}{2}} + h\,[F(x_{n+1},u_n) - \Gamma(x_{n+1},v_{n+\tfrac{1}{2}})]}{1 + h\,\mu(x_{n+1},u_n)} $$

Here wrap(·) enforces topology (e.g., periodic torus). This implicit form is stable under large $\mu$ and matches the fused CUDA/Python reference update. In dimensions, $\mu$ may be per-coordinate; the gate is bounded by a sigmoid scale (maximum friction).

Properties:
- Time-reversibility is conformally broken only by $\mu$; when $\mu\!=\!0$, the scheme is symplectic.  
- Energy production/consumption is localized to transitions, enabling “dash-and-stop” behavior.  
- The dynamic time gate $g(x)$ preserves qualitative trajectories by shrinking steps near strong curvature.



## 4. Implementation Summary (Architecture-Level)

- Friction Gate (Thermodynamic Clutch): A linear mapping over state features (and optionally input force) produces $\mu(x,u)=\alpha\cdot\sigma(\cdot)$ with $\alpha>0$. On compact manifolds (torus), features are periodic [sin(x), cos(x)].  
- Curvature Gate (Dynamic Time): A small MLP over $x$ yields $g(x)\in(0,1]$, scaling $\Delta t$ per head. On torus, inputs use [sin(x), cos(x)] to respect periodicity.  
- Curvature Interaction: A low-rank Christoffel operator computes $\Gamma(x,v)$ with symmetric structure to approximate torsion-free geometry; outputs are softly clamped for numerical safety.  
- Integrators: Leapfrog/Verlet/Yoshida/Forest–Ruth implement the geodesic step; Leapfrog uses the implicit friction update above and respects topology via periodic wrapping.  
- Multi-Head Manifolds: The latent state is split across heads; each head applies its own geometry, gate, and integrator, then results are mixed back.

These elements compose a geodesic layer that performs learned physics-informed computation without violating manifold constraints.



## 5. Topology and Periodic Features

- Toroidal topology $T^n$ bounds coordinates and models cyclic computation naturally (parity, phase, modular arithmetic).  
- Periodic features [sin(x), cos(x)] are used both in friction gating and dynamic time gating to preserve continuity across $2\pi$ boundaries.  
- Position updates apply wrap(·) to maintain coordinates on the manifold; training losses include toroidal distance terms to avoid boundary artifacts.



## 6. Training and Regularization

- Task losses (e.g., cross-entropy or toroidal distance) drive semantic objectives.  
- Hamiltonian stability terms penalize spurious energy creation during coasting segments.  
- Geodesic regularization encourages curvature smoothness and reduces instability under strong gates.  
- Velocity saturation (tanh) limits runaway speeds while keeping gradients well-behaved.



## 7. Information-Theoretic Perspective

Thermodynamic Gating realizes **Landauer’s Principle**: erasure requires dissipating energy. In practice, trained models exhibit:
- Low dissipation during long stationary contexts (memory conservation).  
- Sharp, localized spikes of $\mu$ at semantic transitions (entropy production where rewriting is required).  
- Dynamic time contraction near complex geometric regions to avoid aliasing and numerical instability.

This yields precise state changes without sacrificing long-horizon persistence.



## 8. Practical Notes

- Maximum friction scale is bounded to maintain stability under implicit updates.  
- Gates are per-head and per-batch, enabling heterogeneous regimes across manifold subspaces.  
- CUDA fused paths accelerate integration and gating; Python fallbacks preserve correctness where fused kernels are unavailable.  
- Toroidal wrapping is applied after position updates; gating features remain periodic to avoid discontinuities.



## 9. Conclusion

Thermodynamic Gating unifies conservative memory and decisive updating within a single geometric computation layer. By combining a conformal-symplectic integrator with learned friction and dynamic time gating, Geodesic Flow Networks achieve stability, persistence, and controllability on compact manifolds—enabling symbolic trajectory formation and robust reasoning over long horizons.


**References**

[1] Prigogine, I. (1955). Introduction to Thermodynamics of Irreversible Processes. Thomas.  
[2] Landauer, R. (1961). Irreversibility and Heat Generation in the Computing Process. IBM Journal of Research and Development.  
[3] Greydanus, S., et al. (2019). Hamiltonian Neural Networks. NeurIPS.  
[4] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.  
[5] Ottinger, H. C. (2005). Beyond Equilibrium Thermodynamics. Wiley-Interscience.  
[6] Cranmer, M., et al. (2020). Lagrangian Neural Networks. ICLR.  
[7] Schlögl, F. (1971). Thermodynamic stability of non-equilibrium states. Zeitschrift für Physik.  
