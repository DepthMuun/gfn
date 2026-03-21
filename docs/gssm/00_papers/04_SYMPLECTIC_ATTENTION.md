# Symplectic Attention: Constant-Memory Sequence Modeling via Geodesic Flow

**Author:** Joaquin Stürtz  
**Date:** January 24, 2026

**Abstract**  
Attention computes non-local interactions over a growing memory of keys and values, incurring quadratic time and linear memory with sequence length. We claim that this explicit global lookup is not necessary if the model's recurrence obeys a physically grounded dynamical law. **Symplectic Attention** replaces pairwise token interactions with **local geodesic flow** on a learned manifold: the model carries its causal history in a compact phase state $(x_t, v_t)$, and "attends" by evolving along learned curvature with structure-preserving integrators. A learned friction gate provides selective irreversibility for state rewriting, while a curvature-informed time gate shrinks steps in complex regions. The result is constant-memory inference with linear time, verified by a reference implementation based on multi-head manifolds, toroidal topology, and symplectic integration.



## 1. Motivation: From Global Lookup to Local Flow

### 1.1 The cost of "Action at a Distance"
Transformers maintain a full history of Key–Value pairs ($K, V$) and, to produce the next output, query this entire memory:
$$ Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d}}\right)V $$
This non-local operation scales quadratically in sequence length. It assumes relevant information must be explicitly retrieved from a static cache rather than propagated causally.

### 1.2 The Physical Alternative
In physics, a particle does not query distant points; it moves under local fields. The "memory" of structure is encoded in geometry (metric and connection), and the trajectory reveals which past impulses matter. We reformulate sequence modeling so that "attention" arises from **geodesic deviation**: the state flows along learned curvature that locally encodes how past tokens shape present motion.



## 2. Continuous-Time Formalism

We augment Hamiltonian dynamics with learned curvature and selective dissipation:

$$ \dot{x}^i = v^i, \quad \frac{D v^k}{dt} = \dot{v}^k + \Gamma^k_{ij}(x) v^i v^j = F^k_\theta(u_t) - (\mu_\theta(x, u_t) v)^k $$

Here, $F_\theta$ maps tokens to force impulses, $\Gamma^k_{ij}(x)$ is the Christoffel symbol of the second kind encoding geometric coupling through the Levi-Civita connection, and $\mu_\theta(x, u_t) \geq 0$ is a friction "clutch" that switches between conservative memory and dissipative updating. The covariant derivative $\frac{D v^k}{dt}$ ensures that the equation of motion is properly tensorial, with the Christoffel term being quadratic in velocity components.

### 2.1 Symplectic Attention as Geodesic Flow
Instead of $O(N^2)$ pairwise weighting, past tokens influence the present through the shape of the trajectory. The instantaneous phase state $(x_t, v_t)$ encodes causal history as momentum and winding on a compact manifold. "Attending" is realized by local curvature steering; dissipation opens the clutch to overwrite state during transitions.

### 2.2 Multi-Head Manifolds
Analogous to multi-head attention, we factor the latent into $H$ heads, each with its own geometry and gates. Heads evolve in parallel, then are mixed back into the shared state. This yields diverse relational channels without constructing global attention matrices.

### 2.3 Dynamic Time and Selective Irreversibility
A small gate $g(x) \in (0,1]$ scales the effective step size in "hard" regions (high curvature), while $\mu(x,u)$ controls when to dissipate momentum and rewrite the state. Together they implement stable, controllable computation over long horizons.

### 2.4 Symplectic Stability
For $\mu = 0$, the update is symplectic; phase-space volume is preserved and gradients neither vanish nor explode. Forgetting arises physically through $\mu > 0$, not numerically through unstable mappings.



## 3. Discretization: Symplectic Attention Step

Let $\Delta t$ be the base step, $g(x) \in (0,1]$ the time gate, $\Delta t_{\text{eff}} = g(x) \Delta t$, and $h = \frac{1}{2} \Delta t_{\text{eff}}$. A single head applies a Leapfrog (kick–drift–kick) step with implicit friction:

1) Kick (half-step, implicit damping):
$$ v_{n+\frac{1}{2}}^k = \frac{v_n^k + h\left[F^k_\theta(u_n) - \Gamma^k_{ij}(x_n) v_n^i v_n^j\right]}{1 + h\,\mu_\theta(x_n, u_n)} $$

2) Drift (full-step position with topology):
$$ x_{n+1}^i = \operatorname{wrap}\!\left(x_n^i + \Delta t_{\text{eff}}\, v_{n+\frac{1}{2}}^i\right) $$

3) Kick (half-step at new position):
$$ v_{n+1}^k = \frac{v_{n+\frac{1}{2}}^k + h\left[F^k_\theta(u_n) - \Gamma^k_{ij}(x_{n+1}) v_{n+\frac{1}{2}}^i v_{n+\frac{1}{2}}^j\right]}{1 + h\,\mu_\theta(x_{n+1}, u_n)} $$

Heads evolve in parallel; their states are concatenated and linearly mixed back into $(x, v)$. The scheme generalizes to other symplectic integrators (Yoshida, Forest–Ruth, Omelyan) and high-order explicit schemes (Heun, RK4, RK45) when conservation needs trade off against local accuracy.

Properties:
- When $\mu = 0$, the map is a symplectomorphism (volume-preserving).
- $\mu > 0$ yields controlled dissipation for state rewriting.
- $g(x)$ reduces aliasing in complex regions without breaking topology.

### 3.1 The "Infinite Context" Limit
Liouville's Theorem implies that information about initial conditions is preserved in ideal symplectic systems; causal history becomes phase windings on compact manifolds. The challenge shifts from **storage** (global cache) to **resolution** (readout decoding of state).

### 3.2 Complexity
- **Time:** $O(N)$ per sequence (constant per-token recurrence).
- **Memory:** **$O(1)$** at inference (fixed-size state).
- **Parallelization:** loop-friendly over tokens; sequence scans can reduce training wall-time on accelerators.



## 4. Architecture Mapping

### 4.1 Heads and Geometry
Each head maintains a sub-state and a geometry module (learned low-rank curvature or specialized manifold). Geometries compute $\Gamma^k_{ij}(x)$ and expose gates for $\mu(x,u)$; on compact topologies, gates use periodic features $[\sin(x), \cos(x)]$.

### 4.2 Dynamic Time Gate
A small MLP over state yields $g(x) \in (0,1]$ per head, scaling the effective step. This contracts time in high-curvature regions while allowing coasting elsewhere.

### 4.3 Mixing and Readout
After integration, heads are concatenated and linearly mixed back into the global $(x, v)$. Readout maps the latent to outputs via linear or implicit heads; in "holographic" mode, the latent state itself represents the target (e.g., phases), enabling topology-aligned supervision.

### 4.4 Topology
Toroidal topology bounds coordinates and encodes cyclic computation robustly. Position updates apply periodic wrap; gates consume periodic features to remain continuous at boundaries. Losses include toroidal or circular distance to avoid edge artifacts.

### 4.5 Training
In addition to task loss (e.g., cross-entropy or toroidal distance), physics-informed terms stabilize trajectories:
- Hamiltonian stability loss: penalizes spurious energy creation during coasting.
- Geodesic regularization: smooths curvature excursions.
- Velocity saturation: prevents runaway speeds while keeping gradients bounded.



## 5. Empirical Notes

On algorithmic tasks with long-range structure (parity, modular arithmetic, periodic phase tracking), models with compact topology and symplectic integration maintain stable trajectories and generalize to longer sequences. Dissipation spikes align with semantic transitions, while time gates shrink steps in geometrically complex regions, enabling precise updates without global attention matrices.



## 6. Discussion and Conclusion

Symplectic Attention returns to **local causality**: information moves through geometry, not global lookups. By carrying causal history in a compact phase state and evolving it with structure-preserving updates, we achieve constant-memory inference and robust long-horizon behavior. Selective irreversibility allows decisive state changes, while periodic topology aligns representation with cyclic tasks. This paradigm grounds sequence modeling in physics, offering a scalable alternative to quadratic attention.



**References**

[1] Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.  
[2] Arnold, V. I. (1989). *Mathematical Methods of Classical Mechanics*. Springer-Verlag.  
[3] Katharopoulos, A., et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML.  
[4] Chen, R. T. Q., et al. (2018). *Neural Ordinary Differential Equations*. NeurIPS.  
[5] Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv.  
[6] Noether, E. (1918). *Invariant Variation Problems*. Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen.  
[7] Misner, C. W., Thorne, K. S., & Wheeler, J. A. (1973). *Gravitation*. W. H. Freeman.
