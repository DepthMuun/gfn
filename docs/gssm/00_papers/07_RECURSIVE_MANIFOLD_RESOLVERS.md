# Recursive Manifold Resolvers: Adaptive Mesh Refinement in Neural Geodesic Flows

**Author:** Joaquin Stürtz  
**Date:** January 26, 2026

**Abstract**  
Computational efficiency in sequence modeling typically requires a trade-off between temporal resolution and memory complexity. We propose **Recursive Manifold Resolvers**, a framework for implementing multiscale "Geometric Tunneling" in latent space. By monitoring the local manifold curvature density, the system can dynamically instantiate high-resolution sub-manifolds to resolve semantic ambiguity—a process analogous to Adaptive Mesh Refinement (AMR) in computational fluid dynamics. This allows for the precise resolution of high-frequency logical operations while maintaining a constant-time global integration step. We demonstrate that this architecture successfully bridges the gap between fast, intuitive processing and slow, high-precision deliberative reasoning.



## 1. Introduction

### 1.1 The Multiscale Resolution Challenge
Standard neural integrators employ a fixed temporal resolution $\Delta t$. However, logical operations involving rapid state transitions (e.g., high-frequency signal processing or algorithmic parity) can exceed the Shannon-Nyquist limit of the integration scheme. This leads to **Numerical Aliasing**, where the discrete logic of the task is "smeared" by the coarse integration of the underlying manifold, resulting in catastrophic phase drift.

### 1.2 The Solution: Dynamic Temporal Resolution
To address this, we introduce the concept of a **Recursive Manifold Resolver**. Just as a microscope increases optical resolution only where necessary, our architecture increases *temporal* resolution in regions of the latent space that exhibit high curvature or high semantic density. This is achieved through two complementary mechanisms:
1.  **Learned Time-Dilation:** A neural controller that predicts the optimal $\Delta t$ for each step.
2.  **Recursive Sub-Stepping:** Breaking a single macro-step into multiple micro-steps when a singularity is detected.



## 2. Mathematical Foundation

### 2.1 Manifold Density and Aliasing
We define the **Manifold Density** $\mathcal{D}(x)$ as the local intensity of the geometric coupling encoded in the Christoffel symbols. Since the Christoffel symbols $\Gamma^k_{ij}(x)$ are tensors that depend only on position and encode the curvature of the manifold, we measure the effective geometric acceleration as a function of velocity:

$$ \mathcal{D}(x) = \mathbb{E}_{v \sim p(v)} \left[ \left\| \Gamma^k_{ij}(x) v^i v^j \right\|_2 \right] $$

This expectation over the velocity distribution $p(v)$ captures the typical magnitude of the geometric forces the particle experiences. If the integration step $\Delta t$ is too large relative to this density (i.e., $\Delta t \cdot \mathcal{D}(x) \gg 1$), the linearization of the manifold tangent space fails. The Christoffel symbols $\Gamma^k_{ij}(x)$ are computed from the metric tensor $g_{ij}(x)$ via the Levi-Civita connection formula:

$$ \Gamma^k_{ij}(x) = \frac{1}{2} g^{kl}(x) \left( \partial_i g_{jl}(x) + \partial_j g_{il}(x) - \partial_l g_{ij}(x) \right) $$

The error $\epsilon$ in the trajectory grows as $O(\Delta t^{k+1})$, where $k$ is the order of the integrator, and this error is exacerbated when the Christoffel symbols are large, indicating regions of high curvature where the geodesic deviation is significant.

### 2.2 The Neural Controller
To adapt $\Delta t$ dynamically, we introduce a control policy $\pi_\theta(x, v)$ that modulates the base step size $\Delta t_0$:

$$ \Delta t(x, v) = \Delta t_0 \cdot \text{Softplus}\left( \pi_\theta(x, v) \right) $$

where $\pi_\theta$ is a lightweight Multi-Layer Perceptron (MLP) taking the concatenated state $(x, v)$ as input. The Softplus activation ensures that the predicted step size is strictly positive, $\Delta t(x, v) > 0$. This allows the model to "slow down" (small $\Delta t$) in complex logical regions with high manifold density $\mathcal{D}(x)$ and "fast forward" (large $\Delta t$) through trivial transitions where the Christoffel symbols are near zero.

### 2.3 Recursive Integration Scheme
When $\mathcal{D}(x)$ exceeds a critical threshold $\tau$, the system triggers a **Geometric Tunneling** event. The integration over $[t, t+\Delta t]$ is recursively decomposed:

$$ x_{t+1} = \Phi_{\text{micro}}^{\, m} \left( x_t, \frac{\Delta t}{m} \right) $$

where $\Phi_{\text{micro}}$ is the refined integrator operating with the covariant geodesic equation $\frac{D v^k}{dt} = -\Gamma^k_{ij}(x) v^i v^j$, and $m$ is the number of sub-steps determined by the ratio $\mathcal{D}(x) / \tau$. This ensures that the local curvature is sampled with sufficient frequency to preserve topological invariants and maintain numerical stability in high-curvature regions.



## 3. Implementation

The GFN codebase implements these concepts through specialized integrators and layer configurations.

### 3.1 Neural Integrator (`gfn/integrators/neural.py`)
The `NeuralIntegrator` realizes the learned time-dilation mechanism. It uses a dedicated controller network to predict scalar step-sizes for each item in the batch.

```python
class NeuralIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01, dim=None):
        super().__init__()
        # ...
        self.controller = nn.Sequential(
            nn.Linear(self.dim * 3, self.dim), # Input: [x, v, force]
            nn.GELU(),
            nn.Linear(self.dim, 1),
            nn.Softplus() # Strictly positive dt
        )
        # Initialize to output ~1.0 (neutral scaling)
        nn.init.constant_(self.controller[2].bias, 0.55)

    def forward(self, x, v, force=None):
        # 1. Predict optimal dt based on state
        state_cat = torch.cat([x, v, force], dim=-1)
        dt_scale = self.controller(state_cat)
        dt = self.base_dt * dt_scale
        
        # 2. Integrate with adaptive dt
        # ... (Standard Symplectic/RK update using new dt)
```

The controller takes the full phase space state $(x, v, F)$ where $F^k = F^k_\theta(u_t)$ represents the external force, and predicts a scaling factor for the base timestep. This allows the integrator to slow down precisely where the Christoffel symbols $\Gamma^k_{ij}(x)$ indicate high geometric complexity.

### 3.2 High-Order Validation: Dormand-Prince (`gfn/integrators/runge_kutta/dormand_prince.py`)
To validate that lower-order adaptive methods are not hallucinating trajectories, we provide the `DormandPrinceIntegrator` (RK45). This 5th-order method serves as the "Gold Standard" for detecting numerical aliasing during evaluation.

```python
class DormandPrinceIntegrator(nn.Module):
    r"""
    Dormand-Prince (DP5) Integrator.
    Uses the RK45 tableau and applies the 5th-order solution for updates.
    """
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        # RK45 coefficients (c, a, b5)
        self.c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
        self.b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        # ...
```

The Dormand-Prince method evaluates the acceleration $a^k(x, v) = -\Gamma^k_{ij}(x) v^i v^j$ at multiple intermediate points within each integration step, providing high-order accuracy for tracking geodesics in regions of rapidly varying curvature.

### 3.3 Layer-Level Recursion (`gfn/layers/base.py`)
The `MLayer` class supports a `use_recursive` flag, enabling the recursive geodesic logic defined in the physics configuration.

```python
self.use_recursive = self.physics_config.get('active_inference', {}).get('recursive_geodesics', {}).get('enabled', False)
if self.use_recursive:
    self.context_proj = nn.Linear(heads, dim)
```

In the fused CUDA kernel (`dormand_prince_fused.cu`), this logic is optimized to run entirely on-chip, avoiding the overhead of Python loops for sub-stepping. The recursive subdivision is determined by the manifold density $\mathcal{D}(x)$ computed from the Christoffel symbols $\Gamma^k_{ij}(x)$.

### 3.4 Christoffel Module Interface
The integration schemes rely on a `Christoffel` module that computes the symbols $\Gamma^k_{ij}(x)$ from the learned metric tensor $g_{ij}(x)$. This module is shared across all integrators to ensure consistency:

```python
class Christoffel(nn.Module):
    def forward(self, x):
        # Compute metric g_ij(x) from neural network
        g = self.metric_network(x)
        # Compute inverse metric g^ij
        g_inv = torch.inverse(g)
        # Compute Christoffel symbols from metric derivatives
        # Gamma^k_ij = 0.5 * g^(kl) * (d g_jl/dx^i + d g_il/dx^j - d g_ij/dx^l)
        christoffel = christoffel_symbols(g, g_inv)
        return christoffel  # Shape: [batch, dim, dim, dim]
```

The returned tensor has shape $[B, d, d, d]$ representing $\Gamma^k_{ij}$ for each batch element, where the first index $k$ corresponds to the contravariant component and $i, j$ are the covariant indices of the connection.



## 4. Empirical Implications

### 4.1 Resolution Elasticity
The Recursive Manifold Resolver grants the network **Resolution Elasticity**. It can resolve dynamics faster than the base integration frequency by activating micro-steps only in high-curvature regions. This effectively decouples the "thinking speed" (internal logic steps) from the "reading speed" (token input rate). The manifold density $\mathcal{D}(x)$ serves as a computable proxy for the local "difficulty" of the logical operation being performed.

### 4.2 Computational Economy
By maintaining a coarse global update for flat manifold regions (where $\Gamma^k_{ij}(x) \approx 0$), the model minimizes computational overhead. The expensive high-precision integration is reserved only for "hard" tokens or complex logical transitions, optimizing the FLOPs/entropy ratio. The recursive sub-stepping factor $m$ is dynamically chosen as $m = \lceil \mathcal{D}(x) / \tau \rceil$, ensuring proportional computational investment to geometric complexity.



## 5. Conclusion

Recursive Manifold Resolvers provide a formal framework for enabling "deliberative thought" within continuous neural flows. By treating task complexity as a trigger for local manifold refinement through the Christoffel symbols $\Gamma^k_{ij}(x)$, we enable a new class of architectures that are both computationally efficient and numerically robust across vast temporal scales. The manifold density metric $\mathcal{D}(x)$ provides a principled measure of when to invoke recursive sub-stepping, bridging the gap between the fixed-step integrators used for efficient inference and the adaptive methods required for high-precision deliberative reasoning. This brings the GFN architecture closer to the ideal of a system that can "think fast and slow" purely through geometric adaptation.



**References**

[1] Berger, M. J., & Oliger, J. (1984). *Adaptive mesh refinement for hyperbolic partial differential equations*. Journal of Computational Physics.  
[2] Mandelbrot, B. (1982). *The Fractal Geometry of Nature*. W. H. Freeman.  
[3] Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.  
[4] E, W., & Engquist, B. (2003). *The Heterogeneous Multiscale Methods*. Communications in Mathematical Sciences.  
[5] Chen, R. T. Q., et al. (2018). *Neural Ordinary Differential Equations*. NeurIPS.  
[6] Ames, W. F. (2014). *Numerical Methods for Partial Differential Equations*. Academic Press.
