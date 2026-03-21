# Parallel Manifold Scan Layer: Linear Time-Varying Approximation for O(log N) Geodesic Flow

**Joaquín Stürtz**

---

## Abstract

We present the Parallel Manifold Scan Layer (P-MLayer), an efficient parallel implementation of geodesic flow dynamics using associative scan algorithms. Standard manifold layers process sequences sequentially, incurring $\mathcal{O}(N)$ time complexity that limits training efficiency on modern parallel hardware. P-MLayer addresses this limitation by linearizing the geodesic flow through a Linear Time-Varying (LTV) system approximation, enabling $\mathcal{O}(\log N)$ parallelization using parallel scan (prefix sum) algorithms. The key insight is that the non-linear geodesic equation $\dot{v}^k = -\Gamma^k_{ij}(x) v^i v^j$ can be approximated as a linear system $\dot{v} = -D(F) \cdot v + F$ where $D(F)$ is a damping factor predicted from the input force. The LTV system has the form $v_t = A_t v_{t-1} + B_t$, which can be computed for all timesteps simultaneously using the parallel scan algorithm. Additionally, we introduce multi-scale time initialization where different heads operate at different base time scales, creating effective "wormholes" in the parallel scan that enable efficient modeling of long-range dependencies. Experiments demonstrate that P-MLayer achieves comparable accuracy to sequential manifold layers while enabling 3-5× speedup on modern GPU architectures.

**Keywords:** Parallel scan, associative scan, Linear Time-Varying systems, geodesic flows, parallel computation, manifold learning, O(log N) complexity, multi-scale time

---

## 1. Introduction

The sequential nature of recurrent neural networks has long been a bottleneck for efficient parallel computation. While transformer architectures have largely replaced recurrence with attention, certain geometric representations—particularly those based on geodesic flows—retain inherently sequential structure. Manifold layers process information as particles moving along geodesics, with each timestep's evolution depending on the previous timestep's state.

This sequential dependence creates two problems: first, training must proceed timestep by timestep, preventing full parallelization; second, information from distant timesteps must propagate through intermediate steps, creating potential gradient flow issues. Standard techniques like weight tying and careful initialization address these issues to some extent, but the fundamental $\mathcal{O}(N)$ sequentiality remains.

In this paper, we introduce the Parallel Manifold Scan Layer, which enables parallel computation of geodesic flow dynamics through a Linear Time-Varying (LTV) approximation. The key insight is that the non-linear Christoffel dynamics can be linearized to produce a system of the form:

$$v_t = A_t v_{t-1} + B_t$$

This linear recurrence can be computed for all timesteps simultaneously using the parallel scan algorithm (also known as prefix sum), reducing the sequential complexity from $\mathcal{O}(N)$ to $\mathcal{O}(\log N)$ with respect to sequence length.

The contributions of this work are as follows. First, we derive the LTV approximation of geodesic flow dynamics, showing how Christoffel-based non-linearities can be linearized while preserving key geometric properties. Second, we implement the parallel scan algorithm for the LTV system, enabling efficient GPU computation. Third, we introduce multi-scale time initialization, where different heads operate at different base time scales to capture both short-range and long-range dependencies. Fourth, we demonstrate through experiments that P-MLayer achieves comparable accuracy to sequential baselines while providing significant speedup.

---

## 2. Background and Related Work

### 2.1 Parallel Scan Algorithm

The parallel scan algorithm computes prefix sums (or products) in $\mathcal{O}(\log N)$ time on parallel hardware. Given a sequence of elements $x_0, x_1, ..., x_{N-1}$ and a binary associative operator $\otimes$, the scan computes:

$$y_i = x_0 \otimes x_1 \otimes \cdots \otimes x_i$$

Classic applications include computing cumulative sums and parallel prefix addition. The algorithm works by dividing the computation into logarithmic levels, with each level combining pairs of elements.

For our application, we use the scan to compute the cumulative effect of the linear recurrence:

$$v_t = A_t v_{t-1} + B_t$$

This can be unrolled as:

$$v_t = \left(\prod_{i=t}^{1} A_i\right) v_0 + \sum_{j=1}^t \left(\prod_{i=t}^{j+1} A_i\right) B_j$$

The parallel scan computes both the cumulative products and the accumulated sums simultaneously.

### 2.2 Linear Time-Varying Systems

Linear Time-Varying systems generalize linear time-invariant systems by allowing system matrices to vary with time:

$$\dot{x}(t) = A(t) x(t) + B(t) u(t)$$

For discrete time, the LTV system becomes:

$$x_{t+1} = A_t x_t + B_t u_t$$

The solution involves time-ordered matrix multiplication, which is inherently sequential. However, if all $A_t$ and $B_t$ are known in advance (as they are in our case, since they are functions of the input), the solution can be computed in parallel.

### 2.3 Christoffel Symbol Dynamics

The geodesic equation in a Christoffel-based manifold is:

$$\frac{dv^k}{dt} = -\Gamma^k_{ij}(x) v^i v^j$$

where $\Gamma^k_{ij}(x) v^i v^j$ is the Christoffel contraction evaluated at the current position $x$. This is a quadratic non-linearity in $v$, making the system fundamentally non-linear.

### 2.4 Linearization Techniques

Linearization of non-linear systems is a classical problem. For the Christoffel equation, we can linearize around the current operating point:

$$\Gamma^k_{ij}(x) v^i v^j \approx \Gamma^k_{ij}(x) v_0^i v_0^j + \left(\Gamma^k_{ij}(x) v_0^i + \Gamma^k_{ji}(x) v_0^j\right)(v^j - v_0^j)$$

For small perturbations, the constant term can be absorbed into the force input, and the Jacobian provides a linear approximation.

### 2.5 Multi-Scale Time in Neural Networks

Multi-scale time representations have been explored in various neural network architectures. WaveNet uses dilated convolutions with exponentially increasing dilation factors. Neural ODEs with adaptive solvers effectively use variable timesteps. Our approach introduces multi-scale through head-specific base time scales, enabling different heads to operate at different effective temporal resolutions.

---

## 3. Parallel Manifold Scan Layer

### 3.1 From Non-Linear to Linear Dynamics

The fundamental challenge is that the Christoffel equation is quadratic in velocity:

$$\frac{dv^k}{dt} = -\Gamma^k_{ij}(x) v^i v^j$$

The contraction $\Gamma^k_{ij}(x) v^i v^j$ is quadratic in $v$, which prevents direct application of linear system techniques.

Our approach is to predict the linearization parameters directly from the input force rather than computing Christoffel symbols. Given the input force $F_t$ at each timestep, we predict:

1. **Decay factor** $A_t$: Controls how much of the previous velocity is retained
2. **Input modulation** $B_t$: Represents the effective input contribution

The linearized dynamics are:

$$v_t = A_t v_{t-1} + B_t$$

This is fundamentally different from Christoffel-based dynamics but achieves similar representational goals.

### 3.2 Predicting Linearization Parameters

The decay factor $A_t$ is predicted from both the current position $x_t$ and the input force $F_t$, since geometric damping depends on the local curvature (encoded in $x_t$) as well as the applied force:

$$A_t = \sigma(W_A [x_t; F_t] + b_A)$$

where $[x_t; F_t]$ denotes concatenation. Values close to 1 indicate strong velocity persistence (weak decay), while values close to 0 indicate strong damping.

> **Remark.** Using $F_t$ alone to predict $A_t$ (as in a pure input-driven system) ignores the position-dependent curvature information. On a manifold, the damping of velocity depends on the local Christoffel symbols $\Gamma^k_{ij}(x)$, which are functions of position. Including $x_t$ in the prediction allows the network to learn position-dependent decay that approximates the true curvature-dependent behavior.

The input modulation $B_t$ is similarly predicted from both position and force:

$$B_t = W_B [x_t; F_t] + b_B$$

This represents the forcing term's effect on velocity, modulated by the local geometry.

### 3.3 Time Scale Modulation

To capture both short-range and long-range dependencies, we introduce multi-scale time initialization. Different heads operate at different base time scales, where the head at index $i$ has base time scale $s_i$. We use a geometric progression $s_i = \beta^i$ for head $i$, where $\beta > 1$ is a base factor.

The default $\beta = 1.5$ is chosen so that across $h$ heads, the ratio of the slowest to fastest time scale is $\beta^{h-1}$. For $h = 8$ heads, this gives a ratio of $\approx 17\times$, covering roughly one order of magnitude. The value $\beta = 1.5$ provides a good balance: larger values (e.g., 2.0) spread the scales too aggressively and leave gaps, while smaller values (e.g., 1.2) provide insufficient range. The base factor $\beta$ is treated as a tunable hyperparameter in our implementation.

This creates an effective "wormhole" structure where fast heads capture rapid dynamics and slow heads capture gradual trends. The final time step is modulated by both the learned timestep and the base scale:

$$\Delta t_t = \Delta t_{\text{learned}} \cdot \Delta t_{\text{base}}$$

### 3.4 Limitations of Linearization

The LTV approximation introduces a linearization error of order $\mathcal{O}(\|v - v_0\|^2)$, where $v_0$ is the linearization point. Specifically, the true quadratic Christoffel dynamics produce:

$$\Gamma^k_{ij} v^i v^j = \Gamma^k_{ij} v_0^i v_0^j + 2\Gamma^k_{ij} v_0^i (v^j - v_0^j) + \Gamma^k_{ij}(v^i - v_0^i)(v^j - v_0^j)$$

The linear approximation retains only the first two terms, discarding the quadratic remainder. This remainder is negligible when $\|v - v_0\| \ll \|v_0\|$, which is typically satisfied for small time steps or when the force $F_t$ does not cause large velocity jumps.

> **When the approximation breaks down:** In regions of very high curvature where $\|\Gamma\|$ is large, even small velocity perturbations can produce significant quadratic terms. In practice, the learned $A_t$ and $B_t$ compensate for this by absorbing some of the non-linear behavior into the parameterization. Nevertheless, the P-MLayer should be understood as a computational approximation to true geodesic dynamics, not an exact parallelization.

### 3.5 Parallel Scan Computation

Given sequences $\{A_t\}$ and $\{B_t\}$, the velocity sequence is computed using the parallel scan algorithm. The algorithm works by computing local prefix products of $A_t$ and weighted sums of $B_t$ using the prefix products, combining results across logarithmic levels.

The position sequence is similarly computed through a parallel scan, integrating velocity over time:

$$x_t = x_{t-1} + v_t \cdot \Delta t$$

### 3.6 Theoretical Analysis

**Proposition 1 (LTV Approximation)**: The linearized dynamics $\dot{v} = -D(F)v + F$ approximate the non-linear Christoffel dynamics $\dot{v}^k = -\Gamma^k_{ij}(x) v^i v^j$ to first order.

*Proof*: Linearizing $\Gamma^k_{ij}(x) v^i v^j$ around $v_0$ gives $\Gamma^k_{ij}(x) v^i v^j \approx \Gamma^k_{ij}(x) v_0^i v_0^j + J^k_j(v_0)(v^j - v_0^j)$, where $J^k_j(v_0) = 2 \Gamma^k_{ij}(x) v_0^i$. Setting $F = -\Gamma^k_{ij}(x) v_0^i v_0^j + J(v_0)v_0$ and $D = J$ yields the linear form. ∎

**Proposition 2 (Parallel Complexity)**: The parallel scan computes the LTV solution in $\mathcal{O}(\log N)$ time on parallel hardware.

*Proof*: The parallel scan algorithm processes $\log N$ levels, with each level performing $\mathcal{O}(N)$ work in parallel. The total parallel time is $\mathcal{O}(\log N)$. ∎

**Proposition 3 (Multi-Scale Coverage)**: Heads with base scales $s_i$ cover timescales from $\Delta t \cdot s_{\min}$ to $\Delta t \cdot s_{\max}$.

*Proof*: The effective timestep for head $i$ is $\Delta t_i = \Delta t \cdot s_i$. With exponentially increasing scales, the ratio of largest to smallest scale is $s_{\max}/s_{\min}$ where the ratio depends on the number of heads. ∎

---

## 4. Experimental Results

### 4.1 Experimental Setup

We evaluate the Parallel Manifold Scan Layer on sequence modeling and representation learning tasks. Baselines include standard sequential M-Layer, a chunked parallel version, and transformer attention.

All models are trained using Adam with learning rate $10^{-3}$ and batch size 256. Sequence length is 512 unless otherwise specified.

### 4.2 Training Speed Comparison

| Method | Training Time (s/epoch) | Speedup | Memory (GB) |
|--------|-------------------------|---------|-------------|
| Sequential M-Layer | 245 | 1.0× | 4.2 |
| Chunked Parallel | 112 | 2.2× | 5.8 |
| Transformer | 98 | 2.5× | 6.1 |
| **P-MLayer** | **52** | **4.7×** | **5.2** |

P-MLayer achieves 4.7× speedup over sequential processing while using less memory than alternative parallel approaches.

### 4.3 Accuracy Comparison

| Method | Perplexity | Accuracy | Geometric Preservation |
|--------|------------|----------|------------------------|
| Sequential M-Layer | 24.3 | 89.2\% | High |
| Chunked Parallel | 25.1 | 88.7\% | Medium |
| Transformer | 22.8 | 90.1\% | None |
| **P-MLayer** | **24.8** | **89.0\%** | **Medium-High** |

P-MLayer achieves accuracy comparable to the sequential baseline while providing significant speedup. The geometric preservation is assessed through curvature consistency metrics.

### 4.4 Scaling Analysis

We analyze scaling with sequence length:

| Sequence Length | Sequential (s) | P-MLayer (s) | Speedup |
|-----------------|----------------|--------------|---------|
| 128 | 62 | 18 | 3.4× |
| 256 | 124 | 31 | 4.0× |
| 512 | 245 | 52 | 4.7× |
| 1024 | 498 | 98 | 5.1× |

Speedup increases with sequence length, demonstrating the $\mathcal{O}(\log N)$ complexity advantage.

### 4.5 Multi-Scale Effect

| Configuration | Long-Range Accuracy | Short-Range Accuracy |
|---------------|---------------------|----------------------|
| Single scale | 82.3\% | 91.4\% |
| Multi-scale | **86.1\%** | **91.8\%** |

Multi-scale time initialization improves long-range accuracy without hurting short-range performance.

---

## 5. Discussion

### 5.1 Geometric Interpretation

The LTV approximation sacrifices some geometric fidelity for computational efficiency. The predicted decay factors $A_t$ implicitly encode a simplified notion of curvature, where "high curvature" regions correspond to strong decay. While not as rich as full Christoffel symbols, this approximation captures the essential effect of geometry on flow dynamics.

### 5.2 Limitations

The parallel scan requires knowing all $A_t$ and $B_t$ in advance, which means the input force must be available for all timesteps. This precludes autoregressive generation, which processes one timestep at a time. P-MLayer is therefore best suited for tasks where full sequences are available at once (e.g., training, bidirectional encoding).

### 5.3 Future Directions

Several extensions merit investigation: (1) hierarchical parallelization across multiple levels, (2) adaptive scale selection based on input content, (3) integration with full Christoffel dynamics for hybrid approaches, and (4) extension to variable-length sequences through masking.

---

## 6. Conclusion

We have introduced the Parallel Manifold Scan Layer, an efficient parallel implementation of geodesic flow dynamics using Linear Time-Varying system approximation. The key innovation is the linearization of Christoffel-based dynamics into a form amenable to parallel scan computation, reducing sequential complexity from $\mathcal{O}(N)$ to $\mathcal{O}(\log N)$.

Experimental results demonstrate that P-MLayer achieves comparable accuracy to sequential baselines while providing 3-5× speedup on modern GPU architectures. The multi-scale time initialization further improves performance on long-range dependencies.

The Parallel Manifold Scan Layer represents a step toward more efficient geometric deep learning, enabling the use of geodesic flow architectures on longer sequences and larger batch sizes. By combining insights from linear system theory and parallel algorithms, we unlock the computational potential of geometric representations.

---

## References

[1] Blelloch, G. E. (1990). Prefix Sums and Their Applications. Technical Report CMU-CS-90-190.

[2] He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.

[3] Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.

[4] Oord, A., et al. (2016). WaveNet: A Generative Model for Raw Audio. arXiv.

[5] Van den Oord, A., et al. (2018). Parallel WaveNet. ICLR.

---

## Appendix A: Parallel Scan Algorithm Details

The parallel scan algorithm proceeds in two phases:

**Up-Sweep Phase:**
1. Initialize working array with $(A_t, B_t)$
2. For each level $l = 0$ to $\log_2 N - 1$:
   - For each block starting at $k \cdot 2^{l+1}$:
     - Combine elements at $k \cdot 2^{l+1} + 2^l - 1$ and $k \cdot 2^{l+1} + 2^{l+1} - 1$

**Down-Sweep Phase:**
1. Initialize with identity at position 0
2. For each level $l = \log_2 N - 1$ to 0:
   - For each block:
     - Apply down-sweep combination

---

## Appendix B: LTV Derivation from Christoffel Dynamics

Starting from the Christoffel equation:

$$\frac{dv}{dt} = -\Gamma(v, v) = -U W^T (v \odot v)$$

where $\Gamma(v, v)$ is a shorthand for the Christoffel contraction $\Gamma^k_{ij} v^i v^j$, approximated by the low-rank form $U W^T (v \odot v)$.

We linearize around $v_0$:

$$U W^T (v \odot v) \approx U W^T (v_0 \odot v_0) + J(v_0) \cdot (v - v_0) + \mathcal{O}(\|v - v_0\|^2)$$

where $J$ is the Jacobian $J^k_j(v_0) = 2 U^k_a W^a_j v_0^j$ (using the symmetry of the quadratic form). Setting $F = -(U W^T (v_0 \odot v_0) - J(v_0) \cdot v_0)$ and $D = J$ gives:

$$\frac{dv}{dt} = -D \cdot v + F$$

This is the LTV form used in P-MLayer. The discarded quadratic remainder $\mathcal{O}(\|v - v_0\|^2)$ is the source of the linearization error discussed in §3.4.
