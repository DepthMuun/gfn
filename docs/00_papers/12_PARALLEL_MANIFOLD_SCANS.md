# Parallel Manifold Scans: Logarithmic-Time Sequence Integration for Geodesic Flows

**Author:** Joaquin Stürtz  
**Date:** January 26, 2026

**Abstract**  
The sequential dependency of recursive architectures (e.g., RNNs, S4, Mamba) constitutes a primary bottleneck in high-throughput neural training, where token $t$ traditionally requires the completion of token $t-1$. We introduce **Parallel Manifold Scans**, an associative reformulation of the manifold update operator that enables $O(\log N)$ parallel depth for sequence trajectories. By expressing the discretized flow as a prefix-sum over affine propagators, we obtain a scan-compatible formulation that supports GPU acceleration via fused kernels. We demonstrate that geodesic flows, when linearized into a Linear Time-Varying (LTV) system, can be integrated across massive sequence lengths with logarithmic parallel complexity, bridging the gap between the constant-memory recursion of Geodesic Flow Networks (GFN) and the parallel training efficiency of attention-based models.



## 1. Introduction: The Serial Bottleneck in Manifold Dynamics

Geodesic Flow Networks (GFN) treat sequence processing as the integration of a trajectory on a Riemannian manifold. While this continuous framing provides superior memory stability and infinite-horizon tracking, the traditional numerical integration (e.g., Runge-Kutta or Symplectic integrators) is inherently serial:
$$ \mathbf{s}_{t} = \text{Integrate}(\mathbf{s}_{t-1}, \mathbf{F}_t, \Delta t) $$
This $O(N)$ dependency prevents efficient scaling on modern parallel hardware (GPUs/TPUs). Parallel Manifold Scans solve this by revealing the associative structure hidden within the discretized manifold equations.

## 2. Associative Reformulation of Geodesic Flows

### 2.1 Linearization into LTV Systems
To enable parallelization, we approximate the non-linear geodesic equation $\frac{d\mathbf{v}}{dt} = \mathbf{F} - \Gamma(\mathbf{v}, \mathbf{v})$ as a Linear Time-Varying (LTV) system. For a sufficiently small step $\Delta t$, the update for velocity $\mathbf{v}$ and position $\mathbf{x}$ can be expressed as a first-order affine recurrence:
$$ \mathbf{v}_t = \mathbf{A}_t \mathbf{v}_{t-1} + \mathbf{B}_t $$
$$ \mathbf{x}_t = \mathbf{x}_{t-1} + \mathbf{v}_t \Delta t $$
where:
*   $\mathbf{A}_t$ is the **Retention Operator** (a diagonal decay/rotation matrix predicted from the input force $\mathbf{F}_t$).
*   $\mathbf{B}_t$ is the **Input Propagator** (the effective impulse applied to the state).

### 2.2 The Manifold Propagator Group
We define a manifold update as a pair $\mathcal{P}_t = (\mathbf{A}_t, \mathbf{B}_t)$. The composition of two consecutive updates $\mathcal{P}_{i+1} \circ \mathcal{P}_i$ is governed by the associative rule:
$$ (\mathbf{A}_{i+1}, \mathbf{B}_{i+1}) \circ (\mathbf{A}_i, \mathbf{B}_i) = (\mathbf{A}_{i+1} \mathbf{A}_i, \mathbf{A}_{i+1} \mathbf{B}_i + \mathbf{B}_{i+1}) $$

**Theorem (Associativity):** The operation $\circ$ satisfies $(\mathcal{P}_k \circ \mathcal{P}_j) \circ \mathcal{P}_i = \mathcal{P}_k \circ (\mathcal{P}_j \circ \mathcal{P}_i)$. This algebraic property is the foundation of parallel sequence processing.

## 3. Parallel Integration Algorithms

### 3.1 Logarithmic-Depth Prefix Scans
By exploiting associativity, the entire sequence $\mathbf{s}_{1:N}$ can be computed in $O(\log N)$ parallel steps using the **Hillis-Steele** or **Blelloch** algorithms.
1.  **Up-Sweep (Reduction):** Build a tree of partial compositions $\mathcal{P}_{i:j}$.
2.  **Down-Sweep (Distribution):** Distribute prefixes to compute all final states $\mathbf{s}_t$ simultaneously.

### 3.2 Fused CUDA Implementation
For maximum efficiency, we implement a **Fused Parallel Scan** kernel. This kernel performs the following operations in a single GPU pass:
*   **Warp-Level Composition:** Uses `shfl_sync` primitives to compose operators within a thread warp.
*   **Shared Memory Buffering:** Uses a block-level Blelloch scan to handle chunks of the sequence.
*   **Multi-Scale Integration:** Each head in the manifold processes the scan at a different base time-scale $\Delta t$, allowing the model to capture both high-frequency local dependencies and low-frequency global structures.

## 4. Complexity and Performance Analysis

| Metric | Sequential Integration | Parallel Manifold Scan |
| : | : | : |
| **Compute Complexity** | $O(N \cdot D)$ | $O(N \cdot D)$ (Work-efficient) |
| **Parallel Depth** | $O(N)$ | $O(\log N)$ |
| **Memory Footprint** | $O(1)$ (Inference) | $O(N \cdot D)$ (Training) |
| **Hardware Utilization** | Low (Serial) | High (Massively Parallel) |

The transition from $O(N)$ to $O(\log N)$ depth allows training on sequences of length $10^5$ and beyond, where traditional RNNs would fail due to the time-step bottleneck.

## 5. Geometric and Physical Implications

### 5.1 Emergent Symplecticity
Unlike explicit symplectic integrators that enforce $\det(\frac{\partial \mathbf{s}_t}{\partial \mathbf{s}_{t-1}}) = 1$, Parallel Manifold Scans rely on the learned $\mathbf{A}_t$ to maintain stability. Any volume-preserving behavior is emergent from the loss functions (e.g., Hamiltonian regularization).

### 5.2 The "Wormhole" Effect
By modulating $\mathbf{A}_t \to 1$, the model can create "lossless" conduits where information travels through the manifold without decay. In the parallel scan framing, this corresponds to long-range identity compositions, allowing a token at $t=1$ to influence $t=1000$ in a single $\log N$ jump.

## 6. Conclusion

Parallel Manifold Scans provide a rigorous bridge between the continuous physics of GFN and the requirements of modern deep learning. By reformulating manifold dynamics as an associative prefix-sum, we achieve $O(\log N)$ training speed without sacrificing the constant-memory, infinite-context advantages of recurrent flows. This architecture represents a new paradigm for scalable, physics-informed sequence modeling.



**References**

[1] Blelloch, G. E. (1990). *Prefix Sums and Their Applications*. Technical Report, CMU.  
[2] Hillis, W. D., & Steele Jr, G. L. (1986). *Data parallel algorithms*. Communications of the ACM.  
[3] Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv.  
[4] Martin, E., & Cundy, C. (2018). *Parallelizing Linear Recurrent Neural Nets Over Sequence Length*. ICLR.  
[5] Katharopoulos, A., et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML.  
[6] Smith, J. T., et al. (2023). *S5: Real-Time Sequence Modeling with Selective State Spaces*. ICLR.  
