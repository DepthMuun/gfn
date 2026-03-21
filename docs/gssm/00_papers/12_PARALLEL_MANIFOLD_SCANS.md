# Parallel Manifold Scans: Logarithmic-Time Sequence Integration for Geodesic Flows

**Author:** Joaquin Stürtz  
**Date:** January 26, 2026

**Abstract**  
The sequential dependency of recursive architectures (e.g., RNNs, S4, Mamba) constitutes a primary bottleneck in high-throughput neural training, where token $t$ traditionally requires the completion of token $t-1$. We introduce **Parallel Manifold Scans**, an associative reformulation of the manifold update operator that enables $O(\log N)$ parallel depth for sequence trajectories. By expressing the discretized flow as a prefix-sum over affine propagators, we obtain a scan-compatible formulation that supports GPU acceleration via fused kernels. We demonstrate that geodesic flows, when linearized into a Linear Time-Varying (LTV) system, can be integrated across massive sequence lengths with logarithmic parallel complexity, bridging the gap between the constant-memory recursion of Geodesic Flow Networks (GFN) and the parallel training efficiency of attention-based models.



## 1. Introduction: The Serial Bottleneck in Manifold Dynamics

Geodesic Flow Networks (GFN) treat sequence processing as the integration of a trajectory on a Riemannian manifold. While this continuous framing provides superior memory stability and infinite-horizon tracking, the traditional numerical integration (e.g., Runge-Kutta or Symplectic integrators) is inherently serial:

$$ s_t = \text{Integrate}(s_{t-1}, F_t, \Delta t) $$

This $O(N)$ dependency prevents efficient scaling on modern parallel hardware (GPUs/TPUs). Parallel Manifold Scans solve this by revealing the associative structure hidden within the discretized manifold equations. The phase space state $s_t = (x_t^i, v_t^k)$ represents the position and velocity coordinates on the manifold $\mathcal{M}$ at time $t$.

## 2. Associative Reformulation of Geodesic Flows

### 2.1 Linearization into LTV Systems
To enable parallelization, we approximate the non-linear geodesic equation $\frac{D v^k}{dt} = -\Gamma^k_{ij}(x) v^i v^j$ as a Linear Time-Varying (LTV) system. For a sufficiently small step $\Delta t$, the update for velocity $v^k$ and position $x^i$ can be expressed as a first-order affine recurrence:

$$ v_t^k = A_t^{kj} v_{t-1}^j + B_t^k $$
$$ x_t^i = x_{t-1}^i + v_t^i \Delta t $$

where:
*   $A_t^{kj}$ is the **Retention Operator** (a matrix predicted from the input force $F^k_t$ that encodes the geometric coupling through the Christoffel symbols $\Gamma^k_{ij}(x_{t-1})$).
*   $B_t^k$ is the **Input Propagator** (the effective impulse applied to the state, including the external force and geometric effects).

The retention matrix $A_t$ is derived from a first-order approximation of the geodesic flow:

$$ A_t^{kj} \approx \delta^{kj} - \Delta t \cdot \Gamma^k_{ij}(x_{t-1}) v_{t-1}^i $$

This approximation linearizes the non-linear Christoffel term $\Gamma^k_{ij}(x) v^i v^j$ around the current state, enabling the LTV formulation while preserving the geometric character of the flow.

### 2.2 The Manifold Propagator Group
We define a manifold update as a pair $\mathcal{P}_t = (A_t, B_t)$. The composition of two consecutive updates $\mathcal{P}_{i+1} \circ \mathcal{P}_i$ is governed by the associative rule:

$$ (A_{i+1}, B_{i+1}) \circ (A_i, B_i) = (A_{i+1} A_i, A_{i+1} B_i + B_{i+1}) $$

where matrix multiplication is implied in the first component: $(A_{i+1} A_i)^{kj} = A_{i+1}^{kl} A_l^i$. **Theorem (Associativity):** The operation $\circ$ satisfies $(\mathcal{P}_k \circ \mathcal{P}_j) \circ \mathcal{P}_i = \mathcal{P}_k \circ (\mathcal{P}_j \circ \mathcal{P}_i)$. This algebraic property is the foundation of parallel sequence processing, as it allows the reduction of the entire sequence composition to a binary tree of associative operations that can be computed in parallel.

## 3. Parallel Integration Algorithms

### 3.1 Logarithmic-Depth Prefix Scans
By exploiting associativity, the entire sequence $s_{1:N}$ can be computed in $O(\log N)$ parallel steps using the **Hillis-Steele** or **Blelloch** algorithms. These algorithms exploit the structure of the propagator composition to compute all intermediate states simultaneously:

1.  **Up-Sweep (Reduction):** Build a tree of partial compositions $\mathcal{P}_{i:j} = \mathcal{P}_j \circ \cdots \circ \mathcal{P}_i$ representing the cumulative effect of updates from time $i$ to $j$.
2.  **Down-Sweep (Distribution):** Distribute prefixes to compute all final states $s_t$ simultaneously, using the cumulative propagators to update each position in parallel.

The total work remains $O(N \cdot D^2)$ (where $D$ is the manifold dimension), but the critical path is reduced to $O(\log N)$, enabling efficient GPU utilization.

### 3.2 Fused CUDA Implementation
For maximum efficiency, we implement a **Fused Parallel Scan** kernel. This kernel performs the following operations in a single GPU pass:
*   **Warp-Level Composition:** Uses `shfl_sync` primitives to compose operators within a thread warp, exploiting the associativity of matrix multiplication for $A$ and vector addition for $B$.
*   **Shared Memory Buffering:** Uses a block-level Blelloch scan to handle chunks of the sequence, aggregating propagators $\mathcal{P}_t = (A_t, B_t)$ in shared memory before writing results.
*   **Multi-Scale Integration:** Each head in the manifold processes the scan at a different base time-scale $\Delta t$, allowing the model to capture both high-frequency local dependencies (short $\Delta t$) and low-frequency global structures (long $\Delta t$). The Christoffel symbols $\Gamma^k_{ij}(x)$ are recomputed at each scale to maintain geometric accuracy.

## 4. Complexity and Performance Analysis

| Metric | Sequential Integration | Parallel Manifold Scan |
| : | : | : |
| **Compute Complexity** | $O(N \cdot D^2)$ | $O(N \cdot D^2)$ (Work-efficient) |
| **Parallel Depth** | $O(N)$ | $O(\log N)$ |
| **Memory Footprint** | $O(D)$ (Inference) | $O(N \cdot D^2)$ (Training) |
| **Hardware Utilization** | Low (Serial) | High (Massively Parallel) |

The transition from $O(N)$ to $O(\log N)$ depth allows training on sequences of length $10^5$ and beyond, where traditional RNNs would fail due to the time-step bottleneck. The work complexity remains the same, ensuring that parallel scans do not increase total computational requirements while dramatically reducing wall-clock time.

## 5. Geometric and Physical Implications

### 5.1 Emergent Symplecticity
Unlike explicit symplectic integrators that enforce $\det\left(\frac{\partial s_t}{\partial s_{t-1}}\right) = 1$ through the Störmer-Verlet scheme, Parallel Manifold Scans rely on the learned $A_t$ to maintain stability. Any volume-preserving behavior is emergent from the loss functions (e.g., Hamiltonian regularization) that encourage the propagator matrices to satisfy $A^T A = I$ (orthogonality) and $B = 0$ (no input), which together imply determinant preservation. The Christoffel-symbol-induced geometric structure is preserved through the careful design of the retention operator $A_t$.

### 5.2 The "Wormhole" Effect
By modulating $A_t \to I$ (identity matrix), the model can create "lossless" conduits where information travels through the manifold without decay. In the parallel scan framing, this corresponds to long-range identity compositions $\mathcal{P}_{i:j} \approx (I, 0)$, allowing a token at $t=1$ to influence $t=1000$ in a single $\log N$ jump without geometric distortion. The Christoffel symbols effectively vanish along this trajectory, representing a locally flat region of the manifold that permits direct information transmission.

### 5.3 Phase Space Trajectory Preservation
The parallel scan formulation maintains the phase space trajectory structure of the original continuous system. The composed propagator $\mathcal{P}_{1:N}$ represents an approximation of the flow map $\Phi_{0:T}$ that would be obtained by integrating the geodesic equation $\frac{D v^k}{dt} = -\Gamma^k_{ij}(x) v^i v^j$ continuously. This ensures that the physical interpretation of the model as a geodesic flow remains valid even under parallel computation.

## 6. Conclusion

Parallel Manifold Scans provide a rigorous bridge between the continuous physics of GFN and the requirements of modern deep learning. By reformulating manifold dynamics as an associative prefix-sum over propagators $\mathcal{P}_t = (A_t, B_t)$ that encode the Christoffel-symbol-induced geometric coupling, we achieve $O(\log N)$ training speed without sacrificing the constant-memory, infinite-context advantages of recurrent flows. The linear time-varying (LTV) approximation preserves the geometric character of the original geodesic flow while enabling efficient parallel computation. This architecture represents a new paradigm for scalable, physics-informed sequence modeling that combines the best of continuous dynamics and parallel deep learning.



**References**

[1] Blelloch, G. E. (1990). *Prefix Sums and Their Applications*. Technical Report, CMU.  
[2] Hillis, W. D., & Steele Jr, G. L. (1986). *Data parallel algorithms*. Communications of the ACM.  
[3] Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv.  
[4] Martin, E., & Cundy, C. (2018). *Parallelizing Linear Recurrent Neural Nets Over Sequence Length*. ICLR.  
[5] Katharopoulos, A., et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML.  
[6] Smith, J. T., et al. (2023). *S5: Real-Time Sequence Modeling with Selective State Spaces*. ICLR.
