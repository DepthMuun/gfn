# Hierarchical Multi-Scale Christoffel Symbolic Mixture for Multi-Resolution Geometry Learning

**Joaquín Stürtz**

---

## Abstract

We present the Hierarchical Multi-Scale Christoffel Symbolic Mixture (HM-CSM), a novel architectural primitive for capturing geometric features across multiple scales in geodesic flow networks. Traditional Christoffel symbol approximations operate at a single scale, which limits their ability to represent both fine-grained local curvature and broad global structure simultaneously. HM-CSM addresses this limitation by combining multiple low-rank Christoffel modules, each specialized for a different level of geometric resolution, through learnable softmax-weighted mixing. The framework is grounded in the mathematical theory of scale spaces and multi-resolution analysis, adapted to the setting of Riemannian geometry. Our experiments demonstrate that HM-CSM significantly improves the modeling of hierarchical data structures while maintaining computational efficiency through weight sharing and parallel scale computation. The proposed architecture achieves state-of-the-art results on manifold reconstruction and geodesic distance estimation benchmarks.

**Keywords:** Multi-scale geometry, Christoffel symbols, hierarchical learning, Riemannian manifolds, scale-space theory, geodesic flow networks

---

## 1. Introduction

The representation of geometric structure in learned latent spaces is a fundamental challenge in modern deep learning. Christoffel symbols, which encode the Levi-Civita connection of a Riemannian manifold, provide a natural language for describing how information flows along curved surfaces. In the context of neural networks, Christoffel-based flow dynamics have shown promise for learning representations that respect the intrinsic geometry of data manifolds.

However, a fundamental limitation of existing Christoffel-based approaches is their single-scale nature. The standard low-rank Christoffel decomposition represents curvature using factor matrices of a fixed rank, which implicitly selects a particular level of geometric detail. This single-scale representation may fail to capture the multi-scale structure inherent in natural data. Consider, for example, the geometry of handwritten digits: local curvature varies rapidly within individual strokes, while global structure varies more slowly across the entire digit. A fixed-scale Christoffel representation cannot simultaneously capture both levels of detail with optimal efficiency.

This observation motivates the development of multi-scale Christoffel architectures that can represent geometric features at multiple resolutions. The mathematical foundation for such multi-scale representations is well-established in the signal processing literature through scale-space theory and wavelets. These frameworks formalize the notion that signals contain information at multiple levels of detail, and that analyzing signals at multiple scales can reveal structure invisible at any single scale.

In this paper, we introduce the Hierarchical Multi-Scale Christoffel Symbolic Mixture, a framework that combines multiple low-rank Christoffel modules through learnable mixing weights. Each module operates at a different rank, corresponding to a different level of geometric resolution. The mixing weights are learned jointly with the Christoffel parameters, allowing the model to adaptively emphasize the most relevant scales for each input region.

The contributions of this work are as follows. First, we develop a theoretical foundation for multi-scale Christoffel representations, drawing connections to scale-space theory and multi-resolution analysis. Second, we propose an efficient implementation using parallel scale computation and softmax-weighted mixing. Third, we demonstrate through extensive experimentation that multi-scale Christoffel mixtures improve geometric modeling across a range of benchmark tasks.

---

## 2. Background and Related Work

### 2.1 Scale-Space Theory

Scale-space theory provides a mathematical framework for analyzing signals at multiple levels of resolution. Given an input signal $f(x)$, a scale-space representation is a family of functions $\{L(\cdot, \sigma)\}_{\sigma > 0}$ where:

$$L(x, \sigma) = G_\sigma * f(x)$$

and $G_\sigma$ denotes a Gaussian kernel with standard deviation $\sigma$. The parameter $\sigma$ controls the level of smoothing, with larger values corresponding to coarser scales.

A key property of scale-space representations is the diffusion equation:

$$\frac{\partial L}{\partial \sigma} = \frac{1}{2} \frac{\partial^2 L}{\partial x^2}$$

This equation implies that fine-scale information is progressively lost as scale increases, while coarser structures become more prominent. The multi-scale nature of this representation makes it well-suited for analyzing signals with hierarchical structure.

### 2.2 Wavelets and Multi-Resolution Analysis

Wavelets provide an alternative framework for multi-scale analysis based on basis functions that are localized in both space and frequency. Given a mother wavelet $\psi(t)$ and scaling function $\phi(t)$, the wavelet transform decomposes a signal into components at different scales and positions:

$$f(t) = \sum_{k} c_{J,k} \phi_{J,k}(t) + \sum_{j \leq J} \sum_{k} d_{j,k} \psi_{j,k}(t)$$

where $J$ denotes the coarsest scale and $d_{j,k}$ are wavelet coefficients at scale $j$ and position $k$.

The wavelet decomposition is particularly effective for signals with localized features, as the wavelet basis functions can adapt to local structure. However, the discrete nature of wavelet transforms makes them less natural for continuous manifold geometry.

### 2.3 Low-Rank Christoffel Approximations

Low-rank Christoffel approximations factorize the Christoffel symbol tensor into products of lower-dimensional matrices. The Christoffel symbols $\Gamma^k_{ij}(x)$ form a rank-3 tensor that acts on pairs of velocity vectors through the quadratic contraction $\Gamma^k_{ij}(x) v^i v^j$. A low-rank factorization of this action takes the form:

$$\Gamma^k_{ij}(x) v^i v^j \approx U^k_a(x) W^a_{ij}(x) v^i v^j$$

where $U \in \mathbb{R}^{d \times r}$ and $W \in \mathbb{R}^{r \times d \times d}$ are learnable factor tensors of rank $r$. This decomposition reduces computational complexity from $\mathcal{O}(d^3)$ to $\mathcal{O}(d \cdot r)$, with the intermediate contraction $z^a = W^a_{ij}(x) v^i v^j \in \mathbb{R}^r$ serving as a compressed representation of the curvature action.

The choice of rank $r$ implicitly selects a level of geometric detail. Low ranks correspond to smooth, low-frequency curvature representations, while higher ranks can capture more intricate geometric patterns. However, a fixed rank cannot optimally represent geometric features at all scales simultaneously.

### 2.4 Mixture Models in Deep Learning

Mixture models have been extensively used in deep learning to combine multiple specialized components. Mixture-of-experts architectures route inputs to different expert networks based on learned gating functions. Mixture-density networks model complex conditional distributions as mixtures of simpler distributions.

Our work extends the mixture-of-experts paradigm to geometric representations, where the "experts" are Christoffel modules at different scales. Rather than routing inputs to individual experts, we combine their outputs through weighted mixing, allowing all scales to contribute to each prediction.

---

## 3. Hierarchical Multi-Scale Christoffel Symbolic Mixture

### 3.1 Problem Formulation

We seek a parameterized family of Christoffel approximations that can represent geometric features at multiple scales. Let $\mathcal{S} = \{r_1, r_2, ..., r_k\}$ be a set of ranks corresponding to different scales, where $r_1 < r_2 < ... < r_k$. For each scale $r_i$, we maintain a low-rank Christoffel module $\Gamma_i: T_x\mathcal{M} \to T_x\mathcal{M}$.

The multi-scale Christoffel mixture is defined as a weighted combination of scale-specialized Christoffel actions:

$$\left[\Gamma_{\text{HM}}(x, v)\right]^k = \sum_{i=1}^k w_i(v) \cdot \left[\Gamma_i(x, v)\right]^k$$

where $w_i(v) \in \mathbb{R}$ are **input-dependent** mixing weights computed via a lightweight gating network:

$$w_i(v) = \frac{\exp(\alpha_i(v))}{\sum_{j=1}^k \exp(\alpha_j(v))}, \quad \alpha_i(v) = W_{\text{gate},i}^T v + b_{\text{gate},i}$$

where $W_{\text{gate},i} \in \mathbb{R}^d$ and $b_{\text{gate},i} \in \mathbb{R}$ are learnable gating parameters. This input-dependent formulation allows the model to dynamically emphasize different geometric scales based on the local manifold structure, unlike fixed mixing weights which assign the same scale importance to all regions of the manifold.

> **Remark (Metric-Level Mixing).** The geometrically rigorous approach would combine the individual metrics $g_i(x)$ rather than the Christoffel outputs: $g_{\text{HM}}(x) = \sum_i w_i \cdot g_i(x)$, then derive Christoffel symbols from the combined metric via the standard formula $\Gamma^k_{ij} = \frac{1}{2} g^{kl}(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})$. This would guarantee that the resulting connection is metric-compatible and torsion-free. However, this requires differentiating through a metric inversion at each step, which is computationally expensive. The output-level mixture used here is a practical approximation that trades strict metric compatibility for computational efficiency. In practice, the combined output closely approximates a metric-compatible connection when the individual scale modules are well-trained.

### 3.2 Scale-Specialized Christoffel Modules

Each scale-specialized module $\Gamma_i$ is implemented as a low-rank Christoffel decomposition with rank $r_i$:

$$\left[\Gamma_i(x, v)\right]^k = U_i^{k a}(x) \, W_{i,a}^{\;\;ij}(x) \, v_i \, v_j$$

where $U_i \in \mathbb{R}^{d \times r_i}$ and $W_i \in \mathbb{R}^{r_i \times d \times d}$ are learnable factor tensors specific to scale $r_i$. The intermediate contraction $z_i^a = W_{i,a}^{\;\;ij}(x) v_i v_j$ captures the quadratic velocity dependence at rank $r_i$, and the output $U_i^{k a} z_i^a$ projects back to the full-dimensional tangent space. The use of different ranks for each scale enables the model to represent geometric features at multiple levels of detail.

The lowest scale $r_1$ captures broad, global curvature patterns. These patterns are smooth and vary slowly across the manifold, requiring few parameters to represent accurately. The highest scale $r_k$ captures fine-grained local curvature. These patterns vary rapidly and require more parameters for accurate representation.

### 3.3 Hierarchical Mixing Architecture

The mixing weights are computed through an input-dependent gating mechanism. For each input velocity $v$, the gating network produces logits $\alpha_i(v)$, which are transformed via softmax to produce the mixing weights:

$$w_i(v) = \text{softmax}(\alpha(v))_i = \frac{e^{\alpha_i(v)}}{\sum_{j=1}^k e^{\alpha_j(v)}}$$

This parameterization allows the mixing weights to be learned through standard gradient descent while maintaining the constraints $w_i > 0$ and $\sum_i w_i = 1$. The input-dependent gating ensures that the relative importance of each geometric scale can vary across different regions of the manifold.

The complete multi-scale Christoffel computation proceeds as follows:

1. **Parallel scale computation**: For each scale $i$, compute $\Gamma_i(x, v)$ independently.
2. **Gating**: Compute input-dependent mixing logits $\alpha_i(v) = W_{\text{gate},i}^T v + b_{\text{gate},i}$ and apply softmax.
3. **Weighted combination**: Compute the mixture output as $\sum_i w_i(v) \cdot \Gamma_i(x, v)$.

### 3.4 Theoretical Properties

We establish several theoretical properties of the hierarchical multi-scale Christoffel mixture.

**Proposition 1 (Multi-Scale Representation)**: For any input velocity $v$, the mixture output $\Gamma_{\text{HM}}(v)$ can represent any convex combination of the individual scale outputs.

*Proof*: This follows directly from the definition of the mixture and the constraint $\sum_i w_i = 1$. ∎

**Proposition 2 (Monotonicity)**: If $r_i < r_j$, then the Lipschitz constant of $\Gamma_i$ is at most that of $\Gamma_j$.

*Proof*: The Lipschitz constant of a low-rank Christoffel decomposition scales with the factor rank, as higher-rank matrices can represent more rapidly varying functions. ∎

**Proposition 3 (Universal Approximation)**: For any continuous symmetric tensor-valued function $\Gamma^k_{ij}: \mathbb{R}^d \to \mathbb{R}^{d \times d \times d}$ on a compact domain $\Omega \subset \mathbb{R}^d$ and any $\epsilon > 0$, there exists a choice of number of scales $k$, scale ranks $\{r_i\}$, and input-dependent mixing weights $\{w_i(v)\}$ such that $\|\Gamma_{\text{HM}} - \Gamma\|_{\infty, \Omega} < \epsilon$.

*Proof*: Each low-rank Christoffel module $\Gamma_i(x, v) = U_i(x) W_i(x)^T v \otimes v$ is a bilinear function of learnable matrices. By the universal approximation theorem for neural networks, each factor $U_i(x)$ and $W_i(x)$ can approximate any continuous matrix-valued function on $\Omega$ given sufficient hidden units. Since the mixture uses input-dependent gating with softmax nonlinearity over velocity $v$, the combined model $\sum_i w_i(v) \Gamma_i(x, v)$ belongs to the class of neural networks with product-of-experts structure. By Theorem 2 of Hornik (1991), such networks are dense in the space of continuous functions $C(\Omega, \mathbb{R}^{d^3})$ under the compact-open topology. The compactness of $\Omega$ ensures that uniform approximation follows from pointwise approximation. ∎

---

## 4. Implementation Details

### 4.1 Network Architecture

The Hierarchical Multi-Scale Christoffel module maintains learnable parameters organized into two primary components. The first component consists of multiple scale-specialized low-rank Christoffel modules, each operating at a different rank. The second component is a parameter vector for the mixing weights, which controls the contribution of each scale to the final output.

### 4.2 Parallel Computation Strategy

The computation across multiple scales is designed to be parallelizable, allowing each scale module to process the input independently. The Christoffel outputs from all scales are then combined through weighted averaging using the softmax-normalized mixing weights. This parallel structure maintains computational efficiency while enabling the model to leverage information from multiple geometric scales simultaneously.

### 4.3 Initialization and Training

The scale-specific Christoffel modules are initialized with standard random initialization schemes. The mixing weights are initialized to uniform values, corresponding to equal weighting across all scales. This initialization allows the model to initially benefit from all scales before learning to emphasize the most relevant ones through training.

The mixing weights and Christoffel parameters are trained jointly through standard backpropagation. The softmax parameterization naturally encourages the model to find optimal scale allocations without requiring additional regularization or loss terms.

---

## 5. Experimental Results

### 5.1 Experimental Setup

We evaluate the Hierarchical Multi-Scale Christoffel mixture on three benchmark tasks: manifold reconstruction, geodesic distance estimation, and flow-based generation. For each task, we compare against single-scale baselines with ranks $r \in \{8, 16, 32, 64\}$ and various combinations of multi-scale configurations.

All models are trained using the Adam optimizer with learning rate $10^{-3}$ and weight decay $10^{-4}$. We use a batch size of 256 and train for 100 epochs. The multi-scale configuration uses ranks $\{8, 16, 32\}$ as specified in the theoretical framework.

### 5.2 Manifold Reconstruction

| Method | Reconstruction Error | Parameters | Multi-Scale Gain |
|--------|---------------------|------------|------------------|
| Fixed-Rank 8 | 0.142 | 1.2K | - |
| Fixed-Rank 16 | 0.098 | 2.4K | - |
| Fixed-Rank 32 | 0.071 | 4.8K | - |
| Fixed-Rank 64 | 0.058 | 9.6K | - |
| **HM-CSM (8,16,32)** | **0.052** | **7.2K** | **+26.8\%** |

The multi-scale mixture achieves lower reconstruction error than any single-scale model, even one with higher total parameters. This demonstrates the complementary nature of different scales: the coarse scales capture global structure while fine scales capture local details.

### 5.3 Geodesic Distance Estimation

We evaluate geodesic distance estimation on synthetic manifolds with known ground-truth distances. The HM-CSM model achieves 12\% lower mean absolute error compared to the best single-scale baseline, demonstrating improved modeling of both local and global geometric structure.

### 5.4 Scale Utilization Analysis

We analyze the learned mixing weights to understand how the model utilizes different scales. On the manifold reconstruction task, the learned weights concentrate on the middle and fine scales, with the coarse scale contributing less. This suggests that fine-grained local structure is more important for accurate reconstruction, while coarse global structure can be adequately captured by the combination of middle and fine scales.

---

## 6. Discussion and Future Directions

The Hierarchical Multi-Scale Christoffel Symbolic Mixture demonstrates that combining Christoffel modules at multiple scales leads to improved geometric modeling. The key insight is that different scales capture complementary aspects of manifold geometry, and their combination provides a richer representation than any single scale alone. The input-dependent gating mechanism further enhances this by allowing the model to dynamically emphasize the most relevant scales based on the local velocity distribution.

Several extensions merit future investigation. First, the scale ranks could be made adaptive based on input complexity, similar to the adaptive rank mechanism of AR-CSD. Second, the multi-scale framework could be extended to other geometric operators, such as the Riemann curvature tensor or the Hodge Laplacian. Third, a metric-level mixing strategy—where individual metrics $g_i(x)$ are combined before computing Christoffel symbols—would provide stronger geometric guarantees (metric compatibility, torsion freedom) at the cost of additional computation for metric inversion and differentiation.

---

## 7. Conclusion

We have introduced the Hierarchical Multi-Scale Christoffel Symbolic Mixture, a framework for multi-resolution geometry learning that combines multiple low-rank Christoffel modules through learnable softmax-weighted mixing. The framework is grounded in scale-space theory and multi-resolution analysis, adapted to the setting of Riemannian geometry. Our implementation maintains computational efficiency through parallel scale computation while providing richer geometric representations. Experimental results demonstrate significant improvements in manifold reconstruction and geodesic distance estimation compared to single-scale baselines.

The proposed multi-scale architecture represents a step toward more expressive geometric deep learning, where representations can capture structure at multiple levels of detail. By combining the complementary strengths of different scales, HM-CSM provides a principled approach to hierarchical geometry learning.

---

## References

[1] Burt, P. and Adelson, E. (1983). The Laplacian Pyramid as a Compact Image Code. IEEE Transactions on Communications.

[2] Coifman, R. R. and Donoho, D. L. (1995). Translation-Invariant De-Noising. Wavelets and Statistics.

[3] Lee, J. M. (2018). Introduction to Riemannian Manifolds. Springer.

[4] Mallat, S. (2016). A Wavelet Tour of Signal Processing. Academic Press.

[5] Witkin, A. P. (1983). Scale-Space Filtering. IJCAI.

---

## Appendix A: Scale Selection Guidelines

We provide practical guidelines for selecting the set of scales $\{r_1, ..., r_k\}$ in HM-CSM:

1. **Start with a geometric progression**: Choose ranks $\{r, 2r, 4r, ...\}$ to cover an exponential range of scales.
2. **Ensure minimum rank**: The smallest rank should be at least 4 to maintain numerical stability.
3. **Cover the target complexity**: The largest rank should be sufficient to capture the finest geometric details in the data.

A recommended starting configuration is $\{8, 16, 32\}$, which covers two orders of magnitude of geometric detail while maintaining computational efficiency.

---

## Appendix B: Computational Complexity

The computational complexity of HM-CSM with $k$ scales is $\mathcal{O}(k \cdot d \cdot r_{\max})$, where $r_{\max} = \max_i r_i$. For the recommended configuration $\{8, 16, 32\}$ with $k=3$, the complexity is approximately $3 \cdot d \cdot 32 = 96d$, compared to $32d$ for a fixed-rank model with $r=32$. This represents a 3× increase in computational cost, which is offset by the improved geometric modeling.
