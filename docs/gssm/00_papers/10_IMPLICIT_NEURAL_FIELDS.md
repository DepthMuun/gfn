# Implicit Neural Fields: Continuous Functional Representations for Massive Symbol Vocabularies

**Author:** Joaquin Stürtz  
**Date:** January 26, 2026

**Abstract**  
Contemporary neural network architectures depend on discrete embedding layers whose memory complexity scales linearly with vocabulary size ($O(V)$). This scaling imposes a critical bottleneck for models operating with massive multilingual vocabularies or high-resolution scientific data. We present **Implicit Neural Embeddings (INF)**, a framework where the discrete lookup table is replaced by a continuous neural field $\Psi$ defined on a low-dimensional manifold. Using Sinusoidal Representation Networks (SIREN), we demonstrate that large-scale vocabularies can be compressed into a constant number of parameters $O(1)$ with respect to $V$, while inducing a smooth metric topology between symbols. This approach not only drastically reduces the memory footprint but enables semantic interpolation and reasoning over unseen symbols through the continuity of the functional field.



## 1. Introduction: The Problem of Symbolic Discretization

In traditional deep learning, each symbol or token $i$ in a vocabulary $\mathcal{V}$ is associated with an independent vector $\mathbf{E}_i \in \mathbb{R}^D$. This tabular representation implicitly assumes that the semantic space is a collection of isolated and disjoint points. As $V$ grows to millions of tokens, the embedding matrix dominates the model's parameter budget, limiting deployment on memory-constrained hardware.

We propose a paradigm shift: treating the vocabulary as a **Continuous Semantic Field**. Instead of storing vectors, we learn a function $\Psi$ that maps coordinates $\mathbf{c} \in \mathcal{C}$ in a low-dimensional space to dense vector representations. The coordinate manifold $\mathcal{C}$ inherits a natural metric structure from the embedding space through the pullback metric induced by the neural field $\Psi$.



## 2. Mathematical Framework of Implicit Neural Fields

### 2.1 The Functional Mapping Function
We define the embedding of a token $i$ as the value of a neural field evaluated at its corresponding coordinate $\mathbf{c}_i$:

$$ \mathbf{E}_i = \Psi(\mathbf{c}_i; \Theta) $$

where $\Psi: \mathcal{C} \to \mathbb{R}^D$ is a multi-layer perceptron (MLP) parameterized by $\Theta$, and $\mathbf{c}_i$ is a low-rank coordinate vector ($d \ll D$) lying on the coordinate manifold $\mathcal{C}$. In this scheme, knowledge about vocabulary structure is stored in the network weights $\Theta$, while token identities are preserved in the coordinate space $\mathcal{C}$. The neural field $\Psi$ is designed to be smooth (continuously differentiable) to ensure that nearby coordinates produce semantically similar embeddings.

### 2.2 Periodic Activations and High-Frequency Detail
Standard ReLU-based MLPs suffer from "spectral bias," where the network prioritizes learning low-frequency functions, failing to capture the sharp discontinuities necessary to distinguish between thousands of unique symbols in a small coordinate space. To mitigate this, we adopt architectures with periodic activation functions:

$$ \phi(x) = \sin(\omega_0 \cdot x) $$

where $\omega_0$ is a frequency factor controlling the field's bandwidth. SIREN networks enable the neural field to capture high-frequency details and represent complex functions with superior accuracy compared to traditional architectures, maintaining the high-order differentiability required for physics-based optimizations. The periodic activations ensure that the neural field can represent the full spectrum of semantic variations across the vocabulary manifold.



## 3. Numerical Stability and Spectral Initialization

The convergence of networks with sinusoidal activations is extremely sensitive to weight scaling. To ensure signal variance remains constant across layers, we use an initialization scheme where weights $W$ are drawn from a distribution scaled by the frequency:

$$ W \sim \mathcal{U}\left(-\frac{\sqrt{6/n}}{\omega_0}, \frac{\sqrt{6/n}}{\omega_0}\right) $$

where $n$ denotes the fan-in (number of input connections) for each weight. This initialization prevents phase collapse and allows the neural field to distribute uniformly over the coordinate space, maximizing the model's expressive capacity to represent the complete vocabulary. The scaling by $\omega_0$ ensures that the gradient magnitudes remain stable throughout the network, regardless of the frequency of the sinusoidal activations.



## 4. Metric Topology and Semantic Interpolation

Unlike lookup tables, INFs induce a natural topological structure. If two tokens $i, j$ have coordinates $\mathbf{c}_i, \mathbf{c}_j$ that are close in the coordinate manifold $\mathcal{C}$ (with respect to the induced metric $g_{kl}(\mathbf{c}) = \partial_k \Psi \cdot \partial_l \Psi$), their resulting embeddings will be semantically similar due to the continuity of $\Psi$. The induced metric tensor $g_{kl}(\mathbf{c})$ on $\mathcal{C}$ is given by the pullback of the Euclidean metric on $\mathbb{R}^D$:

$$ g_{kl}(\mathbf{c}) = \sum_{\alpha=1}^D \frac{\partial \Psi^\alpha}{\partial c^k} \frac{\partial \Psi^\alpha}{\partial c^l} $$

where $\Psi^\alpha$ denotes the $\alpha$-th component of the embedding vector. This metric structure provides a principled measure of semantic distance that respects the geometry of the learned embedding space.

This property enables:
1.  **Semantic Interpolation**: It is possible to explore "intermediate concepts" by evaluating the field at coordinates not assigned to specific tokens, moving along geodesics in the induced metric.
2.  **Noise Robustness**: Small perturbations in coordinates result in smooth embedding changes, improving training stability, as the Christoffel symbols derived from $g_{kl}(\mathbf{c})$ remain well-behaved.
3.  **Zero-Shot Synthesis**: The ability to generate representations for new symbols simply by assigning them a position in the existing coordinate landscape, leveraging the continuous geometry of the manifold.



## 5. Implicit Readout Mechanisms

The inverse process—mapping a latent state back to a symbol—can also be formulated as a neural field. Instead of a massive linear projection toward $V$ logits, we use an **Implicit Readout** that projects the latent state to coordinates in $\mathcal{C}$. The readout field $\Phi: \mathbb{R}^D \to \mathcal{C}$ learns to invert the embedding function, and symbol selection is performed by finding the coordinate $\mathbf{c}_i$ closest to the predicted coordinate $\hat{\mathbf{c}} = \Phi(\mathbf{h})$.

To handle the discrete nature of token selection, we employ a temperature-annealed sigmoid function that defines a soft attention over the coordinate space:

$$ P(i | \mathbf{h}) \propto \exp\left( -\frac{\| \Phi(\mathbf{h}; \Theta_{read}) - \mathbf{c}_i \|^2_{g}}{\tau} \right) $$

where $\| \cdot \|^2_{g} = g^{kl}(\mathbf{c}_i) (\Phi_k - c_{i,k}) (\Phi_l - c_{i,l})$ denotes the squared distance in the induced metric at coordinate $\mathbf{c}_i$, and $\tau$ is the temperature parameter. As temperature $\tau$ decreases during training, the distribution becomes sharper, enabling a smooth transition from continuous exploration regime to discrete symbol selection. The metric-aware distance ensures that the softmax attention respects the semantic geometry of the embedding space.



## 6. Complexity Analysis and Parameter Efficiency

Consider a vocabulary of $V = 100,000$ tokens with an embedding dimension $D = 512$.
*   **Traditional Model**: $100,000 \times 512 \approx 51.2$ million parameters.
*   **INF Model**: Requires a coordinate table of $100,000 \times d$ (where $d = 16$) plus a small $\Psi$ network (~100k parameters). Total $\approx 1.7$ million parameters.

The INF achieves an approximately **30x parameter reduction**, decoupling vocabulary growth from model complexity. In the limit, if coordinates are derived from hash functions or fixed structures (e.g., $\mathbf{c}_i = \text{hash}(token_i)$), the embedding memory complexity becomes $O(1)$ with respect to $V$, as the coordinate assignment is computed on-the-fly rather than stored. The geometric structure of the embedding manifold is fully encoded in the neural field parameters $\Theta$, providing a compact and continuous representation of the vocabulary topology.



## 7. Conclusion

Implicit Neural Fields represent a fundamental shift in symbolic information management. By treating the vocabulary as a continuous field $\Psi: \mathcal{C} \to \mathbb{R}^D$ with an induced metric structure $g_{kl}(\mathbf{c})$, we not only optimize computational resource utilization but endow the model with an intrinsic geometric understanding of semantics. The induced metric and its associated Christoffel symbols $\Gamma^k_{ij}(\mathbf{c})$ provide a principled framework for understanding semantic relationships as geodesic distances on the embedding manifold. This approach lays the foundation for architectures capable of processing vocabularies of unlimited scale with unprecedented efficiency, while maintaining the mathematical rigor of differential geometry in the embedding space.



**References**

[1] Sitzmann, V., et al. (2020). *Implicit Neural Representations with Periodic Activation Functions*. NeurIPS.  
[2] Mildenhall, B., et al. (2020). *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*. ECCV.  
[3] Bronstein, M. M., et al. (2021). *Geometric Deep Learning*. arXiv.  
[4] Tancik, M., et al. (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*. NeurIPS.  
[5] Belkin, M., & Niyogi, P. (2003). *Laplacian Eigenmaps for Dimensionality Reduction and Data Representation*. Neural Computation.  
[6] Bengio, Y., et al. (2013). *Representation Learning: A Review and New Perspectives*. IEEE TPAMI.
