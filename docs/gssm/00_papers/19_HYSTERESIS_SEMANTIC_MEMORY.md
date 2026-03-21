# Hysteresis and Ghost Forces: Trajectory-Dependent Semantic Memory in Neural Manifolds

**Joaquin Stürtz**  
*Independent Researcher*  
January 2026

---

## Abstract

We introduce a novel mechanism for long-term semantic memory in Geodesic Flow Networks (GFN) based on the physical principle of **hysteresis**. Unlike traditional recurrent memory (which stores flat vectors) or attention (pivoting on token history), Hysteresis implements memory as a **dynamic deformation of the manifold**. We model this as a "plastic" state that accumulates deformation from the particle's trajectory and generates a **Ghost Force** (or self-gravity) that influences future dynamics. Our results show that Hysteresis improves performance on long-range dependency tasks by 14% and enables the model to resolve semantic ambiguities through path-dependent context. The ghost force couples to the geodesic dynamics through the Christoffel symbols $\Gamma^k_{ij}(x)$, modifying the effective geometry of the latent manifold.

---

## 1. Introduction

In Geodesic Flow Networks, computation is the evolution of a state $(x_t, v_t)$ on a manifold $\mathcal{M}$ with metric $g_{ij}(x)$. While the "Clutch" (thermodynamic gating) allows for instantaneous context switching, it lacks a mechanism for persistent, path-dependent memory that doesn't rely on the immediate state.

We propose **Hysteresis**, a property where the state of a system depends on its history. In our framework, the manifold itself possesses "memory" through a hidden deformation field. As the neural state moves along a geodesic, it leaves a "wake" or "path" that perturbs the geometry, creating a self-consistent semantic context. The accumulated deformation modifies the effective Christoffel symbols $\Gamma^k_{ij}(x, H_t)$, where $H_t$ is the hysteresis state, creating a trajectory-dependent geometry.

---

## 2. Mathematical Framework

### 2.1 The Hysteresis State

We define a hidden memory state $H_t \in \mathbb{R}^d$ that evolves according to the trajectory $(x_t, v_t)$:

$$ H_{t+1}^i = \lambda H_t^i + \phi^i(x_t, v_t) $$

where:
- $\lambda \in [0, 1]$ is a learnable **decay coefficient** (memory retention)
- $\phi^i(x, v)$ is a learned **plasticity mapping** that encodes the current state into a deformation update
- The index $i$ denotes the component of the hysteresis state vector

The hysteresis state $H_t^i$ accumulates the trajectory history, with the decay coefficient $\lambda$ controlling how quickly past information is forgotten.

### 2.2 Plasticity Mapping

The mapping $\phi^i$ depends on the manifold topology:
- **Euclidean:** $\phi^i(x, v) = \tanh\left((W_{up}^{ij} [x_j, v_j] + b_{up}^i)\right)$
- **Toroidal:** $\phi^i(x, v) = \tanh\left(W_{up}^{ij} [\sin(x_j), \cos(x_j), v_j] + b_{up}^i\right)$

where the Einstein summation convention is used for repeated indices. This ensures that the memory is informed by both position $x_j$ (semantics) and velocity $v_j$ (rate of change/confusion), with the Christoffel symbols $\Gamma^k_{ij}(x)$ providing the geometric context for the deformation.

### 2.3 The Ghost Force

The accumulated hysteresis state generates a **Ghost Force** $F^k_{ghost}$ through a readout mapping:

$$ F^k_{ghost} = W_{out}^{ki} H_t^i + b_{out}^k $$

This force is added to the external force $F^k_{ext}$ (from token embeddings) in the covariant geodesic equation:

$$ \frac{D v^k}{dt} = \dot{v}^k + \Gamma^k_{ij}(x) v^i v^j = F^k_{ext} + F^k_{ghost} - \mu v^k $$

where $\mu$ is the friction coefficient. The ghost force $F^k_{ghost}$ acts as a "self-gravity" field created by the history of the trajectory, pulling the state toward semantically relevant regions visited in the past. The modified Christoffel symbols due to hysteresis can be expressed as:

$$ \Gamma^k_{ij}(x, H_t) = \Gamma^k_{ij}(x) + \alpha \frac{\partial H_t^k}{\partial x^i} \frac{\partial H_t^l}{\partial x^j} g_{lk} $$

which encodes the trajectory-dependent deformation in the geometric structure.

**Interpretation:** The ghost force modifies the effective geometry through the Christoffel symbols, creating an attractive potential well around previously visited semantic regions. This path-dependent geometry is the geometric manifestation of hysteresis.

---

## 3. Physical Intuition

### 3.1 Self-Gravity and Semantic Attractors
Hysteresis creates a "well" in the latent space. If the model frequently visits a specific semantic region (e.g., the concept of "Quantum Physics"), the ghost force $F^k_{ghost}$ will create an attractor that makes it easier for the particle to return to or stay in that semantic regime, even if individual token forces $F^k_{ext}$ are noisy. The effective Christoffel symbols $\Gamma^k_{ij}(x, H_t)$ are modified to favor geodesics that pass through these semantically rich regions.

### 3.2 Friction and Plasticity
The learnable decay $\lambda$ allows the model to balance between **brittle memory** ($\lambda \to 0$, only immediate past matters) and **plastic memory** ($\lambda \to 1$, long-term history is preserved). The friction coefficient $\mu$ in the geodesic equation $\frac{D v^k}{dt} = \dots - \mu v^k$ complements this by controlling how quickly the velocity—and thus the immediate dynamics—decays.

### 3.3 Geometric Interpretation
The hysteresis state $H_t^i$ can be interpreted as a deformation field that modifies the metric tensor $g_{ij}(x)$:

$$ g_{ij}(x, H_t) = g_{ij}(x) + \beta H_t^k \partial_k g_{ij}(x) $$

where $\beta$ is a deformation scaling parameter. The Christoffel symbols derived from this deformed metric,

$$ \Gamma^k_{ij}(x, H_t) = \frac{1}{2} g^{kl}(x, H_t) \left( \partial_i g_{jl}(x, H_t) + \partial_j g_{il}(x, H_t) - \partial_l g_{ij}(x, H_t) \right) $$

encode the trajectory-dependent geometry that underlies the hysteresis mechanism.

---

## 4. Implementation details

The integration of Hysteresis into the symplectic framework requires careful attention to preserving the geometric structure of the manifold dynamics. We incorporate the hysteresis update within a symplectic integration scheme to ensure that the volume preservation properties of the geodesic flow are maintained even with the addition of the memory-dependent ghost force.

The hysteresis update introduces a time-asymmetric component to the dynamics, which must be reconciled with the reversible nature of symplectic integrators. This is achieved through a careful partitioning of the update steps, where the plasticity mapping $\phi^i(x, v)$ is applied as a bounded perturbation that does not violate the underlying Hamiltonian structure of the system. The decay coefficient $\lambda$ provides a mechanism for controlling the temporal horizon of the memory, effectively implementing a low-pass filtering of the trajectory history. The symplectic leapfrog integration with hysteresis becomes:

$$ v_{n+\frac{1}{2}}^k = v_n^k + \frac{h}{2} \left( F^k_{ext} + F^k_{ghost}(H_n) - \Gamma^k_{ij}(x_n) v_n^i v_n^j - \mu v_n^k \right) $$
$$ x_{n+1}^i = x_n^i + h v_{n+\frac{1}{2}}^i $$
$$ H_{n+1}^i = \lambda H_n^i + \phi^i(x_{n+1}, v_{n+\frac{1}{2}}) $$
$$ v_{n+1}^k = v_{n+\frac{1}{2}}^k + \frac{h}{2} \left( F^k_{ext} + F^k_{ghost}(H_{n+1}) - \Gamma^k_{ij}(x_{n+1}) v_{n+\frac{1}{2}}^i v_{n+\frac{1}{2}}^j - \mu v_{n+\frac{1}{2}}^k \right) $$

where $h$ is the integration step size. This scheme preserves the symplectic structure while incorporating the hysteresis state $H_t^i$ and its associated ghost force.

---

## 5. Experimental Results

### 5.1 Long-Range Dependency (LongBench)
We evaluate the model on the Needle-In-A-Haystack test (32k context).

| Model | Memory Type | Accuracy (4k) | Accuracy (32k) |
|-------|-------------|---------------|----------------|
| GFN (Base) | Phase Only | 89.2% | 61.5% |
| GFN + Hysteresis | Plasticity | **92.4%** | **78.6%** (+17%) |
| Transformer | KV-Cache | 91.8% | 76.2% |

Hysteresis allows the fixed-size state to maintain context over significantly longer horizons than raw $(x, v)$ dynamics. The ghost force $F^k_{ghost}$ provides a continuous signal that encodes the semantic trajectory history, enabling the model to "remember" relevant context even when the immediate state vector $(x_t, v_t)$ has moved on.

### 5.2 Semantic Consistency
We measure the variance of the Ghost Force during a sequence. In coherent text, $F^k_{ghost}$ tends to point in a stable direction, indicating the formation of a "semantic theme." The magnitude $\|F_{ghost}\| = \sqrt{g_{kl} F^k_{ghost} F^l_{ghost}}$ drops and then re-aligns during context shifts, providing a natural signal for segmentation. The Christoffel symbols $\Gamma^k_{ij}(x)$ remain well-behaved during these transitions, ensuring smooth geodesic dynamics.

---

## 6. Discussion

Hysteresis provides a physically-grounded alternative to the KV-cache. While the KV-cache stores every token explicitly, Hysteresis stores the **integrated semantic trajectory** through the accumulated state $H_t^i$ and its associated ghost force. This is more efficient ($O(1)$ memory) and more reflective of how physical systems (and potentially biological neurons) maintain state through gradual physical changes. The modified Christoffel symbols $\Gamma^k_{ij}(x, H_t)$ encode this memory in the geometric structure of the latent manifold.

**Advantages:**
- $O(1)$ memory complexity with $O(1)$ inference latency
- Physically grounded mechanism with clear geometric interpretation
- Natural path-dependence enables semantic context resolution
- Compatible with symplectic integration and existing GFN framework

**Limitations:**
- The learnable decay $\lambda$ can be sensitive to initialization
- Very long-term recall (e.g., specific names after 100k tokens) still benefits from external retrieval/attention
- The ghost force $F^k_{ghost}$ may create unwanted attractors for irrelevant past context

**Future Work:**
- Hierarchical hysteresis with multiple time scales
- Connection to information geometry and capacity bounds
- Integration with attention for explicit retrieval from the hysteresis state

---

## 7. Conclusion

By incorporating Hysteresis into the Manifold/GFN framework, we bridge the gap between continuous dynamics and symbolic memory. Ghost forces enable a form of "internal dialogue" where the past trajectory actively shapes the present computation through the modified Christoffel symbols $\Gamma^k_{ij}(x, H_t)$, leading to more robust and context-aware sequence modeling. The hysteresis state $H_t^i$ provides a geometrically grounded mechanism for trajectory-dependent memory that complements the instantaneous context switching of the thermodynamic clutch.

---

## References

[1] Jiles, D. C., & Atherton, D. L. (1986). Theory of ferromagnetic hysteresis. *Journal of Magnetism and Magnetic Materials*.

[2] Sturtz, J. (2026). *Geodesic Flow Networks: A Physics-Informed Paradigm*.

[3] Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*.
