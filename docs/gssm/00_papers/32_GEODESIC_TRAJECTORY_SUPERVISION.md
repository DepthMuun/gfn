# Geodesic Trajectory Supervision: Forcing Logical Reasoning via Dense Manifold Alignment

**Joaquín Stürtz**  
*Independent Researcher*  
February 2026

---

## Abstract

Standard Chain-of-Thought (CoT) methods train models to produce correct graphical tokens, treating the reasoning process as a sequence of discrete classifications. We propose **Geodesic Trajectory Supervision (GTS)**, a continuous supervision paradigm where the *entire latent trajectory* is forced to align with the geodesic path corresponding to the ground truth logic. By minimizing the toroidal/manifold distance between the latent state $x_t$ and the "Answer Manifold" at every timestep, we compel the neural dynamics to physically enact the logical transformation. We demonstrate that this method, unlike "Last Token Only" supervision, eliminates "reasoning gaps" and ensures that intermediate states encode valid partial computations, enabling O(1) verification of logical correctness.

**Keywords:** Dense Supervision, Toroidal Manifold, Geodesic Distance, Neural ODEs, Reasoning Dynamics.

---

## 1. Introduction

In transformers, intermediate layers often act as "scratchpads" with no semantic guarantees. The model is only penalized if the final logits are wrong. This allows "Clever Hans" behavior where the model memorizes spurious correlations rather than maximizing the validity of the reasoning steps.

GTS changes the objective from "Predict the next token" to "Be in the right place". We define a target manifold region $\mathcal{M}_{target}$ (e.g., the region corresponding to "Odd" parity). We then supervise the *entire flow* $x(t)$ such that the distance $d_{\mathcal{M}}(x(t), \mathcal{M}_{target})$ is minimized throughout the sequence.

---

## 2. Method

### 2.1 Logic as Geometry

We map logical states to geometric locations. For example, in a parity task:
*   **Even:** Angle $\theta \approx -\pi/2$
*   **Odd:** Angle $\theta \approx +\pi/2$

### 2.2 Dense Geodesic Loss

Instead of effective loss $L = \text{CE}(y_T, \hat{y}_T)$, we optimize:
$$ L = \frac{1}{T} \int_0^T d_{\mathcal{M}}(x(t), y(t))^2 dt $$
where $y(t)$ is the target geodesic (or target region) at time $t$. For parity, $y(t)$ is the current parity of the partial sum.

This forces the Neural ODE to effectively "simulate" the logical operation as a continuous physical process (e.g., rotation on a torus).

### 2.3 Benefits

1.  **Interpretability:** Deviations from the path indicate logical errors *before* the answer is effectively generated.
2.  **Robustness:** The model cannot "jump" to the answer; it must traverse the manifold, building momentum and resistance to perturbations.
3.  **Data Efficiency:** Every timestep provides a supervision signal, not just the end of the sequence.

---

## 3. Implementation

We implement GTS in the `ManifoldTrainer` via a custom `compute_loss` method:

```python
def compute_loss(self, out, y, ...):
    logits = out[0] # [B, L, D]
    target_vec = y.unsqueeze(-1).expand_as(logits) # Dense targets
    
    # Toroidal Distance on Full Sequence
    loss = toroidal_distance_loss(logits, target_vec)
    return loss
```

---

## 4. Conclusion

GTS represents a shift from "discrete symbolic mimicking" to "continuous semantic simulation". By supervising the geometry of the thought process, we produce models that not only answer correctly but reason robustly.
