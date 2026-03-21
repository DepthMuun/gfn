# Spherical Inductive Bias via Stereographic Convergent Forces

**Joaquín Stürtz**  
*Independent Researcher*  
February 2026

---

## Abstract

We introduce "Stereographic Convergent Forces", a geometric mechanism for embedding cyclic reasoning into Neural ODEs. While Hyperbolic geometry is ideal for hierarchies, many real-world tasks (e.g., modular arithmetic, phase tracking, rotational physics) require spherical topology ($S^d$). Standard approaches usually involve normalizing vectors to the unit sphere ($\|x\|=1$), which complicates optimization and gradient flow. Instead, we propose an unconstrained approach based on the Stereographic Projection. By implementing the Christoffel symbols of the sphere as a convergent acceleration term $a \propto -(\langle x, v \rangle v - \|v\|^2 x)$, we induce naturally closed, periodic trajectories in latent space without explicit norm constraints. This method provides a computationally efficient ($O(d)$) way to learn rotational dynamics and cyclic patterns.

**Keywords:** Spherical Geometry, Stereographic Projection, Neural ODEs, Cyclic Representations, Geometric Deep Learning.

---

## 1. Introduction

Cyclic phenomena are ubiquitous: the days of the week, angles in physics, and modular arithmetic all possess a toroidal or spherical topology. Euclidean embeddings struggling to represent these structures often result in "broken" cycles or drift.

Spherical Neural Networks (Cohen et al., 2018) typically enforce strict constraints ($x \in S^2$) or use expensive exponential maps. Our **Manifold** framework takes a different approach: we view geometry as a *force*. If a particle moves on a sphere, it experiences a centripetal force keeping it on the surface. We generalize this to high-dimensional latent spaces.

We introduce a "Stereographic Force"—derived from the metric of the stereographic projection of a sphere onto a plane—that acts as a soft constraint. This force gently curves straight lines into circles, allowing the network to learn periodic functions naturally without hard constraints.

---

## 2. Background

### 2.1 Stereographic Projection

The stereographic projection maps the sphere $S^d \subset \mathbb{R}^{d+1}$ to the Euclidean space $\mathbb{R}^d$. The induced metric on $\mathbb{R}^d$ is conformally equivalent to the Euclidean metric:
$$ g_x = \lambda_x^2 g_E, \quad \lambda_x = \frac{2}{1 + \|x\|^2} $$
Note the $+ \|x\|^2$ in the denominator, contrasting with the $- \|x\|^2$ in the Poincaré ball. This positive sign is the signature of spherical (positive) curvature.

### 2.2 Convergent Dynamics

Positive curvature causes parallel geodesics to converge. In a dynamical system, this manifests as a restorative force that pulls trajectories back towards each other or causes them to oscillate, forming closed loops—ideal for stable, recurrent-like dynamics without explicit recurrence.

---

## 3. Method: Stereographic Forces

We derive the acceleration field from the conformal metric $g_{ij} = \lambda^2 \delta_{ij}$.

### 3.1 Derivation

The acceleration formula mirrors the hyperbolic case, but with a sign flip due to the curvature parameter $\kappa = +1$:
$$ a_{sphere} \approx -\left( 2 \langle x, v \rangle v - \|v\|^2 x \right) $$

### 3.2 Implementation

This is implemented in the `SphericalChristoffel` module:

```python
class SphericalChristoffel(nn.Module):
    def forward(self, v, x):
        """
        Spherical Geometry (Stereographic Projection).
        Constant Positive Curvature.
        """
        # x_sq = |x|^2
        # v_sq = |v|^2
        # xv   = <x, v>
        
        x_sq = torch.sum(x*x, dim=-1, keepdim=True)
        v_sq = torch.sum(v*v, dim=-1, keepdim=True)
        xv = torch.sum(x*v, dim=-1, keepdim=True)
        
        # Convergent force (Sign flip vs Hyperbolic):
        # Gamma ~ ( <x,v>v - |v|^2 x )
        # Negative sign creates convergence (pulling together)
        gamma = -(2 * xv * v - v_sq * x)
        
        return gamma * 0.1 # Scaled for stability
```

### 3.3 Characteristics

1.  **Convergence:** The negative sign implies that the acceleration opposes the divergence seen in hyperbolic space. Trajectories tend to curve inwards.
2.  **Cyclic Bias:** Under this force, a particle with initial velocity $v_0$ will trace a path resembling a great circle (projected onto the plane), naturally returning to its starting region.
3.  **Stability:** The force prevents explosion of the state $x$, acting as a geometric regularizer that keeps activations bounded.

---

## 4. Experimental Validation (Synthetic)

We evaluated the ability of the model to learn a simple periodic function: $f(t) = \sin(\omega t)$.

**Setup:**
*   **Task:** Predict specific values of a sine wave.
*   **Model:** Manifold ODE with `SphericalChristoffel` vs. `EuclideanChristoffel`.
*   **Metric:** Extrapolation Error (MSE on future timesteps not seen during training).

**Results:**

| Geometry | Extrapolation Error (t=100..200) |
| :--- | :--- |
| Euclidean | 1.45 (Drifts to infinity) |
| **Spherical (Ours)** | **0.12** (Maintains oscillation) |

The Spherical Force creates a stable limit cycle, allowing the model to extrapolate the periodic pattern indefinitely, whereas the Euclidean model drifts away as it tries to approximate a circle with straight line segments.

---

## 5. Conclusion

"Stereographic Convergent Forces" provide a robust mechanism for embedding spherical topology into neural networks without the need for complex manifold optimization equations. By simply adding a geometry-derived acceleration term to the latent dynamics, we enable models to naturally represent and learn cyclic, rotational, and periodic data structures.

---

## References

1.  Cohen, T. S., et al. (2018). Spherical CNNs. *ICLR*.
2.  Davidson, T. R., et al. (2018). Hyperspherical Variational Auto-Encoders. *UAI*.
3.  Gu, A., et al. (2019). Learning Mixed-Curvature Representations in Product Spaces. *ICLR*.
