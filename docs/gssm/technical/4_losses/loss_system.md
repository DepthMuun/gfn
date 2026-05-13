# GFN Loss System: Geometric & Path Optimization

The `gfn/losses` module defines how the model is trained. Since GFN is a physical system, losses often target "Paths" rather than just isolated outputs.

## 1. Probabilistic vs. Geometric Losses

| Philosophy | Class | Use Case |
| :--- | :--- | :--- |
| **Probabilistic** | `ManifoldGenerativeLoss` | Token prediction, NLP, Entropy-heavy tasks. |
| **Geometric** | `ToroidalDistanceLoss` | Exact coordinate regression, Logic (XOR). |
| **Hybrid** | `PhysicsInformedLoss` | The "Gold Standard" for GFN. NLL + Physical Guards. |

---

## 2. Physics-Informed Loss (`physics.py`)
The most advanced training objective in the framework. It ensures the model is both accurate and physically consistent.

$$L = L_{NLL} + \lambda_{geo} L_{geo} + \lambda_{ham} L_{ham} + \lambda_{kin} L_{kin}$$

### Components:
- **Geodesic ($L_{geo}$)**: Penalizes local curvature. Forces the model to find the most "natural" (straightest) paths in coordinates.
- **Hamiltonian ($L_{ham}$)**: Penalizes energy fluctuations. Ensures the symplectic integrator is operating in a stable regime.
- **Kinetic ($L_{kin}$)**: Prevents "Velocidad Fugitiva" (exploding kinetic energy).

---

## 3. Toroidal Specific Losses (`toroidal.py`)
Optimized for the $S^1 \times S^1 ... \times S^1$ topology.

- **ToroidalDistance**: Uses `atan2(sin, cos)` to measure the shortest angular path, respecting periodic boundaries.
- **ToroidalVelocity**: Regularizes angular velocity in the tangent space of the torus.

---

## 4. Loss Factory & Registration
Losses are modular. New losses can be registered via the `@register_loss` decorator and instantiated via `LossFactory.create(loss_cfg)`.

---

## 5. Dynamic Balancing (`regularization.py`)
The `DynamicLossBalancer` is used when multiple losses are active. It uses the magnitude of gradients to automatically scale the $\lambda$ weights, ensuring that the NLL doesn't overwhelm the Physics terms (or vice-versa).
