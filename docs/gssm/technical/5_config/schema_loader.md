# Configuration & Registry System

GFN V5 is designed for maximum configurability. Every aspect of the physics and architecture can be controlled via a unified schema.

## 1. Schema System (`schema.py`)
Uses Python Dataclasses to define a hierarchical configuration structure:

- **TopologyConfig**: Type (Torus, Euclidean), $R$, $r$.
- **StabilityConfig**: $dt$, integrators, trace normalization.
- **ActiveInferenceConfig**: Dynamic time, singularities, holographic flags.
- **ManifoldConfig**: The top-level config (Depth, Dim, Heads, Physics).

---

## 2. Configuration Loader (`loader.py`)
Translates flat YAML/JSON files or CLI arguments into the complex `ManifoldConfig` structure.

- Supports **Aliases**: `dt` $\rightarrow$ `config.physics.stability.base_dt`.
- Supports **Overrides**: Dynamically patching parts of the physics config during experimentation.

---

## 3. Validation & Serialization
- **Validator (`validator.py`)**: Checks for physical consistency (e.g., "Cannot use Riemannian Gating on Euclidean geometry").
- **Serialization (`serialization.py`)**: Converts configs to/from JSON for model saving/loading (`save_pretrained`).

---

## 4. The Registry Pattern (`registry.py`)
Registry is the secret to GFN's modularity. It decouples the core model from its components.

### Available Registries:
- `GEOMETRY_REGISTRY`: Euclidean, Torus, LowRank, etc.
- `INTEGRATOR_REGISTRY`: Yoshida, RK4, Leapfrog.
- `LOSS_REGISTRY`: PhysicsInformed, Toroidal, etc.
- `DYNAMICS_REGISTRY`: Direct, Residual, Mix.

### How to extend:
Simply decorate your new class:
```python
@register_integrator('my_new_integrator')
class MyIntegrator(BaseIntegrator):
    ...
```
Now it can be used anywhere by passing `integrator='my_new_integrator'` in the config.
