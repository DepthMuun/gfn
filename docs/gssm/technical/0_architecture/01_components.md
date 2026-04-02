# Architecture Components

Detailed breakdown of GSSM components and their interactions.

## Component Hierarchy

```
ManifoldModel
├── Embedding (FunctionalEmbedding)
│   └── Maps token IDs to manifold impulses
├── Layers (ModuleList of ManifoldLayer)
│   ├── Integrator (Leapfrog by default)
│   │   └── PhysicsEngine
│   │       └── Geometry (Torus/LowRank/etc)
│   ├── Mixer (FlowMixer)
│   ├── Dynamics (direct/residual/gated)
│   └── Plugins (optional)
└── Readout (CategoricalReadout)
    └── Projects manifold state to logits
```

## ManifoldLayer Details

Each `ManifoldLayer` performs:

1. **Pre-integrate hooks**: Dynamic time adjustment
2. **Integration**: `integrator.step(x, v, force, dt)`
3. **Post-integrate hooks**: Post-processing
4. **Mixing**: `mixer(x, v)` for head interaction
5. **Dynamics routing**: Apply mixing proposal
6. **Topology resolution**: Wrap to manifold bounds
7. **Finalize hooks**: Fractal steps, etc.

## Key Parameters by Component

### Integrator Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_dt` | Time step | 0.1 |
| `steps` | Steps per call | 1 |
| `integrator_type` | Method | leapfrog |

### PhysicsEngine Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `friction` | Base friction | 0.01 |
| `velocity_friction_scale` | Velocity-dependent scaling | 0.0 |
| `velocity_saturation` | Max velocity (0=off) | 0.0 |

### Geometry Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `topology.type` | Manifold type | torus |
| `topology.major_radius` | R (torus) | 1.0 |
| `topology.minor_radius` | r (torus) | 1.0 |

---

*Last Updated: 2026-04-02*
