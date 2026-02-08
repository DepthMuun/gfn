# Changelog

## Current Development Version

### Architectural Changes

**Modular Configuration System**

The current version introduces a fully redesigned configuration system. Parameters that were previously hardcoded are now externally configurable. This includes hysteresis parameters, velocity-dependent friction, trace normalization, and singularity thresholds. The benefit is extreme flexibility; the downside is that default values require careful tuning.

**New CUDA/Python Parity Policy**

Automated parity verification between backends was implemented. Tests compare Python and CUDA outputs with defined tolerances. This prevents drift between implementations but requires any constant changes to be reflected in both backends.

**Implicit Integrator with Friction**

The velocity update changed from explicit to implicit:

Previous form: v_new = v + h * (force - friction)

Current form: v_new = (v + h * force) / (1 + h * mu)

The change improves numerical stability, especially with high friction. The system can now handle friction values that previously caused divergence.

### New Features

**Semantic Hysteresis**

Long-term memory system that allows the model to "remember" previous states. Controlled by constants HYSTERESIS_FORGET_GATE_INIT, HYSTERESIS_STATE_MOMENTUM, and HYSTERESIS_GHOST_FORCE_SCALE.

**Reactive Curvature**

The metric can now adapt dynamically to data. The system responds to "perturbations" by adjusting curvature. It prevents singularities via the "black hole" mechanism.

**Trace Normalization**

Option to normalize the metric trace before Christoffel computation. Prevents the metric from becoming ill-conditioned in some configurations.

### Constant Changes

Constant changes are documented in detail in the reversion viability analysis. The most significant:

| Constant | Previous | Current | Reason |
|-----------|----------|--------|-------|
| FRICTION_SCALE | 5.0 | 0.02 | Exploration |
| DEFAULT_LR | 1e-3 | 1e-4 | Stability |
| CURVATURE_CLAMP | 20.0 | 3.0 | Avoid singularities |
| EPSILON_STANDARD | 1e-6 | 1e-7 | Precision |

### API Changes

**Manifold Constructor**

New constructor with additional parameters:

```python
class Manifold(
    vocab_size, dim, depth, rank, heads,
    # New parameters
    hyst_alpha=0.9,
    hyst_beta=0.1,
    enable_trace_normalization=True,
    velocity_friction_scale=0.02,
    position_friction_scale=0.0,
    plasticity_coef=0.02,
    singularity_threshold=0.5,
)
```

**New Geometries**

Added: AdaptiveGeometry, HierarchicalGeometry, ReactiveGeometry

**New Integrators**

Added: OmelyanIntegrator (alternative to Leapfrog)

### CUDA Backend Changes

**Fused Kernels**

The leapfrog_fused and christoffel_fused kernels are now available. They provide a 2-3x speedup over the Python implementation.

**Extended Backward Pass**

The leapfrog_fused backward now returns 11 values (previously 7) to include hysteresis gradients.

**Automated Verification**

Python/CUDA parity tests integrated into the CI suite.

## v2.6.5 Version (Reference)

### State of the Art

This version was the last before the rewrite. It was the last to converge consistently on the viz superiority benchmark.

### Features

- Functional architecture but with technical debt
- Functional CUDA kernels but without parity tests
- Constants optimized for convergence, not precision
- Existing but fragmented documentation

### Known Limitations

- No hysteresis support
- No reactive curvature
- Friction was binary (on/off)
- No trace normalization
- Python/CUDA parity not verified

### Notable Constants

| Constant | Value | Notes |
|-----------|-------|-------|
| FRICTION_SCALE | 5.0 | Extremely high |
| DEFAULT_LR | 1e-3 | Aggressive |
| CURVATURE_CLAMP | 20.0 | Allowed extreme curvature |
| LAMBDA_H_DEFAULT | 0.01 | Hamiltonian loss active |

### Reason for the Rewrite

v2.6.5 worked but had structural issues:

- The code was monolithic and hard to maintain
- Excessive friction limited model capacity
- No support for planned new features
- Technical debt prevented fast iteration

The rewrite aimed to solve these problems while maintaining or improving performance.

## Performance Comparison

### Convergence in Viz Superiority

| Version | Steps to Convergence | Stability |
|---------|-------------------------|-------------|
| v2.6.5 | ~5000 | High |
| Development | ~8000 (with tuning) | Medium |

The development version converges more slowly in the default configuration due to more conservative values. With proper tuning, the gap shrinks significantly.

### Speed (Throughput)

| Version | Tokens/sec/GPU | Speedup |
|---------|----------------|---------|
| v2.6.5 | 1500 | 1.0x |
| Development (CPU) | 1400 | 0.93x |
| Development (CUDA) | 2800 | 1.87x |

The development version with CUDA is significantly faster due to fused kernels.

### Memory

| Version | Peak GPU Memory |
|---------|-----------------|
| v2.6.5 | 4.2 GB |
| Development | 4.8 GB |

The development version uses more memory due to additional intermediate states for hysteresis and reactive curvature.

## Migration Notes

### From v2.6.5 to Development

If you migrate code from v2.6.5:

1. Review the Manifold constructor parameters
2. Verify that your constants are within valid ranges
3. Run parity tests after changes
4. Consider that the physics may behave differently

### Recommended Default Values

To reproduce behavior similar to v2.6.5:

```yaml
physics:
  friction_scale: 0.5    # Entre 0.02 y 5.0
  dt: 0.08               # Entre 0.05 y 0.1
  lambda_g: 0.0002       # Entre 0.00005 y 0.001

training:
  learning_rate: 5e-4    # Entre 1e-4 y 1e-3
```

## Known Bugs

### Current Version

1. The hysteresis mechanism can cause a memory leak in long training (>100k steps)
2. Trace normalization does not work correctly with analytical geometries
3. The Omelyan integrator has a backward pass error (avoid using it)

### Workarounds

1. Restart training periodically for hysteresis
2. Disable trace normalization for analytical geometries
3. Use Leapfrog or Yoshida instead of Omelyan

## Future Plans

### Upcoming Changes

- 6th-order integrator for better energy preservation
- Geometry that emerges from data (metric learning)
- bf16 precision support in CUDA
- Integration with Hugging Face Transformers

### Planned Deprecations

- The old `step()` method will be replaced in v3.0
- The current checkpoint format will change to a standard format
- The geometry API will be simplified

## Acknowledgments

Thanks to the contributors who reported bugs and proposed improvements during development of this version.

---

**Manifold Labs (Joaquín Stürtz)**
