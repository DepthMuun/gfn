# Architecture Overview

## Table of Contents

1. [Overview](00_overview.md) - Complete architecture guide with diagnostic results
2. [Components](01_components.md) - Detailed component breakdown
3. [Data Flow](02_data_flow.md) - How data moves through the model

## Quick Links

- **Models**: See `../3_models/`
- **Physics**: See `../2_physics/`
- **Geometry**: See `../1_geometry/`
- **Config**: See `../5_config/`

## Key Findings from Diagnostics

### Stability Recommendations

| Parameter | Default | Recommended |
|-----------|---------|-------------|
| `initial_spread` | 0.0 | 0.1-0.5 |
| `base_dt` | 0.1 | 0.05-0.1 |
| `integrator_type` | leapfrog | leapfrog |
| `topology_type` | torus | torus |

### Known Issues

1. **Rank Deficiency**: Layer 0 has condition number ~17M - use `initial_spread=0.1`
2. **Vanishing Gradients**: min gradient ~1e-08 - use larger initial spread
3. **Numerical Instability**: dt > 0.2 may cause NaN - keep dt <= 0.1

---

*Last Updated: 2026-04-02*
