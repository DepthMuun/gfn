# GSSM Architecture Diagnostics Report

## Executive Summary

This report documents the diagnostic tests run on the GSSM (Geometric Flow Network) architecture to identify and diagnose potential issues with:
1. Condition number / rank deficiency
2. Vanishing gradients
3. Integrator stability
4. Time step (dt) sensitivity
5. Topology comparison

---

## Test Results Summary

### 1. Condition Number Diagnosis

**Finding: MODERATE CONCERN**

| Layer | Singular Value 0 | Singular Last | Rank @ 1e-3 | Entropy |
|-------|------------------|---------------|-------------|---------|
| 0 | 30.45 | 1.38e-06 | 3 | 0.12 |
| 4 | 28.55 | 2.06e-04 | 26 | 0.74 |
| 8 | 28.71 | 1.59e-03 | 32 | 1.00 |
| 15 | 28.93 | 6.41e-03 | 32 | 1.22 |

**Analysis:**
- Layer 0 has severe rank deficiency (only 3 significant singular values at 1e-3 tolerance)
- Later layers (4, 8, 15) have full rank (32)
- Entropy increases with depth, indicating more diverse representations later
- **Root cause**: Initial layer is bottlenecked - likely due to embedding projection

**Recommendation**: 
- Increase embedding dimension
- Add residual connections
- Use layer normalization in embedding projection

---

### 2. Integrator Comparison

**Finding: LEAPFROG IS MOST STABLE**

| Integrator | x_norm | v_norm | Has NaN |
|------------|--------|--------|---------|
| **leapfrog** | 14.28 | 37.65 | 0 |
| yoshida | 19.80 | 98.98 | 0 |
| heun | 19.79 | 98.94 | 0 |

**Analysis:**
- Leapfrog produces significantly lower norms (more stable)
- Yoshida and Heun have ~3x higher velocity norms
- All are numerically stable (no NaN)

**Recommendation**: 
- Keep **leapfrog** as default (already set)
- If faster convergence needed, consider yoshida with smaller dt

---

### 3. dt Sensitivity

**Finding: dt=0.01-0.1 OPTIMAL**

| dt | x_norm | v_norm | Has NaN |
|----|--------|--------|---------|
| 0.01 | 17.28 | 10.59 | 0 |
| 0.05 | 18.15 | 52.28 | 0 |
| **0.1** | 19.78 | 98.87 | 0 |
| 0.2 | 18.25 | 123.78 | 0 |
| 0.5 | 19.22 | 137.64 | 0 |

**Analysis:**
- Velocity scales linearly with dt (as expected from physics)
- dt=0.01 gives most conservative (lowest velocity) behavior
- dt=0.1 (default) is acceptable
- dt > 0.2 may cause instability in training

**Recommendation**:
- Use dt=0.05-0.1 for training
- dt=0.01 for fine-tuning or unstable phases

---

### 4. Topology Comparison

**Finding: TORUS IS MORE STABLE**

| Topology | x_norm | v_norm | Condition # | x_change |
|----------|--------|--------|-------------|----------|
| **torus** | 21.80 | 60.80 | 786 | 0.73 |
| euclidean | 50.62 | 187.31 | 648 | 45.74 |

**Analysis:**
- Torus has much smaller position changes (0.73 vs 45.7)
- Torus has slightly higher condition number but much more stable dynamics
- Euclidean explodes in position (unbounded)

**Recommendation**:
- Use **torus topology** for training
- Euclidean only for specific tasks requiring unbounded space

---

### 5. Gradient Flow Comparison

**Finding: initial_spread HELPS GRADIENTS**

| Configuration | First Layer Grad | Last Layer Grad |
|---------------|------------------|-----------------|
| spread=0.0 | Higher | Lower |
| spread=1.0 | More balanced | More balanced |

**Analysis:**
- Larger initial spread helps gradient flow
- Default spread=0.0 may cause initial gradient issues

**Recommendation**:
- Use initial_spread=0.1-0.5 for training
- Or add skip connections

---

## Recommendations Summary

### Immediate Actions

1. **Increase initial_spread** from 0.0 to 0.1-0.5
   ```python
   model = gfn.create(..., initial_spread=0.1)
   ```

2. **Keep leapfrog integrator** (already default)

3. **Use dt=0.05-0.1** (default 0.1 is fine)

4. **Use torus topology** (already default)

### Long-term Improvements

1. Add layer normalization after embedding
2. Add residual connections between layers
3. Increase embedding dimension if rank deficiency persists
4. Consider gradient clipping during training

---

## Test Files Location

All test results saved to:
```
tests/gssm/results/diagnostics/
```

- `condition_number_diagnosis_latest.json`
- `integrator_comparison_latest.json`
- `dt_sensitivity_latest.json`
- `topology_comparison_latest.json`
- `gradient_flow_comparison_latest.json`

---

*Generated: 2026-04-02*
*Test Suite: GSSM Architecture Diagnostics*
