# Numerical Validation

## Purpose

This guide documents procedures to verify the system's numerical correctness. Validation is crucial because the system depends on physical properties (energy conservation, geodesic trajectories) that can break subtly.

## Python-CUDA Consistency Tests

The implementation has two backends: pure Python and custom CUDA kernels. For consistent results, both must produce the same values within numerical tolerance.

### Execution

```bash
python tests/test_cuda_python_consistency.py
```

This test compares outputs of key operations across backends.

### Reported Metrics

The test reports the maximum absolute and relative error for each operation:

```
Operation: leapfrog_step
  Max abs diff: 1.2e-7
  Max rel diff: 3.4e-6
  Status: PASS

Operation: christoffel_computation
  Max abs diff: 8.5e-8
  Max rel diff: 2.1e-5
  Status: PASS
```

### Tolerances

Per-operation tolerances:

| Operation | Absolute | Relative |
|-----------|----------|----------|
| Leapfrog | 1e-5 | 1e-4 |
| Christoffel | 1e-6 | 1e-3 |
| Metric | 1e-6 | 1e-3 |
| Gradients | 1e-4 | 1e-2 |

If errors exceed these tolerances, the system may still work but with different behavior on GPU.

### Causes of Failures

**Out-of-sync constants.** Verify that EPSILON_STANDARD, FRICTION_SCALE, and other values match between Python and CUDA.

**Operation order.** Reduction operations can differ in floating-point associativity.

**Data types.** Verify both use float32 (not float16).

## Energy Conservation Test

In the absence of friction and external forces, the Hamiltonian H(q,p) should stay constant.

### Execution

```bash
python tests/diagnostics/conservation_audit.py --steps 1000
```

### Metrics

The test reports:

- **Energy drift**: Percentage change of the Hamiltonian over the simulation
- **Energy oscillation**: Amplitude of fluctuations
- **Energy variance**: Hamiltonian variance

### Interpretation

Typical results for Leapfrog with low friction:

```
Energy drift: 0.023%  (expected: < 1%)
Energy oscillation: 0.15%  (expected: < 1%)
Status: PASS
```

Acceptable results:

- Drift < 1% for 1000 steps
- Oscillation < 1% of the initial value
- No monotonic trend (up or down)

### Failure Diagnosis

**Large positive drift.** Indicates spurious energy gain. Check:
- That external force is zero
- That there are no incorrect force terms

**Large negative drift.** Indicates excessive energy loss. Check:
- That friction is not too high
- That the timestep is not too large

**Growing oscillation.** Indicates integrator instability. Solution:
- Reduce DEFAULT_DT
- Increase LEAPFROG_SUBSTEPS
- Switch to a more stable integrator

## Geodesic Trajectory Test

Geodesics are minimum-length paths. We verify that model trajectories are geodesic.

### Execution

```bash
python tests/diagnostics/test_suite_comprehensive.py --test geodesic
```

### Metrics

The test reports:

- **Geodesic deviation**: Deviation from the geodesic equation
- **Length optimality**: Ratio between actual length and geodesic distance
- **Parallel transport**: Rotation of transported vectors

### Interpretation

```
Geodesic deviation: 0.0023  (expected: < 0.01)
Length optimality: 1.0012  (expected: < 1.01)
Parallel transport error: 0.0015  (expected: < 0.01)
Status: PASS
```

### Failure Diagnosis

**High geodesic deviation.** Indicates the system does not follow geodesics. Possible causes:
- LAMBDA_G_DEFAULT too low
- FRICTION_SCALE too high
- Incorrect integrator

**Poor length optimality.** Indicates trajectories are not optimal. Solution:
- Increase LAMBDA_G_DEFAULT
- Reduce DEFAULT_DT
- Increase LEAPFROG_SUBSTEPS

## Differentiability Test

All operations must be differentiable for training.

### Execution

```bash
python tests/architecture/test_differentiability.py
```

### Checks

The test verifies that:
- The forward pass produces gradient tensors
- Gradients are not NaN or Inf
- Gradients have correct shapes
- The gradient of gradients (Hessian) exists where applicable

### Failure Diagnosis

**NaN gradients.** Indicates non-differentiable operations. Check:
- Non-differentiable operations in the forward
- Division by tensors that can be zero

**Zero gradients.** Indicates the computation graph is broken. Check:
- That there are no incorrect detach() or no_grad()
- That operations are part of the graph

## Gradient Parity Test

CUDA backend gradients must match Python.

### Execution

```bash
python tests/cuda/verify_cuda_autograd.py
```

### Metrics

Reports differences between gradients:

```
Gradient: dL/dq
  Max abs diff: 2.3e-7
  Status: PASS

Gradient: dL/dp
  Max abs diff: 1.8e-7
  Status: PASS
```

### Failure Diagnosis

**Large gradient differences.** Indicates errors in the CUDA backward pass. Check:
- That the CUDA backward matches the analytic formula
- That there are no errors in gradient propagation

## Full Validation Suite

For exhaustive validation:

```bash
python tests/run_suite.py --all
```

This suite runs all validation tests and generates a report.

### Suite Report

The report includes:

- Summary of passed/failed tests
- Performance metrics
- Alerts about potential issues
- Recommendations

### Approval Criteria

For the build to be approved:

- 100% of unit tests passed
- 100% of parity tests passed (or within tolerance)
- 100% of differentiability tests passed
- Conservation audit: drift < 1%
- Geodesic test: deviation < 0.01

## Continuous Monitoring

During development, run the suite regularly:

```bash
# Before commit
python tests/run_suite.py --quick

# Integration
python tests/run_suite.py --all
```

The quick test includes only the most critical tests.

## Problem Reporting

If you encounter validation issues:

1. Document the exact command and output
2. Include the code version (git log)
3. Include CUDA and driver versions
4. Include hardware (GPU, RAM)
5. Create an issue with this information

---

**Manifold Labs (Joaquín Stürtz)**
