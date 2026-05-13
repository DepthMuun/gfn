# DepthMuun Benchmarking Guide

## Overview

This document describes the benchmarking framework integrated within the DepthMuun (GFN v2) architecture. The suite evaluates model performance natively against state-of-the-art baselines like `micro_gpt` across several functional pillars: convergence proofs, hardware matrix testing, and system stress testing.

The benchmarks are specifically designed to test the O(1) continuous state-space claims, energy conservation over deep trajectories, and the capacity for symbolic learning through geometric representation.

### Benchmark Directory Structure

The benchmark suite is fundamentally split into four functional pillars:

```text
tests/benchmarks/
├── baselines/               # Baseline model implementations (e.g., micro_gpt.py)
├── convergence/             # Generalization and logical deduction proofs
│   ├── xor/
│   ├── language/
│   ├── needle_haystack_real/
│   └── ...
├── matrix/                  # Hyperparameter permutation and stability validation
└── stress/                  # Computational overhead, latency, and hardware constraints
```

---

## Convergence Benchmarks

Convergence testing ensures the model can empirically solve mathematically rigorous problems that historically stump Recurrent Neural Networks and standard Transformers.

### Current Test Scenarios

1. **XOR Parity Bounds (`convergence/xor/logic_xor.py`)**  
   Evaluates pure combinatorial logic. Models are assessed on their ability to solve parity arrays of arbitrary bounds using `ToroidalRiemannianGeometry`, proving the model can isolate and recall alternating state sequences over deep step counts.

2. **Language Context (`convergence/language/lang_context.py`)**  
   Validates standard autoregressive perplexity parameters using synthetic dictionaries. Ensures the internal geometric representation translates cleanly to sequence prediction and language modeling.

3. **Needle In A Haystack (`convergence/needle_haystack_real/`)**  
   Measures memory capacity retrieval within large document chunks. Evaluates if the geometric system can correctly preserve and isolate singular signal spikes over massive contextual noise thresholds.

4. **SHA256 Mapping (`convergence/sha256/`)**  
   A brutal cryptographic hashing simulation designed to test collision boundaries. Evaluates the pure continuous capacity of the state space by forcing chaotic input resolution.

---

## Matrix Testing

The `tests/benchmarks/matrix/` module systematically validates the interaction between different system settings by permuting over wide hyperparameter domains (Integrator types, geometries, physics variables). 

### The Run Suite (`run_suite.py`)

The matrix suite uses a unified generator -> runner -> analyser pipeline:

- `MatrixGenerator`: Spawns combinatorial combinations of active `ManifoldConfig` states.
- `MatrixRunner`: Executes isolated mini-epochs capturing gradient norms, explosion rates, and convergence speeds.
- `MatrixAnalyser`: Consolidates the permutations to mathematically declare the most robust architecture setups.

**Execution:**
```bash
python tests/benchmarks/matrix/run_suite.py --limit 10
```

*Tip: Use `--filter-integrator leapfrog` to restrict the search space during iterative debugging.*

---

## Stress & Hardware Profiling

The `tests/benchmarks/stress/` suite evaluates the underlying hardware overhead and scaling characteristics. 

### Latency & Performance Suites

- **Overhead Tests (`bench_overhead.py`)**: Precisely clocks framework initialization versus computation time.
- **CUDA Live Interaction (`bench_cuda_live.py`)**: Tests raw throughput via native PyTorch bindings against standard Python autograd graphs. Ensures the CUDA custom C++ engines correctly execute step operations.
- **Performance (`bench_performance.py`)**: Determines tokens-per-second capabilities when scaling `model.dim` from embedded (256) architectures to dense cluster models (1024+).

### Precision Checks (`bench_integrators.py`)

Verifies the mathematical rigor of different integrators (Yoshida vs Leapfrog vs Heun). It calculates energy drift percentiles in continuous steps.

```
Expected Metric Standard:
- Energy drift percentage < 1% over 1000 trajectory steps.
- Zero spurious monotonic energy gains under constant boundary interactions.
```

---

## Model Baselines

Benchmarks are routinely checked against matched structural baselines mapping traditional network topologies.

- **`micro_gpt.py`**: A clean, modern transformer baseline configured with exact parameter boundaries to evaluate perplexity metrics directly against the Geometry Flow standard.

To ensure fair scientific evaluation, parameter counts (`dim`, `depth`, `heads`) are tightly coupled between models to analyze architectural advantages independent of parameter sizes.

---

**DepthMuun (GFN v2)**
