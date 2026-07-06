# Benchmarking Guide

This guide summarizes the benchmark structure that actually exists today under `tests/gssm/benchmarks/`.

It does not treat every script as equally canonical. Some benchmark scripts are current and useful, while others still carry older experimental assumptions.

## Benchmark Areas

The repository currently contains these main benchmark areas:

```text
tests/gssm/benchmarks/
├── baselines/
├── convergence/
├── matrix/
├── physics/
└── stress/
```

## 1. Convergence Benchmarks

`convergence/` contains task-oriented runs that check whether a model can actually learn or generalize on specific problems.

Current examples in the tree include:

- `xor/logic_xor.py`
- `language/run.py`
- `math/train_math.py`
- `MQAR/`
- `needle_haystack_*`
- `arc-agi-2/`
- `drone_detection/`

These are useful when you want to answer:

- does the model train at all on a real task?
- does a config change break convergence?
- does a geometry/readout/loss choice still fit the task?

## 2. Matrix Benchmarks

`matrix/` is the combinatorial benchmark area.

The current matrix flow is real and script-backed:

- `generator.py`
- `runner.py`
- `analyser.py`
- `run_suite.py`

Example command:

```bash
python tests/gssm/benchmarks/matrix/run_suite.py --limit 10
```

You can also restrict the search space:

```bash
python tests/gssm/benchmarks/matrix/run_suite.py --limit 10 --filter-integrator leapfrog
```

This area is useful for:

- comparing integrators
- comparing topology choices
- screening robustness across many small trials

## 3. Physics Audits

`physics/` contains targeted audits rather than broad training benchmarks.

One current example is:

```bash
python tests/gssm/benchmarks/physics/integration_audit.py
```

Use this area when you care about:

- friction behavior
- integration stability
- update-form comparisons

## 4. Stress And Performance Benchmarks

`stress/` contains runtime-oriented scripts such as:

- `performance/bench_overhead.py`
- `performance/bench_performance.py`
- `performance/bench_integrators.py`
- `performance/bench_cuda_live.py`
- `scale/bench_scaling.py`
- `scale/test_batch_scaling.py`

Use these when you want to inspect:

- throughput
- memory pressure
- scaling behavior
- backend-specific performance

Important caveat:

- some performance scripts still include older code paths or legacy assumptions, so treat them as investigative tools rather than as the final word on current runtime behavior

## 5. Baselines

The `baselines/` folder currently includes at least:

- `micro_gpt.py`

This is useful when you want a rough comparison point, but benchmark conclusions should still be checked carefully because fairness depends on:

- parameter count
- tokenization or input format
- task framing
- loss/readout contract
- training budget

## Recommended Workflow

Use benchmarks in this order:

1. health tests for correctness
2. convergence benchmark for the relevant task family
3. matrix or stress benchmark only if you are comparing configurations or runtime cost

This avoids using a large benchmark script to debug a basic shape or config problem.

## Example Commands

### Small convergence run

```bash
python tests/gssm/benchmarks/convergence/math/train_math.py
```

### Matrix screening

```bash
python tests/gssm/benchmarks/matrix/run_suite.py --limit 10
```

### Physics audit

```bash
python tests/gssm/benchmarks/physics/integration_audit.py
```

### Performance check

```bash
python tests/gssm/benchmarks/stress/performance/bench_overhead.py
```

## How To Read Results

For GSSM, benchmark results are only meaningful if you keep these aligned:

- task target
- readout type
- loss family
- topology / geometry choice
- integrator settings

If those are mismatched, a benchmark may “run” while still measuring the wrong thing.

## Practical Caveats

- Do not treat every benchmark script as a maintained production harness.
- Do not use old benchmark claims as framework-wide defaults.
- Do not compare runs unless the supervision and output contracts are actually equivalent.
