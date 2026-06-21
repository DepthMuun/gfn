# Numerical Validation

This guide describes the validation paths that actually exist in the current repository.

It does not assume a single monolithic orchestrator. The current tree is better understood as a set of focused health tests, audits, and benchmark scripts.

## What To Validate First

For GSSM, the most useful validation layers are:

- component health tests
- integration tests across topology / integrator / mixer combinations
- targeted physics audits
- benchmark scripts for runtime or convergence behavior

## 1. Health Test Suite

The main automated validation entry point that clearly exists today is:

```bash
pytest tests/gssm/health -v
```

This area contains:

- unit tests for geometries and integrators
- config override tests
- force and attention tests
- integration tests for model combinations
- pipeline-level checks

Useful sub-runs include:

```bash
pytest tests/gssm/health/unit/test_geometries.py -v
pytest tests/gssm/health/unit/test_integrators.py -v
pytest tests/gssm/health/integration/test_combinations.py -v
pytest tests/gssm/health/integration/test_pipeline.py -v
```

## 2. Forward/Backward Compatibility Checks

Two important questions for GSSM are:

- can the model be instantiated across relevant config combinations?
- does a forward and backward pass still work?

The integration suite under `tests/gssm/health/integration/` is a better current source of truth for that than older docs that referenced generic “numeric validation” directories that no longer match the tree cleanly.

## 3. Physics Audit Scripts

There are also focused audit scripts under `tests/gssm/benchmarks/physics/`.

One current example is:

```bash
python tests/gssm/benchmarks/physics/integration_audit.py
```

This script is not a universal certification suite. It is a focused comparative audit for explicit versus semi-implicit friction handling under a simple oscillatory setup.

Use it to inspect:

- trajectory behavior
- energy variance trends
- relative stability of update formulas

## 4. Integrator Benchmark Scripts

There are performance-oriented scripts under:

- `tests/gssm/benchmarks/stress/performance/`

For example:

```bash
python tests/gssm/benchmarks/stress/performance/bench_integrators.py
```

Important caveat:

- some of these scripts contain older benchmarking code paths and should be treated as exploratory tooling, not as the sole source of runtime truth

They are useful for investigation, but the health tests and the current model/runtime code remain the authority.

## 5. CUDA And Backend Validation

If you are debugging CUDA-specific behavior, validate in layers:

1. first run the general health suite
2. then run the relevant stress/performance script
3. compare the behavior against CPU or simpler paths

Do not assume one repository path like `tests/gssm/unit/cuda/` exists as a canonical suite just because older docs referenced it.

## 6. What Counts As A Good Result

A practically good result for current GSSM work looks like this:

- model instantiates under the intended config
- forward pass returns the expected contract
- backward pass runs without NaNs
- health tests for the changed area pass
- any targeted audit you care about behaves consistently before and after the change

That is more reliable than using one fixed numeric threshold copied from an older version of the docs.

## 7. Minimal Validation Order

When you change runtime code, this is the safest order:

1. run the smallest relevant health test
2. run the matching integration test
3. run a targeted audit or benchmark only if the change affects runtime behavior materially

Examples:

```bash
pytest tests/gssm/health/unit/test_integrators.py -v
pytest tests/gssm/health/integration/test_combinations.py -v
python tests/gssm/benchmarks/physics/integration_audit.py
```

## 8. What To Record When Something Fails

When a numerical issue appears, record:

- exact command used
- affected topology / integrator / mixer
- input shape and dtype
- whether CUDA was enabled
- whether the failure is in instantiation, forward, backward, or a benchmark script

That is usually enough to reproduce the issue without relying on vague “the suite failed” reports.
