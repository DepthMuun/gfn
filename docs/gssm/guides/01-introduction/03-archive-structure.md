# File Structure

This guide maps the current repository structure in broad strokes so you can find the parts that matter for GSSM work.

It is intentionally practical. It does not try to describe every historical directory or every experimental artifact as if all of them were equally current.

## Repository Root

At the root of the repository you will find:

- project metadata such as `README.md`, `LICENSE`, `pyproject.toml`
- the main Python package under `gfn/`
- documentation under `docs/`
- tests under `tests/`
- some config material under `configs/`

For most GSSM work, the directories that matter are `gfn/`, `docs/`, and `tests/`.

## `gfn/`

This is the Python package root.

Important top-level files include:

- `gfn/__init__.py`
- `gfn/api.py`
- `gfn/constants.py`
- `gfn/errors.py`

This layer exposes the public API used by docs such as:

```python
import gfn

model = gfn.create("gssm", vocab_size=256)
```

## `gfn/realizations/`

This is where the concrete model families live.

Current major realizations include:

- `gfn/realizations/gssm/`
- `gfn/realizations/isn/`

If you are working on GSSM, this is the main code tree you will read.

## `gfn/realizations/gssm/`

The GSSM realization is organized by subsystem.

### Main runtime areas

- `config/`: schema, loader, normalizer, validator, serialization
- `geometry/`: analytical and learned geometries
- `physics/`: engine, dynamics, integrators, normalization
- `models/`: base model, manifold model, manifold layer, builders, components, plugins
- `losses/`: generative, physics, toroidal, detection, regularization
- `training/`: optimizer, metrics, trainer, scheduler, callbacks
- `utils/`: lower-level helpers and diagnostics

### Supporting areas

- `cuda/`: Python-side CUDA helpers
- `csrc/`: native and CUDA extension sources
- `core/`: internal state/types helpers
- `data/`: dataset and transform utilities
- `interfaces/`: shared protocols and interfaces
- `math/`: math helpers used by the realization

## `docs/`

The documentation tree currently includes:

- `docs/gssm/`
- `docs/isn/`
- `docs/dev/`

Within `docs/gssm/`, the current intended split is:

- `guides/`: user-facing conceptual and practical docs
- `technical/`: runtime-aligned implementation docs
- `00_papers/`: research and idea notes

If you want current runtime truth, prefer `docs/gssm/technical/` plus the code itself.

## `tests/`

The GSSM tests are mainly under:

- `tests/gssm/health/`
- `tests/gssm/benchmarks/`

### `tests/gssm/health/`

This is the best current place for correctness-oriented checks:

- unit tests
- integration tests
- component compatibility checks

### `tests/gssm/benchmarks/`

This is the broader experiment and benchmarking area:

- `convergence/`
- `matrix/`
- `physics/`
- `stress/`
- `baselines/`

Use it for runtime experiments, task-oriented checks, and comparative runs.

## `configs/`

There is some configuration material in the repository, but the most important config logic for GSSM is in code under:

- `gfn/realizations/gssm/config/schema.py`
- `gfn/realizations/gssm/config/loader.py`
- `gfn/realizations/gssm/config/normalizer.py`

So for GSSM, do not assume repository YAML folders are the main source of truth for defaults.

## How To Navigate Efficiently

If you are debugging behavior, read in this order:

1. `gfn/realizations/gssm/config/`
2. `gfn/realizations/gssm/models/factory.py`
3. `gfn/realizations/gssm/models/base.py`
4. `gfn/realizations/gssm/models/manifold_layer.py`
5. `gfn/realizations/gssm/physics/engine.py`
6. the matching geometry, loss, or integrator module

If you are debugging docs, read in this order:

1. current guide
2. matching `technical/` document
3. corresponding runtime file

## Naming Conventions

The current tree broadly follows:

- Python modules: `snake_case`
- classes: `PascalCase`
- tests: usually `test_*.py`
- ordered documentation: numeric prefixes such as `00-`, `01-`, `02-`

## Practical Warning

Some directories still include older experimental or historical material.

When in doubt, treat these as the current GSSM anchors:

- `gfn/realizations/gssm/`
- `docs/gssm/technical/`
- `tests/gssm/health/`
- `tests/gssm/benchmarks/`
