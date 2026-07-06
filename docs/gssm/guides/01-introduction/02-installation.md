# Installation

This guide covers a practical setup path for working with GSSM inside the current `gfn` repository.

It avoids older installation instructions that referenced historical demos or repository layouts that no longer match the tree cleanly.

## System Requirements

Recommended baseline:

- Python `3.10+`
- PyTorch installed and importable
- optional NVIDIA CUDA setup if you want GPU acceleration

You can work on CPU, but many GSSM experiments and benchmarks are much slower there.

## Create A Virtual Environment

Use a clean virtual environment before installing dependencies.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

If the project is being used as a local editable package in your environment, also ensure the repo root is the active working tree when running scripts.

## Verify The Core Import

Run the simplest import check first:

```bash
python -c "import gfn; print('gfn import ok')"
```

If this fails, solve that before trying any benchmark or training script.

## Verify PyTorch

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

This confirms:

- PyTorch imports correctly
- CUDA is visible if you expect GPU support

## Optional CUDA / Native Paths

The repository contains CUDA and native extension code under the GSSM realization, for example:

- `gfn/realizations/gssm/csrc/`
- `gfn/realizations/gssm/cuda/`

These paths are relevant if you are explicitly working on backend performance or low-level kernels.

For normal model usage and documentation work, the Python runtime path is enough to get started.

## First Runtime Check

Once the import works, verify the public API:

```bash
python -c "import gfn; m = gfn.create('gssm', vocab_size=16, dim=64, depth=2, heads=4); print(type(m).__name__)"
```

This is a much better current sanity check than relying on older demo scripts outside the present GSSM workflow.

## First Validation Check

After the import and model-construction checks work, run the health suite:

```bash
pytest tests/gssm/health -v
```

That is the most direct current validation path in the repository for GSSM installation and runtime consistency.

## First Benchmark Check

If you want a real end-to-end task after the health suite:

```bash
python tests/gssm/benchmarks/convergence/math/train_math.py
```

This benchmark uses the current public API and is a better match for the present codebase than historical demo paths.

## Common Issues

### `ModuleNotFoundError: No module named 'gfn'`

Make sure:

- you are in the repository environment
- dependencies are installed
- you are launching Python from the project root or from an environment that can resolve the package

### `torch` import errors

Reinstall or upgrade PyTorch in the active environment.

### CUDA is unavailable

That does not block basic GSSM usage. Start on CPU, then fix the CUDA environment separately.

### Health tests fail immediately

Check:

- Python version
- PyTorch version
- whether the environment is mixing incompatible packages
- whether the repo root and imports are resolving consistently

## Recommended Order

Use this order every time:

1. create environment
2. install dependencies
3. verify `import gfn`
4. verify `gfn.create("gssm", ...)`
5. run `pytest tests/gssm/health -v`
6. run one benchmark script if needed
