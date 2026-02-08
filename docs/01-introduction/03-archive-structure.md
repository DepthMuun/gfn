# File Structure

## Root Directory

The root directory contains configuration files and project metadata. LICENSE contains the terms of use for the code. README.md provides a quick overview of the project. requirements.txt lists all Python dependencies. STRUCTURE.md is a legacy document describing the original structure.

Build files (SConstruct, compile.bat) are for users who manually compile CUDA kernels. Most users can ignore these files.

The dist directory contains distributed builds of the project. The .whl files are installable Python wheels. The .tar.gz files are source code for manual installation.

## gfn/ Directory

This is the main source code directory. gfn/__init__.py exports the main classes and functions for external use.

gfn/constants.py defines all numeric constants in the system. Any parameter changes should be made here or in the YAML configuration files.

gfn/exceptions.py defines project-specific exceptions for error handling.

## gfn/ Subdirectories

The code is organized into subdirectories by functionality. This organization makes navigation and code maintenance easier.

The core/ directory contains the main Manifold class and the adjoint system for differentiation. The manifold.py file defines the public class that users instantiate. adjoint.py implements automatic differentiation for integrator operations.

The cuda/ directory contains custom CUDA kernels and their Python wrappers. autograd.py defines differentiable functions that call CUDA kernels. cuda_kernels.cpp is the C++ code for the kernels. core.py and ops.py are low-level wrappers.

The geometry/ directory contains metric implementations and Christoffel symbol computation. lowrank.py is the main low-rank implementation. hyper.py defines hyperbolic geometry. toroidal.py defines toroidal geometry. adaptive.py and hierarchical.py implement adaptive variants.

The integrators/ directory contains numerical integrator implementations. symplectic/ includes symplectic integrators (leapfrog, verlet, yoshida, forest_ruth). runge_kutta/ includes generic Runge–Kutta integrators (euler, heun, rk4, dormand_prince).

The losses/ directory contains loss functions. hamiltonian.py implements the energy conservation term. geodesic.py implements geodesic regularization. combined.py combines multiple loss terms.

The layers/ directory contains custom neural layers. base.py defines the base layer. fractal.py implements fractal composition. gating.py implements activation gates.

The embeddings/ directory contains embedding initializations. siren.py implements SIREN for implicit embeddings. implicit.py and functional.py include other initializations.

The utils/ directory contains miscellaneous utilities. visualization.py contains plotting functions. scan.py implements parallel scan. safety.py includes stability checks.

## configs/ Directory

configs/ contains YAML configuration files for different experiments and hardware.

configs/model/ contains model architecture configurations (gfn_small, gfn_medium, gfn_large). Each file defines dimension, depth, number of heads, and other structural parameters.

configs/training/ contains training configurations (learning rate, batch size, number of steps). experiment_medium.yaml is the standard configuration for experiments.

configs/hardware/ contains GPU-specific optimizations (gtx_1650, rtx_4090). They adjust batch size and precision based on hardware capacity.

configs/demos/ contains configurations for specific demos (copy task, sorting, wikitext).

## demos/ Directory

demos/ contains example scripts for different tasks.

demos/sorting/ contains sorting experiments. train_sorting.py trains a model to learn sequence sorting. Variants (train_hyper_sorting.py, train_inf_sorting.py) use alternative configurations.

demos/tinystories/ contains training on the TinyStories dataset. This dataset is useful for quick tests because it is small but demonstrates text generation capabilities.

demos/wikitext/ contains training on Wikitext for more serious benchmarks.

## tests/ Directory

tests/ contains tests organized by type.

tests/unit/ contains unit tests for individual components. test_geometry.py verifies Christoffel symbol computation. test_integrators.py verifies integrator correctness.

tests/architecture/ contains architecture tests. test_differentiability.py verifies that all operations are differentiable. test_learning_dynamics.py monitors training metrics.

tests/cuda/ contains GPU-specific tests. verify_cuda_autograd.py verifies CUDA autograd correctness.

tests/diagnostics/ contains diagnostic tools. conservation_audit.py verifies energy preservation. parity_probe.py verifies Python/CUDA parity.

tests/benchmarks/ contains benchmarking scripts. benchmark_performance.py measures throughput. benchmark_precision_stability.py verifies numerical stability.

## docs/ Directory

docs/ contains all project documentation.

docs/00_papers/ contains research papers that underpin the design. Specific papers address different theoretical aspects.

docs/00_AUDITS/ contains technical code audits and stability analysis.

docs/00_HISTORY/ contains historical documentation of breakthroughs and technical decisions.

## Naming Conventions

Python files use snake_case (lowercase with underscores). Classes use PascalCase (capitalized words).

Configuration files use kebab-case (lowercase with hyphens) for compatibility with command-line tools.

Tests use the test_ prefix followed by the component name they test.

Documentation files use the NN-descriptive-name.md format to facilitate ordering.

---

**Manifold Labs (Joaquín Stürtz)**
