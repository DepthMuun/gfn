# DepthMuun GFN Test Suite

This directory contains the verification suite for the Geodesic Flow Networks (GFN) framework by **DepthMuun**. The suite is organized into three central pillars for maximum clarity and maintainability.

## Structure

### 1. 🧪 `unit/` (Atomic Logic)
Tests individual classes and functions in isolation. Includes geometry checks, shape validation, and core mathematical primitives.

### 2. 🧩 `integration/` (System Flow)
Ensures multiple components work together correctly. Includes full forward-backward passes, model-to-integrator wiring, and state state management.

### 3. 🚀 `benchmarks/` (Performance & Convergence)
Quantitative analysis of model capabilities, scaling limits, and long-term convergence properties (e.g., Matrix scaling, Stress tests).

## Usage

The suite is managed via a professional orchestrator:

```bash
# List test structure
python tests/orchestrator.py --list

# Run unit tests (Atomic)
python tests/orchestrator.py --unit

# Run integration tests (System)
python tests/orchestrator.py --integration

# Run benchmarks
python tests/orchestrator.py --benchmarks
```
