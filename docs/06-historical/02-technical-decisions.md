# Technical Decisions

## General Philosophy

This document justifies the architectural and technical decisions made during the project's development. The goal is for future maintainers to understand the reasoning behind the choices, not just the what.

## Decision: Leapfrog Integrator over Other Integrators

### Context

The original version (v2.6.5) used Leapfrog without clear documentation of why. During the rewrite, we considered multiple options: Euler, Runge-Kutta 4, Yoshida, Forest-Ruth, and Leapfrog.

### Analysis

Euler was too unstable for Hamiltonian systems. RK4 was precise but did not preserve symplectic structure. Yoshida and Forest-Ruth offered better precision but at 3x the computational cost.

Leapfrog offers the best balance for training. Its second-order accuracy is sufficient for optimizer gradients. Its symplectic structure preserves Hamiltonian system properties. Its cost is similar to Euler but with better stability.

### Decision

Keep Leapfrog as the default integrator. Implement Yoshida and Forest-Ruth as options for users who need higher precision.

### Implications

- Users who want more precision can change integrator_type to "yoshida"
- The integrator code is isolated to make it easier to add new types
- Tests verify correctness for all integrators

## Decision: Implicit Friction over Explicit

### Context

Version v2.6.5 used explicit friction:

v_new = v + h * (force - friction)

This caused issues when friction was high: the denominator in subsequent operations could be small.

### Analysis

Implicit friction:

v_new = (v + h * force) / (1 + h * mu)

prevents instability because the denominator never approaches zero. Tested with friction values from 0.1 to 1.0, the implicit form is always stable.

### Decision

Switch to implicit friction with protective EPSILON.

### Implications

- The backward pass changed to include the denominator gradient
- Friction values that previously caused divergence now work
- Users migrating from v2.6.5 needed to adjust their configs

## Decision: Hysteresis System

### Context

The original model had no long-term memory. Each token was processed independently, without reference to prior states beyond attention.

### Analysis

Hysteresis adds a memory mechanism that persists across tokens. The forget gate controls how much of the previous state is retained. The ghost force allows previous states to influence current dynamics.

This allows the model to capture long-range dependencies in a structured way, not only through attention.

### Decision

Add a hysteresis system with configurable parameters.

### Implications

- The model state grew to include hysteresis variables
- The backward pass grew to include hysteresis gradients
- Users can disable hysteresis if they do not need it

## Decision: Trace Normalization

### Context

The metric computed by the neural network could become singular (eigenvalues near zero) in some regions of space.

### Analysis

Trace normalization:

g_normalized = g / trace(g) * dim

preserves the shape of the metric while ensuring the trace is constant. This prevents singularities without changing the metric's principal directions.

### Decision

Enable trace normalization by default, with an option to disable it.

### Implications

- Christoffel computation is more stable
- The physical interpretation of the metric changes slightly
- Users who need the original metric can disable it

## Decision: Fused CUDA Kernels

### Context

The original implementation had separate kernels for kick, drift, and kick. This required three passes over GPU memory.

### Analysis

By fusing operations into a single kernel:

- The number of memory transfers is reduced
- GPU throughput is improved
- Latency per token is reduced

The cost is greater kernel code complexity and more difficult debugging.

### Decision

Implement fused kernels for critical operations (leapfrog, Christoffel).

### Implications

- The CUDA code is harder to maintain
- Python/CUDA parity verification is needed
- The speedup justifies the complexity (2-3x)

## Decision: Python/CUDA Parity Policy

### Context

In v2.6.5, Python and CUDA could produce slightly different results without anyone noticing. This caused different behavior between development (CPU) and production (GPU).

### Analysis

Parity is critical for debugging and reproducibility. Without it, you cannot be sure that what you test on CPU is what you run on GPU.

We established:
1. Automated parity tests
2. Per-operation tolerances
3. CI fails if parity breaks

### Decision

Make parity a non-negotiable requirement.

### Implications

- Constant changes must be reflected in both backends
- Operations must stay synchronized
- Development is slower but more robust

## Decision: Modular Configuration System

### Context

v2.6.5 had parameters hardcoded in multiple places. Changing something required searching throughout the code.

### Analysis

A centralized configuration system offers:
- A single place to change parameters
- YAML files for reproducibility
- Environment variables for deployment
- Centralized documentation

### Decision

Create a centralized constants.py and YAML configuration system.

### Implications

- constants.py is now the source of truth
- Values can be overridden by YAML or CLI
- Constants documentation is in one place

## Decision: Documentation Reorganization

### Context

v2.6.5 documentation was scattered:
- Basic README.md
- docs/ with files without clear structure
- Papers in docs/00_papers/
- No index

### Analysis

Documentation without structure is hard to navigate. Users do not know where to look for information.

### Decision

Reorganize into thematic modules with a centralized index.

### Chosen Structure

- 01-introduccion/: For new users
- 02-conceptos-core/: Theoretical explanations
- 03-referencia/: API and constants
- 04-guias/: Practical tutorials
- 05-analisis/: Technical studies
- 06-historico/: Project evolution

### Implications

- Users can navigate by level
- Documentation maintenance is easier
- The structure guides writing

## Decision: Conservative Constants

### Context

v2.6.5 had aggressive values:
- FRICTION_SCALE = 5.0 (very high)
- DEFAULT_LR = 1e-3 (aggressive)
- DEFAULT_DT = 0.1 (large)

### Analysis

Aggressive values give fast convergence but can:
- Cause divergence on difficult problems
- Limit model capacity
- Make training fragile

Conservative values give:
- More stability
- More manifold exploration
- Better generalization

### Decision

Change default values to conservative, with documentation on how to increase them.

### Implications

- Training may be slower initially
- v2.6.5 users need to adjust configs
- Convergence is more robust

## Decision: Backward-Compatible API

### Context

The development version introduced significant changes that break the v2.6.5 API.

### Analysis

Breaking compatibility has costs:
- Existing users have to rewrite code
- Old benchmarks are not comparable
- There is migration work to do

But maintaining compatibility also has costs:
- More complex code with branches
- Fundamental changes cannot be made
- Technical debt accumulates

### Decision

Break compatibility explicitly (major version), with migration documentation.

### Implications

- Version 3.0 marks the change
- Migration documentation is available
- v2.6.5 users must adapt their code

## Pending Decisions

The following decisions are under consideration:

1. **bf16 precision**: Is it worth the effort to support bf16?
2. **Transformers integration**: Should it integrate with Hugging Face?
3. **Checkpoint format**: Change to an industry standard format?
4. **Distribution**: Offer an official pip distribution?

## Acknowledgments

The decisions documented here reflect discussions with the development team and user feedback. Thanks to everyone who contributed.

---

**Manifold Labs (Joaquín Stürtz)**
