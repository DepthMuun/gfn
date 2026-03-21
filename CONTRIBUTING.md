# Contributing to GFN Framework

## Adding a New Geometric Flow Network Realization

This document establishes the standards for contributing new GFN realizations to the framework. All contributions must implement valid Geometric Flow Networks adhering to the **Five Pillars of GFN**.

---

## Table of Contents

1. [The Five Pillars of GFN](#the-five-pillars-of-gfn)
2. [What is a Valid GFN Realization?](#what-is-a-valid-gfn-realization)
3. [Required Components](#required-components)
4. [Directory Structure](#directory-structure)
5. [Test Requirements](#test-requirements)
6. [Documentation Standards](#documentation-standards)
7. [Submission Process](#submission-process)

---

## The Five Pillars of GFN

For an architecture to be considered a valid GFN realization, it must satisfy **all five pillars**:

### Pillar 1: Persistent Internal World

The system must maintain a **simulator**, not a memory buffer.

**What this means:**
- State evolves through geometry, not merely accumulates inputs
- No KV-cache or history buffer
- Memory is intrinsic to the space structure, not external storage

**Anti-pattern:**
```python
# INVALID: Accumulating history in a buffer
for token in sequence:
    kv_cache.append(token)  # This is a memory buffer, not a world
    output = attention(query, kv_cache)
```

**Valid pattern:**
```python
# VALID: Evolving the world state
for token in sequence:
    world_state = evolve(world_state, token)  # State evolves, doesn't accumulate
```

### Pillar 2: At Least One Invariant

The system must encode at least one physical/mathematical invariant.

**What this means:**
- The invariant acts as "gravity" - prevents the world from disintegrating
- Examples: Casimir operators, Hamiltonian conservation, norm preservation, group symmetries
- Without invariants, the world-state becomes a "latent hallucination"

**Required documentation:**
```python
# Each realization must declare its invariant(s)
class MyGFN:
    invariant_type: str = "hamiltonian_conservation"  # or "casimir", "norm_preservation", etc.
    invariant_description: str = "Energy is preserved through symplectic integration"
```

### Pillar 3: Structural Integrity

The state cannot collapse to zero or explode to infinity.

**What this means:**
- Gradient stability is guaranteed by geometry, not by patching
- The invariant ensures informational "volume" remains constant
- No gradient vanishing or explosion by construction

**Verification:**
- Numerical validation that invariants hold over long sequences
- No artificial gradient clipping required for stability

### Pillar 4: Temporal Locality (For True O(1))

The computational cost of updating state must be independent of history length.

**What this means:**
```python
# VALID: O(1) per step
new_state = f(current_state, present_input)

# INVALID: O(N) - breaks O(1) promise
new_state = f(current_state, all_previous_tokens)  # Loops over history
```

**Rule:** "The flow is calculated over the current state and present input, nothing more."

**This is what enables true O(1) memory complexity.**

### Pillar 5: Geometric Differentiability

All states must exist in a manifold where "distance" is physically coherent.

**What this means:**
- For "forces" and "flows" to exist, distance must be defined
- The metric must be coherent with the state evolution
- Enables meaningful gradient flow through the geometry

---

## What is a Valid GFN Realization?

A GFN realization must satisfy **all five pillars** without exception.

### Mandatory Requirements Summary

| Pillar | Requirement | Verification |
|--------|-------------|--------------|
| 1 | Persistent Internal World | State evolves, doesn't accumulate |
| 2 | At Least One Invariant | Invariant documented and preserved |
| 3 | Structural Integrity | Gradient stability by geometry |
| 4 | Temporal Locality | O(1) state update (no history loops) |
| 5 | Geometric Differentiability | Metric coherence defined |

### Anti-Patterns (Invalid for GFN)

The following are **not** valid GFN realizations:

- **Transformer with KV-cache**: Buffer, not world; no invariants; O(N) update
- **Standard SSM without invariants**: Has persistent state but no "gravity"
- **World-state networks without invariants**: Degrades over time, no structural integrity
- **Any architecture looping over history**: Breaks Temporal Locality

### Examples of Valid Invariants

- **Casimir Operators**: Commute with all generators of a Lie algebra
- **Hamiltonian Conservation**: Total energy preserved through symplectic integration
- **Norm Preservation**: State norm remains bounded
- **Phase Space Volume**: Liouville's theorem
- **Topological Constraints**: Invalid states are geometrically impossible

---

## Required Components

Every GFN realization must include:

### 1. Core Implementation

```
realizations/[your_realization]/
├── __init__.py                 # Package initialization
├── api.py                      # Realization API
├── core/
│   ├── __init__.py
│   ├── world.py               # World state representation
│   └── types.py               # Type definitions
├── invariants/
│   ├── __init__.py
│   └── [your_invariant].py   # Invariant implementation
├── evolution/
│   ├── __init__.py
│   └── dynamics.py            # State evolution dynamics
├── metrics/
│   ├── __init__.py
│   └── coherence.py           # Metric/distance definitions
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── losses.py
└── utils/
    ├── __init__.py
    └── diagnostics.py
```

### 2. Invariant Declaration

```python
# invariants/base.py
from abc import ABC, abstractmethod

class Invariant(ABC):
    """Base class for GFN invariants."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the invariant (e.g., 'hamiltonian_conservation')."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the invariant."""
        pass
    
    @abstractmethod
    def measure(self, world_state) -> float:
        """Measure the invariant value for a given state."""
        pass
    
    @abstractmethod
    def is_preserved(self, world_state, tolerance=1e-6) -> bool:
        """Check if the invariant is preserved within tolerance."""
        pass
```

### 3. World State Definition

```python
# core/world.py
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class WorldState:
    """Represents the internal world state of a GFN."""
    
    # The geometric state (varies by realization)
    geometry: torch.Tensor
    
    # Invariant values (should remain constant)
    invariant_values: dict[str, float]
    
    # Metadata
    step: int = 0
    device: str = "cpu"
    
    def evolve(self, input perturbation: torch.Tensor) -> "WorldState":
        """Evolve the world state by applying an input perturbation."""
        # Implementation specific to the realization
        pass
    
    def measure_distance(self, other: "WorldState") -> torch.Tensor:
        """Measure geometric distance to another world state."""
        # Must be physically coherent
        pass
```

### 4. Registry Registration

```python
# registry.py additions
register_realization(
    name="your_realization",
    class_path="gfn.realizations.your_realization.YourGFN",
    config_class=YourRealizationConfig,
    pillars={
        "persistent_world": True,
        "invariant_type": "your_invariant_name",
        "structural_integrity": True,
        "temporal_locality": True,  # O(1) update
        "geometric_differentiability": True,
    },
    complexity={
        "inference_memory": "O(1)",
        "forward_pass": "O(1) per step",
    },
)
```

---

## Directory Structure

```
gfn_framework/
├── gfn/
│   └── realizations/
│       ├── api.py              # DO NOT MODIFY
│       ├── registry.py         # Extend with new registration
│       └── [your_realization]/ # Your implementation
├── docs/
│   └── [your_realization]/    # Your documentation
│       ├── README.md          # Realization overview
│       ├── theory.md          # Theoretical basis
│       ├── invariants.md       # Invariant documentation
│       ├── api.md             # API reference
│       └── examples.md        # Usage examples
├── tests/
│   └── [your_realization]/   # Your tests
│       ├── unit/
│       ├── integration/
│       └── benchmarks/
└── configs/
    └── [your_realization]/    # Example configurations
```

---

## Test Requirements

### Unit Tests (Mandatory)

Location: `tests/[your_realization]/unit/`

**Required Tests:**

1. **World Evolution Test** (`test_world_evolution.py`)
   ```python
   def test_state_evolves_not_accumulates():
       """World state must evolve, not accumulate like a buffer."""
       model = create_realization(config)
       world = model.init_world(batch_size=1)
       
       initial_geometry = world.geometry.clone()
       
       # Process sequence
       for token in sequence:
           world = model.evolve(world, token)
       
       # State should evolve, not just append
       assert world.geometry.shape == initial_geometry.shape
       assert not torch.allclose(world.geometry, initial_geometry)
   ```

2. **Invariant Preservation Test** (`test_invariant_preservation.py`)
   ```python
   def test_invariant_preserved_over_long_sequence():
       """The declared invariant must be preserved."""
       model = create_realization(config)
       world = model.init_world()
       
       initial_invariant = model.invariant.measure(world)
       
       # Process long sequence
       for _ in range(10000):
           world = model.evolve(world, random_input())
       
       final_invariant = model.invariant.measure(world)
       
       # Invariant should be preserved
       assert model.invariant.is_preserved(world, tolerance=1e-4)
   ```

3. **Temporal Locality Test** (`test_temporal_locality.py`)
   ```python
   def test_forward_pass_does_not_loop_history():
       """Forward pass must be O(1) - no loops over history."""
       model = create_realization(config)
       
       # Track operations in forward pass
       world = model.init_world()
       with track_operations() as ops:
           world = model.evolve(world, token)
       
       # Should NOT contain any history-scanning operations
       assert not any("scan" in op for op in ops)
       assert not any("reduce" in op and "history" in op for op in ops)
   ```

4. **Structural Integrity Test** (`test_structural_integrity.py`)
   ```python
   def test_no_gradient_collapse_or_explosion():
       """Gradients should be stable without explicit clipping."""
       model = create_realization(config)
       
       # Train for many steps
       for _ in range(1000):
           loss = model.training_step(data)
       
       # Gradients should remain bounded
       for param in model.parameters():
           grad_norm = param.grad.norm() if param.grad is not None else 0
           assert grad_norm < 10.0  # Should not explode
           assert grad_norm > 1e-8  # Should not vanish
   ```

5. **Geometric Distance Test** (`test_geometric_metrics.py`)
   ```python
   def test_distance_is_metric_coherent():
       """Geodesic distance must satisfy metric axioms."""
       model = create_realization(config)
       w1 = model.init_world()
       w2 = model.init_world()
       w3 = model.init_world()
       
       d12 = model.distance(w1, w2)
       d21 = model.distance(w2, w1)
       d13 = model.distance(w1, w3)
       d23 = model.distance(w2, w3)
       
       # Symmetry
       assert torch.isclose(d12, d21)
       
       # Triangle inequality
       assert d13 <= d12 + d23
       
       # Non-negativity and identity
       assert d12 >= 0
       assert d12 == 0 if w1 is w2 else True
   ```

### Integration Tests (Mandatory)

Location: `tests/[your_realization]/integration/`

**Required Tests:**

1. **End-to-End Training Test** - Train on simple task, verify convergence
2. **State Reset Test** - Verify `reset()` reinitializes properly
3. **Serialization Test** - Save/load produces equivalent model
4. **Long Sequence Test** - Verify invariants hold over very long sequences (10k+ steps)

### Benchmark Tests (Recommended)

Location: `tests/[your_realization]/benchmarks/`

- Parity/XOR extrapolation (structural law learning)
- Induction tasks (persistent memory)
- Custom domain-specific tasks

---

## Documentation Standards

### README.md Template

```markdown
# [Realization Name]

## Overview

Brief description of the realization.

## The Five Pillars Compliance

| Pillar | Status | Implementation |
|--------|--------|----------------|
| 1. Persistent World | ✅ | [Description] |
| 2. Invariant | ✅ | [Invariant name and type] |
| 3. Structural Integrity | ✅ | [How guaranteed] |
| 4. Temporal Locality | ✅ | [O(1) verification] |
| 5. Geometric Differentiability | ✅ | [Metric definition] |

## Invariant Documentation

**Invariant Type**: [casimir/hamiltonian/norm/other]

**Description**: [What is preserved and why]

**Verification**: [How invariance is measured]

## Verified Results

| Task | Result | Conditions |
|------|--------|------------|
| Task 1 | Metric | Details |

## Usage

```python
import gfn

model = gfn.create("[realization_name]")
# ...
```
```

### Invariants Documentation (`invariants.md`)

```markdown
# Invariants: [Realization Name]

## Declared Invariants

### Primary Invariant: [Name]

**Type**: [Physical/Mathematical]

**Physical Meaning**: [What this invariant represents in physics terms]

**Implementation**: [How it's computed in code]

**Preservation Mechanism**: [How the architecture ensures it's preserved]

## Invariant Measurement

[Code showing how to measure and verify the invariant]
```

---

## Submission Process

### Pre-Submission Checklist

- [ ] All five pillars satisfied (document in README)
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Invariant documented and verified
- [ ] Temporal locality verified (no history loops)
- [ ] Structural integrity verified (gradient stability)
- [ ] Geometric metrics defined and tested
- [ ] Registry registration complete
- [ ] Documentation complete

### Submission

Create a pull request with:
1. Implementation in `gfn/realizations/[your_realization]/`
2. Tests in `tests/[your_realization]/`
3. Documentation in `docs/[your_realization]/`
4. This checklist completed

---

## Questions?

Open an issue or contact the maintainers.

---

## Appendix: Invariant Examples

See existing realizations for reference implementations:

- **G-SSM**: Hamiltonian conservation via symplectic integration
- **ISN**: World coherence as invariant metric

Each demonstrates proper invariant implementation and verification.
