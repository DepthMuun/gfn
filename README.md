<div align="center">

# GFN: Geometric Flow Networks

### A Physics-Informed Paradigm for Sequential Intelligence

[![Framework: GFN](https://img.shields.io/badge/Paradigm-GFN_2.7.2-blue.svg?style=for-the-badge)](https://github.com/DepthMuun/gfn)
[![Models: Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-orange.svg?style=for-the-badge)](https://huggingface.co/DepthMuun)
[![DOI: 10.5281/zenodo.19141133](https://img.shields.io/badge/DOI-10.5281/zenodo.19141133-blue.svg?style=for-the-badge)](https://doi.org/10.5281/zenodo.19141133)
[![Status: Active](https://img.shields.io/badge/Status-Active-green.svg?style=for-the-badge)]()
[![Package: GFN PyPi](https://static.pepy.tech/personalized-badge/gfn?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pypi.org/project/gfn/)

</div>

---

## What is GFN?

**Geometric Flow Networks (GFN)** represent a fundamental shift in how neural architectures process sequential information. Instead of relying on token-to-token correlation mechanisms (Attention), GFN treats intelligence as a trajectory through a high-integrity dynamical system.

### Core Concept

```
┌─────────────────────────────────────────────────────────────┐
│  Traditional (Attention)          GFN (Geometric Flow)      │
├─────────────────────────────────────────────────────────────┤
│  Token → Correlation → Token      Token → Perturbation →    │
│  (with KV-cache buffer)           World State Evolution     │
│                                                             │
│  ❌ Memory buffer (crutch)        ✅ Persistent simulator  │
│  ❌ O(N^2) complexity             ✅ O(1) per step         │
│  ❌ Statistical guessing          ✅ Physical orbits       │
└─────────────────────────────────────────────────────────────┘
```

---

## The Five Pillars

A valid GFN implementation must satisfy all five requirements:

### 1. Persistent Internal World

The system maintains a simulator rather than a memory buffer. State evolves through geometric space rather than being stored in a cache.

```
Transformer KV-cache: "Remove the cache → instant amnesia"
GFN World-State: "The geometry itself IS the memory"
```

### 2. At Least One Invariant

Physical or mathematical invariants provide constraints that prevent state collapse or divergence, similar to how conservation laws operate in physics.

| Invariant Type | Description |
|:--------------|:------------|
| **Casimir Operators** | Commute with all generators of a Lie algebra |
| **Hamiltonian Conservation** | Total energy preserved through symplectic integration |
| **Norm Preservation** | State norm remains bounded |
| **Group Symmetries** | Topological constraints make invalid states impossible |
| **Phase Space Volume** | Liouville's theorem preservation |

### 3. Structural Integrity

The state cannot collapse to zero or explode to infinity. Gradient stability emerges from geometric properties rather than architectural patches:

- No gradient vanishing (singular values ≈ 1)
- No gradient explosion (volume preserved)
- Stable by design, not by intervention

### 4. Temporal Locality (True O(1))

Computational cost per step is independent of sequence length.

```python
# Valid GFN: O(1) per step
state_update = f(current_state, present_input)

# Invalid: O(N) - violates temporal locality
state_update = f(current_state, all_previous_tokens)
```

> "The flow is calculated over the current state and present input, nothing more."

### 5. Geometric Differentiability

All states exist in a manifold where distance is physically coherent. This enables meaningful gradients and coherent state evolution.

> "For there to be forces and flows, you need to know how far one concept is from another in that curved space."

---

## Formal Definition

A Geometric Flow Network is a neural architecture satisfying all five pillars above.

Mathematically:

$$
W_{t+1} = \mathcal{T}(W_t, f_{ext}; \theta)
$$

Where:
- $W_t$ is the internal world state at time $t$
- $f_{ext}$ is the external input (perturbation)
- $\mathcal{T}$ is a transfer operator that:
  - Preserves at least one invariant
  - Operates on $W_t$ and $f_{ext}$ ONLY (no history access)
  - Is differentiable with respect to a coherent metric

---

## Comparison to Related Approaches

| Architecture | Persistent World | Invariant | Integrity | O(1) Update | Metric |
|:-------------|:----------------:|:---------:|:---------:|:-----------:|:------:|
| **Transformer + KV-cache** | No (buffer) | No | No | No | No |
| **Mamba / SSM** | Yes | No | No | Yes | No |
| **World-State Networks** | Yes | No | No | Yes | No |
| **GFN** | Yes | Yes | Yes | Yes | Yes |

### Key Distinctions

**GFN vs Transformer:**
- Transformer: Relies on statistical correlation → Pattern matching
- GFN: Follows physical trajectories → Physics-constrained evolution

**GFN vs SSM:**
- SSM: State can collapse or explode without constraints
- GFN: State orbits around physically coherent solutions

**GFN vs World-State Networks:**
- World-state without invariants degrades over time
- GFN maintains coherence through geometric constraints

---

## Latent Planning Capability

The five pillars enable a significant emergent property:

> GFN can anticipate future states without token-by-token generation, by moving the state vector through the geometric flow.

This allows:
- Future state computation through manifold flow
- Non-autoregressive planning
- Causal structure encoded in geometry

---

## Complexity Characteristics

| Requirement | Impact |
|:------------|:-------|
| **Temporal Locality** | O(1) inference memory |
| **Structural Integrity** | Intrinsic gradient stability |
| **All Pillars** | No KV-cache, no O(N^2) attention |

---

## Documentation

- [THEORY.md](docs/THEORY.md) - Mathematical foundations
- [ARCHITECTURES.md](ARCHITECTURES.md) - Available implementations
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
- [Zenodo Preprint](https://doi.org/10.5281/zenodo.19141133) - Research paper

### Citation

```bibtex
@article{sturtz2026geometry,
  title={Geometric Flow Networks: A Physics-Informed Paradigm for Sequential Intelligence},
  author={St{\"u}rtz, Joaqu{\'i}n},
  journal={Zenodo Preprints},
  year={2026},
  doi={10.5281/zenodo.19141133},
  url={https://doi.org/10.5281/zenodo.19141133}
}
```

---

## Quick Start

```bash
pip install gfn
```

Or from source:

```bash
git clone https://github.com/DepthMuun/gfn.git
cd gfn
pip install -e .
```

```python
from gfn import GFNModel, PhysicsConfig

config = PhysicsConfig(dim=512, depth=6)
model = GFNModel(config)
output = model(input_tokens)
```

---

<div align="center">

*Intelligence flows through geometry*

**Author**: Joaquín Stürtz, DepthMuun Research  
**Version**: 2.7.2  
**License**: Apache 2.0

</div>
