# GFN: Geometric Flow Networks

## A Physics-Informed Paradigm for Sequential Intelligence

[![Framework: GFN](https://img.shields.io/badge/Paradigm-GFN_2.7.1-blue.svg)](https://github.com/DepthMuun/gfn)
[![Models: Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-orange.svg)](https://huggingface.co/DepthMuun)
[![DOI: 10.5281/zenodo.19141133](https://img.shields.io/badge/DOI-10.5281/zenodo.19141133-blue.svg)](https://doi.org/10.5281/zenodo.19141133)
[![Status: Active](https://img.shields.io/badge/Status-Active-green.svg)]()
[![Package: GFN PyPi](https://static.pepy.tech/personalized-badge/gfn?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pypi.org/project/gfn/)

> "Intelligence is not statistical correlation; it is the continuous evolution of a persistent world-state governed by physical invariants."

**Geometric Flow Networks (GFN)** represents a fundamental shift in neural architecture. Instead of treating computation as token-to-token correlation (Attention), GFN treats it as a **trajectory** within a high-integrity dynamical system where inputs act as external perturbations that drive state evolution according to structural invariants.

---

## The GFN Paradigm

The GFN paradigm (Stürtz, 2026) formalizes reasoning as structural flow. In this framework, inputs act as external perturbations $\mathbf{f}_{ext}$ that drive a persistent state toward a configuration satisfying global conservation laws.

---

## The Five Pillars of GFN

For an architecture to be considered a valid GFN realization, it must satisfy **all five pillars**:

### Pillar 1: Persistent Internal World

The system must maintain a **simulator**, not a memory buffer. The state evolves through the space, not merely accumulates inputs.

```
INVALID (Transformer with KV-cache):     Token → Correlation → Token (with memory buffer)
GFN (Valid):                             Token → Perturbation → World State Evolution

The KV-cache is a "memory crutch" - if you cut the cache, the Transformer forgets everything.
The GFN world-state is the geometry itself. Memory is intrinsic to the curvature, not an external buffer.
```

### Pillar 2: At Least One Invariant

The system must encode at least one physical/mathematical invariant that acts as the "gravity" of the model. This prevents the internal world from becoming a latent hallucination.

Examples of valid invariants:
- **Casimir Operators**: Commute with all generators of a Lie algebra
- **Hamiltonian Conservation**: Total energy preserved through symplectic integration
- **Norm Preservation**: State norm remains bounded
- **Group Symmetries**: Topological constraints that make invalid states geometrically impossible
- **Phase Space Volume**: Liouville's theorem preservation

```
WITHOUT INVARIANTS:  "A map without a compass. Without invariants, the world-state is a latent hallucination."
WITH INVARIANTS:     "The laws of gravity in your model. They prevent the internal world from disintegrating."
```

### Pillar 3: Structural Integrity

The state cannot collapse to zero or explode to infinity. The invariant guarantees that informational "volume" remains constant.

This resolves the gradient problem intrinsically:
- No gradient vanishing (singular values = 1)
- No gradient explosion (volume preserved)
- The system is stable by geometry, not by patching

### Pillar 4: Temporal Locality (For True O(1))

**Requirement**: The computational cost of updating the state must be independent of how many tokens came before.

```
VALID:   state_update = f(current_state, present_input)  # O(1) per step
INVALID: state_update = f(current_state, all_previous_tokens)  # O(N) - breaks O(1) promise

"The flow is calculated over the current state and present input, nothing more."
```

If an architecture's forward pass loops over history, it breaks the O(1) promise.

### Pillar 5: Geometric Differentiability (For Metric Coherence)

**Requirement**: All states must exist in a manifold where "distance" is physically coherent. This defines how "forces" and "flows" operate.

```
For there to be "forces" and "flows", you need to know how far one concept is from another in that curved space.

Rule: "Every state must exist in a variety where the notion of 'distance' is physically coherent."
```

---

## The GFN Definition

A **Geometric Flow Network (GFN)** is a neural architecture satisfying all five pillars above:

1. Persistent Internal World (Simulator, not buffer)
2. At Least One Physical/Mathematical Invariant
3. Structural Integrity (gradient stability by geometry)
4. Temporal Locality (O(1) state update)
5. Geometric Differentiability (metric coherence)

### Formal Definition

$$
\mathbf{W}_{t+1} = \mathcal{T}(\mathbf{W}_t, \mathbf{f}_{ext}; \theta)
$$

Where:
- $\mathbf{W}_t$ is the internal world state at time $t$
- $\mathbf{f}_{ext}$ is the external input (perturbation)
- $\mathcal{T}$ is a transfer operator that:
  - Preserves at least one invariant
  - Operates on $\mathbf{W}_t$ and $\mathbf{f}_{ext}$ ONLY (no history)
  - Is differentiable with respect to a coherent metric

---

## Why Geometric Flow Networks?

### Comparison to Related Approaches

| Architecture | Persistent World | Invariant | Integrity | O(1) Update | Metric |
|-------------|-----------------|-----------|-----------|--------------|--------|
| **Transformer + KV-cache** | ❌ (buffer) | ❌ | ❌ | ❌ | ❌ |
| **Mamba/SSM** | ✅ | ❌ | ❌ | ✅ | ❌ |
| **World-State Networks** | ✅ | ❌ | ❌ | ✅ | ❌ |
| **GFN** | ✅ | ✅ | ✅ | ✅ | ✅ |

### Key Distinctions

**GFN vs Transformer:**
- Transformer: "Guesses" by statistics
- GFN: "Orbits" solutions because geometric structure doesn't allow stepping outside physics

**GFN vs SSM:**
- SSM: "Radio signal" - state can collapse or explode
- GFN: "River flow" - state orbits around physically coherent solutions

**GFN vs World-State Networks:**
- World-state without invariants: "Photo on a post-it" - blurry, degrades over time
- GFN: "Full simulator" - the world exists as geometry, not memory

---

## Latent Planning Capability

A critical consequence of the five pillars:

> GFN can "predict" the future without generating token by token, simply by moving the state vector through the geometric flow.

The internal world enables latent planning:
- Future states can be computed by flowing through the manifold
- No need to autoregressively generate each token to "see" what comes next
- The geometry of the world encodes causal structure

---

## Complexity Characteristics

**GFN is a paradigm with five mandatory requirements.** Complexity characteristics depend on implementation:

| Requirement | Complexity Impact |
|-------------|------------------|
| Temporal Locality (Pillar 4) | Enables O(1) inference memory |
| Structural Integrity (Pillar 3) | Intrinsic gradient stability |
| All Pillars Combined | No KV-cache, no O(N²) attention |

---
# Install

Install with PyPi:
```
pip install  gfn
```

## Paradigm Documentation

For detailed theoretical foundations and mathematical formalism, see:

- [THEORY.md](docs/THEORY.md) - Complete mathematical foundations
- [ARCHITECTURES.md](ARCHITECTURES.md) - Available realizations
- [CONTRIBUTING.md](CONTRIBUTING.md) - Adding new realizations
- [GEOMETRY_IS_ALL_YOU_NEED.tex](GEOMETRY_IS_ALL_YOU_NEED.tex) - Primary research paper

### Citation

```latex
@article{sturtz2026geometry,
  title={Geometric Flow Networks: A Physics-Informed Paradigm for Sequential Intelligence},
  author={Stürtz, Joaquín},
  journal={Zenodo Preprints},
  year={2026},
  doi={10.5281/zenodo.19141133},
  url={https://doi.org/10.5281/zenodo.19141133}
}
```

---

## License

This project is proprietary to DepthMuun Research. See the `LICENSE` file for details.

**Author**: Joaquín Stürtz, DepthMuun Research  
**Version**: 2.7.1  
**Date**: March 2026
