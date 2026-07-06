<div align="center">


*The current project documentation is out of date; in the next version, we will focus more on documentation, usage, testing, etc.*

# GFN: Geometric Flow Networks

### A Physics-Informed Paradigm for Sequential Intelligence

[![Framework: GFN](https://img.shields.io/badge/Paradigm-GFN_2.7.2-blue.svg?style=for-the-badge)](https://github.com/DepthMuun/gfn)
[![Models: Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-orange.svg?style=for-the-badge)](https://huggingface.co/DepthMuun)
[![DOI: 10.5281/zenodo.19141133](https://img.shields.io/badge/DOI-10.5281/zenodo.19141132-blue.svg?style=for-the-badge)](https://doi.org/10.5281/zenodo.19141132)
[![Status: Active](https://img.shields.io/badge/Status-Active-green.svg?style=for-the-badge)]()
[![Package: GFN PyPi](https://static.pepy.tech/personalized-badge/gfn?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pypi.org/project/gfn/)

</div>

---

## What is GFN?

**Geometric Flow Networks (GFN)** treat intelligence not as statistical pattern matching, but as the evolution of a state through a learned geometry. Where traditional architectures compute correlations between tokens, GFN computes trajectories through a manifold where state transitions follow paths of minimal resistance.

### Core Distinction

```
┌─────────────────────────────────────────────────────────────┐
│  Statistical Models                 GFN (Geometric Flow)      │
├─────────────────────────────────────────────────────────────┤
│  Token → Correlation → Token      State → Geodesic Flow →   │
│  (weighted similarity)            (trajectory through space) │
│                                                             │
│  ❌ Statistical transformation      ✅ Geodesic state flow     │
│  ❌ Memory buffer (KV-cache)        ✅ Persistent world      │
│  ❌ Global correlation (O(N^2))     ✅ Causal locality       │
│  ❌ Probabilistic constraints       ✅ Structural invariance │
│  ❌ Likelihood coherence            ✅ Physics-grounded      │
└─────────────────────────────────────────────────────────────┘
```

The difference is ontological: statistical models manipulate tokens; GFN evolves a world.

---

## The Five Pillars of Geometric Flow Networks

The GFN paradigm is defined by five structural pillars that capture its philosophical essence rather than prescribing specific implementations. These pillars distinguish GFN from both statistical models (attention, SSMs) and generic continuous models (Neural ODEs) by articulating what the paradigm fundamentally *is*, not how it must be *computed*:

### 1. Geodesic State Flow

The state evolves as a continuous trajectory through a learned geometry, not as a statistical transformation of tokens. This pillar captures the essential nature of GFN computation: the system computes transitions as flow along geodesics in a manifold where valid transitions correspond to trajectories of minimal resistance.

Unlike attention which computes weighted correlations, GFN computes how state moves through semantic curvature. The state possesses "semantic inertia": history manifests as trajectory shape, not as explicit storage.

### 2. Persistent Internal World

The state exists as a geometric configuration that persists independently of inputs; inputs perturb the trajectory without replacing the state. This pillar articulates an ontological distinction: the internal world is not a memory buffer that stores history (like a KV-cache), but a reality that *is*.

Inputs do not add information to a list; they curve the space-time where the state orbits. A Transformer without KV-cache forgets everything; the GFN world exists as geometry itself, not as a record of events.

### 3. Structural Invariance

At least one conservation law (physical, logical, or topological) governs valid transitions, making certain states structurally impossible rather than merely improbable. This is the paradigm's most philosophically distinctive pillar.

Invariants are not soft regularization or probabilistic normalization (like softmax in attention), but physical laws encoded in geometry. In logical domains (XOR), the space has toroidal topology: invalid transitions are geometrically impossible. In semantic domains, invariants are "soft" but remain structural, not statistical.

### 4. Causal Locality

Dynamics emerge from local interactions (forces, curvature, couplings), not from global correlation over the entire sequence. This pillar distinguishes GFN from architectures requiring simultaneous access to all historical context.

Locality here is not necessarily spatial (as in CNNs) but causal: the next state depends on forces acting on the current state, not on computing similarity with all prior tokens. This enables memory without memory buffers.

### 5. Physics-Grounded Computation

Validity constraints are geometric and topological, not statistical; coherence is measured in terms of curvature and conservation, not likelihood. This pillar articulates that GFN is not "more of the same with a different name".

Constraints are not learned statistics or probabilistic normalization, but validity conditions encoded in geometry. The system "knows" which states are invalid because they are topologically inconsistent, not because they are statistically improbable.

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
  doi={10.5281/zenodo.19141132},
  url={https://doi.org/10.5281/zenodo.19141132}
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

---

## Reproducible Benchmarks

GFN v2.7.3 includes fully reproducible benchmark implementations demonstrating zero-shot length generalization and O(1) memory complexity:

### 🌀 XOR Parity Solver
- **Repository**: [gssm-xor-extrapolation](https://github.com/DepthMuun/gssm-xor-extrapolation)
- **Task**: Binary sequence parity (odd/even count of 1s)
- **Performance**: Trained on length 20, generalizes to 1M+ tokens with 100% accuracy
- **Key Result**: Constant O(1) VRAM footprint regardless of sequence length
- **Hugging Face**: [gfn-gssm-xor-parity](https://huggingface.co/DepthMuun/gfn-gssm-xor-parity)

### 📍 Multi-Needle-in-a-Haystack (mNIAH)
- **Repository**: [gssm-mniah-extrapolation](https://github.com/DepthMuun/gssm-mniah-extrapolation)
- **Task**: K-needle retrieval with AND semantics (output flips to 1 only after all K needles observed)
- **Performance**: 100% accuracy on sequences up to 32,000+ tokens
- **Key Result**: ~100x parameter reduction compared to attention-based architectures
- **Hugging Face**: [gfn-gssm-mniah-k2](https://huggingface.co/DepthMuun/gfn-gssm-mniah-k2)

Both benchmarks demonstrate the five pillars of GFN: geodesic state flow, persistent internal world, structural invariance, causal locality, and physics-grounded computation.

---

<div align="center">

*Intelligence flows through geometry*

**Author**: Joaquín Stürtz, DepthMuun Research  
**Version**: 2.7.2  
**License**: Apache 2.0

</div>
