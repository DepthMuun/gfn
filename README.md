<div align="center">

# 🌊 GFN: Geometric Flow Networks

### *A Physics-Informed Paradigm for Sequential Intelligence*

[![Framework: GFN](https://img.shields.io/badge/Paradigm-GFN_2.7.2-blue.svg?style=for-the-badge)](https://github.com/DepthMuun/gfn)
[![Models: Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-orange.svg?style=for-the-badge)](https://huggingface.co/DepthMuun)
[![DOI: 10.5281/zenodo.19141133](https://img.shields.io/badge/DOI-10.5281/zenodo.19141133-blue.svg?style=for-the-badge)](https://doi.org/10.5281/zenodo.19141133)
[![Status: Active](https://img.shields.io/badge/Status-Active-green.svg?style=for-the-badge)]()
[![Package: GFN PyPi](https://static.pepy.tech/personalized-badge/gfn?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pypi.org/project/gfn/)

</div>

> 💡 **"Intelligence is not statistical correlation; it is the continuous evolution of a persistent world-state governed by physical invariants."**
> 
> — *The GFN Paradigm*

---

## 🎯 What is GFN?

**Geometric Flow Networks (GFN)** represent a fundamental paradigm shift in neural architecture. Instead of treating computation as token-to-token correlation (Attention), **GFN treats intelligence as a trajectory** within a high-integrity dynamical system.

### The Core Insight

```
┌─────────────────────────────────────────────────────────────┐
│  Traditional (Attention)          GFN (Geometric Flow)  │
├─────────────────────────────────────────────────────────────┤
│  Token → Correlation → Token      Token → Perturbation →    │
│  (with KV-cache buffer)           World State Evolution     │
│                                                             │
│  ❌ Memory buffer (crutch)        ✅ Persistent simulator   │
│  ❌ O(N^2) complexity           ✅ O(1) per step            │
│  ❌ Statistical guessing          ✅ Physical orbits        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏛️ The Five Pillars of GFN

For an architecture to be a valid **GFN realization**, it must satisfy **all five pillars**:

### 🧠 Pillar 1: Persistent Internal World
The system maintains a **simulator**, not a memory buffer. The state evolves through geometric space.

```
❌ Transformer KV-cache: "Cut the cache → instant amnesia"
✅ GFN World-State: "The geometry itself IS the memory"
```

### ⚖️ Pillar 2: At Least One Invariant
Physical/mathematical invariants act as the "gravity" preventing latent hallucinations.

| Invariant Type | Description |
|:--------------|:------------|
| **🔷 Casimir Operators** | Commute with all generators of a Lie algebra |
| **⚡ Hamiltonian Conservation** | Total energy preserved through symplectic integration |
| **📐 Norm Preservation** | State norm remains bounded |
| **🔄 Group Symmetries** | Topological constraints make invalid states impossible |
| **📊 Phase Space Volume** | Liouville's theorem preservation |

### 🛡️ Pillar 3: Structural Integrity
The state cannot collapse to zero or explode to infinity.

**Gradient stability by geometry:**
- ✅ No gradient vanishing (singular values ≈ 1)
- ✅ No gradient explosion (volume preserved)
- ✅ Stable by design, not by patching

### ⏱️ Pillar 4: Temporal Locality (True $O(1)$)
Computational cost is **independent** of sequence length.

```python
# ✅ VALID GFN: O(1) per step
state_update = f(current_state, present_input)

# ❌ INVALID: O(N) - breaks the promise
state_update = f(current_state, all_previous_tokens)
```

> 📝 **"The flow is calculated over the current state and present input, nothing more."**

### 📏 Pillar 5: Geometric Differentiability
All states exist in a manifold where "distance" is physically coherent.

> 💭 *"For there to be 'forces' and 'flows', you need to know how far one concept is from another in that curved space."*

---

## The GFN Definition

A **Geometric Flow Network (GFN)** is a neural architecture satisfying all five pillars above:

1. Persistent Internal World (Simulator, not buffer)
2. At Least One Physical/Mathematical Invariant
3. Structural Integrity (gradient stability by geometry)
4. Temporal Locality ($O(1)$ state update)
5. Geometric Differentiability (metric coherence)

### Formal Definition

$$
W_{t+1} = \mathcal{T}(W_t, f_{ext}; \theta)
$$

Where:
- $W_t$ is the internal world state at time $t$
- $f_{ext}$ is the external input (perturbation)
- $\mathcal{T}$ is a transfer operator that:
  - Preserves at least one invariant
  - Operates on $W_t$ and $f_{ext}$ ONLY (no history)
  - Is differentiable with respect to a coherent metric

---

## Why Geometric Flow Networks?

### Comparison to Related Approaches

| Architecture | 🌍 Persistent World | ⚖️ Invariant | 🛡️ Integrity | ⏱️ $O(1)$ Update | 📏 Metric |
|:-------------|:------------------:|:----------:|:-----------:|:---------------:|:---------:|
| **Transformer + KV-cache** | ❌ (buffer) | ❌ | ❌ | ❌ | ❌ |
| **Mamba / SSM** | ✅ | ❌ | ❌ | ✅ | ❌ |
| **World-State Networks** | ✅ | ❌ | ❌ | ✅ | ❌ |
| **🌊 GFN** | ✅ | ✅ | ✅ | ✅ | ✅ |

### Key Distinctions Explained

**GFN vs Transformer:**
- 🤖 Transformer: "Guesses" by statistics → *Correlation-based*
- 🌊 GFN: "Orbits" solutions → *Physics-constrained trajectories*

**GFN vs SSM:**
- 📡 SSM: "Radio signal" → State can collapse or explode
- 🌊 GFN: "River flow" → State orbits around physically coherent solutions

**GFN vs World-State Networks:**
- 🖼️ World-state without invariants: "Photo on a post-it" — blurry, degrades
- 🌍 GFN: "Full simulator" — The world exists as geometry, not memory

---

## Latent Planning Capability

A critical consequence of the five pillars:

> ### ✨ **GFN can "predict" the future without generating token by token, simply by moving the state vector through the geometric flow.**

**The internal world enables latent planning:**
- 🔮 Future states computed by flowing through the manifold
- 🚀 No need to autoregressively generate each token
- 🗺️ The geometry encodes causal structure

---

## Complexity Characteristics

**GFN is a paradigm with five mandatory requirements.** Complexity characteristics depend on implementation:

| Requirement | Complexity Impact |
|:------------|:----------------|
| **Temporal Locality** (Pillar 4) | Enables $O(1)$ inference memory |
| **Structural Integrity** (Pillar 3) | Intrinsic gradient stability |
| **All Pillars Combined** | No KV-cache, no $O(N^2)$ attention |

---

## Paradigm Documentation

For detailed theoretical foundations and mathematical formalism, see:

- [THEORY.md](docs/THEORY.md) - Complete mathematical foundations
- [ARCHITECTURES.md](ARCHITECTURES.md) - Available realizations
- [CONTRIBUTING.md](CONTRIBUTING.md) - Adding new realizations
- [Zenodo Preprint](https://doi.org/10.5281/zenodo.19141133) - Primary research paper (DOI: 10.5281/zenodo.19141133)

### 📖 Citation

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

## ⚡ Quick Start

```bash
# Install from PyPI
pip install gfn

# Or install from source
git clone https://github.com/DepthMuun/gfn.git
cd gfn
pip install -e .
```

```python
from gfn import GFNModel, PhysicsConfig

# Create a GFN model
config = PhysicsConfig(dim=512, depth=6)
model = GFNModel(config)

# Forward pass (stateful, O(1) memory)
output = model(input_tokens)
```

---

<div align="center">

## 🌊 *Intelligence flows through geometry*

**Author**: Joaquín Stürtz, DepthMuun Research  
**Version**: 2.7.2  
**License**: Apache 2.0

</div>
