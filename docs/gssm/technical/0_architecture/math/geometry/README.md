# Geometries

This directory contains mathematical explanations of all manifold geometries available in GSSM.

---

## Available Geometries

| File | Geometry | Curvature | Bounded | Use Case |
|------|----------|-----------|---------|----------|
| `torus.md` | Torus $T^n$ | Variable | Yes | **Default** - General use |
| `euclidean.md` | Euclidean $\mathbb{R}^n$ | Flat (0) | No | Simple tasks |
| `low_rank.md` | Low-Rank Approximation | Variable | Yes | High dimensions |

---

## Quick Selection Guide

### For Most Applications

**Use Torus** (default)
- Best stability
- Bounded state
- Rich structure
- Works for most problems

### For Specific Needs

| Need | Geometry | Why |
|------|----------|-----|
| Maximum simplicity | Euclidean | Flat, easy to understand |
| High dimension ($D>256$) | Low-Rank | Memory efficient |
| Production stability | Torus | Battle-tested |

---

## What is a Geometry?

A geometry defines:

1. **The space**: Where the state $(x, v)$ lives
2. **The metric**: How to measure distances
3. **The Christoffel symbols**: How curvature affects motion
4. **The boundary**: What happens at edges

### Components

| Component | Purpose | Example (Torus) |
|-----------|---------|-----------------|
| Metric $g_{ij}$ | Measure distances | $g = \text{diag}(r^2, (R+r\cos\theta)^2)$ |
| Christoffel $\Gamma$ | Curvature force | $\Gamma^\theta_{\phi\phi} = (R+r\cos\theta)\sin\theta/r$ |
| Projection | Handle boundaries | Wrap to $[-\pi, \pi]$ |
| Friction | Damping | $\mu(\theta) = \mu_0 + \alpha \cdot \text{curv}(\theta)$ |

---

## Mathematical Summary

### Common Equation

All geometries compute the **geometric acceleration**:

$$a_{geo} = -\Gamma(x, v)$$

Where the Christoffel symbols depend on the specific geometry.

### Space Comparison

| Property | Torus | Euclidean | Low-Rank |
|----------|-------|-----------|----------|
| $\Gamma$ | Non-zero | Zero | Approximate |
| Boundary | Periodic | None | Periodic |
| Cost | Medium | Low | Low |
| Stability | High | Low | Medium |

---

## Reading Order

1. **Torus** (most important, default)
2. **Euclidean** (for comparison)
3. **Low-Rank** (for high dimensions)

---

*Last Updated: 2026-04-02*
