# Omelyan Integrator

This document describes the **current `OmelyanIntegrator` implementation**.

The authoritative code is:

- `gfn/realizations/gssm/physics/integrators/symplectic/omelyan.py`

## Important Runtime Correction

The old documentation described Omelyan as a second-order method.

That is **not** what the current code implements.

The current runtime explicitly documents and implements:

- **Omelyan PEFRL**
- **4th-order symplectic integration**

So any doc that still calls it second-order is outdated.

## What The Current Code Implements

The solver uses five drift coefficients and four kick coefficients:

```text
xi  =  0.1786178958448091
lam = -0.2123418310626054
chi = -0.06626458266981849

c1 = xi
c2 = chi
c3 = 1 - 2*(chi + xi)
c4 = chi
c5 = xi

d1 = (1 - 2*lam)/2
d2 = lam
d3 = lam
d4 = (1 - 2*lam)/2
```

This is the PEFRL-style multi-stage structure used by the current implementation.

## Current Step Pattern

For each step, the code performs:

1. drift with `c1`, evaluate acceleration, kick with `d1`,
2. drift with `c2`, evaluate acceleration, kick with `d2`,
3. drift with `c3`, evaluate acceleration, kick with `d3`,
4. drift with `c4`, evaluate acceleration, kick with `d4`,
5. final drift with `c5`.

At every drift:

- topology is resolved.

At every kick:

- velocity is clamped through the shared base helper.

So the current runtime Omelyan path is:

- fourth-order,
- symplectic,
- topology-aware,
- velocity-saturation-aware.

## Relationship To The Older Description

The old text about a single `zeta ~= 0.1932` optimized second-order scheme does not match the present implementation.

That older description refers to a different Omelyan-style formulation than the one currently in the repo.

## Practical Interpretation

Use Omelyan when:

- you want a higher-order symplectic method,
- you want an alternative to Yoshida or Forest-Ruth,
- you are willing to pay for a richer multi-stage integrator.

It is less attractive when:

- you want the default path,
- you want the cheapest robust choice,
- leapfrog already gives enough quality.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- the current Omelyan implementation is second-order,
- it is parameterized only by a single `zeta`,
- it has the same stage structure as the old doc version,
- it ignores topology or shared velocity controls.

Those claims do not match the current code.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/integrators/symplectic/omelyan.py`
- `gfn/realizations/gssm/physics/integrators/base.py`
- `docs/gssm/technical/0_architecture/math/02_integrators.md`
