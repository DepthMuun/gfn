# Forest-Ruth Integrator

This document describes the **current `ForestRuthIntegrator` implementation**.

The authoritative code is:

- `gfn/realizations/gssm/physics/integrators/symplectic/forest_ruth.py`

## What It Is In The Current Runtime

`ForestRuthIntegrator` is a fourth-order symplectic solver implemented as:

- drift with `c1`, kick with `d1`,
- drift with `c2`, kick with `d2`,
- drift with `c3`, kick with `d3`,
- final drift with `c4`.

Like the other current symplectic integrators, it also uses:

- topology resolution through the shared base helper,
- velocity clamping through the shared base helper.

## Coefficients Used By The Code

The runtime sets:

```text
theta = 1.3512071919596576

c1 = c4 = theta / 2
c2 = c3 = (1 - theta) / 2
d1 = d3 = theta
d2 = 1 - 2 * theta
```

Those are the exact values used in the current implementation.

## Current Step Pattern

For each step, the code performs:

1. drift, wrap topology,
2. evaluate acceleration,
3. kick, clamp velocity,
4. repeat across the three sub-steps,
5. finish with a final drift and topology wrap.

So the runtime behavior is best described as:

- fourth-order symplectic,
- topology-aware,
- velocity-saturation-aware.

## Relationship To Yoshida

Forest-Ruth and Yoshida are both fourth-order symplectic schemes, but in the current repo:

- they are separate implementations,
- they use different coefficients,
- Yoshida has an optional fused CUDA path for low-rank geometries,
- Forest-Ruth currently does not expose that extra fast path here.

So the docs should not present them as fully interchangeable implementation-wise, even if they are closely related mathematically.

## When To Use It

Use Forest-Ruth when:

- you want a fourth-order symplectic alternative,
- you want to compare higher-order symplectic schemes,
- you accept higher cost than the leapfrog default.

It is less likely to be the first choice when:

- you want the default path,
- you specifically want Yoshida's runtime specialization,
- you prioritize simpler and cheaper training.

## What This Document Should Not Claim

It would be inaccurate to claim that:

- Forest-Ruth is the default integrator,
- it has the same runtime specialization as Yoshida,
- the current implementation is just a symbolic description without shared topology or velocity helpers.

Those claims do not match the code.

## Runtime Cross-References

- `gfn/realizations/gssm/physics/integrators/symplectic/forest_ruth.py`
- `gfn/realizations/gssm/physics/integrators/base.py`
- `docs/gssm/technical/0_architecture/math/integrators/yoshida.md`
