# Architecture Overview

This folder contains the architecture-level documentation for GSSM.

It should now be read as an index into the updated runtime-aligned docs, not as a frozen dump of old diagnostics.

## Main Entry Points

Start here:

1. [00_overview.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/00_overview.md) for the current architecture overview.
2. [math/README.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/README.md) for the mathematical and component-level layer.
3. [runtime/README.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/runtime/README.md) for effective defaults and runtime hyperparameters.

## Current Folder Structure

- [00_overview.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/00_overview.md): current technical overview
- [01_components.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/01_components.md): component-oriented architecture notes
- [02_code_analysis.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/02_code_analysis.md): legacy analysis notes that should be read cautiously
- [math/README.md](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/0_architecture/math/README.md): mathematics and deep component explanations

Important correction:

- the older reference to `02_data_flow.md` was stale; that file is not part of the maintained structure here.

## Quick Links

- [models](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/3_models/manifold_model.md)
- [physics](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/2_physics/engine.md)
- [geometry](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/1_geometry/base.md)
- [config](file:///D:/ASAS/principal_proyects/manifold_mini/dev/dev/gfn/docs/gssm/technical/5_config/schema_loader.md)

## How To Use These Docs

Use `0_architecture/` when you want:

- a system-level picture of how GSSM fits together,
- entry points into the deeper math and runtime docs,
- architecture context before reading detailed component pages.

Use `technical/runtime/` when you want:

- effective defaults,
- hyperparameter behavior,
- precedence and runtime truth.

Use `guides/` when you want:

- user-facing recipes,
- practical configuration examples,
- public-reference style explanations.

## Important Caveat

This folder previously mixed:

- architecture notes,
- snapshot diagnostics,
- stale links,
- and recommendations frozen to older runtime behavior.

The maintained path now prefers:

- runtime-aligned docs,
- explicit caveats,
- and links to the current canonical pages.
