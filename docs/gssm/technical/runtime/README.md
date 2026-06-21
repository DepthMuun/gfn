# GSSM Runtime Notes

This folder documents the **actual runtime behavior** of GSSM as implemented in the current codebase.

Use this section when you need answers to questions such as:

- which defaults are truly effective after config normalization,
- how geometry selection is resolved,
- how integrators apply damping and wrapping,
- how embedding and readout are constructed,
- how hyperparameters interact in practice.

This is the technical counterpart to the user-facing material in `docs/gssm/guides/`.

## Reading Order

1. `00-effective-defaults.md`
2. `01-hyperparameters.md`

## Documentation Rule

When a statement in an older document conflicts with this folder, prefer the runtime-derived explanation here and then update the higher-level guide accordingly.
