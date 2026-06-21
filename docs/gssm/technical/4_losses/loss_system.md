# GSSM Loss System

This document describes the current registered loss family for GSSM and the runtime caveats that matter when wiring losses into training code.

For a code-aligned overview of nearby runtime behavior, also see:

- `technical/0_architecture/math/training/losses.md`
- `technical/3_models/manifold_model.md`

## Factory Entry Point

The current loss factory lives in `gfn/realizations/gssm/losses/factory.py`.

Its behavior is simple:

- if given a string, it wraps it as `{"type": ...}`
- it looks up the class in `LOSS_REGISTRY`
- if the type is missing, it falls back to `ManifoldGenerativeLoss`

It also imports the loss modules to ensure registration happens:

- `generative`
- `physics`
- `toroidal`
- `detection`
- `regularization`

## Main Registered Loss Families

The most important runtime-visible families are:

- generative losses
- physics losses
- toroidal losses
- task-specific losses such as detection-oriented heads when present

This is a more accurate framing than older docs that presented one single "gold standard" loss path.

## Generative Losses

The default fallback path is generative.

This is the right fit when:

- the readout is categorical
- the target is token-space or class-space
- you want the loss to operate directly over logits

If your task target is not token-space, generative loss is often the wrong default even if the script compiles.

## Physics Loss

`PhysicsLoss` is a regularization-style loss, not a standalone task objective.

Current components:

- geodesic regularization
- Hamiltonian conservation
- kinetic regularization

Current runtime caveat:

- `PhysicsLoss` only contributes if the required keys exist in `state_info`
- geodesic regularization currently depends on `state_info["christoffels"]`, which is not part of the default `BaseModel.forward()` output contract

That means the geodesic term is not automatically active in a plain baseline training loop unless some upstream code explicitly provides the necessary data.

## Physics-Informed Loss

`PhysicsInformedLoss` combines:

- cross-entropy over logits
- optional entropy bonus
- `PhysicsLoss` as a regularizer

This is useful only when both sides of that combination make sense:

- logits really are categorical outputs
- `state_info` contains the physical information you want to regularize

It should not be described as the universal best choice for all GSSM tasks.

## Toroidal Losses

The toroidal module registers several losses:

- `toroidal`
- `toroidal_distance`
- `toroidal_categorical`
- `toroidal_velocity`

### `ToroidalLoss`

This is the main toroidal distance loss.

It supports multiple modes, including:

- `circular`
- `mse`
- `riemannian`
- `hybrid`
- `phase`

Current runtime details:

- CUDA acceleration exists for the circular path when the required CUDA ops are available
- the `riemannian` mode assumes even-dimensional paired torus coordinates
- if the target arrives as integer IDs, the implementation converts it to float before computing angular differences

### `ToroidalCategoricalLoss`

This is for cases where the model produces categorical logits but the supervision is interpreted on a toroidal angular space.

It is a specialized bridge loss, not a default language-modeling objective.

### `ToroidalVelocityLoss`

This regularizes velocity magnitude from `state_info["v_seq"]`.

Like other state-aware regularizers, it only works when the required trajectory data is available.

## Loss / Readout Alignment

The most important practical rule is that the loss must match the readout contract.

Typical valid pairings:

- `standard` readout -> generative or categorical-style objectives
- `implicit` readout -> task-specific projection-space losses
- `identity` readout -> latent-space or manifold-aware objectives
- toroidal latent targets -> toroidal loss family

Most "the model learns the wrong thing" failures come from mismatched loss/readout/target spaces rather than from the registry itself.

## State Information Dependency

Several nontrivial losses depend on `state_info`.

The default `BaseModel.forward()` currently provides:

- `x_seq`
- `v_seq`
- `forces`
- `x_final`
- `v_final`
- `mask`
- `plugin_results`

If a loss needs more than that, the training pipeline must supply it explicitly or a plugin must populate it.

## Practical Caveats

- Do not present `PhysicsInformedLoss` as the universal best training objective.
- Do not assume physics terms are automatically active just because the class exists.
- Do not use toroidal losses unless the target actually lives on periodic coordinates or an equivalent angular representation.
