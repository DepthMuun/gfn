# ISN Training Guide

Training an **Inertial State Network (ISN)** uses the modular trainer in `gfn.realizations.isn.training.trainer`. The current runtime supports standard supervised token training plus optional auxiliary terms tied to the latent world outputs.

## The Multi-Dimensional Loss

The trainer constructs a `MultiDimensionalLoss` instance. Its full interface includes multiple terms:

$$ \mathcal{L} = \lambda_1 \mathcal{L}_{outcome} + \lambda_2 \mathcal{L}_{coherence} + \lambda_3 \mathcal{L}_{efficiency} + \dots $$

1.  **Outcome Loss ($ \mathcal{L}_{outcome} $)**: Standard Cross-Entropy between predicted logits and target tokens.
2.  **Coherence Loss ($ \mathcal{L}_{coherence} $)**: Uses the reported `world_coherence` score.
3.  **Grounding Loss ($ \mathcal{L}_{grounding} $)**: Available in the loss module, but only becomes meaningful if richer `world_states` metadata is provided.
4.  **Validity Loss ($ \mathcal{L}_{validity} $)**: Present in the loss interface, currently lightweight.
5.  **Emergence Loss ($ \mathcal{L}_{emergence} $)**: Present in the loss interface, currently lightweight.
6.  **Efficiency Loss ($ \mathcal{L}_{efficiency} $)**: Present in the loss interface, currently lightweight.

Important runtime note:

- the default `Trainer.validate()` path uses `MultiDimensionalLoss`,
- the training loss actually depends on the selected backpropagation strategy,
- the `adjoint` strategy currently computes plain cross-entropy in its `compute_loss()` implementation.

So, today, the most reliable statement is that ISN training is **strategy-driven**, with `MultiDimensionalLoss` available as the general validation/composite criterion.

## Trainer API

The runtime trainer class is `Trainer`, not `ISNTrainer`:

```python
from gfn.realizations.isn.training.trainer import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    config=config,
    device=device,
    checkpoint_dir="./checkpoints",
)

trainer.train(train_loader, val_loader, num_epochs=50)
```

## Stability & Optimization

- **Gradient Clipping**: Supported directly by `Trainer` through `config["training"]["gradient_clip"]`. The implementation defaults to `1.0`.
- **Backprop Strategy**: Selected through `config["training"]["backprop_strategy"]` and resolved from the ISN strategy registry.
- **Adjoint Method**: Available as `backprop_strategy="adjoint"`. It requires `torchdiffeq` and wraps the world engine with an adjoint ODE solve.
- **Integrator Choice**: When the model uses `GFNPhysics`, the world integrator can be selected independently through `world_kwargs={"integrator": "euler" | "leapfrog" | "yoshida"}`.

## Checkpoint Management

`Trainer` saves `best_model.pt` whenever validation loss improves:

```python
trainer.train(train_loader, val_loader, num_epochs=50)
```

At the moment, the built-in save path stores the model weights for the best checkpoint. If you need optimizer state or richer experiment metadata, that should be handled explicitly by the surrounding training script.
