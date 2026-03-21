# ISN Training Guide

Training an **Inertial State Network (ISN)** requires a departure from standard Cross-Entropy regimens. Because ISN is a physical simulation, we supervise not just the *output* but the *integrity* of the internal world.

## The Multi-Dimensional Loss

The `ISNTrainer` uses a composite loss function (`MultiDimensionalLoss`) that balances several objectives:

$$ \mathcal{L} = \lambda_1 \mathcal{L}_{outcome} + \lambda_2 \mathcal{L}_{coherence} + \lambda_3 \mathcal{L}_{efficiency} + \dots $$

1.  **Outcome Loss ($ \mathcal{L}_{outcome} $)**: Standard Cross-Entropy between predicted logits and target tokens.
2.  **Coherence Loss ($ \mathcal{L}_{coherence} $)**: Penalizes non-physical state transitions and ensures that entities stay within their manifold bounds.
3.  **Grounding Loss ($ \mathcal{L}_{grounding} $)**: Measures the alignment between the latent world properties and verified symbolic facts.
4.  **Validity Loss ($ \mathcal{L}_{validity} $)**: Checks if the **Conservation Laws** (e.g., type or parity preservation) are respected post-interaction.
5.  **Emergence Loss ($ \mathcal{L}_{emergence} $)**: Supervises the creation of new entities from multi-entity interactions.
6.  **Efficiency Loss ($ \mathcal{L}_{efficiency} $)**: Penalizes state drift and excessive complexity (Least Action Principle).

## Curriculum Learning

ISN models benefit significantly from weight curriculum. In the early stages of training, we prioritize **Coherence** (learning the physics); in later stages, we prioritize **Outcome** (learning the task).

Example configuration:
```json
"curriculum": {
    "phase_1": {
        "epochs": [0, 10],
        "lambda_weights": {
            "lambda_outcome": 0.1,
            "lambda_coherence": 1.0,
            "lambda_efficiency": 0.5
        }
    },
    "phase_2": {
        "epochs": [10, 50],
        "lambda_weights": {
            "lambda_outcome": 1.0,
            "lambda_coherence": 0.5,
            "lambda_efficiency": 0.1
        }
    }
}
```

## Stability & Optimization

- **Gradient Clipping**: Essential for maintaining symplectic stability. Recommended value: `1.0`.
- **Integrator Choice**: Use `GFNWorld` (Leapfrog) for standard training. High-precision tasks may require `PEFRL`.
- **Adjoint Method**: Support for `torchdiffeq`-style adjoint backprop is available for $O(1)$ memory training on extremely long sequences.

## Checkpoint Management

The `Trainer` automatically saves the best model based on validation loss.
```python
trainer.train(train_loader, val_loader, num_epochs=50)
```
Checkpoints include the model state, optimizer state, and the exact `config` used for reproducibility.
