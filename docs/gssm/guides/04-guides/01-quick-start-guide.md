# Quick Start Guide

## Your First Experiment

This guide lets you run a basic experiment in under 10 minutes. We assume you already have the environment configured according to the installation instructions.

### Step 1: Verify the Installation

Before starting, verify that everything works correctly.

```bash
python -c "import gfn; print('Version:', gfn.__version__)"
```

If you see a version number, the installation is correct. If there are errors, review the installation section.

### Step 2: Run a Simple Experiment

The project includes an example configuration for a quick overfitting experiment. This experiment trains on 10,000 examples to verify the system works.

```bash
python demos/tinystories/train_tinystories.py \
    --config configs/demos/tinystories.yaml
```

This command:
- Loads the overfitting configuration from the embedded yaml.
- Trains using the parameters specified in the config (`max_steps`, `batch_size`).
- Limits the HuggingFace dataset to avoid memory issues automatically.

If you want to manually adjust steps or batch size, you will need to edit `configs/demos/tinystories.yaml` directly before running.

### Step 3: Monitor Training

The system reports metrics every certain number of steps. Look for in the output:

- `loss`: Total loss (should decrease)
- `h_loss`: Hamiltonian loss (should stay stable)
- `g_loss`: Geodesic loss (should decrease slowly)
- `grad_norm`: Gradient norm (should stay bounded)

If the loss diverges (goes to NaN or infinity), stop training with Ctrl+C and review the common problems section.

### Step 4: Inspect the Model

After training, the model saves checkpoints. Inspect the latest checkpoint:

```bash
python scripts/inspect_checkpoint.py \
    --checkpoint logs/tinystories/run_xxx/checkpoints/last.ckpt \
    --output inspection.json
```

The JSON file contains model metrics and architecture.

## Anatomy of a Configuration

The YAML configuration files control all aspects of the experiment. Examine `configs/demos/tinystories.yaml`:

```yaml
model:
  vocab_size: 10000
  dim: 512
  depth: 6
  rank: 64
  heads: 8
  integrator_type: "leapfrog"

training:
  learning_rate: 0.0001
  batch_size: 16
  max_steps: 1000
  warmup_steps: 100
  
physics:
  friction_scale: 0.02
  dt: 0.05
  lambda_h: 0.0
  lambda_g: 0.00005
```

Main sections:
- `model`: Model architecture
- `training`: Optimizer parameters
- `physics`: Physical and integrator constants
- `data`: Dataset configuration

## Your Own Experiment

To create your own experiment, copy the base configuration to a new file:

```bash
cp configs/demos/tinystories.yaml configs/demos/my_experiment.yaml
```

Edit the file with your parameters. For example, for a larger model:

```yaml
model:
  vocab_size: 50000
  dim: 1024
  depth: 12
  rank: 128
  heads: 16

training:
  learning_rate: 0.00005  # Lower LR for larger models
  batch_size: 8           # Smaller batch due to memory
  max_steps: 5000
```

Run your experiment:

```bash
python demos/tinystories/train_tinystories.py \
    --config configs/demos/my_experiment.yaml
```

## Log Structure

Results are saved in the directory specified by --output. Typical structure:

```
logs/tinystories/mi_experimento/
â”œâ”€â”€ checkpoints/
â”‚   â”œâ”€â”€ last.ckpt
â”‚   â””â”€â”€ best.ckpt
â”œâ”€â”€ events.out.tfevents.xxx
â”œâ”€â”€ config.yaml
â””â”€â”€ metrics.json
```

- `checkpoint_epoch_X.pt`: Models saved during training. Include state, loss, and vocab.
- `best_model.pt`: The model state scoring the lowest validation loss.

## Monitoring

To manually inspect convergence, read the console outputs for `PPL` (Perplexity) and `Val Loss`. Look for the "New best model saved" checkmark indicator.
## Next Steps

Now that you have an experiment running, you can:

1. Read the advanced configuration guide to tune parameters
2. Review the constants reference to understand each parameter
3. Explore demos/sorting/ for other experiment types
4. Run the test suite to verify correctness

If you run into problems, consult the troubleshooting guide.

---

**DepthMuun (GFN v2)**
