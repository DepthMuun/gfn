# Troubleshooting

## Diverging Loss

Loss diverges when values become NaN (Not a Number) or Inf (infinite). This is the most common problem and usually has a few causes.

### Symptoms

- The loss reports `nan` or values that grow exponentially
- Gradients are `nan` or extremely large
- The model stops learning

### Common Causes

**Timestep too large.** A high DEFAULT_DT can make the integrator unstable.

Solution: Reduce DEFAULT_DT from 0.05 to 0.02 or lower.

**Friction too low.** Without enough friction, the system can oscillate and diverge.

Solution: Increase FRICTION_SCALE to 0.05 or 0.1.

**Learning rate too high.** The optimizer takes steps that are too large.

Solution: Reduce DEFAULT_LR from 1e-4 to 5e-5 or 1e-5.

**Extreme curvature.** The metric becomes singular in some regions.

Solution: Reduce CURVATURE_CLAMP to 2.0 or enable trace normalization.

### Diagnosis

Add logging to see which part of the loss diverges:

```python
# In your training script
total_loss = h_loss + g_loss + cross_entropy
print(f"Total: {total_loss.item():.4f}")
print(f"Hamiltonian: {h_loss.item():.4f}")
print(f"Geodesic: {g_loss.item():.4f}")
print(f"Cross-entropy: {cross_entropy.item():.4f}")
```

If only Hamiltonian diverges, the problem is energy conservation. If only Geodesic diverges, the problem is trajectories.

## CUDA Memory Error

The memory error occurs when the GPU cannot allocate tensors.

### Symptoms

- `RuntimeError: CUDA out of memory`
- The process is killed by the system

### Solutions

**Reduce batch size.** The most effective change.

```yaml
training:
  batch_size: 4  # Reduce from 16 to 4
```

**Reduce model dimension.**

```yaml
model:
  dim: 256  # Reduce from 512 to 256
  rank: 32  # Reduce from 64 to 32
```

**Disable CUDA kernels.** Use the pure Python implementation.

```python
import os
os.environ["GFN_USE_CUDA"] = "0"
```

**Reduce input seq_len.** If the dataset allows.

**Free GPU memory before training.**

```python
import torch
torch.cuda.empty_cache()
```

### Prevention

Monitor memory usage:

```bash
nvidia-smi -l 1
```

If memory fills gradually, there is a memory leak. Report this bug.

## NaN in Gradients

NaN gradients indicate problems in the backward pass.

### Common Causes

**Division by zero.** The integrator tries to divide by a very small number.

Solution: Increase EPSILON_STANDARD to 1e-6.

**Extreme activations.** Activations that overflow in float16.

Solution: Reduce READOUT_GAIN, reduce IMPULSE_SCALE.

**Spurious metric gradients.** The low-rank approximation can produce unstable gradients.

Solution: Increase the factorization rank.

### Diagnosis

Enable gradient checking:

```python
torch.autograd.set_detect_anomaly(True)
```

This slows down training but helps identify the NaN source.

## Python-CUDA Parity Failure

Parity tests fail when implementations produce different results.

### Symptoms

```
AssertionError: max diff Python vs CUDA = 1e-3 (tolerance 1e-5)
```

### Causes

**Out-of-sync constants.** FRICTION_SCALE or epsilon differ between Python and CUDA.

Solution: Verify that constants.py and CudaConstants match.

**Different operation order.** Reduction operations can differ.

Solution: Review the CUDA kernel code.

**Different data types.** float32 vs float16.

Solution: Force float32 in both backends.

### Verification

Run the parity test:

```bash
python tests/test_cuda_python_consistency.py
```

If it fails, the system may still work but with different behavior on GPU.

## Unstable Integrator

The integrator produces oscillations or erratic behavior.

### Symptoms

- Position or momentum oscillate without converging
- The system energy does not stabilize

### Solutions

**Increase friction.**

```yaml
physics:
  friction_scale: 0.05
  default_friction: 0.005
```

**Reduce timestep.**

```yaml
physics:
  dt: 0.02
```

**Increase substeps.**

```yaml
physics:
  leapfrog_substeps: 5
```

**Switch to a more stable integrator.**

```yaml
model:
  integrator_type: "heun"  # En lugar de leapfrog
```

## Import Error

The gfn module cannot be imported.

### Causes

**Virtual environment not activated.**

```bash
source manifold-env/bin/activate
```

**Missing dependencies.**

```bash
pip install -r requirements.txt
**Incorrect Python path.**

```bash
python -c "import sys; print(sys.path)"
```

**Corrupted installation.**

```bash
pip uninstall gfn
pip install -e .
```

## Loss Does Not Converge

The loss decreases very slowly or stalls.

### Diagnosis

Check which loss term dominates:

```python
print(f"CE: {ce_loss.item():.4f}")
print(f"Hamiltonian: {h_loss.item():.4f}")
print(f"Geodesic: {g_loss.item():.4f}")
```

### Solutions

**Learning rate too low.** Increase it.

```yaml
training:
  learning_rate: 0.0005
```

**Regularization too strong.** Reduce the loss weights.

```yaml
physics:
  lambda_g: 0.00001
  lambda_h: 0.0
```

**Model too small.** Increase dimension.

```yaml
model:
  dim: 1024
  depth: 12
```

**Data too difficult.** Try a simpler dataset.

## Slow Speed

Training is slower than expected.

### Causes

**No CUDA acceleration.** Verify that CUDA is available.

```python
import torch
print(torch.cuda.is_available())
```

**CUDA kernels not compiled.** Compile the kernels.

```bash
python -m gfn.cuda.precompile_kernels
```

**Batch size too small.** Increase it for better throughput.

**Integrator with too many substeps.** Reduce LEAPFROG_SUBSTEPS.

### Optimization

For a modern NVIDIA GPU:

```bash
export PYTORCH_CUDA_ALLOCATOR=max
export CUDA_LAUNCH_BLOCKING=0
```

These flags improve memory allocation.

## File and Path Errors

Files not found or incorrect paths.

### Common Causes

**Incorrect relative path.** Run from the project root directory.

**Missing configuration files.** Verify that the file exists:

```bash
ls configs/training/experiment_overfit_10k.yaml
```

**File permissions.** Make sure you can read/write.

```bash
chmod 644 configs/training/*.yaml
```

## Reporting Bugs

If none of these solutions work:

1. Collect system information:

```bash
python -c "import gfn; print(gfn.__version__)"
nvidia-smi
pip freeze | grep torch
```

2. Create a minimal script that reproduces the error:

```python
# save as bug_report.py
import gfn
# código mínimo que falla
```

3. Open an issue with:
- Problem description
- Reproduction script
- Full error outputs
- System information

---

**Manifold Labs (Joaquín Stürtz)**
