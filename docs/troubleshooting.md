# Manifold Troubleshooting Guide


**Last Updated:** February 2026

This document provides comprehensive troubleshooting guidance for the Manifold Geometric Flow Network implementation. It covers common issues, diagnostic procedures, and solutions for problems that users may encounter during installation, training, and inference.

---

## 1. Installation Issues

### 1.1 Python Version Compatibility

**Problem:** The installation fails with version compatibility errors.

**Symptoms:** Error messages mentioning incompatible Python version or missing dependencies.

**Solution:** Manifold requires Python 3.10 or higher. Check your Python version:

```bash
python --version
```

If the version is below 3.10, install a newer Python version using your preferred method:

```bash
# Using pyenv
pyenv install 3.10.x
pyenv local 3.10.x

# Using conda
conda create -n manifold python=3.10
conda activate manifold
```

### 1.2 PyTorch Installation Issues

**Problem:** PyTorch is not found or has incompatible version.

**Symptoms:** Import errors when importing Manifold, or errors about missing torch modules.

**Solution:** Install PyTorch 2.3 or higher:

```bash
# For CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio
```

Verify the installation:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 1.3 CUDA Compilation Failures

**Problem:** CUDA kernels fail to compile during installation.

**Symptoms:** Errors during `compile_cuda_kernels.py` execution, or warnings about missing nvcc.

**Solution:** Ensure CUDA Toolkit is properly installed:

```bash
nvcc --version
```

If nvcc is not found, install CUDA Toolkit:

```bash
# For Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit

# Verify installation
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
```

If compilation still fails, try manual compilation:

```bash
cd gfn/cuda
python setup.py build
```

For systems without CUDA compiler, use the pure PyTorch implementation which does not require CUDA compilation.

### 1.4 Import Errors

**Problem:** Importing Manifold fails with ModuleNotFoundError.

**Symptoms:** Error message: "ModuleNotFoundError: No module named 'gfn'"

**Solution:** Ensure the package is installed:

```bash
pip install -e .
```

If the package is installed but not found, check your Python path:

```python
import sys
print(sys.path)
```

Ensure the Manifold directory is in your Python path, or use a virtual environment.

---

## 2. Training Issues

### 2.1 Loss Oscillation or Divergence

**Problem:** Training loss oscillates chaotically or diverges to infinity.

**Symptoms:** Loss values fluctuate wildly, increase over time, or become NaN.

**Diagnosis:** This is often caused by using the wrong optimizer or incorrect configuration.

**Solution Steps:**

1. Verify you are using RiemannianAdam instead of standard optimizers:

```python
from gfn.optim import RiemannianAdam

optimizer = RiemannianAdam(model.parameters(), lr=1e-4)
```

2. Check the configuration for missing stability settings:

```python
physics_config = {
    'stability': {
        'base_dt': 0.4,
        'curvature_clamp': 5.0
    }
}
```

3. Enable stricter gradient clipping:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
```

4. Verify the integrator type is appropriate:

```python
model = Manifold(
    ...
    integrator_type='leapfrog',  # Recommended for stability
    ...
)
```

### 2.2 Slow Convergence

**Problem:** Model takes too many epochs to converge or never reaches good performance.

**Symptoms:** Loss decreases very slowly, or plateaus at a high value.

**Diagnosis:** This may indicate incorrect configuration or suboptimal hyperparameters.

**Solution Steps:**

1. Verify embedding configuration uses linear mode:

```python
physics_config = {
    'embedding': {
        'type': 'functional',
        'mode': 'linear',  # Superior to 'binary'
        'coord_dim': 16
    }
}
```

2. Check learning rate is appropriate:

```python
optimizer = RiemannianAdam(model.parameters(), lr=1e-4)  # Typical range: 1e-5 to 1e-3
```

3. Enable active inference for faster adaptation:

```python
physics_config = {
    'active_inference': {
        'enabled': True,
        'dynamic_time': {'enabled': True},
        'reactive_curvature': {'enabled': True, 'plasticity': 0.2}
    }
}
```

4. Consider using differential learning rates:

```python
optimizer = RiemannianAdam([
    {'params': [p for n, p in model.named_parameters()
                if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
     'lr': 1e-3, 'weight_decay': 1e-4},
    {'params': [p for n, p in model.named_parameters()
                if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
     'lr': 1e-2, 'weight_decay': 0}
])
```

### 2.3 Gradient Issues

**Problem:** Gradients become NaN, Inf, or have unusual values.

**Symptoms:** Training crashes with NaN values, or gradients are extremely large or small.

**Solution Steps:**

1. Enable gradient clipping:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)  # Stricter than typical 1.0
```

2. Check for NaN in inputs:

```python
assert not torch.isnan(inputs).any(), "Input contains NaN values"
```

3. Verify curvature clamping is enabled:

```python
physics_config = {
    'stability': {
        'curvature_clamp': 5.0  # Prevents curvature explosion
    }
}
```

4. Reduce integration timestep if gradients are unstable:

```python
physics_config = {
    'stability': {
        'base_dt': 0.3  # More conservative timestep
    }
}
```

5. Use gradient checkpointing for memory-constrained training:

```python
from torch.utils.checkpoint import checkpoint

# Wrap layer forward pass with checkpointing
x = checkpoint(layer.forward, x, v, F)
```

### 2.4 Memory Issues

**Problem:** Running out of GPU memory during training.

**Symptoms:** CUDA out-of-memory errors, process killed, or excessive memory usage.

**Solution Steps:**

1. Reduce batch size:

```python
dataloader = DataLoader(dataset, batch_size=8)  # Reduce from 32 or 16
```

2. Enable gradient accumulation:

```python
# Simulate larger batch size through accumulation
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch).loss
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. Use gradient checkpointing:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedManifold(nn.Module):
    def forward(self, x):
        return checkpoint(self.manifold, x)
```

4. Reduce model size:

```python
model = Manifold(
    dim=256,   # Reduce from 512
    depth=4,   # Reduce from 6
    heads=4,   # Reduce from 8
    ...
)
```

5. Use mixed precision training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    optimizer.zero_grad()
    with autocast():
        loss = model(batch).loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2.5 Poor Generalization

**Problem:** Model performs well on training data but poorly on test data.

**Symptoms:** High training accuracy but low validation accuracy.

**Diagnosis:** This may indicate overfitting or incorrect use of embedding mode.

**Solution Steps:**

1. Verify linear embedding mode is used:

```python
physics_config = {
    'embedding': {
        'type': 'functional',
        'mode': 'linear',  # Superior generalization
        'coord_dim': 16
    }
}
```

2. Enable active inference:

```python
physics_config = {
    'active_inference': {
        'enabled': True,
        'dynamic_time': {'enabled': True}
    }
}
```

3. Add regularization:

```python
optimizer = RiemannianAdam(model.parameters(), lr=1e-4, weight_decay=0.01)
```

4. Use early stopping:

```python
best_val_loss = float('inf')
patience_counter = 0
for epoch in range(max_epochs):
    train_loss = train_epoch()
    val_loss = val_epoch()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
```

---

## 3. Inference Issues

### 3.1 OOM During Inference

**Problem:** Out of memory errors during autoregressive generation.

**Symptoms:** CUDA out-of-memory after generating some tokens.

**Diagnosis:** This can occur if state accumulation or batched generation exceeds memory.

**Solution Steps:**

1. Reduce batch size for generation:

```python
model.eval()
with torch.no_grad():
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        generated = model.generate(batch, max_tokens=100)
```

2. Clear CUDA cache between generations:

```python
import torch
model.eval()
with torch.no_grad():
    for prompt in prompts:
        result = model.generate(prompt, max_tokens=100)
        torch.cuda.empty_cache()  # Clear cache between generations
```

3. Use CPU offloading for very long sequences:

```python
model = model.cpu()
with torch.no_grad():
    result = model.generate(prompt, max_tokens=1000)
```

### 3.2 Incorrect Output

**Problem:** Generated output is garbled, repetitive, or incorrect.

**Symptoms:** Output tokens don't form coherent sequences, or model repeats tokens.

**Diagnosis:** This may indicate incorrect temperature, sampling settings, or model configuration.

**Solution Steps:**

1. Adjust temperature:

```python
# Higher temperature = more random
# Lower temperature = more deterministic
generated = model.generate(prompt, max_tokens=100, temperature=0.7)
```

2. Use top-k or top-p filtering:

```python
# Top-k sampling
generated = model.generate(prompt, max_tokens=100, top_k=40)

# Top-p (nucleus) sampling
generated = model.generate(prompt, max_tokens=100, top_p=0.9)
```

3. Verify model configuration matches training:

```python
# Ensure the same configuration is used
assert model.physics_config == training_config
```

4. Check for state contamination between generations:

```python
# Initialize state explicitly for each generation
state = None
for prompt in prompts:
    state = None  # Reset state
    result = model.generate(prompt, max_tokens=100, state=state)
```

### 3.3 Slow Inference

**Problem:** Generation is too slow for production use.

**Symptoms:** High latency per token, or low tokens per second.

**Solution Steps:**

1. Enable CUDA kernels:

```python
# Ensure CUDA kernels are compiled
from gfn.cuda import load_kernels
load_kernels()
```

2. Use larger batch sizes during generation:

```python
# Process multiple prompts in parallel
prompts = [prompt1, prompt2, prompt3, prompt4]
batch = tokenizer(prompts, padding=True, return_tensors='pt')
output = model.generate(**batch, max_tokens=100)
```

3. Consider model distillation or quantization:

```python
# Post-training quantization (example for demonstration)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## 4. Configuration Issues

### 4.1 Invalid Configuration

**Problem:** Model raises error about invalid configuration.

**Symptoms:** ConfigurationError or ValueError during model creation.

**Solution Steps:**

1. Verify all required keys are present:

```python
physics_config = {
    'embedding': {...},
    'readout': {...},
    'active_inference': {...},
    'fractal': {...},
    'topology': {...},
    'stability': {...}
}
```

2. Check for valid option values:

```python
# Valid embedding types
'functional', 'implicit', 'standard'

# Valid integrator types
'leapfrog', 'forest_ruth', 'heun', 'rk4', 'yoshida', 'omelyan', 'verlet', 'euler'

# Valid topology types
'torus', 'sphere', 'plane'
```

3. Use the default configuration as a template:

```python
from gfn.constants import DEFAULT_CONFIG
physics_config = DEFAULT_CONFIG.copy()
```

### 4.2 Dimension Mismatch

**Problem:** Error about dimension mismatch during forward pass.

**Symptoms:** RuntimeError about tensor shapes not matching.

**Solution Steps:**

1. Verify vocab_size matches dataset:

```python
model = Manifold(vocab_size=50257)  # Standard GPT-2 vocabulary
# or
model = Manifold(vocab_size=len(tokenizer))  # Custom vocabulary
```

2. Check input dimensions:

```python
# Input should be [batch_size, seq_len] of token IDs
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
```

3. Verify heads divides dim evenly:

```python
model = Manifold(dim=512, heads=8)  # 512 / 8 = 64, valid
# vs
model = Manifold(dim=512, heads=7)  # Error: 512 not divisible by 7
```

### 4.3 State Persistence Issues

**Problem:** State not being maintained correctly during inference.

**Symptoms:** Context is lost between tokens, or O(1) memory benefit is not achieved.

**Solution Steps:**

1. Properly initialize and pass state:

```python
state = None
for t in range(sequence_length):
    logits, state, _ = model(input_ids[:, t:t+1], state=state)
    # state must be passed to next iteration
```

2. Verify state shape:

```python
# State should be (position, velocity) each of shape [batch, dim]
x, v = state
assert x.shape == (batch_size, dim)
assert v.shape == (batch_size, dim)
```

3. Avoid mixing training and inference state:

```python
model.eval()  # Switch to evaluation mode
with torch.no_grad():  # Disable gradient computation
    state = None
    for t in range(sequence_length):
        logits, state, _ = model(input_ids[:, t:t+1], state=state)
```

---

## 5. Performance Optimization

### 5.1 Profiling Model Performance

Use PyTorch profiling to identify bottlenecks:

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler')
) as prof:
    for batch in dataloader:
        model(batch)
        prof.step()
```

### 5.2 CUDA Optimization Checklist

For maximum CUDA performance:

1. Precompile CUDA kernels before training
2. Use mixed precision training with GradScaler
3. Enable cudnn benchmarking for fixed input sizes
4. Use torch.jit.script for inference optimization

```python
# Enable cudnn benchmarking
torch.backends.cudnn.benchmark = True

# Script model for inference
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')
```

### 5.3 Data Pipeline Optimization

Ensure the data pipeline is not a bottleneck:

```python
# Use multiple workers for data loading
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster GPU transfer
    prefetch_factor=2       # Prefetch batches
)
```

---

## 6. Common Error Messages

### 6.1 RuntimeError: Expected tensor with dimension...

**Cause:** Dimension mismatch in tensor operations.

**Solution:** Check input shapes match expected dimensions. Verify vocab_size, dim, and other parameters are consistent.

### 6.2 ValueError: Invalid configuration key...

**Cause:** Unknown configuration key or invalid value.

**Solution:** Verify all configuration keys are valid. Check documentation for valid option values.

### 6.3 torch.cuda.OutOfMemoryError

**Cause:** GPU memory exceeded.

**Solution:** Reduce batch size, enable gradient checkpointing, or use CPU offloading.

### 6.4 AssertionError: Input contains NaN values

**Cause:** NaN values in input data.

**Solution:** Check data preprocessing pipeline for NaN values. Normalize data properly.

### 6.5 RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED

**Cause:** CUDA operation failed, often due to configuration or memory issues.

**Solution:** Disable cudnn benchmarking, reduce batch size, or update CUDA drivers.

---

## 7. Getting Help

### 7.1 Before Asking for Help

Before submitting an issue or asking for help, please:

1. Search existing issues in the GitHub repository
2. Review this troubleshooting guide
3. Verify the issue is not in the FAQ
4. Collect relevant information:
   - Operating system and version
   - Python version
   - PyTorch version
   - CUDA version
   - GPU model
   - Complete error message
   - Minimal reproduction code

### 7.2 Reporting Issues

When reporting issues, include:

1. Complete error traceback
2. Minimal code to reproduce the issue
3. System information (use `gfn.utils.get_system_info()`)
4. Steps you have already tried

### 7.3 Debugging Tips

Enable verbose logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Use assertion checks in your code:

```python
assert input_ids.shape[0] == batch_size, "Batch size mismatch"
assert not torch.isnan(logits).any(), "NaN in output"
```

---

## 8. Best Practices Summary

### Installation Best Practices

- Use virtual environments to isolate dependencies
- Verify PyTorch installation before installing Manifold
- Compile CUDA kernels for production deployments
- Test installation with a minimal example before training

### Training Best Practices

- Always use RiemannianAdam optimizer
- Enable gradient clipping (0.05 threshold)
- Use linear embedding mode for generalization
- Start with the default configuration before tuning
- Monitor training with tensorboard or similar tools

### Inference Best Practices

- Switch to eval mode before inference
- Use torch.no_grad() to disable gradients
- Clear CUDA cache between long generations
- Match configuration between training and inference

### Configuration Best Practices

- Use the default configuration as a starting point
- Document any configuration changes
- Verify configuration matches between training and inference
- Test configuration changes on small scale before full training

---

**Document Version:** 2.6.4  
**For additional resources, see:** [API.md](API.md), [ARCHITECTURE.md](ARCHITECTURE.md), [BENCHMARKS.md](BENCHMARKS.md)
