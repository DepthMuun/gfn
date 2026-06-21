# Quick Start Guide

This guide gives you the shortest reliable path to a working GSSM run using the current public API.

It intentionally starts from code, not from older YAML snapshots.

## 1. Verify The Import

From the project root:

```bash
python -c "import gfn; print('gfn import ok')"
```

If that fails, fix the environment before continuing.

## 2. Create A Minimal GSSM Model

The preferred public entry point is:

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=256,
    dim=128,
    depth=2,
    heads=4,
)
```

That path goes through the current config normalization and model factory.

## 3. Run A Sanity Forward Pass

```python
import torch
import gfn

model = gfn.create("gssm", vocab_size=256, dim=128, depth=2, heads=4)
input_ids = torch.randint(0, 256, (2, 8))

logits, (x_final, v_final), state_info = model(input_ids)

print(logits.shape)
print(x_final.shape, v_final.shape)
print(state_info.keys())
```

Expected shape-level behavior:

- `logits` has `[batch, seq, vocab]`
- `x_final` and `v_final` have matching latent shapes
- `state_info` includes trajectory information such as `x_seq` and `v_seq`

## 4. Start From Effective Defaults

A fresh `gfn.create("gssm", ...)` currently resolves close to:

- topology: `torus`
- geometry: analytical torus
- integrator: `leapfrog`
- `base_dt = 0.1`
- `friction = 0.01`
- embedding mode: `linear`
- readout type: `standard`

These are safe starting values because they match the current runtime path.

## 5. Minimal Training Loop

```python
import torch
import gfn

model = gfn.create("gssm", vocab_size=256, dim=128, depth=2, heads=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

input_ids = torch.randint(0, 256, (4, 16))
targets = input_ids.roll(shifts=-1, dims=1)

logits, (_, _), state_info = model(input_ids)
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.shape[-1]),
    targets.reshape(-1),
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

This is only a shape-valid example. Real tasks should use targets and losses that actually match the task semantics.

## 6. Use A Real Benchmark Script

If you want a repository example instead of a handcrafted loop, there are real scripts under `tests/gssm/benchmarks/`.

One current example is:

```bash
python tests/gssm/benchmarks/convergence/math/train_math.py
```

That script uses the public `gfn.create(...)` path and a generative loss over a small symbolic math task.

## 7. What To Watch First

At the beginning, check only the basics:

- loss decreases at all
- logits and targets actually live in the same space
- no NaNs appear
- the model return contract matches what your training script expects

Do not assume every benchmark prints the same metric names.

## 8. First Safe Overrides

If you need to customize the model, start with explicit but conservative overrides:

```python
import gfn

model = gfn.create(
    "gssm",
    vocab_size=1024,
    dim=256,
    depth=3,
    heads=4,
    physics={
        "topology": {"type": "torus"},
        "stability": {
            "integrator_type": "leapfrog",
            "base_dt": 0.1,
            "friction": 0.01,
        },
        "dynamics": {"type": "direct"},
        "readout": {"type": "standard"},
    },
)
```

## 9. When To Change More

Only start changing these after the baseline path works:

- geometry override
- readout type
- continuous embedding mode
- optional physics modules
- non-default integrators

Most early failures come from mismatching readout, target, and loss, not from lacking an exotic geometry setting.

## 10. Next Steps

After the sanity run works, continue with:

1. `03-reference/00-handbook.md`
2. `03-reference/02-api-classes.md`
3. `04-guides/02-advanced-configuration.md`
4. `04-guides/03-problem-solving.md`
