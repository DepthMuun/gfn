# Code and Quality Standards

For professional development, code must be readable, typed, and self-documented.

## 1. Code Style
- **Python**: We follow PEP 8.
- **Formatting**: We use `black` or `ruff`. Please format your files before committing.
- **Naming**: 
  - Classes: `PascalCase` (e.g., `TopologicalIntegrator`).
  - Functions/Variables: `snake_case` (e.g., `compute_geodesic_loss`).

## 2. Documentation (Docstrings)
We use the Google style for docstrings. Every public function must have explanation of arguments and return.

```python
def compute_flow(state: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute the geodesic flow for a given state.

    Args:
        state: The current state tensor (Batch, Dim).
        dt: The integrator time step.

    Returns:
        The new state after integration.
    """
    ...
```

## 3. Static Typing (Type Hinting)
The use of `typing` is mandatory in public function signatures. It helps catch errors early and improves IDE autocompletion.

## 4. Testing
- Pull Requests that break the `pytest` suite are not accepted.
- If you add new functionality, add at least one unit test in the `tests/` folder.
- Execution: `pytest tests/` from the project root.
