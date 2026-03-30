# Estándares de Código y Calidad

Para un desarrollo profesional, el código debe ser legible, tipado y autodocumentado.

## 1. Estilo de Código
- **Python**: Seguimos PEP 8.
- **Formateo**: Usamos `black` o `ruff`. Por favor, formatea tus archivos antes de commitear.
- **Naming**: 
  - Clases: `PascalCase` (ej: `TopologicalIntegrator`).
  - Funciones/Variables: `snake_case` (ej: `compute_geodesic_loss`).

## 2. Documentación (Docstrings)
Usamos el estilo de Google para docstrings. Cada función pública debe tener explicación de argumentos y retorno.

```python
def compute_flow(state: torch.Tensor, dt: float) -> torch.Tensor:
    """Calcula el flujo geodésico para un estado dado.

    Args:
        state: El tensor de estado actual (Batch, Dim).
        dt: El paso de tiempo del integrador.

    Returns:
        El nuevo estado tras la integración.
    """
    ...
```

## 3. Tipado Estático (Type Hinting)
El uso de `typing` es obligatorio en firmas de funciones públicas. Ayuda a evitar errores de forma temprana y mejora el autocompletado en el IDE.

## 4. Pruebas (Testing)
- No se aceptan Pull Requests que rompan la suite de `pytest`.
- Si añades una nueva funcionalidad, añade al menos un test unitario en la carpeta `tests/`.
- Ejecución: `pytest tests/` desde la raíz del proyecto.
