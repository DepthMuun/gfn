# GFN CUDA-Python Consistency Test Suite

## Resumen

Este documento describe el stack de testing completo creado para validar la consistencia entre las implementaciones CUDA y Python del proyecto GFN.

## Archivos Creados

### 1. `tests/test_cuda_python_consistency.py`

Test suite principal con 11 clases de test y más de 40 tests individuales:

| Clase de Test | Descripción | Tests |
|---------------|-------------|-------|
| `TestCUDAAvailability` | Verificación de dispositivos CUDA y constantes | 3 |
| `TestChristoffelOperation` | Tests de símbolos de Christoffel | 4 |
| `TestLeapfrogIntegration` | Tests del integrador Leapfrog | 4 |
| `TestGradientConsistency` | Verificación de gradientes | 3 |
| `TestCUDAVsPythonEquivalence` | Equivalencia numérica CUDA-Python | 2 |
| `TestConvergenceBehavior` | Tests de convergencia | 2 |
| `TestEdgeCases` | Casos límite y estabilidad | 5 |
| `TestPerformanceBenchmarks` | Benchmarks de rendimiento | 3 |
| `TestTopologyBehavior` | Tests de topología | 3 |
| `TestAutogradFunctionality` | Funciones autograd | 3 |
| `TestFullPipeline` | Tests de integración | 2 |

### 2. `tests/run_consistency_tests.py`

Script ejecutor con opciones flexibles:

```bash
# Ver resumen de tests disponibles
python tests/run_consistency_tests.py --summary

# Quick sanity checks
python tests/run_consistency_tests.py --quick

# Ejecutar suite completa
python tests/run_consistency_tests.py

# Ejecutar solo CPU (sin GPU)
python tests/run_consistency_tests.py --cpu-only

# Modo verbose
python tests/run_consistency_tests.py --verbose
```

## Categorías de Test

### 1. Tests de Equivalencia Numérica

Verifican que CUDA y Python producen resultados idénticos:

```python
def test_christoffel_cuda_python_equivalence(self, cuda_config, test_tensors):
    gamma_cuda = christoffel_fused(v_cuda, U_cuda, W_cuda, ...)
    gamma_python = python_op.forward(v, U, W, ...)
    
    max_diff = compute_max_abs_diff(gamma_cuda_cpu, gamma_python)
    assert max_diff < tolerance * 10
```

**Métricas rastreadas:**
- `max_diff`: Diferencia máxima absoluta
- `mean_diff`: Diferencia media absoluta
- Tolerancia: 1e-4 (forward), 1e-3 (gradientes)

### 2. Tests de Consistencia de Gradientes

Verifican que los gradientes son correctos mediante diferenciación numérica:

```python
def test_gradient_numerical_verification(self, config, test_tensors):
    # Autograd reference
    grad_ref = v_ref.grad
    
    # Numerical gradient
    grad_numerical = (gamma_plus - gamma_minus) / (2 * eps)
    
    # Compare
    relative_diff = compute_relative_error(grad_ref, grad_numerical)
    assert relative_diff < 1e-3
```

### 3. Tests de Convergencia

Verifican que la optimización converge correctamente:

```python
def test_learning_curve_convergence(self, config, test_tensors):
    tracker = ConvergenceTracker(tolerance=1e-6, max_iterations=50)
    
    for i in range(50):
        # Training step
        loss = torch.sum(gamma * gamma)
        loss.backward()
        optimizer.step()
        
        tracker.step(float(loss))
        if tracker.converged():
            break
    
    assert stats['iterations'] <= 45
    assert stats['final_loss'] < stats['initial_loss']
```

### 4. Tests de Rendimiento

Benchmarks para medir throughput y speedup:

```python
def test_christoffel_throughput(self, config):
    iterations = 100
    start = time.perf_counter()
    
    for _ in range(iterations):
        _ = python_op.forward(v, U, W, None, None, plasticity=0.0)
    
    avg_time = elapsed / iterations * 1000
    assert avg_time < 100  # ms
```

### 5. Tests de Casos Límite

Verifican comportamiento en condiciones extremas:

- Velocidad cero
- Velocidad unitaria
- Valores de entrada grandes
- Paso de tiempo muy pequeño
- Muchos substeps

### 6. Tests de Topología

Verifican comportamiento con diferentes topologies:

- Euclidiana (plana)
- Tórica (periódica)
- Condiciones de frontera

## Métricas de Consistencia

### Tolerancias Esperadas

| Tipo de Test | Tolerancia | Notas |
|--------------|------------|-------|
| Forward pass (CUDA vs Python) | 1e-4 | Diferencias debido a precisión |
| Gradientes | 1e-3 | Verificación numérica |
| Energía | 10.0 | Rate de cambio de Hamiltoniano |
| Rendimiento | < 100ms | Por operación Christoffel |

### Criterios de Éxito

```
✓ CONSISTENTE:     max_diff < 1e-4
⚠️ PEQUEÑAS DIFERENCIAS: 1e-4 ≤ max_diff < 1e-2
❌ DIVERGENCIA:    max_diff ≥ 1e-2
```

## Ejemplo de Uso

```python
# Importar test suite
import pytest
from tests.test_cuda_python_consistency import *

# Ejecutar todos los tests
pytest tests/test_cuda_python_consistency.py -v

# Ejecutar solo tests de equivalencia
pytest tests/test_cuda_python_consistency.py -v -k "equivalence"

# Ejecutar tests de gradientes
pytest tests/test_cuda_python_consistency.py -v -k "gradient"
```

## Configuración de Test

Los tests usan `TestConfig` con valores por defecto:

```python
@dataclass
class TestConfig:
    device: str = 'cpu'
    dtype: torch.dtype = torch.float32
    batch_size: int = 32
    dimension: int = 64
    rank: int = 8
    tolerance: float = 1e-4
    gradient_tolerance: float = 1e-3
    max_iterations: int = 100
    seed: int = 42
```

## Funciones de Utilidad

### Cálculo de Errores

```python
def compute_relative_error(tensor1, tensor2):
    """Error relativo entre dos tensores."""
    diff = torch.abs(tensor1 - tensor2)
    norm = torch.max(torch.abs(tensor1))
    return float(torch.max(diff / (norm + 1e-8)))

def compute_max_abs_diff(tensor1, tensor2):
    """Diferencia máxima absoluta."""
    return float(torch.max(torch.abs(tensor1 - tensor2)).item())

def compute_mean_abs_diff(tensor1, tensor2):
    """Diferencia media absoluta."""
    return float(torch.mean(torch.abs(tensor1 - tensor2)).item())
```

### Tracking de Convergencia

```python
tracker = ConvergenceTracker(tolerance=1e-6, max_iterations=100)

for epoch in range(100):
    # Training step...
    tracker.step(loss_value, grad_norm)
    
    if tracker.converged():
        break

stats = tracker.get_stats()
# stats = {
#     'iterations': 50,
#     'initial_loss': 1.234,
#     'final_loss': 0.123,
#     'loss_reduction': 1.111,
#     'converged': True,
#     'max_grad_norm': 0.5
# }
```

## Próximos Pasos

1. **Ejecutar tests iniciales:** `python tests/run_consistency_tests.py --quick`
2. **Ejecutar suite completa:** `python tests/run_consistency_tests.py`
3. **Verificar CUDA disponible:** `python -c "import torch; print(torch.cuda.is_available())"`
4. **Analizar resultados:** Revisar métricas de consistencia

## Notas

- Los tests usan PyTorch 2.x con `torch.autograd`
- Los tests de CUDA se saltan si no hay GPU disponible
- Seeds aleatorios garantizan reproducibilidad
- Los tests están diseñados para ser rápidos (< 1min total)
