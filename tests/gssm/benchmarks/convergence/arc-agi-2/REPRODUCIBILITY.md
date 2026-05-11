# ARC-AGI-2 Reproducibility Guide

Guía completa para reproducir los resultados del benchmark GSSM en ARC-AGI-2.

---

## Requisitos

### Software
```bash
Python >= 3.10
PyTorch >= 2.0
CUDA >= 11.8 (para GPU)
Git
```

### Dependencias
```bash
pip install torch numpy tqdm tensorboard
pip install -e /path/to/gfn  # Instalar GSSM
```

### Hardware Recomendado
- GPU: NVIDIA con 8GB+ VRAM
- RAM: 16GB+
- Storage: 10GB libre

---

## Quick Start

### 1. Preparar Datos

```bash
cd tests/gssm/benchmarks/convergence/arc-agi-2

# Descargar y preparar dataset
python scripts/prepare_data.py \
    --data_dir data \
    --download \
    --split_ratios 0.8/0.1/0.1 \
    --seed 42
```

Esto creará:
- `data/arc_agi_2_data/` - Dataset original
- `data/processed/` - Datos procesados con splits

### 2. Verificar Tests

```bash
# Verificar métricas (CRÍTICO antes de entrenar)
python tests/test_metrics.py

# Verificar integración
python tests/test_integration.py
```

**Ambos deben pasar 100%** antes de publicar resultados.

### 3. Entrenar Modelo

```bash
python scripts/train.py \
    --data_path data/processed \
    --config medium \
    --epochs 100 \
    --lr 0.001 \
    --max_train_pairs 3 \
    --device cuda \
    --output_dir results
```

### 4. Evaluar

```bash
python scripts/evaluate.py \
    --data_path data/processed \
    --checkpoint results/checkpoints/best_model.pt \
    --split test \
    --save_predictions \
    --output_dir results/predictions
```

---

## Métricas Oficiales

### La Métrica ARC-AGI-2

**Strict Match**: Una task cuenta como correcta si y solo si:
1. El tamaño del grid coincide exactamente
2. TODOS los pixels coinciden exactamente

**NO hay partial credit**.

### Verificación de Métricas

```python
# En src/evaluation/metrics.py
ARCMetrics.strict_match(pred, gt)  # True si exactamente igual
```

Para verificar que tu implementación es correcta:
```bash
python tests/test_metrics.py
```

Debe mostrar:
```
🎉 ALL TESTS PASSED!
Metrics implementation is verified and ready for publication.
```

---

## Configuraciones

### Small (Rápida prueba)
```bash
python scripts/train.py --config small --epochs 50
```
- dim: 128
- heads: 4
- depth: 4

### Medium (Balance)
```bash
python scripts/train.py --config medium --epochs 100
```
- dim: 256
- heads: 8
- depth: 6

### Large (Máxima capacidad)
```bash
python scripts/train.py --config large --epochs 200
```
- dim: 512
- heads: 16
- depth: 8

---

## Reproducibilidad

### Seeds
Por defecto, seed=42 en todos los scripts.

Para reproducir exactamente:
```bash
python scripts/train.py --seed 42 ...
python scripts/evaluate.py --seed 42 ...
```

### Determinismo
- `num_workers=0` (single-threaded)
- No hay augmentations aleatorias
- Shuffling reproducible con seed fijo

### Checkpointing
Los checkpoints guardan:
- Model weights
- Optimizer state
- Época actual
- Métricas

---

## Resultados Esperados

### Baselines

| Config | Task Accuracy | Epochs | Time |
|--------|--------------|--------|------|
| Random | ~0% | - | - |
| Small | 1-5% | 50 | 1h |
| Medium | 5-15% | 100 | 4h |
| Large | 10-25% | 200 | 12h |

*Nota: ARC-AGI-2 es extremadamente difícil. SOTA humano ~80%, SOTA ML ~30%.*

### Guardar Resultados

Los resultados se guardan en:
```
results/
├── checkpoints/
│   ├── best_model.pt
│   └── checkpoint_epoch_N.pt
├── logs/           # TensorBoard
├── predictions/
│   ├── predictions_test.json
│   └── metrics_test.json
└── config.json     # Config exacta usada
```

---

## Verificación Externa

Para permitir verificación externa:

### 1. Guardar Predicciones
```bash
python scripts/evaluate.py --save_predictions ...
```

### 2. Compartir Archivos
- `predictions_test.json` - Predicciones en formato ARC
- `metrics_test.json` - Métricas detalladas
- `config.json` - Configuración exacta
- `best_model.pt` - Checkpoint (opcional)

### 3. Script de Verificación
Cualquiera puede verificar:
```python
from src.evaluation.metrics import verify_predictions

accuracy = verify_predictions(
    "predictions_test.json",
    "ground_truth.json"  # De ARC-AGI-2 oficial
)
```

---

## Troubleshooting

### Out of Memory
```bash
# Usar config small
python scripts/train.py --config small

# O usar CPU
python scripts/train.py --device cpu
```

### Métricas Dudosas
```bash
# Siempre verificar antes de publicar
python tests/test_metrics.py
python tests/test_integration.py
```

### Dataset No Descarga
```bash
# Manualmente
git clone https://github.com/arcprize/ARC-AGI-2.git data/arc_agi_2_data
python scripts/prepare_data.py --data_dir data
```

---

## Publicación de Resultados

### Checklist Antes de Publicar

- [ ] Tests de métricas pasan 100%
- [ ] Tests de integración pasan 100%
- [ ] Seed documentado
- [ ] Config exacta guardada
- [ ] Predicciones guardadas (para verificación)
- [ ] Checkpoint disponible (opcional pero recomendado)
- [ ] Métricas reportadas con error estimado (si aplica)

### Formato de Reporte

```
Modelo: GSSM-{config}
Dataset: ARC-AGI-2 (split 0.8/0.1/0.1, seed=42)
Métrica: Task Accuracy (strict match)
Resultado: XX.X% (X/Y tasks correct)
Seed: 42
Config: Ver archivo config.json
Checkpoint: Disponible en {url}
Predicciones: Disponibles en {url} para verificación
```

---

## Contacto y Issues

Para problemas con el benchmark:
1. Verificar tests pasan
2. Checkear logs en `results/logs/`
3. Reportar con config.json adjunto

---

*Última actualización: 2026-04-02*
