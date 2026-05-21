# ARC-AGI-2 GSSM Benchmark

Benchmark de GSSM en ARC-AGI-2 con few-shot learning y métricas verificables.

---

## Estructura

```
arc-agi-2/
├── README.md                    # Este archivo
├── data/                        # Dataset ARC-AGI-2
│   ├── arc_agi_2_data/          # Repositorio clonado
│   ├── processed/               # Datos procesados
│   └── splits/                  # Splits train/val/test
├── src/                         # Código fuente
│   ├── data/                    # Data loaders
│   ├── models/                  # Configuraciones de modelo
│   ├── training/                # Lógica de entrenamiento
│   ├── evaluation/              # Métricas y evaluación
│   └── utils/                   # Utilidades
├── configs/                     # Archivos de configuración
│   ├── model/                   # Configs de modelo GSSM
│   ├── training/                # Configs de entrenamiento
│   └── data/                    # Configs de datos
├── scripts/                     # Scripts ejecutables
│   ├── train.py                 # Entrenamiento
│   ├── evaluate.py              # Evaluación
│   ├── prepare_data.py          # Preparar dataset
│   └── visualize.py             # Visualización
├── tests/                       # Tests unitarios
│   ├── test_data_loader.py
│   ├── test_metrics.py
│   └── test_model.py
└── results/                     # Resultados (generado)
    ├── checkpoints/             # Modelos guardados
    ├── logs/                    # Logs de entrenamiento
    ├── metrics/                 # Métricas calculadas
    └── predictions/             # Predicciones
```

---

## Dataset ARC-AGI-2

- **Fuente**: https://github.com/arcprize/ARC-AGI-2.git
- **Formato**: JSON con pairs de input/output grids
- **Tamaño**: ~1000+ tasks con train/eval splits
- **Características**: 
  - Grids de 0-9 colores (10 valores)
  - Tamaños variables (1x1 hasta 30x30)
  - Few-shot: 1-5 ejemplos de entrenamiento por task
  - Test: 1 input grid, predecir output grid

---

## Métricas (Verificables)

### Primary: Task Accuracy
```python
# Una task está "correcta" si TODOS los pixels coinciden exactamente
task_correct = np.all(pred_grid == true_grid)
accuracy = mean(task_correct across all tasks)
```

### Secondary Metrics
- **Pixel Accuracy**: % de pixels correctos (pero no es la métrica principal)
- **Grid Size Accuracy**: % de tasks con tamaño correcto
- **Color Accuracy**: % de pixels con color correcto

### ARC-AGI-2 Official Metric
La métrica oficial de ARC-AGI-2 es **strict match**: 
- La predicción debe ser EXACTAMENTE igual al ground truth
- No hay partial credit
- Size, colors, positions deben coincidir perfectamente

---

## Configuración GSSM

### Input Representation
- **Mode**: `continuous` (grids son imágenes)
- **Dim**: Modelo procesa grids como input continuo
- **Tokenization**: Flatten grid + positional encoding

### Model Architecture
```python
dim: 256          # Hidden dimension
heads: 8          # Attention heads
depth: 6          # Manifold layers
topology: 'torus' # Default (stable)
integrator: 'leapfrog' # Symplectic
```

### Training Strategy
- **Few-shot**: 1-3 ejemplos por task durante entrenamiento
- **Batch**: 1 task = 1 batch (varía por tamaño)
- **Loss**: Grid prediction (MSE para colores + structure)
- **Optimizer**: RiemannianAdam con dual-group

---

## Workflow

### 1. Preparar Datos
```bash
python scripts/prepare_data.py --download --split 0.8/0.1/0.1
```

### 2. Entrenar
```bash
python scripts/train.py --config configs/training/few_shot.yaml --epochs 100
```

### 3. Evaluar
```bash
python scripts/evaluate.py --checkpoint results/checkpoints/best.pt --split test
```

### 4. Verificar Métricas
```bash
python tests/test_metrics.py --predictions results/predictions/test.json
```

---

## Verificación de Reproducibilidad

Para asegurar que las métricas son correctas:

1. **Ground Truth Loading**: Cargar directamente del JSON oficial
2. **Prediction Format**: Exactamente mismo formato que GT
3. **Comparison**: Element-wise numpy comparison
4. **No preprocessing**: No aplicar transforms al GT
5. **Logging**: Guardar predicciones raw para audit

---

## Requisitos

```
torch>=2.0
gfn (local)
numpy
json
tqdm
matplotlib  # Para visualización
```

---

*Última actualización: 2026-04-02*
