"""
gfn/training/checkpoint.py
===========================
Checkpoint multi-módulo para pipelines GFN con proyectores y cabezas externas.

El gfn.api.save() usa formato Hugging Face (directorio). Este módulo cubre
el caso de múltiples módulos distintos (projector + manifold + detection_head)
donde se quiere un solo archivo .pt con metadata.

Uso:
    from ..training.checkpoint import save_checkpoint, load_checkpoint

    # Guardar
    save_checkpoint(
        "checkpoints/best.pt",
        modules={'projector': proj, 'manifold': manifold, 'det_head': head},
        metadata={'epoch': 3, 'score': 0.95, 'img_size': 64, 'dim': 64},
    )

    # Cargar
    states, meta = load_checkpoint("checkpoints/best.pt")
    projector.load_state_dict(states['projector'])
    manifold.load_state_dict(states['manifold'])
    det_head.load_state_dict(states['det_head'])
    print(meta['epoch'], meta['score'])

    # Cargar con módulos ya instanciados (más cómodo):
    load_checkpoint("best.pt", modules={'projector': proj, 'manifold': manifold, 'det_head': head})
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ['save_checkpoint', 'load_checkpoint']

# Prefijos de submodules internos del GFN que NO deben guardarse en el
# checkpoint del manifold (son diagnóstico, no arquitectura).
_SKIP_PREFIXES = ('physics_monitor.',)


def save_checkpoint(
    path: str | Path,
    modules:  Dict[str, nn.Module],
    metadata: Optional[Dict[str, Any]] = None,
    best:     bool = False,
) -> None:
    """
    Guarda un checkpoint multi-módulo en un único archivo .pt.

    Args:
        path:     Ruta destino. Los directorios padres se crean automáticamente.
        modules:  Diccionario { nombre: módulo }. Ej: {'projector': proj, 'manifold': m}.
        metadata: Cualquier dato serializable adicional (epoch, score, config, …).
        best:     Si True, también guarda una copia en <stem>_best.pt al lado.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt: Dict[str, Any] = {}

    for name, module in modules.items():
        sd = module.state_dict()
        # Filtrar claves internas que no son arquitectura (physics_monitor, etc.)
        filtered = {k: v for k, v in sd.items()
                    if not any(k.startswith(pfx) for pfx in _SKIP_PREFIXES)}
        ckpt[name] = filtered

    if metadata:
        ckpt.update(metadata)

    torch.save(ckpt, path)

    if best:
        best_path = path.parent / f"{path.stem}_best{path.suffix}"
        torch.save(ckpt, best_path)


def load_checkpoint(
    path:    str | Path,
    modules: Optional[Dict[str, nn.Module]] = None,
    device:  str = 'cpu',
    strict:  bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Carga un checkpoint multi-módulo.

    Args:
        path:    Ruta al archivo .pt.
        modules: Opcional. Si se pasa, carga los state_dicts directamente en
                 los módulos (in-place). Módulos no encontrados en el ckpt
                 se ignoran con un warning.
        device:  Dispositivo donde mapear los tensores.
        strict:  Si True (default), falla si hay keys inesperadas en el módulo.
                 Útil ponerlo en False al cargar checkpoints legados.

    Returns:
        (states, metadata) donde:
            states:   Dict[str, state_dict] — state dicts de los módulos presentes.
            metadata: Dict[str, Any]        — todo lo demás (epoch, score, config…).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {path}")

    raw = torch.load(path, map_location=device)

    # Separar state_dicts de metadata.
    # Un valor es un state_dict si es un dict cuyos valores son mayoritariamente tensores.
    states:   Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

    for key, value in raw.items():
        if _is_state_dict(value):
            states[key] = value
        else:
            metadata[key] = value

    # Cargar in-place si se pasaron módulos
    if modules is not None:
        for name, module in modules.items():
            if name not in states:
                warnings.warn(
                    f"[load_checkpoint] Módulo '{name}' no encontrado en el checkpoint. "
                    f"Keys disponibles: {list(states.keys())}",
                    UserWarning, stacklevel=2,
                )
                continue
            try:
                module.load_state_dict(states[name], strict=strict)
            except RuntimeError as e:
                if strict:
                    raise
                warnings.warn(
                    f"[load_checkpoint] Error al cargar '{name}' (strict=False): {e}",
                    UserWarning, stacklevel=2,
                )

    return states, metadata


def _is_state_dict(value: Any) -> bool:
    """Heurística: un state dict es un dict cuyos values son tensores o dicts de tensores."""
    if not isinstance(value, dict):
        return False
    if len(value) == 0:
        return False
    # Chequear que la mayoría de los valores son tensores
    tensor_count = sum(1 for v in value.values() if isinstance(v, torch.Tensor))
    return tensor_count > len(value) // 2
