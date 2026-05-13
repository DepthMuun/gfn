"""
Optimizadores — GFN V5
Incluye optimizadores estándar y Riemannian.
Migrado de gfn/optim/riemannian_adam.py y riemannian_sgd.py
"""

import torch
import torch.optim as optim
from typing import Optional, List, Dict, Any, Iterable
from ..constants import TOPOLOGY_TORUS, TOPOLOGY_EUCLIDEAN


class RiemannianAdam(optim.Adam):
    """
    Riemannian Adam — extensión del Adam estándar para manifolds.
    En la versión V5, actúa como Adam estándar pero está preparado
    para recibir métricas de manifold para ajustar la actualización.

    Para manifolds simples (Euclidean), es idéntico a Adam.
    Para torus y manifolds curvados, aplica un retract en el espacio de parámetros.
    """

    def __init__(self, params: Iterable, lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, geometry_type: str = TOPOLOGY_EUCLIDEAN):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.geometry_type = str(geometry_type).lower().strip()

    def step(self, closure=None):
        """
        Standard Adam step.
        For torus geometry, parameters are optionally wrapped after update.
        """
        loss = super().step(closure)

        if self.geometry_type == TOPOLOGY_TORUS:
            # Wrap position parameters to [-π, π] after gradient update
            with torch.no_grad():
                for group in self.param_groups:
                    if group.get('is_position', False):
                        for p in group['params']:
                            if p.grad is not None:
                                p.data = torch.atan2(torch.sin(p.data), torch.cos(p.data))

        return loss


class RiemannianSGD(optim.SGD):
    """
    Riemannian SGD con retract opcional para manifolds.
    """

    def __init__(self, params: Iterable, lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0,
                 geometry_type: str = TOPOLOGY_EUCLIDEAN):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.geometry_type = str(geometry_type).lower().strip()

    def step(self, closure=None):
        loss = super().step(closure)
        if self.geometry_type == 'torus':
            with torch.no_grad():
                for group in self.param_groups:
                    if group.get('is_position', False):
                        for p in group['params']:
                            p.data = torch.atan2(torch.sin(p.data), torch.cos(p.data))
        return loss


ManifoldSGD = RiemannianSGD


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Factory para crear optimizadores desde configuración.

    config keys:
    - 'type':         'adam' | 'riemannian_adam' | 'sgd' | 'adamw'
    - 'lr':           learning rate (default 1e-3)
    - 'weight_decay': weight decay (default 0.0)
    - 'geometry':     geometry type para Riemannian optimizers
    """
    opt_type = config.get('type', 'adam').lower()
    lr = config.get('lr', 1e-3)
    wd = config.get('weight_decay', 0.0)
    geometry = config.get('geometry', TOPOLOGY_EUCLIDEAN)

    # Group parameters to identify position variables for Riemannian wrapping
    pos_params = []
    other_params = []
    for name, p in model.named_parameters():
        # Identify x0 or anything explicitly marked as position
        if 'x0' in name or 'position' in name:
            pos_params.append(p)
        else:
            other_params.append(p)

    param_groups = [
        {'params': other_params},
        {'params': pos_params, 'is_position': True}
    ] if pos_params else model.parameters()

    if opt_type == 'riemannian_adam':
        return RiemannianAdam(param_groups, lr=lr, weight_decay=wd, geometry_type=geometry)
    elif opt_type == 'riemannian_sgd':
        return RiemannianSGD(param_groups, lr=lr, weight_decay=wd, geometry_type=geometry)
    elif opt_type == 'adamw':
        return optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    elif opt_type == 'sgd':
        return optim.SGD(param_groups, lr=lr, weight_decay=wd)
    else:
        return optim.Adam(param_groups, lr=lr, weight_decay=wd)


# ══════════════════════════════════════════════════════════════════════════════
# MAKE_GFN_OPTIMIZER — Dual-group estándar para pipelines GFN
# ══════════════════════════════════════════════════════════════════════════════

# Parámetros que pertenecen al grupo de física y necesitan lr más alto.
# Son los grados de libertad fundamentales del sistema hamiltoniano.
_PHYSICS_PARAM_NAMES = frozenset({'x0', 'v0', 'impulse_scale', 'gate'})


def make_gfn_optimizer(
    manifold:          torch.nn.Module,
    lr:                float = 1e-3,
    max_lr:            float = None,
    weight_decay:      float = 1e-4,
    physics_lr_scale:  float = 10.0,
    extra_modules:     list  = None,
    optimizer_cls:     type  = optim.AdamW,
    physics_param_names: frozenset = None,
) -> optim.Optimizer:
    """
    Crea un optimizador AdamW dual-group para pipelines GFN.

    El framework GFN distingue dos tipos de parámetros:

    **Grupo 1 — Parámetros de red** (proyectores, heads, weights de layers):
        Tasa de aprendizaje normal `lr`, weight decay `weight_decay`.

    **Grupo 2 — Parámetros de física** (x0, v0, impulse_scale, gate):
        Tasa de aprendizaje escalada `lr × physics_lr_scale`, sin weight decay.
        Necesitan lr más alto porque son pocos parámetros con gradientes pequeños.

    Ejemplo de uso (reemplaza ~15 líneas en cada benchmark):
        optimizer = make_gfn_optimizer(
            manifold,
            lr=1e-3,
            extra_modules=[projector, detection_head],
        )

    Args:
        manifold:           El modelo GFN principal (ManifoldModel).
        lr:                 Learning rate base. Default: 1e-3.
        max_lr:             Ignorado aquí — pasarlo al scheduler OneCycleLR.
        weight_decay:       Weight decay para parámetros de red. Default: 1e-4.
        physics_lr_scale:   Factor de escala para parámetros de física. Default: 10.
        extra_modules:      Lista de módulos adicionales (projectors, heads).
                            Sus parámetros van al grupo 1 (lr normal).
        optimizer_cls:      Clase de optimizador. Default: AdamW.
        physics_param_names: Set de nombres de parámetros de física.
                             Default: {'x0', 'v0', 'impulse_scale', 'gate'}.

    Returns:
        Optimizador configurado con dos param_groups.
    """
    _phys_names = physics_param_names or _PHYSICS_PARAM_NAMES

    # Recolectar todos los parámetros con nombre
    all_named: list[tuple[str, torch.nn.Parameter]] = []
    all_named.extend(manifold.named_parameters())
    if extra_modules:
        for mod in extra_modules:
            all_named.extend(mod.named_parameters())

    # Separar en dos grupos
    physics_params = [
        p for n, p in manifold.named_parameters()  # solo del manifold
        if p.requires_grad and any(phys in n for phys in _phys_names)
    ]
    physics_ids = {id(p) for p in physics_params}

    network_params = [
        p for _, p in all_named
        if p.requires_grad and id(p) not in physics_ids
    ]

    param_groups = [
        {
            'params':       network_params,
            'lr':           lr,
            'weight_decay': weight_decay,
        },
    ]
    if physics_params:
        param_groups.append({
            'params':       physics_params,
            'lr':           lr * physics_lr_scale,
            'weight_decay': 0.0,
        })

    return optimizer_cls(param_groups)


def all_parameters(*modules: torch.nn.Module) -> list:
    """
    Retorna todos los parámetros entrenables de una lista de módulos,
    sin duplicados (por id).

    Útil para gradient clipping:
        all_params = all_parameters(projector, manifold, det_head)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    """
    seen = set()
    params = []
    for mod in modules:
        for p in mod.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params
