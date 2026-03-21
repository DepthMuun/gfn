"""
PhysicsLoss — GFN V5
Pérdidas physics-informed consolidadas.
Migradas y consolidadas de gfn/losses/orchestration/physics_informed.py
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from ..losses.base import BaseLoss
from ..registry import register_loss
from ..constants import EPS


# ─── Primitivas de pérdida físicas ──────────────────────────────────────────

def geodesic_regularization(christoffels: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
    """
    Penaliza la magnitud de la aceleración geodésica.
    L_geo = ||Γ(v,v)||² — mide cuánto 'curva' la trayectoria.
    """
    if christoffels is None or velocities is None:
        return torch.zeros(1, device='cpu', requires_grad=True)
    return (christoffels ** 2).mean()


def hamiltonian_conservation(x_seq: torch.Tensor, v_seq: torch.Tensor) -> torch.Tensor:
    """
    Penaliza la variación de la energía de Hamiltonian a lo largo de la trayectoria.
    H = 0.5 * ||v||² (energía cinética, en ausencia de potencial explícito).
    Una trayectoria simpléctica debería conservar H.
    """
    if x_seq is None or v_seq is None:
        return torch.zeros(1, device='cpu', requires_grad=True)
    # H en cada tiempo [B, S]
    H = 0.5 * (v_seq ** 2).sum(dim=-1)
    # Varianza temporal de H como medida de não-conservación
    H_var = H.var(dim=1).mean()
    return H_var


def kinetic_regularization(v_seq: torch.Tensor, max_kinetic: float = 10.0) -> torch.Tensor:
    """
    Penaliza energía cinética excesiva.
    Previene explosión de velocidades.
    """
    if v_seq is None:
        return torch.zeros(1, device='cpu', requires_grad=True)
    ke = 0.5 * (v_seq ** 2).sum(dim=-1)
    excess = F.relu(ke - max_kinetic)
    return excess.mean()


# ─── Clases de pérdida ──────────────────────────────────────────────────────

@register_loss('physics')
class PhysicsLoss(BaseLoss):
    """
    Pérdida física combinada, configurable por componentes.

    Componentes disponibles:
    - 'geodesic':   Regularización geodésica (penaliza curvatura excesiva)
    - 'hamiltonian': Conservación del Hamiltoniano
    - 'kinetic':    Regularización de energía cinética
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.lambda_geo = self.config.get('lambda_geo', 0.001)
        self.lambda_ham = self.config.get('lambda_ham', 0.0)
        self.lambda_kin = self.config.get('lambda_kin', 0.0)
        self.max_kinetic = self.config.get('max_kinetic', 10.0)

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor,
                state_info: Optional[Dict[str, Any]] = None, **kwargs) -> torch.Tensor:
        if state_info is None:
            return torch.zeros(1, device=x_pred.device, requires_grad=True)

        losses: List[torch.Tensor] = []
        dev = x_pred.device

        if self.lambda_geo > 0 and 'christoffels' in state_info and 'v_seq' in state_info:
            l = geodesic_regularization(state_info['christoffels'], state_info['v_seq'])
            losses.append(l.to(dev) * self.lambda_geo)

        if self.lambda_ham > 0 and 'x_seq' in state_info and 'v_seq' in state_info:
            l = hamiltonian_conservation(state_info['x_seq'], state_info['v_seq'])
            losses.append(l.to(dev) * self.lambda_ham)

        if self.lambda_kin > 0 and 'v_seq' in state_info:
            l = kinetic_regularization(state_info['v_seq'], self.max_kinetic)
            losses.append(l.to(dev) * self.lambda_kin)

        if not losses:
            return torch.zeros(1, device=dev, requires_grad=True)

        return torch.stack(losses).sum()


@register_loss('generative_physics')
class PhysicsInformedLoss(BaseLoss):
    """
    Pérdida generativa + regularización física combinada.
    La pérdida principal es CrossEntropy (sobre logits, no representaciones toroidales).
    El término físico actúa como regularizador.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.lambda_physics = self.config.get('lambda_physics', 0.01)
        self.physics_loss = PhysicsLoss(config)
        self.entropy_coef = self.config.get('entropy_coef', 0.0)

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor,
                state_info: Optional[Dict[str, Any]] = None, **kwargs) -> torch.Tensor:
        # CrossEntropy sobre logits [B, S, V] → scalar
        ce = F.cross_entropy(x_pred.view(-1, x_pred.size(-1)), x_target.view(-1))

        # Entropy bonus (opcional)
        if self.entropy_coef > 0:
            probs = F.softmax(x_pred, dim=-1)
            entropy = -(probs * torch.log(probs + EPS)).sum(dim=-1).mean()
            ce = ce - self.entropy_coef * entropy

        # Physics regularizer
        phys = self.physics_loss(x_pred, x_target, state_info=state_info)

        return ce + self.lambda_physics * phys
