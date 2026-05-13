"""
ManifoldGenerativeLoss — GFN V5
Pérdida generativa para el modelo Manifold.

ARQUITECTURA: El modelo puede usar diferentes estrategias de salida:
1. Holographic: El estado final se usa directamente como logits
2. Readout: Proyección lineal del estado al vocabulario
3. Toroidal: Coordenadas angulares para espacio toroidal

Opciones:
- 'nll':        CrossEntropy sobre logits (default para readout categórico)
- 'mse':        L2 sobre el espacio de salida (para representaciones continuas)
- 'cosine':     Distancia coseno (embeddings normalizados)
- 'toroidal':   Distancia angular geodésica (para manifold toroidal)
- 'hybrid':     Combina NLL con regularización toroidal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from ..losses.base import BaseLoss
from ..registry import register_loss
from ..constants import EPS


@register_loss('generative')
class ManifoldGenerativeLoss(BaseLoss):
    """
    Pérdida generativa para GFN V5.
    
    Maneja múltiples modos de salida del manifold:
    - 'nll':      CrossEntropy sobre logits proyectados
    - 'mse':      MSE sobre vectores continuos
    - 'cosine':   Distancia coseno
    - 'toroidal': Distancia angular geodésica
    - 'hybrid':   Combina NLL + toroidal
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.mode = self.config.get('mode', 'nll')
        self.entropy_coef = self.config.get('entropy_coef', 0.0)
        self.label_smoothing = self.config.get('label_smoothing', 0.0)
        
        # Parámetros para modo toroidal
        self.toroidal_scale = self.config.get('toroidal_scale', 1.0)
        self.toroidal_weight = self.config.get('toroidal_weight', 0.3)
        
        # Parámetros para modo híbrido
        self.hybrid_nll_weight = self.config.get('hybrid_nll_weight', 0.7)

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor,
                state_info: Optional[Dict[str, Any]] = None, **kwargs) -> torch.Tensor:
        """
        Args:
            x_pred:   Logits o vectores de salida del readout [B, S, V] o [B, S, D]
            x_target: Token IDs objetivo [B, S]
            state_info: Información del estado para pérdidas adicionales

        Returns:
            Escalar de pérdida.
        """
        if self.mode == 'mse':
            return self._mse_loss(x_pred, x_target)
        
        elif self.mode == 'cosine':
            return self._cosine_loss(x_pred, x_target)
        
        elif self.mode == 'toroidal':
            return self._toroidal_loss(x_pred, x_target)
        
        elif self.mode == 'hybrid':
            return self._hybrid_loss(x_pred, x_target, state_info)
        
        else:
            # mode == 'nll' (default)
            return self._nll(x_pred, x_target)

    def _nll(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, S, V] or [B, S, H, HD], targets: [B, S]
        if logits.dim() == 2:
            # Case where it's already flattened or single step
            return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
            
        if logits.dim() == 4:
            # Multi-head holographic case [B, S, H, HD] -> flatten heads for loss
            B, S, H, HD = logits.shape
            logits = logits.reshape(B, S, H * HD)

        B, S, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * S, V),
            targets.reshape(B * S),
            label_smoothing=self.label_smoothing
        )

        if self.entropy_coef > 0:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + EPS)).sum(dim=-1).mean()
            loss = loss - self.entropy_coef * entropy

        return loss

    def _mse_loss(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Pérdida L2 sobre vectores continuos."""
        # Asegurar targets en float para permitir flujo de gradiente
        y = x_target.float()
        
        # Si pred es [B, S, V] y target es [B, S], promediamos logits si no hay readout
        if x_pred.dim() == 3 and y.dim() == 2:
             # Caso de regresión sobre canal: promediamos o ajustamos
             # Por ahora, si el usuario pide MSE sobre logits, promediamos el vocab
             y = y.unsqueeze(-1)
        
        return F.mse_loss(x_pred, y)

    def _cosine_loss(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Distancia coseno."""
        if x_target.dtype in (torch.long, torch.int):
            return self._nll(x_pred, x_target)
        
        x_pred_n = F.normalize(x_pred, dim=-1)
        x_tgt_n = F.normalize(x_target.float(), dim=-1)
        return (1 - (x_pred_n * x_tgt_n).sum(dim=-1)).mean()

    def _toroidal_loss(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Pérdida toroidal: distancia angular geodésica.
        Requiere que x_pred sean coordenadas angulares.
        """
        # Si x_pred son logits, convertir a coordenadas
        if x_pred.dim() == 3 and x_pred.shape[-1] > 1:
            # Logits -> coordenadas angulares
            probs = F.softmax(x_pred, dim=-1)
            # Asumir vocabulario evenly spaced en [0, 2π]
            num_classes = x_pred.shape[-1]
            angles = torch.linspace(0, 2 * torch.pi, num_classes, 
                                   device=x_pred.device).unsqueeze(0)
            x_pred_coords = torch.sum(probs * angles.unsqueeze(0), dim=-1)
        else:
            x_pred_coords = x_pred.squeeze(-1) if x_pred.dim() > 2 else x_pred
        
        # Convertir targets a coordenadas angulares
        if x_target.dtype in (torch.long, torch.int):
            num_classes = x_pred.shape[-1] if x_pred.dim() == 3 else 100
            x_target_coords = 2 * torch.pi * x_target.float() / num_classes
        else:
            x_target_coords = x_target
        
        # Calcular distancia angular
        diff = x_pred_coords - x_target_coords
        diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        return self.toroidal_scale * (diff_wrapped ** 2).mean()

    def _hybrid_loss(self, x_pred: torch.Tensor, x_target: torch.Tensor,
                    state_info: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Pérdida híbrida: combina NLL con regularización toroidal.
        Útil cuando el modelo produce logits pero el espacio latente es toroidal.
        """
        # Componente NLL
        nll = self._nll(x_pred, x_target)
        
        # Componente toroidal (si tenemos información del estado)
        toroidal_loss = torch.tensor(0.0, device=x_pred.device)
        
        if state_info is not None and 'x_seq' in state_info:
            x_seq = state_info['x_seq']
            
            # Obtener coordenadas del último estado
            if x_seq.dim() == 4:
                # [B, S, H, D] -> obtener estado final
                x_final = x_seq[:, -1, :, :]  # [B, H, D]
            else:
                x_final = x_seq
            
            # Calcular distancia a target
            if x_target.dtype in (torch.long, torch.int):
                num_classes = x_pred.shape[-1]
                tgt_coords = 2 * torch.pi * x_target.float() / num_classes
            else:
                tgt_coords = x_target.float()
            
            # Distancia geodésica
            diff = x_final - tgt_coords.unsqueeze(-1) if tgt_coords.dim() > 1 else x_final - tgt_coords
            diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
            toroidal_loss = (diff_wrapped ** 2).mean()
        
        # Combinar
        return self.hybrid_nll_weight * nll + self.toroidal_weight * toroidal_loss
