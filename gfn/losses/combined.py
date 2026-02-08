"""
Combined Loss
=============

Main GFN loss combining multiple physics-informed components.

IMPORTANT NOTES FROM AUDITORIA LOGICA (2026-02-06):

1. PHYSICS-GROUNDED SUPERVISION:
   The loss now supports configured modes for physics components:
   - Hamiltonian: 'none', 'adaptive', 'structural', 'relative'
   - Geodesic: 'magnitude', 'structural', 'normalized'
   
2. DEFAULT SETTINGS:
   For stable training, use:
   - hamiltonian_mode='none' or 'adaptive'
   - geodesic_mode='structural'
   
3. SEPARATION OF CONCERNS:
   Cross-entropy handles prediction quality.
   Physics losses handle geometric consistency.
"""

import torch
import torch.nn as nn
from .hamiltonian import hamiltonian_loss
from .geodesic import geodesic_regularization
from .kinetic import kinetic_energy_penalty
from .noether import noether_loss
from .curiosity import curiosity_loss
from ..constants import LAMBDA_H_DEFAULT, LAMBDA_G_DEFAULT, LAMBDA_K_DEFAULT


class GFNLoss(nn.Module):
    """
    Combined loss for GFN training.
    
    Components:
        1. Cross-Entropy (prediction accuracy)
        2. Hamiltonian Loss (energy conservation - with configurable modes)
        3. Geodesic Regularization (curvature - with configurable modes)
        4. Noether Loss (semantic symmetries)
    
    AUDIT FIX: Added configurable modes for physics losses.
    
    Args:
        lambda_h: Hamiltonian loss weight (default: 0.01)
        lambda_g: Geodesic regularization weight (default: 0.001)
        lambda_k: Kinetic energy penalty weight (default: 0.0)
        lambda_c: Curiosity loss weight (default: 0.0)
        lambda_n: Noether symmetry weight (default: 0.0)
        ignore_index: Padding token index for CE loss
        hamiltonian_mode: Mode for energy conservation ('none', 'adaptive', 'structural', 'relative')
        geodesic_mode: Mode for curvature regularization ('magnitude', 'structural', 'normalized')
    """
    
    def __init__(self, lambda_h: float = LAMBDA_H_DEFAULT, 
                 lambda_g: float = LAMBDA_G_DEFAULT,
                 lambda_k: float = LAMBDA_K_DEFAULT,
                 lambda_c: float = 0.0,
                 lambda_n: float = 0.0,
                 ignore_index: int = -100,
                 hamiltonian_mode: str = 'adaptive',
                 geodesic_mode: str = 'structural'):
        super().__init__()
        self.lambda_h = lambda_h
        self.lambda_g = lambda_g
        self.lambda_k = lambda_k
        self.lambda_c = lambda_c
        self.lambda_n = lambda_n
        self.hamiltonian_mode = hamiltonian_mode
        self.geodesic_mode = geodesic_mode
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, logits, targets, velocities=None, christoffel_outputs=None, 
                isomeric_groups=None, states=None, forces=None):
        """
        Compute combined loss.
        
        AUDIT FIX: Now supports configured modes for physics losses.
        
        Args:
            logits: Model output [batch, seq_len, vocab_size]
            targets: Target tokens [batch, seq_len]
            velocities: Optional list of velocity tensors for Hamiltonian loss
            christoffel_outputs: Optional list of curvature tensors
            isomeric_groups: Optional list of isomeric head groups for Noether loss
            states: Optional list of position tensors for physics losses
            forces: Optional list of force tensors for adaptive Hamiltonian loss
            
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary with individual loss components
        """
        # Primary loss: Cross-Entropy
        batch_size, seq_len, vocab_size = logits.shape
        ce = self.ce_loss(logits.reshape(-1, vocab_size), targets.reshape(-1))
        
        loss_dict = {
            "ce": ce.item(),
            "hamiltonian_mode": self.hamiltonian_mode,
            "geodesic_mode": self.geodesic_mode
        }
        total = ce
        
        # AUDIT FIX: Hamiltonian regularization with configurable mode
        if velocities and len(velocities) > 1 and self.lambda_h > 0:
            h_loss = hamiltonian_loss(
                velocities, 
                states=states, 
                lambda_h=self.lambda_h, 
                forces=forces,
                mode=self.hamiltonian_mode
            )
            total = total + h_loss
            loss_dict["hamiltonian"] = h_loss.item()
        
        # AUDIT FIX: Geodesic regularization with configurable mode
        if christoffel_outputs and self.lambda_g > 0:
            g_loss = geodesic_regularization(
                christoffel_outputs,
                velocities=velocities,
                lambda_g=self.lambda_g,
                mode=self.geodesic_mode
            )
            total = total + g_loss
            loss_dict["geodesic"] = g_loss.item()

        # Curiosity (Entropy Production)
        if self.lambda_c > 0 and velocities:
            c_loss = curiosity_loss(velocities, self.lambda_c)
            total = total + c_loss
            loss_dict["curiosity"] = c_loss.item()

        # Noether (Semantic Symmetries)
        if self.lambda_n > 0 and christoffel_outputs:
            n_loss = noether_loss(christoffel_outputs, isomeric_groups=isomeric_groups, lambda_n=self.lambda_n)
            total = total + n_loss
            loss_dict["noether"] = n_loss.item()
            
        loss_dict["total"] = total.item()
        
        return total, loss_dict
