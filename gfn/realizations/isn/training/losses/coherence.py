"""
Loss Functions for Reality Model.

Implements the 6-dimensional loss:
L_total = Σ λ_i · L_i

1. Outcome Loss (L₁)
2. World Coherence Loss (L₂)
3. Entity Grounding Loss (L₃)
4. Operation Validity Loss (L₄)
5. Emergence Consistency Loss (L₅)
6. Process Efficiency Loss (L₆)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .semantic_distance import SemanticDistanceLoss
from .energy_threshold import ThresholdModulationLoss


class MultiDimensionalLoss(nn.Module):
    """
    Complete multi-dimensional loss for Reality Model.
    
    L_total = Σ_{i=1}^6 λ_i · L_i
    """
    
    def __init__(
        self,
        lambda_outcome: float = 1.0,
        lambda_coherence: float = 0.8,
        lambda_grounding: float = 0.6,
        lambda_validity: float = 0.5,
        lambda_emergence: float = 0.4,
        lambda_efficiency: float = 0.3,
        vocab_size: int = 10000
    ):
        super().__init__()
        
        # Loss weights
        self.lambda_outcome = lambda_outcome
        self.lambda_coherence = lambda_coherence
        self.lambda_grounding = lambda_grounding
        self.lambda_validity = lambda_validity
        self.lambda_emergence = lambda_emergence
        self.lambda_efficiency = lambda_efficiency
        
        self.vocab_size = vocab_size
        
        # Individual loss components (Physics & Topologic)
        self.semantic_loss = SemanticDistanceLoss(use_l2=False)
        self.energy_loss = ThresholdModulationLoss(margin=0.5)
        
        self.coherence_loss = CoherenceLoss()
        self.grounding_loss = GroundingLoss()
        self.validity_loss = ValidityLoss()
        self.emergence_loss = EmergenceLoss()
        self.efficiency_loss = EfficiencyLoss()
        
        # Standard Language Loss helper
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        world_coherence: torch.Tensor,
        world_states: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and components.
        
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            targets: Target token IDs (batch_size, seq_len)
            world_coherence: Coherence scores (batch_size,)
            world_states: Optional world state dicts
            
        Returns:
            Dictionary of losses
        """
        # Extract Physical Arguments
        emitted_embeddings = kwargs.get('emitted_embeddings')
        vocab_basis = kwargs.get('vocab_basis')
        energy_trace = kwargs.get('energy_trace')
        threshold = kwargs.get('threshold')
        emission_mask = kwargs.get('emission_mask')
        
        l1 = torch.tensor(0.0, device=logits.device)
        
        # L1-a: Standard CrossEntropy (from logits)
        if logits is not None and targets is not None:
             # Flatten for CE
             l_ce = self.ce_loss(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
             l1 = l1 + l_ce
             
        # L1-b: Geodetic Outcome (Semantic Distance + Energy Modulation)
        if emitted_embeddings is not None and vocab_basis is not None:
            l_semantic = self.semantic_loss(emitted_embeddings, targets, vocab_basis)
            l1 = l1 + l_semantic
            
        if energy_trace is not None and emission_mask is not None and threshold is not None:
            l_energy = self.energy_loss(energy_trace, emission_mask, threshold)
            l1 = l1 + l_energy
        
        # L₂: World Coherence Loss
        l2 = self.coherence_loss(world_coherence)
        
        # L₃: Entity Grounding Loss
        if world_states is not None:
            l3 = self.grounding_loss(world_states)
        else:
            l3 = torch.tensor(0.0, device=logits.device)
        
        # L₄: Operation Validity Loss
        if world_states is not None:
            l4 = self.validity_loss(world_states)
        else:
            l4 = torch.tensor(0.0, device=logits.device)
        
        # L₅: Emergence Consistency Loss
        if world_states is not None:
            l5 = self.emergence_loss(logits, world_states)
        else:
            l5 = torch.tensor(0.0, device=logits.device)
        
        # L₆: Process Efficiency Loss
        if world_states is not None:
            l6 = self.efficiency_loss(world_states)
        else:
            l6 = torch.tensor(0.0, device=logits.device)
        
        # Total loss
        l_total = (
            self.lambda_outcome * l1 +
            self.lambda_coherence * l2 +
            self.lambda_grounding * l3 +
            self.lambda_validity * l4 +
            self.lambda_emergence * l5 +
            self.lambda_efficiency * l6
        )
        
        return {
            'loss': l_total,
            'outcome_loss': l1,
            'coherence_loss': l2,
            'grounding_loss': l3,
            'validity_loss': l4,
            'emergence_loss': l5,
            'efficiency_loss': l6
        }
    
    def update_weights(self, epoch: int, initial_weights: Dict[str, float]):
        """
        Update loss weights dynamically during training.
        
        λ_i^(epoch) = λ_i^(0) · exp(-β_i · L_i^(epoch) / L_i^(0))
        """
        # Simplified: keep weights constant for now
        # In full implementation, track loss history and adjust
        pass


# (Removed legacy OutcomeLoss due to pure geometrical shift)


class CoherenceLoss(nn.Module):
    """
    L₂: World Coherence Loss
    
    L_coherence = 1 - C_world(W^(T))
    
    Penalizes incoherent world states.
    """
    
    def forward(self, world_coherence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            world_coherence: Coherence scores (batch_size,)
        """
        # L = 1 - C
        loss = torch.tensor(1.0, device=world_coherence.device) - world_coherence.mean()
        
        return loss


class GroundingLoss(nn.Module):
    """
    L₃: Entity Grounding Loss
    
    L_grounding = (1/|E|) Σ D_ground(E_i, s_i)
    
    Ensures entities correspond to their source symbols.
    """
    
    def forward(self, world_states: List[Dict]) -> torch.Tensor:
        """
        Args:
            world_states: List of world state dicts
        """
        # Simplified: check entity coherence
        total_loss = 0.0
        count = 0
        
        for state in world_states:
            entities = state.get('entities', {})
            for entity_data in entities.values():
                # Entity coherence is already computed
                # Grounding loss is inverse of coherence
                # (This is simplified - full version would check symbol correspondence)
                count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        # Placeholder: return small constant
        return torch.tensor(0.1, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


class ValidityLoss(nn.Module):
    """
    L₄: Operation Validity Loss
    
    L_validity = (1/|O|) Σ (1 - V(O_j, Args(O_j)))
    
    Penalizes invalid operations.
    """
    
    def forward(self, world_states: List[Dict]) -> torch.Tensor:
        """
        Args:
            world_states: List of world state dicts
        """
        # Simplified: assume operations are valid if world coherence is high
        # Full implementation would track operation validity
        return torch.tensor(0.0, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


class EmergenceLoss(nn.Module):
    """
    L₅: Emergence Consistency Loss
    
    L_emergence = -log P(y | E_result) + β · 1[E_result ∉ E^(T)]
    
    Ensures output emerges from world state.
    """
    
    def forward(
        self,
        logits: torch.Tensor,
        world_states: List[Dict]
    ) -> torch.Tensor:
        """
        Args:
            logits: Output logits
            world_states: World states
        """
        # Simplified: check if result entity exists
        # Full implementation would verify output is derived from result entity
        return torch.tensor(0.0, device=logits.device)


class EfficiencyLoss(nn.Module):
    """
    L₆: Process Efficiency Loss
    
    L_efficiency = γ₁·|E^(T)| + γ₂·T + γ₃·|E|
    
    Penalizes overly complex processes.
    """
    
    def __init__(
        self,
        gamma_entities: float = 0.01,
        gamma_timesteps: float = 0.01,
        gamma_edges: float = 0.005
    ):
        super().__init__()
        self.gamma_entities = gamma_entities
        self.gamma_timesteps = gamma_timesteps
        self.gamma_edges = gamma_edges
    
    def forward(self, world_states: List[Dict]) -> torch.Tensor:
        """
        Args:
            world_states: List of world state dicts
        """
        total_loss = 0.0
        
        for state in world_states:
            n_entities = len(state.get('entities', {}))
            timestep = state.get('timestep', 0)
            n_edges = len(state.get('graph', {}))
            
            loss = (
                self.gamma_entities * n_entities +
                self.gamma_timesteps * timestep +
                self.gamma_edges * n_edges
            )
            
            total_loss += loss
        
        if len(world_states) == 0:
            return torch.tensor(0.0, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        return torch.tensor(total_loss / len(world_states), device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
