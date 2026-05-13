"""
ISN Topological World Component
Purified C++ Topological Engine Backend.
"""

import torch
import torch.nn as nn
import os
import sys
from typing import Dict, Any, Optional

from ...interfaces import BaseWorldEngine

try:
    # Attempt to load the native C++ extension
    import gfn_topology
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

class BilinearGeodesicOp(nn.Module):
    """
    Purified Bilinear Geodesic Operator.
    res = W1(e1) + W2(e2) + W3(e1 * e2)
    """
    def __init__(self, d_embedding: int):
        super().__init__()
        self.w1 = nn.Linear(d_embedding, d_embedding, bias=True)
        self.w2 = nn.Linear(d_embedding, d_embedding, bias=True)
        self.w3 = nn.Linear(d_embedding, d_embedding, bias=True)

class TopologicalWorld(BaseWorldEngine):
    """
    ISN World Engine based on the C++ Topological Simulation.
    """
    def __init__(self, d_model: int, d_embedding: int, d_properties: int):
        super().__init__()
        self.d_model = d_model
        self.d_embedding = d_embedding
        
        # Ontology & Alignment
        self.embedding_projector = nn.Linear(d_model, d_embedding)
        self.type_classifier = nn.Linear(d_model, 5) # 5 types: Entity, Property, Op...
        
        # Physics (Bilinear Operators)
        self.operation_network = BilinearGeodesicOp(d_embedding)

    def forward(
        self, 
        impulses: torch.Tensor,
        noise_std: float = 0.0,
        max_burst: int = 5,
        state: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        device = impulses.device
        
        # 1. Project Topology from Impulses
        type_logits = self.type_classifier(impulses)
        type_idx = type_logits.argmax(dim=-1)
        embeddings_seq = self.embedding_projector(impulses)
        
        mask_op = (type_idx == 3)
        mask_entity = ~mask_op
        
        # 2. Extract weights
        op = self.operation_network
        
        if HAS_CPP:
            # Native C++ stimulation (Note: C++ should also be updated if state is needed there)
            em_w_energy = kwargs.get('em_w_energy')
            em_b_energy = kwargs.get('em_b_energy')
            em_w_out = kwargs.get('em_w_out')
            em_b_out = kwargs.get('em_b_out')
            threshold = kwargs.get('threshold', 0.5)
            
            _, final_embs, energies = gfn_topology.forward(
                embeddings_seq,
                mask_op,
                mask_entity,
                op.w1.weight, op.w1.bias, 
                op.w2.weight, op.w2.bias, 
                op.w3.weight, op.w3.bias,
                em_w_energy, em_b_energy,
                em_w_out, em_b_out,
                threshold,
                noise_std,
                max_burst
            )
            final_state = state # Placeholder if no state returned from C++ yet
        else:
            # 3. Pure Python Fallback (O(1) Memory compatible iteration)
            b, l, d = embeddings_seq.shape
            final_embs = []
            energies = []
            
            if state is None:
                state = torch.zeros(b, d, device=device)
            
            for t in range(l):
                # If it's an operator, it transforms the state
                if mask_op[:, t].any():
                    state = op.w1(state) + op.w2(embeddings_seq[:, t, :])
                else:
                    # If it's an entity, it imprints on the state
                    state = 0.9 * state + 0.1 * embeddings_seq[:, t, :]
                
                final_embs.append(state.unsqueeze(1))
                energies.append(torch.norm(state, dim=-1, keepdim=True))
                
            final_embs = torch.cat(final_embs, dim=1)
            energies = torch.cat(energies, dim=1)
            final_state = state
            
        return {
            'emitted_embeddings': final_embs,
            'energy_trace': energies,
            'mask_op': mask_op,
            'final_state': final_state
        }
        
        return {
            'emitted_embeddings': final_embs,
            'energy_trace': energies,
            'mask_op': mask_op
        }
