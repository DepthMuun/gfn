"""
ISN Topological Physics Engine — Modular V5
======================================
O(1) Memory Causal Topological Engine.
Uses high-performance C++ Bilinear Geodesic Operators.
Includes Reconciliation Layer for 1:1 Training Compatibility.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List
from ..interfaces.base import WorldEngineProtocol
from ..registry import physics

@physics.register("topological")
@physics.register("bilinear")
class TopologicalPhysics(nn.Module):
    """
    World Engine that evolves state based on topological connectivity.
    Implements a Stack-based Bilinear Geodesic Operator.
    """
    def __init__(self, d_model: int, d_embedding: int, vocab_size: int = 65, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_embedding = d_embedding
        self.vocab_size = vocab_size
        
        # 1. Bilinear Geodesic Operator Weights
        self.op_w1 = nn.Parameter(torch.randn(d_embedding, d_embedding) * 0.02)
        self.op_b1 = nn.Parameter(torch.zeros(d_embedding))
        self.op_w2 = nn.Parameter(torch.randn(d_embedding, d_embedding) * 0.02)
        self.op_b2 = nn.Parameter(torch.zeros(d_embedding))
        self.op_w3 = nn.Parameter(torch.randn(d_embedding, d_embedding) * 0.02)
        self.op_b3 = nn.Parameter(torch.zeros(d_embedding))
        
        # 2. Purified Emitter Weights (Integrated into Physics for C++ speed)
        self.em_w_energy = nn.Parameter(torch.randn(1, d_embedding) * 0.02)
        self.em_b_energy = nn.Parameter(torch.zeros(1))
        self.em_w_out = nn.Parameter(torch.randn(vocab_size, d_embedding) * 0.02)
        self.em_b_out = nn.Parameter(torch.zeros(vocab_size))

        self.threshold_base = kwargs.get('threshold_base', 0.5)
        self.max_burst = kwargs.get('max_burst', 5)

    def forward(
        self,
        impulses: torch.Tensor,
        noise_std: float = 0.0,
        world_state: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        b, l, d = impulses.shape
        device = impulses.device
        
        mask_op = kwargs.get('mask_op', torch.ones(b, l, device=device, dtype=torch.bool))
        mask_entity = kwargs.get('mask_entity', torch.ones(b, l, device=device, dtype=torch.bool))

        # ----------------------------------------------------------------------
        # FAST PATH: C++ Extension
        # ----------------------------------------------------------------------
        try:
            csrc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../csrc/topology'))
            if csrc_path not in sys.path:
                sys.path.append(csrc_path)
            
            import gfn_topology
            
            logits, embs, energies = gfn_topology.forward(
                impulses, 
                mask_op, 
                mask_entity,
                self.op_w1, self.op_b1,
                self.op_w2, self.op_b2,
                self.op_w3, self.op_b3,
                self.em_w_energy, self.em_b_energy,
                self.em_w_out, self.em_b_out,
                float(self.threshold_base),
                float(noise_std),
                int(self.max_burst)
            )
            
            # --- Reconciliation Layer ---
            # Standard training (CrossEntropy) expects 1:1 length (L).
            if logits.size(1) < l:
                pad_len = l - logits.size(1)
                logits = torch.cat([logits, torch.zeros(b, pad_len, self.vocab_size, device=device)], dim=1)
                embs = torch.cat([embs, torch.zeros(b, pad_len, self.d_embedding, device=device)], dim=1)
            elif logits.size(1) > l:
                logits = logits[:, :l, :]
                embs = embs[:, :l, :]

            return {
                'logits': logits,
                'emitted_embeddings': embs,
                'energy_trace': energies
            }
            
        except ImportError:
            # Simple fallback for structural verification
            return {
                'emitted_embeddings': torch.zeros(b, l, self.d_embedding, device=device),
                'energy_trace': torch.zeros(b, l, 1, device=device)
            }
