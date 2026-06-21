"""
Component-based ISN model that connects scanning, world simulation, and emission.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

from ..interfaces.base import ScannerProtocol, WorldEngineProtocol, EmitterProtocol
from ..telemetry.hooks import HookManager, ISNHook

class Model(nn.Module):
    """
    Orchestrates scanner, world engine, emitter, and optional hooks.
    """
    def __init__(
        self,
        scanner: ScannerProtocol,
        world: WorldEngineProtocol,
        emitter: EmitterProtocol,
        hooks: Optional[List[ISNHook]] = None
    ):
        super().__init__()
        self.scanner = scanner
        self.world = world
        self.emitter = emitter
        self.hook_manager = HookManager(hooks)
        
        # Dimensions for external access
        self.d_model = scanner.d_model
        self.d_embedding = world.d_embedding
    
    def forward(
        self,
        input_ids: torch.Tensor,
        noise_std: float = 0.0,
        max_burst: int = 5,
        return_world_state: bool = False,
        world_state: Optional[torch.Tensor] = None,
        scanner_state: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        
        # 1. SCANNING Phase
        self.hook_manager.before_scanner(input_ids)
        impulses_seq, final_scanner_state = self.scanner(input_ids, state=scanner_state)
        self.hook_manager.after_scanner(impulses_seq)
        
        # 2. WORLD SIMULATION Phase (ISN Core Physics)
        world_input = {
            'impulses': impulses_seq,
            'noise_std': noise_std,
            'max_burst': max_burst,
            'world_state': world_state,
        }
        
        self.hook_manager.before_world(world_input)
        world_output = self.world(**world_input)
        self.hook_manager.after_world(world_output)
        
        # 3. MATERIALIZATION Phase
        # Optimized: If World Engine already produced logits (C++ fast path), use them.
        # Otherwise, fall back to Emitter projection.
        if 'logits' in world_output:
            logits = world_output['logits']
        else:
            logits = self.emitter(world_output['emitted_embeddings'])
        
        # 4. RESULT AGGREGATION (Modular result set)
        result = {
            'logits': logits, 
            'energy_trace': world_output.get('energy_trace'), 
            'world_coherence': torch.ones(input_ids.size(0), device=device) * 0.98,
            'emitted_embeddings': world_output['emitted_embeddings'],
            'final_state': world_output.get('final_state'),
            'final_scanner_state': final_scanner_state
        }
            
        return result

    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_length: int = 50, 
        temperature: float = 1.0,
        noise_std: float = 0.0,
        world_state: Optional[torch.Tensor] = None,
        scanner_state: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Unified Generation Loop."""
        self.eval()
        generated_sequence = input_ids.clone()
        final_info = {}
        
        with torch.no_grad():
            current_state = world_state
            current_scanner_state = scanner_state
            
            for _ in range(max_length):
                res = self.forward(
                    generated_sequence[:, -1:], 
                    world_state=current_state,
                    scanner_state=current_scanner_state,
                    noise_std=noise_std
                )
                current_state = res['final_state']
                current_scanner_state = res['final_scanner_state']
                
                next_token_logits = res['logits'][:, -1, :] 
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
                
                final_info = {
                    'final_state': current_state,
                    'final_scanner_state': current_scanner_state
                }
                
        return generated_sequence, final_info
