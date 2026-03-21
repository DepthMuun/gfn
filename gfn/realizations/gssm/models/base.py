import torch
import torch.nn as nn
import json
import os
from typing import Optional, List, Any, Dict, Tuple, Union
from .hooks import HookManager

class BaseModel(nn.Module):
    """
    Core Evolution Engine for GFN V5.
    Sequentially evolves (x, v) states through manifold layers.
    """
    def __init__(self, layers: nn.ModuleList, embedding: nn.Module, 
                 x0: nn.Parameter, v0: nn.Parameter, holographic: bool = False,
                 config: Optional[Any] = None):
        super().__init__()
        self.layers = layers
        self.embedding = embedding
        self.x0 = x0
        self.v0 = v0
        self.holographic = holographic
        self.config = config
        self.hooks = HookManager()
        
        # Set seed for reproducible initialization if requested in config
        seed = 42
        if config and hasattr(config, 'seed'):
            seed = config.seed
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.n_trajectories = 1
        self.initial_spread = getattr(config, 'initial_spread', 1e-3) if config else 1e-3

    def forward(self, input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                force_manual: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        
        # 1. Resolve Forces
        if force_manual is not None:
            all_forces = force_manual
        elif input_ids is not None:
            all_forces = self.embedding(input_ids)
        else:
            raise ValueError("BaseModel requires input_ids or force_manual.")
            
        batch_size, seq_len = all_forces.shape[0], all_forces.shape[1]
        mask = attention_mask.unsqueeze(-1).float() if attention_mask is not None \
               else torch.ones(batch_size, seq_len, 1, device=all_forces.device)

        # Hook to modify/broadcast forces (e.g. for Ensembles)
        forces_res = self.hooks.trigger("on_resolve_forces", all_forces=all_forces, mask=mask)
        if forces_res:
             all_forces, mask = forces_res[-1]

        # 2. Lifecycle Hooks
        self.hooks.trigger("on_batch_start", batch_size=batch_size, device=all_forces.device)
        
        # P0.3 Architecture Fix: Reset all stateful physical components across layers
        # This prevents hysteresis/momentum bleed between unrelated batches.
        for layer in self.layers:
            if hasattr(layer, 'integrator') and hasattr(layer.integrator, 'physics_engine'):
                engine = layer.integrator.physics_engine
                if hasattr(engine, 'reset_hysteresis'):
                    engine.reset_hysteresis()
                # Clean up any other registered curiosity or stochastic modules
                if hasattr(engine, 'curiosity_module') and hasattr(engine.curiosity_module, 'reset'):
                    engine.curiosity_module.reset()
        
        # 3. State Initialization (FIXED for Autoregressive State Persistence)
        if state is not None:
             x, v = state
        else:
             init_res = self.hooks.trigger("state_init", batch_size=batch_size)
             if init_res:
                  x, v = init_res[-1]
             else:
                  # Default fallback
                  x = self.x0.expand(batch_size, self.x0.shape[1], self.x0.shape[2])
                  v = self.v0.expand(batch_size, self.v0.shape[1], self.v0.shape[2])
                  if self.initial_spread > 0:
                      x = x + torch.randn_like(x) * self.initial_spread

        assert x.shape == v.shape, f"Initial state shapes mismatch: x {x.shape}, v {v.shape}"

        # 4. Execute Evolution
        logits_total, (x_final, v_final), (x_seq_total, v_seq_total) = self._evolve_sequence(
            x, v, all_forces, mask, **kwargs
        )

        # 5. Result Assembly
        res_x_seq = torch.stack(x_seq_total, dim=1) if isinstance(x_seq_total, list) else x_seq_total
        res_v_seq = torch.stack(v_seq_total, dim=1) if isinstance(v_seq_total, list) else v_seq_total
        res_logits = torch.stack(logits_total, dim=1) if isinstance(logits_total, list) and logits_total \
                     else (logits_total if not isinstance(logits_total, list) else x_final)

        # 6. Lifecycle End & Plugin Results
        plugin_res = self.hooks.trigger("on_batch_end")
        
        # 7. State Info for physics losses
        state_info = {
            "x_seq": res_x_seq,
            "v_seq": res_v_seq,
            "forces": all_forces,
            "x_final": x_final,
            "v_final": v_final,
            "mask": mask,
            "plugin_results": plugin_res
        }

        return res_logits, (x_final, v_final), state_info

    def _evolve_sequence(self, x_in, v_in, f_seq, m_seq, **kwargs):
        """
        Internal evolution loop. Can be overridden for different integration schemes.
        """
        def run_evolution(x_curr, v_curr, fs, ms, **inner_kwargs):
            local_x, local_v = x_curr, v_curr
            l_logits, l_x_seq, l_v_seq = [], [], []
            l_seq_len = fs.shape[1]
            
            for i in range(l_seq_len):
                force = fs[:, i] * ms[:, i]
                
                # Timestep Start Hooks
                step_start_res = self.hooks.trigger("on_timestep_start", x=local_x, v=local_v, force=force)
                for res in step_start_res:
                    if isinstance(res, torch.Tensor):
                        force = force + res
                    elif isinstance(res, dict) and "force" in res:
                        force = res["force"]

                # Layer Pass
                for layer in self.layers:
                    layer_kwargs = {}
                    self.hooks.trigger("on_layer_start", layer=layer, layer_kwargs=layer_kwargs, x=local_x, v=local_v)
                    res = layer(local_x, local_v, force, **layer_kwargs)
                    
                    if isinstance(res, tuple) and len(res) >= 2:
                        local_x, local_v = res[0], res[1]
                        extra_info = res[2] if len(res) > 2 else {}
                    else:
                        local_x, local_v = res, local_v
                        extra_info = {}

                    self.hooks.trigger("on_layer_end", layer=layer, x=local_x, v=local_v, extra_info=extra_info)
                
                # Timestep End Hooks (Readouts)
                step_res = self.hooks.trigger("on_timestep_end", x=local_x, v=local_v)
                for r in step_res:
                    if isinstance(r, torch.Tensor):
                        l_logits.append(r)
                
                l_x_seq.append(local_x)
                l_v_seq.append(local_v)
                
            return l_logits, (local_x, local_v), (l_x_seq, l_v_seq)

        # Hook to wrap evolution (e.g. for checkpointing)
        evolve_fn = run_evolution
        wrapped_evolution = self.hooks.trigger("wrap_evolution", evolution_fn=evolve_fn)
        if wrapped_evolution:
            evolve_fn = wrapped_evolution[-1]

        return evolve_fn(x_in, v_in, f_seq, m_seq, layers=self.layers, hooks=self.hooks, **kwargs)

    def save_pretrained(self, save_directory: str):
        """
        Saves the model and its configuration in a directory.
        Creates config.json and pytorch_model.bin.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 1. Save Config
        if self.config is not None:
            config_path = os.path.join(save_directory, "config.json")
            # PhysicsConfig/ManifoldConfig should have to_dict()
            config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=4)

        # 2. Save Weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        print(f"Model saved to {save_directory}")
