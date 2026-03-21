import torch
from torch.utils.checkpoint import checkpoint
from typing import Callable, Any, Tuple, List, Dict
from ...models.hooks import Plugin, HookManager

class CheckpointingPlugin(Plugin):
    """
    Plugin for Gradient Checkpointing.
    Intercepts the evolution loop and wraps it in chunked checkpointing calls
    to significantly reduce VRAM usage on long sequences.
    """
    def __init__(self, chunk_size: int = 32):
        super().__init__()
        self.chunk_size = chunk_size

    def register_hooks(self, manager: HookManager):
        manager.register("wrap_evolution", self.wrap_evolution)

    def wrap_evolution(self, evolution_fn: Callable) -> Callable:
        """Returns a chunked checkpointed wrapper for the evolution function."""
        
        def checkpointed_worker(x, v, all_forces, mask, **inner_kwargs):
            seq_len = all_forces.shape[1]
            x_total, v_total, logits_total = [], [], []

            # Ensure inputs have gradients for checkpointing
            if not x.requires_grad: x = x.detach().requires_grad_(True)
            if not v.requires_grad: v = v.detach().requires_grad_(True)

            for c_start in range(0, seq_len, self.chunk_size):
                c_end = min(c_start + self.chunk_size, seq_len)
                c_forces = all_forces[:, c_start:c_end]
                c_mask = mask[:, c_start:c_end]

                # Evolution function must follow the interface in BaseModel
                res = checkpoint(
                    evolution_fn,
                    x, v, c_forces, c_mask,
                    use_reentrant=False,
                    **inner_kwargs
                )
                
                # res is (logits_list, (x_final, v_final), (x_seq, v_seq))
                c_logits_list, (x_final, v_final), (c_x_seq, c_v_seq) = res
                
                # Update current state for next chunk
                x, v = x_final, v_final
                
                # Aggregate results
                logits_total.extend(c_logits_list)
                x_total.extend(c_x_seq)
                v_total.extend(c_v_seq)

            return logits_total, (x, v), (x_total, v_total)

        return checkpointed_worker
