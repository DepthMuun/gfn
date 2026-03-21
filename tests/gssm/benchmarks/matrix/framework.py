import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import itertools
import time
from .metrics import MatrixMetrics, save_matrix_results
import gfn

class MatrixRunner:
    """
    Orchestrates the execution of multiple GFN configurations 
    across different axes (dynamics, geometry, etc).
    """
    def __init__(self, task_name: str, device: str = "cuda"):
        self.task_name = task_name
        self.device = device
        self.results = []

    def run_axes(self, 
                 axes: Dict[str, List[Any]],
                 train_fn,
                 eval_fn,
                 base_overrides: Dict[str, Any] = None):
        """
        axes example: {
            "dynamics_type": ["direct", "residual", "mix"],
            "physics.topology.type": ["torus", "euclidean"]
        }
        """
        keys = list(axes.keys())
        values = list(axes.values())
        combinations = list(itertools.product(*values))

        print(f"=== Starting GFN Matrix: {self.task_name} ===")
        print(f"Total Combinations: {len(combinations)}")

        for combination in combinations:
            combo_overrides = dict(zip(keys, combination))
            if base_overrides:
                # Merge base overrides
                full_overrides = {**base_overrides, **combo_overrides}
            else:
                full_overrides = combo_overrides
                
            print(f"\n[RUN] {combo_overrides}")
            
            # 1. Reset Metrics
            MatrixMetrics.reset_peak_vram()
            start_time = time.time()

            # 2. Create Model using public API for realism
            try:
                # We use 'stable-torus' as base if not specified
                model = gfn.create(preset_name=full_overrides.get('preset', 'stable-torus'), 
                                 **full_overrides).to(self.device)
                
                # 3. Train
                import inspect
                sig_train = inspect.signature(train_fn)
                if 'overrides' in sig_train.parameters:
                    history = train_fn(model, overrides=combo_overrides)
                else:
                    history = train_fn(model)
                
                # 4. Eval
                sig_eval = inspect.signature(eval_fn)
                if 'adapter' in sig_eval.parameters:
                    acc = eval_fn(model, adapter=history.get("adapter"))
                else:
                    acc = eval_fn(model)
                duration = time.time() - start_time

                # 5. Capture
                m = MatrixMetrics.capture(model, metadata={
                    "task": self.task_name,
                    "config": combo_overrides,
                    "accuracy": acc,
                    "duration_sec": round(duration, 2),
                    "steps": history.get("steps", 0),
                    "loss_final": history.get("loss", 0.0)
                })
                self.results.append(m)
            except Exception as e:
                import traceback
                print(f"[ERROR] Combo {combo_overrides} failed: {e}")
                traceback.print_exc()
                if 'logits' in locals(): print(f"DEBUG: Logits shape: {logits.shape}")
                if 'y' in locals(): print(f"DEBUG: Targets shape: {y.shape}")
                self.results.append({
                    "task": self.task_name,
                    "config": combo_overrides,
                    "error": str(e)
                })
            
            # Free memory
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()

        save_matrix_results(self.results, f"matrix_{self.task_name}.json")

    def _prepare_config(self, overrides: Dict[str, Any]):
        # This would use ConfigurationLoader to merge defaults with overrides
        # Placeholder for now, returning a basic GFN config structure
        from gfn.realizations.gssm.config.schema import ManifoldConfig, PhysicsConfig
        cfg = ManifoldConfig(
            dim=64, depth=4, heads=4,
            vocab_size=10, # Small for micro-tests
            **overrides
        )
        return cfg
