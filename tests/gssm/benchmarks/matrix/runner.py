import torch
import torch.nn as nn
import torch.optim as optim
import time
import traceback
import json
import os
from typing import Dict, Any, Tuple
from pathlib import Path
from gfn import Manifold
from gfn.realizations.gssm.config import ManifoldConfig

class MatrixRunner:
    """
    Executes a single trial for a given configuration.
    """
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def run_trial(self, config: ManifoldConfig, trial_id: int) -> Dict[str, Any]:
        """
        Run a standardized convergence task.
        Task: Vector Field Alignment (Synthetic).
        Goal: Map random inputs X to target Y = Function(X) 
        (e.g. Identity or Permutation).
        """
        trial_dir = self.results_dir / f"trial_{trial_id:04d}"
        if not trial_dir.exists():
            trial_dir.mkdir(parents=True, exist_ok=True)
            
        metrics = {
            "trial_id": trial_id,
            "status": "PENDING",
            "final_loss": float('inf'),
            "steps": 0,
            "time": 0.0,
            "stability_score": 0.0,
            "config_summary": str(config)
        }
        
        try:
            start_time = time.time()
            
            # 1. Instantiate Model
            model = Manifold(config).to(self.device)
            
            # 2. Optimizer
            optimizer = optim.AdamW(model.parameters(), lr=0.005)
            
            # 3. Simple Task: Sequence Identity Copy
            batch_size = 16
            seq_len = 10
            vocab_size = config.vocab_size
            
            max_steps = 200 # Fast pass
            target_loss = 0.01
            
            loss_history = []
            
            model.train()
            
            for step in range(max_steps):
                # Generate simple batch
                x = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
                
                # Forward
                logits, struct, _, _, _, _ = model(x)
                
                # Target: Predict self (Identity map) or shifted
                loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), x.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isnan(grad_norm) or torch.isnan(loss):
                    raise ValueError("NaN Gradients/Loss Detected")
                    
                optimizer.step()
                
                loss_val = loss.item()
                loss_history.append(loss_val)
                
                if loss_val < target_loss:
                    break
            
            duration = time.time() - start_time
            
            # Calculate Stability Score (Variance of last 10 steps)
            variance = 0.0
            if len(loss_history) > 10:
                last_window = torch.tensor(loss_history[-10:])
                variance = torch.var(last_window).item()
            
            metrics.update({
                "status": "SUCCESS",
                "final_loss": loss_history[-1],
                "steps": len(loss_history),
                "time": duration,
                "stability_score": variance
            })
            
            # Save Loss Curve
            with open(trial_dir / "loss.json", "w", encoding='utf-8') as f:
                json.dump(loss_history, f, indent=2)
                
        except Exception as e:
            metrics["status"] = "CRASH"
            metrics["error"] = str(e)
            metrics["traceback"] = traceback.format_exc()
            
            with open(trial_dir / "crash.log", "w", encoding='utf-8') as f:
                f.write(traceback.format_exc())
        
        # Save Metrics
        with open(trial_dir / "metrics.json", "w", encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
            
        return metrics
