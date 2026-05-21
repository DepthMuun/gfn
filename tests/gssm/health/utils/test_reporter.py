"""
Test Results Reporter — GFN GSSM
================================

Handles saving test results to JSON format for analysis.
Generates reports in tests/results/ directory.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class ResultsReporter:
    """Reporter for saving test results to JSON."""
    
    def __init__(self, test_name: str, category: str = "health"):
        self.test_name = test_name
        self.category = category
        self.start_time = time.time()
        self.metrics: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.passed = False
        self.errors: List[str] = []
        
        # Results directory - goes up 4 levels from utils/ to reach tests/
        # utils/ -> health/ -> gssm/ -> tests/
        self.results_dir = Path(__file__).resolve().parent.parent.parent.parent / "results" / category
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create category results dir within each test area
        self.category_results_dir = Path(__file__).resolve().parent.parent / "results"
        self.category_results_dir.mkdir(parents=True, exist_ok=True)
    
    def log_metric(self, name: str, value: Any, step: Optional[int] = None):
        """Log a metric value."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        
        entry = {
            "name": name,
            "value": value,
            "timestamp": time.time(),
            "step": step
        }
        self.history.append(entry)
        
        # Keep latest value in metrics
        if step is None:
            self.metrics[name] = value
    
    def log_plot_data(self, name: str, x_data: List[float], y_data: List[float]):
        """Log data for plotting."""
        self.metrics[f"{name}_plot"] = {
            "x": x_data,
            "y": y_data,
            "type": "line"
        }
    
    def mark_passed(self, passed: bool = True):
        """Mark test as passed or failed."""
        self.passed = passed
    
    def add_error(self, error_msg: str):
        """Add error message."""
        self.errors.append(error_msg)
        self.passed = False
    
    def save(self) -> Path:
        """Save results to JSON file in both global and category directories."""
        elapsed = time.time() - self.start_time
        
        report = {
            "test_name": self.test_name,
            "category": self.category,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "passed": self.passed,
            "metrics": self.metrics,
            "history": self.history,
            "errors": self.errors
        }
        
        # Save to global results directory
        filename = f"{self.test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save as latest in global
        latest_path = self.results_dir / f"{self.test_name}_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save to category-specific directory for easy access
        category_filepath = self.category_results_dir / f"{self.test_name}_latest.json"
        with open(category_filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return filepath
    
    def print_summary(self):
        """Print test summary to console."""
        status = "PASS" if self.passed else "FAIL"
        print(f"\n{'='*60}")
        print(f"Test: {self.test_name}")
        print(f"Category: {self.category}")
        print(f"Status: {status}")
        print(f"Time: {time.time() - self.start_time:.2f}s")
        print(f"{'='*60}")
        
        for name, value in self.metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {name}: {value:.6f}")
            else:
                print(f"  {name}: {value}")
        
        if self.errors:
            print(f"\n  Errors:")
            for err in self.errors:
                print(f"    - {err}")
        
        print(f"{'='*60}\n")


class ConvergenceTracker:
    """Track convergence metrics over training."""
    
    def __init__(self, target_metric: str = "loss", target_value: float = 0.01):
        self.target_metric = target_metric
        self.target_value = target_value
        self.history: List[Dict[str, Any]] = []
        self.converged = False
        self.convergence_step = -1
    
    def step(self, step_num: int, metrics: Dict[str, Any]):
        """Record a training step."""
        entry = {"step": step_num, **metrics}
        self.history.append(entry)
        
        # Check convergence
        if not self.converged and self.target_metric in metrics:
            value = metrics[self.target_metric]
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            if value <= self.target_value:
                self.converged = True
                self.convergence_step = step_num
    
    def get_convergence_data(self) -> Dict[str, Any]:
        """Get convergence statistics."""
        if not self.history:
            return {}
        
        losses = [h.get(self.target_metric, float('inf')) for h in self.history]
        losses = [l.item() if isinstance(l, torch.Tensor) else l for l in losses]
        
        return {
            "converged": self.converged,
            "convergence_step": self.convergence_step,
            "final_value": losses[-1] if losses else None,
            "best_value": min(losses) if losses else None,
            "total_steps": len(self.history),
            "loss_curve": losses
        }
