import torch
import time
import json
import os
from typing import Dict, Any, List
from gfn.realizations.gssm.utils.diagnostics import check_model_health

class MatrixMetrics:
    """Helper to track VRAM, Parameters, and Performance for GFN Matrix."""
    
    @staticmethod
    def get_vram_usage() -> float:
        """Returns current VRAM usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0

    @staticmethod
    def get_peak_vram() -> float:
        """Returns peak VRAM usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0

    @staticmethod
    def reset_peak_vram():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @classmethod
    def capture(cls, model: torch.nn.Module, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        params = cls.count_parameters(model)
        vram = cls.get_vram_usage()
        peak = cls.get_peak_vram()
        
        health = check_model_health(model)
        
        metrics = {
            "parameters": params,
            "vram_mb": round(vram, 2),
            "peak_vram_mb": round(peak, 2),
            "health": health,
            "timestamp": time.time()
        }
        if metadata:
            metrics.update(metadata)
        return metrics

def save_matrix_results(results: List[Dict[str, Any]], filename: str = "results_matrix.json"):
    path = os.path.join("tests/benchmarks/matrix", filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[MATRIX] Results saved to {path}")
