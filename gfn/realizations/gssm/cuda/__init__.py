"""
gfn/cuda/__init__.py
Infraestructura CUDA para GFN V5.
"""
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

def is_cuda_active(tensor: torch.Tensor) -> bool:
    """Verifica si CUDA está disponible y el tensor está en un dispositivo GPU."""
    return CUDA_AVAILABLE and tensor.is_cuda
