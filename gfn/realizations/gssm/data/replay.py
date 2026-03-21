"""
Replay / Trajectory Buffer — GFN V5
Maneja el almacenamiento y muestreo de estados físicos (x, v, forces) 
para soporte de entrenamiento Off-Policy y exploración de GFlowNets reales.
"""

import torch
from typing import Optional, Tuple

class TrajectoryReplayBuffer:
    """
    A persistent buffer for storing and managing manifold trajectories (x, v states).
    Serves as replay memory for Hamiltonian/Geodesic flows in V5.
    """
    def __init__(
        self, 
        capacity: int, 
        dim: int, 
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):
        self.capacity = capacity
        self.dim = dim
        self.device = device
        self.dtype = dtype
        
        # Buffers for state (x), velocity (v), and optional force
        # Shape: [capacity, dim] or [capacity, heads, head_dim] depending on input
        # Note: we flatten the capacity dimension but keep the geometry shape.
        self._initialized_shape = False
        
        self.pointer = 0
        self.size = 0
        self.is_full = False

    def _init_buffers(self, example_shape: torch.Size):
        """Initializes the tensor buffers based on the first observed shape."""
        # example_shape might be [Batch, Dim] or [Batch, Heads, HeadDim] 
        # We need [Capacity, *shape[1:]]
        element_shape = example_shape[1:]
        
        # Memory check safeguard
        import math
        bytes_per_el = 4 if self.dtype == torch.float32 else 8
        total_elements = self.capacity * math.prod(element_shape)
        # 3 buffers (x, v, force)
        total_mb = (3 * total_elements * bytes_per_el) / (1024 ** 2)
        if total_mb > 1024 and self.device.type == 'cuda':
             import logging
             logging.warning(f"ReplayBuffer: Allocating {total_mb:.1f} MB on CUDA. Risk of OOM.")

        self.x_buffer = torch.zeros((self.capacity, *element_shape), device=self.device, dtype=self.dtype)
        self.v_buffer = torch.zeros((self.capacity, *element_shape), device=self.device, dtype=self.dtype)
        self.force_buffer = torch.zeros((self.capacity, *element_shape), device=self.device, dtype=self.dtype)
        self._initialized_shape = True

    def add(
        self, 
        x: torch.Tensor, 
        v: torch.Tensor, 
        force: Optional[torch.Tensor] = None
    ):
        """
        Adds a batch of transitions to the buffer.
        """
        batch_size = x.size(0)
        
        if not self._initialized_shape:
            self._init_buffers(x.shape)
            
        # Handle wrap-around indexing
        indices = torch.arange(self.pointer, self.pointer + batch_size) % self.capacity
        
        self.x_buffer[indices] = x.to(self.device).detach()
        self.v_buffer[indices] = v.to(self.device).detach()
        if force is not None:
            self.force_buffer[indices] = force.to(self.device).detach()
            
        self.pointer = (self.pointer + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
        if self.size == self.capacity:
            self.is_full = True

    def sample_random(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly samples a batch of states from the buffer.
        Returns: (x, v, force)
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
            
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.x_buffer[indices],
            self.v_buffer[indices],
            self.force_buffer[indices]
        )

    def sample_recent(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples the most recently added transitions."""
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
            
        if self.size < batch_size:
            idx = torch.arange(0, self.size, device=self.device)
        else:
            idx = (torch.arange(self.pointer - batch_size, self.pointer, device=self.device) % self.capacity)
            
        return (
            self.x_buffer[idx], 
            self.v_buffer[idx],
            self.force_buffer[idx]
        )

    def sample_with_noise(self, batch_size: int, noise_std: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples with Gaussian jitter to improve robust training."""
        x, v, _ = self.sample_random(batch_size)
        x_noisy = x + torch.randn_like(x) * noise_std
        return x_noisy, v

    def clear(self):
        """Resets the buffer."""
        self.pointer = 0
        self.size = 0
        self.is_full = False
        self._initialized_shape = False

    def __len__(self):
        return self.size
