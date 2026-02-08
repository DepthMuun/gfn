"""
GFN CUDA Kernels - Módulo de Operaciones Base
==============================================

Este módulo proporciona las operaciones fundamentales de CUDA para el proyecto GFN.
Diseñado de forma modular para facilitar la extensión y el testing.

Autor: MiniMax Agent
Fecha: 2026-02-07
"""

import torch
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


# ============================================================================
# GESTOR DE DISPOSITIVOS Y CONSTANTES
# ============================================================================

class CudaDeviceManager:
    """
    Gestiona la disponibilidad y estado de los dispositivos CUDA.
    
    Proporciona una interfaz unificada para:
    - Detección de dispositivos
    - Gestión de memoria
    - Sincronización
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not CudaDeviceManager._initialized:
            self._cuda_available = False
            self._device_name = "CPU (Fallback)"
            self._device_count = 0
            self._compute_capability = None
            self._init_device()
            CudaDeviceManager._initialized = True
    
    def _init_device(self):
        """Inicializa la información del dispositivo."""
        self._cuda_available = torch.cuda.is_available()
        
        if self._cuda_available:
            self._device_count = torch.cuda.device_count()
            self._device_name = torch.cuda.get_device_name(0)
            self._compute_capability = torch.cuda.get_device_capability(0)
            torch.cuda.set_device(0)
    
    @property
    def is_available(self) -> bool:
        return self._cuda_available
    
    @property
    def device_name(self) -> str:
        return self._device_name
    
    @property
    def device_count(self) -> int:
        return self._device_count
    
    @property
    def compute_capability(self) -> Tuple[int, int]:
        return self._compute_capability
    
    def get_device(self, index: int = 0) -> torch.device:
        """Obtiene el dispositivo en el índice especificado."""
        if self._cuda_available and index < self._device_count:
            return torch.device(f'cuda:{index}')
        return torch.device('cpu')
    
    def empty_cache(self):
        """Libera memoria CUDA no utilizada."""
        if self._cuda_available:
            torch.cuda.empty_cache()
    
    def synchronize(self, device: Optional[torch.device] = None):
        """Sincroniza el dispositivo especificado."""
        if self._cuda_available:
            if device is None:
                device = self.get_device()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
    
    def memory_info(self) -> Dict[str, Any]:
        """Obtiene información de memoria del dispositivo."""
        if not self._cuda_available:
            return {"available": 0, "total": 0, "used": 0}
        
        return {
            "available": torch.cuda.get_device_properties(0).total_memory,
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated()
        }


# Instancia global del gestor de dispositivos
device_manager = CudaDeviceManager()


# ============================================================================
# CONSTANTES CENTRALIZADAS
# ============================================================================

class CudaConstants:
    """
    Constantes específicas para operaciones CUDA.

    Estas constantes están sincronizadas con las constantes Python para
    garantizar consistencia entre las implementaciones.

    AUDIT FIX (2026-02-07): Updated to match optimized Python constants.
    """

    # Física - OPTIMIZED for proper symplectic behavior
    FRICTION_SCALE = 0.02
    VELOCITY_FRICTION_SCALE = 0.02
    DEFAULT_FRICTION = 0.002

    # Estabilidad numérica - OPTIMIZED for better gradient flow
    EPSILON_STANDARD = 1e-7
    EPSILON_STRONG = 1e-7
    EPSILON_SMOOTH = 1e-7

    # Geometría - OPTIMIZED for stability
    CURVATURE_CLAMP = 2.5
    SINGULARITY_GATE_SLOPE = 0.5

    # Integración - OPTIMIZED for exploration
    DEFAULT_DT = 0.05
    LEAPFROG_SUBSTEPS = 3

    # Velocidad
    VELOCITY_SATURATION = 100.0

    # Topología
    TOROIDAL_PERIOD = 6.283185307179586  # 2 * π
    
    @classmethod
    def to_dict(cls) -> Dict[str, float]:
        """Retorna las constantes como diccionario."""
        return {
            'FRICTION_SCALE': cls.FRICTION_SCALE,
            'VELOCITY_FRICTION_SCALE': cls.VELOCITY_FRICTION_SCALE,
            'DEFAULT_FRICTION': cls.DEFAULT_FRICTION,
            'EPSILON_STANDARD': cls.EPSILON_STANDARD,
            'EPSILON_STRONG': cls.EPSILON_STRONG,
            'EPSILON_SMOOTH': cls.EPSILON_SMOOTH,
            'CURVATURE_CLAMP': cls.CURVATURE_CLAMP,
            'SINGULARITY_GATE_SLOPE': cls.SINGULARITY_GATE_SLOPE,
            'DEFAULT_DT': cls.DEFAULT_DT,
            'LEAPFROG_SUBSTEPS': cls.LEAPFROG_SUBSTEPS,
            'VELOCITY_SATURATION': cls.VELOCITY_SATURATION,
            'TOROIDAL_PERIOD': cls.TOROIDAL_PERIOD
        }


# ============================================================================
# REGISTRO DE OPERACIONES
# ============================================================================

class OperationRegistry:
    """
    Registro de operaciones CUDA disponibles.
    
    Permite:
    - Registrar nuevas operaciones
    - Verificar disponibilidad
    - Obtener información de operaciones
    """
    
    def __init__(self):
        self._operations: Dict[str, Dict[str, Any]] = {}
        self._register_standard_operations()
    
    def _register_standard_operations(self):
        """Registra las operaciones estándar."""
        standard_ops = [
            'christoffel_fused',
            'lowrank_christoffel_fused',
            'leapfrog_fused',
            'heun_fused',
            'euler_fused',
            'rk4_fused',
            'verlet_fused',
            'head_mixing_fused',
            'dynamic_gating_fused',
            'recurrent_manifold_fused'
        ]
        
        for op in standard_ops:
            self._operations[op] = {
                'available': False,
                'cuda_available': False,
                'python_fallback': True,
                'description': ''
            }
    
    def register(self, name: str, cuda_impl: bool, python_fallback: bool, description: str = ''):
        """Registra una nueva operación."""
        self._operations[name] = {
            'available': cuda_impl or python_fallback,
            'cuda_available': cuda_impl,
            'python_fallback': python_fallback,
            'description': description
        }
    
    def is_available(self, name: str) -> bool:
        """Verifica si una operación está disponible."""
        return self._operations.get(name, {}).get('available', False)
    
    def has_cuda(self, name: str) -> bool:
        """Verifica si una operación tiene implementación CUDA."""
        return self._operations.get(name, {}).get('cuda_available', False)
    
    def get_info(self, name: str) -> Dict[str, Any]:
        """Obtiene información de una operación."""
        return self._operations.get(name, {})
    
    def list_available(self) -> Dict[str, Dict[str, Any]]:
        """Lista todas las operaciones disponibles."""
        return self._operations
    
    def summary(self) -> Dict[str, int]:
        """Retorna un resumen de disponibilidad."""
        cuda_count = sum(1 for op in self._operations.values() if op['cuda_available'])
        fallback_count = sum(1 for op in self._operations.values() if op['python_fallback'])
        return {
            'total': len(self._operations),
            'cuda': cuda_count,
            'python_fallback': fallback_count,
            'missing': len(self._operations) - cuda_count
        }


# Instancia global del registro
operation_registry = OperationRegistry()


# ============================================================================
# INTERFAZ PÚBLICA
# ============================================================================

def get_device_manager() -> CudaDeviceManager:
    """Obtiene el gestor de dispositivos."""
    return device_manager


def get_constants() -> CudaConstants:
    """Obtiene las constantes CUDA."""
    return CudaConstants


def get_operation_registry() -> OperationRegistry:
    """Obtiene el registro de operaciones."""
    return operation_registry


def check_cuda_availability() -> bool:
    """Verifica si CUDA está disponible."""
    return device_manager.is_available


def get_device_info() -> Dict[str, Any]:
    """Obtiene información completa del dispositivo."""
    return {
        'available': device_manager.is_available,
        'name': device_manager.device_name,
        'count': device_manager.device_count,
        'compute_capability': device_manager.compute_capability,
        'memory': device_manager.memory_info(),
        'operations': operation_registry.summary()
    }
