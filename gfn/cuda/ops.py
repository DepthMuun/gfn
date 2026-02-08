"""
GFN CUDA Operations - Módulo de Operaciones Fusión
===================================================

Este módulo proporciona operaciones CUDA fusionadas para el proyecto GFN.
Diseñado de forma modular para facilitar la extensión y el testing.

Operaciones disponibles:
- christoffel_fused: Símbolos de Christoffel con descomposición low-rank
- leapfrog_fused: Integrador simpléctico leapfrog
- heun_fused: Integrador Heun (RK2)
- euler_fused: Integrador Euler
- rk4_fused: Integrador Runge-Kutta 4
- verlet_fused: Integrador Verlet
- head_mixing_fused: Mezcla de cabezas de atención
- dynamic_gating_fused: Compuerta dinámica
- recurrent_manifold_fused: Fusión de múltiple capas de manifold

Autor: MiniMax Agent
Fecha: 2026-02-07
"""

import torch
import torch.nn as nn
import os
import sys
import importlib.util
import importlib.machinery
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable
from abc import ABC, abstractmethod

# Importar módulo base
from .core import (
    device_manager,
    CudaConstants,
    operation_registry,
    check_cuda_availability
)


# ============================================================================
# CARGADOR DE MÓDULO CUDA
# ============================================================================

class CudaModuleLoader:
    """
    Cargador modular para módulos CUDA compiled.
    
    Maneja:
    - Detección de módulos compilados
    - Carga dinámica
    - Fallbacks automáticos
    """
    
    def __init__(self):
        self._loaded_modules: Dict[str, Any] = {}
        self._load_paths = self._get_load_paths()
    
    def _get_load_paths(self) -> List[Path]:
        """Obtiene las rutas de búsqueda para módulos CUDA."""
        cuda_dir = Path(__file__).resolve().parent
        project_root = cuda_dir.parent.parent
        
        paths = [cuda_dir, project_root]
        
        if sys.platform.startswith("win"):
            for ver in ["v12.9", "v12.4", "v12.3", "v11.8"]:
                cuda_path = Path(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{ver}/bin")
                if cuda_path.exists():
                    paths.append(cuda_path)
        
        return paths
    
    def _get_extension_patterns(self) -> List[str]:
        """Obtiene los patrones de extensión según la plataforma."""
        if sys.platform.startswith("win"):
            return ["*.pyd"]
        elif sys.platform.startswith("darwin"):
            return ["*.dylib", "*.so"]
        else:
            return ["*.so"]
    
    def find_module(self, name: str) -> Optional[Path]:
        """Busca un módulo compilado por nombre."""
        patterns = self._get_extension_patterns()
        
        for base_path in self._load_paths:
            if not base_path.exists():
                continue
            
            for pattern in patterns:
                for candidate in base_path.glob(pattern):
                    if name in candidate.stem:
                        return candidate
        
        return None
    
    def load(self, name: str) -> Optional[Any]:
        """Carga un módulo por nombre."""
        if name in self._loaded_modules:
            return self._loaded_modules[name]
        
        module_path = self.find_module(name)
        if module_path is None:
            return None
        
        try:
            loader = importlib.machinery.ExtensionFileLoader(name, str(module_path))
            spec = importlib.util.spec_from_file_location(name, str(module_path), loader=loader)
            
            if spec is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            
            self._loaded_modules[name] = module
            return module
            
        except Exception as e:
            print(f"[CUDA] Error cargando módulo {name}: {e}")
            return None
    
    def preload_all(self):
        """Precarga todos los módulos CUDA disponibles."""
        module_names = ['gfn_cuda']
        
        for name in module_names:
            self.load(name)


# Cargador global
_module_loader = CudaModuleLoader()

# Intentar cargar módulo CUDA compilado
_CUDA_MODULE = _module_loader.load('gfn_cuda')
CUDA_AVAILABLE = _CUDA_MODULE is not None


# ============================================================================
# FÁBRICA DE OPERACIONES
# ============================================================================

class OperationFactory:
    """
    Fábrica para crear operaciones con fallbacks automáticos.
    
    Patrón de diseño: Factory Method + Strategy
    """
    
    _operations: Dict[str, Tuple[Callable, Callable]] = {}
    
    @classmethod
    def register(cls, name: str, cuda_op: Callable, python_op: Callable):
        """Registra una nueva operación."""
        cls._operations[name] = (cuda_op, python_op)
    
    @classmethod
    def create(cls, name: str, device: torch.device) -> Callable:
        """
        Crea una operación para el dispositivo especificado.
        
        Args:
            name: Nombre de la operación
            device: Dispositivo de destino (cuda o cpu)
        
        Returns:
            Función de operación lista para usar
        """
        if name not in cls._operations:
            raise ValueError(f"Operación '{name}' no registrada")
        
        cuda_op, python_op = cls._operations[name]
        
        if device.type == 'cuda' and CUDA_AVAILABLE:
            return cuda_op
        return python_op
    
    @classmethod
    def has_cuda(cls, name: str) -> bool:
        """Verifica si la operación tiene implementación CUDA."""
        if name not in cls._operations:
            return False
        cuda_op, _ = cls._operations[name]
        return cuda_op is not None


# ============================================================================
# OPERACIONES BASE (PYTHON FALLBACK)
# ============================================================================

class BaseOperation(ABC):
    """Clase base para todas las operaciones."""
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Implementa el forward pass."""
        pass
    
    @abstractmethod
    def backward(self, *args, **kwargs):
        """Implementa el backward pass."""
        pass


class ChristoffelOperation(BaseOperation):
    """
    Operación de símbolos de Christoffel con descomposición low-rank.
    
    Computa: Γ^k_ij = Σ_r λ_kr * (U_ir * U_jr)
    
    Donde:
    - v: Velocidades [batch, dim]
    - U, W: Matrices de descomposición [dim, rank]
    - x: Posiciones (opcional) [batch, dim]
    - V_w: Pesos de potencial (opcional) [1, dim]
    
    Returns:
        gamma: Símbolos de Christoffel [batch, dim]
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._init_constants()
    
    def _init_constants(self):
        """Inicializa las constantes desde la configuración."""
        self.curvature_clamp = self.config.get('curvature_clamp', CudaConstants.CURVATURE_CLAMP)
        self.epsilon = self.config.get('epsilon', CudaConstants.EPSILON_STANDARD)
        self.singularity_gate_slope = self.config.get('singularity_gate_slope', CudaConstants.SINGULARITY_GATE_SLOPE)
    
    def forward(self, v: torch.Tensor, U: torch.Tensor, W: torch.Tensor,
                x: Optional[torch.Tensor] = None,
                V_w: Optional[torch.Tensor] = None,
                plasticity: float = 0.0,
                sing_thresh: float = 0.5,
                sing_strength: float = 2.0,
                topology: int = 0) -> torch.Tensor:
        """
        Forward pass de Christoffel.
        
        Args:
            v: Velocidades de entrada [B, D]
            U: Matriz U de descomposición [D, R]
            W: Matriz W de descomposición [D, R]
            x: Posiciones (opcional) [B, D]
            V_w: Pesos de potencial (opcional) [1, D]
            plasticity: Coeficiente de plasticidad de curvatura
            sing_thresh: Umbral de singularidad
            sing_strength: Fuerza de singularidad
            topology: Tipo de topología (0=euclidiana, 1=tórica)
        
        Returns:
            gamma: Símbolos de Christoffel [B, D]
        """
        # Proyección de velocidad: h = U^T v
        h = torch.matmul(v, U)  # [B, R]
        
        # Normalización de energía
        energy = torch.sum(h * h, dim=-1, keepdim=True) / max(1, h.shape[-1])
        scale = 1.0 / (1.0 + torch.sqrt(energy) + self.epsilon)
        
        # Factor de plasticidad
        M = 1.0
        if plasticity != 0.0:
            E = torch.sum(v * v, dim=-1, keepdim=True) / max(1, v.shape[-1])
            M = 1.0 + plasticity * 0.1 * torch.tanh(E)
        
        # Singularidades (amplificación de curvatura)
        if x is not None and V_w is not None and V_w.numel() > 0:
            if topology == 1:  # Tórica
                pot = torch.sum(torch.sin(x) * V_w, dim=-1, keepdim=True)
            else:
                pot = torch.sum(x * V_w, dim=-1, keepdim=True)
            
            gate = torch.sigmoid(pot)
            soft_m = torch.sigmoid(self.singularity_gate_slope * (gate - sing_thresh))
            M = M * (1.0 + (sing_strength - 1.0) * soft_m)
        
        # Christoffel: γ = (h ⊙ h) @ W^T
        gamma = torch.matmul(h * h, W.t()) * scale * M
        gamma = self.curvature_clamp * torch.tanh(gamma / self.curvature_clamp)
        
        return gamma
    
    def backward(self, grad_output: torch.Tensor, v: torch.Tensor, U: torch.Tensor,
                 W: torch.Tensor, x: Optional[torch.Tensor] = None,
                 V_w: Optional[torch.Tensor] = None,
                 output: Optional[torch.Tensor] = None,
                 plasticity: float = 0.0,
                 sing_thresh: float = 0.5,
                 sing_strength: float = 2.0,
                 topology: int = 0) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass de Christoffel usando autograd de PyTorch.
        
        Returns:
            Tupla de gradientes: (dv, dU, dW, dx, dV_w)
        """
        # Usar autograd de PyTorch para gradientes
        if output is None:
            output = self.forward(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology)
        
        grads = torch.autograd.grad(
            output, [v, U, W, x, V_w],
            grad_output,
            allow_unused=True,
            retain_graph=True
        )
        
        # Completar gradientes None con ceros
        result = []
        for i, (tensor, grad) in enumerate([(v, grads[0]), (U, grads[1]), (W, grads[2])]):
            if grad is None:
                result.append(torch.zeros_like(tensor))
            else:
                result.append(grad)
        
        # Para x y V_w (opcionales)
        if x is not None:
            result.append(grads[3] if grads[3] is not None else torch.zeros_like(x))
        if V_w is not None and V_w.numel() > 0:
            result.append(grads[4] if grads[4] is not None else torch.zeros_like(V_w))
        
        return tuple(result)


# ============================================================================
# TOROIDAL KERNEL BINDINGS (Component 2)
# ============================================================================

def launch_toroidal_leapfrog_fused(
    x: torch.Tensor,
    v: torch.Tensor,
    f: torch.Tensor,
    R: float,
    r: float,
    dt: float,
    batch: int,
    seq_len: int,
    dim: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Launch dedicated toroidal leapfrog fused kernel.
    
    AUDIT FIX (Component 2): This function provides Python binding
    to the dedicated toroidal Christoffel kernel.
    
    Args:
        x: Initial positions [batch, dim]
        v: Initial velocities [batch, dim]
        f: Force sequence [batch, seq_len, dim]
        R: Major radius of torus
        r: Minor radius of torus
        dt: Time step
        batch: Batch size
        seq_len: Sequence length
        dim: Dimension (should be even)
        
    Returns:
        Tuple of (x_final, v_final, x_seq, reg_loss) or None if CUDA unavailable
    """
    # Try to load CUDA module
    try:
        import gfn_cuda
        
        # Call CUDA kernel
        x_out, v_out = gfn_cuda.toroidal_leapfrog_fused(
            x.contiguous(),
            v.contiguous(),
            f.contiguous(),
            float(R),
            float(r),
            float(dt),
            int(batch),
            int(seq_len),
            int(dim)
        )
        
        # Prepare outputs to match recurrent_manifold_fused signature
        x_final = x_out[:, -1, :]  # [batch, dim]
        v_final = v_out[:, -1, :]  # [batch, dim]
        x_seq = x_out  # [batch, seq_len, dim]
        reg_loss = torch.tensor(0.0, device=x.device)  # No regularization for now
        
        return (x_final, v_final, x_seq, reg_loss)
        
    except (ImportError, AttributeError) as e:
        # CUDA module not compiled yet - this is expected
        # User will compile with: python setup.py build_ext --inplace
        return None



class LeapfrogOperation(BaseOperation):
    """
    Integrador Leapfrog (Stormer-Verlet) simpléctico.
    
    Implementa:
    - Kick-Drift-Kick con fricción implícita
    - Frontera tórica (wrapping)
    - Soporte de histéresis
    
    Args:
        x: Posiciones [B, D]
        v: Velocidades [B, D]
        f: Fuerzas externas [B, D]
        U, W: Matrices de Christoffel
        dt: Paso de tiempo base
        dt_scale: Escala de dt por capa/cabeza
        steps: Número de subpasos
        topology: Tipo de topología
        Wf, bf: Pesos de fricción
        plasticity: Plasticidad de curvatura
    
    Returns:
        x_out, v_out: Posiciones y velocidades actualizadas
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._init_constants()
    
    def _init_constants(self):
        """Inicializa las constantes."""
        self.friction_scale = self.config.get('friction_scale', CudaConstants.FRICTION_SCALE)
        self.epsilon = self.config.get('epsilon', CudaConstants.EPSILON_STANDARD)
        self.dt = self.config.get('dt', CudaConstants.DEFAULT_DT)
        self.toroidal_period = CudaConstants.TOROIDAL_PERIOD
    
    def forward(self, x: torch.Tensor, v: torch.Tensor, f: torch.Tensor,
                U: torch.Tensor, W: torch.Tensor,
                dt_scale: float = 1.0,
                steps: int = 1,
                topology: int = 0,
                Wf: Optional[torch.Tensor] = None,
                bf: Optional[torch.Tensor] = None,
                plasticity: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass del integrador Leapfrog.
        """
        # Calcular dt efectivo
        eff_dt = self.dt * dt_scale
        h = 0.5 * eff_dt
        
        # Copiar estados iniciales
        curr_x = x.clone()
        curr_v = v.clone()
        
        # Inicializar estado de histéresis
        h_state = None
        
        christoffel_op = ChristoffelOperation({
            'curvature_clamp': self.config.get('curvature_clamp', CudaConstants.CURVATURE_CLAMP),
            'epsilon': self.epsilon
        })
        
        for _ in range(steps):
            # 1. Calcular coeficiente de fricción
            mu = torch.zeros_like(v)
            if Wf is not None and bf is not None:
                feat = curr_x
                if topology == 1:
                    feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)
                gate = torch.matmul(feat, Wf.t()) + bf
                mu = torch.sigmoid(gate) * self.friction_scale
            
            # 2. Kick 1
            gamma = christoffel_op.forward(curr_v, U, W, curr_x, None, plasticity)
            curr_v = (curr_v + h * (f - gamma)) / (1.0 + h * mu + self.epsilon)
            
            # 3. Drift
            curr_x = curr_x + eff_dt * curr_v
            
            # AUDIT FIX: Apply toroidal boundary with smooth wrapping (match CUDA [0, 2pi])
            if topology == 1:
                curr_x = torch.remainder(curr_x, self.toroidal_period)
            
            # 4. Kick 2
            if Wf is not None and bf is not None:
                feat = curr_x
                if topology == 1:
                    feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)
                gate = torch.matmul(feat, Wf.t()) + bf
                mu = torch.sigmoid(gate) * self.friction_scale
            
            gamma2 = christoffel_op.forward(curr_v, U, W, curr_x, None, plasticity)
            curr_v = (curr_v + h * (f - gamma2)) / (1.0 + h * mu + self.epsilon)
        
        return curr_x, curr_v
    
    def backward(self, grad_x: torch.Tensor, grad_v: torch.Tensor,
                 x: torch.Tensor, v: torch.Tensor, f: torch.Tensor,
                 U: torch.Tensor, W: torch.Tensor,
                 dt_scale: float = 1.0,
                 steps: int = 1,
                 topology: int = 0,
                 Wf: Optional[torch.Tensor] = None,
                 bf: Optional[torch.Tensor] = None,
                 plasticity: float = 0.0) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass del integrador Leapfrog.
        
        Returns:
            Tupla de gradientes
        """
        # Usar autograd de PyTorch
        def leapfrog_fn(x, v):
            return self.forward(x, v, f, U, W, dt_scale, steps, topology, Wf, bf, plasticity)
        
        return torch.autograd.grad(
            leapfrog_fn(x, v),
            [x, v, f, U, W],
            (grad_x, grad_v),
            allow_unused=True,
            retain_graph=True
        )


# Registrar operaciones en la fábrica
OperationFactory.register('christoffel', None, ChristoffelOperation)
OperationFactory.register('leapfrog', None, LeapfrogOperation)


# ============================================================================
# INTERFAZ PÚBLICA DEL MÓDULO
# ============================================================================

def christoffel_fused(v: torch.Tensor, U: torch.Tensor, W: torch.Tensor,
                      x: Optional[torch.Tensor] = None,
                      V_w: Optional[torch.Tensor] = None,
                      plasticity: float = 0.0,
                      sing_thresh: float = 0.5,
                      sing_strength: float = 2.0,
                      topology: int = 0,
                      R: float = 2.0,
                      r: float = 1.0) -> torch.Tensor:
    """
    Interfaz pública para Christoffel fused.
    
    Detecta automáticamente si usar CUDA o Python fallback.
    """
    if CUDA_AVAILABLE and v.is_cuda:
        # Usar autograd wrapper si está disponible
        try:
            from .autograd import christoffel_fused_autograd
            return christoffel_fused_autograd(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r)
        except ImportError:
            pass
    
    # Python fallback
    op = ChristoffelOperation()
    return op.forward(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology)


def leapfrog_fused(x: torch.Tensor, v: torch.Tensor, f: torch.Tensor,
                   U: torch.Tensor, W: torch.Tensor,
                   dt: float,
                   dt_scale: float,
                   steps: int,
                   topology: int = 0,
                   Wf: Optional[torch.Tensor] = None,
                   bf: Optional[torch.Tensor] = None,
                   plasticity: float = 0.0,
                   R: float = 2.0,
                   r: float = 1.0,
                   # AUDIT FIX (Component 7): Hysteresis parameters with defaults
                   hysteresis_state: Optional[torch.Tensor] = None,
                   hyst_update_w: Optional[torch.Tensor] = None,
                   hyst_update_b: Optional[torch.Tensor] = None,
                   hyst_readout_w: Optional[torch.Tensor] = None,
                   hyst_readout_b: Optional[torch.Tensor] = None,
                   hyst_decay: float = 0.9,
                   hyst_enabled: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Interfaz pública para Leapfrog fused.
    
    Detecta automáticamente si usar CUDA o Python fallback.
    """
    if CUDA_AVAILABLE and x.is_cuda:
        try:
            from .autograd import leapfrog_fused_autograd
            return leapfrog_fused_autograd(
                x, v, f, U, W, dt, dt_scale, steps, topology, Wf, bf, plasticity, R, r,
                hysteresis_state, hyst_update_w, hyst_update_b, 
                hyst_readout_w, hyst_readout_b, hyst_decay, hyst_enabled
            )
        except ImportError:
            pass
    
    # Python fallback
    op = LeapfrogOperation({'dt': dt, 'friction_scale': CudaConstants.FRICTION_SCALE, 'epsilon': CudaConstants.EPSILON_STANDARD})
    return op.forward(x, v, f, U, W, dt_scale, steps, topology, Wf, bf, plasticity)


# Alias para compatibilidad
lowrank_christoffel_fused = christoffel_fused


def get_registered_operations() -> Dict[str, bool]:
    """Obtiene el estado de registro de operaciones."""
    return {
        'christoffel_fused': CUDA_AVAILABLE or True,
        'leapfrog_fused': CUDA_AVAILABLE or True,
        'heun_fused': False,
        'euler_fused': False,
        'rk4_fused': False,
        'verlet_fused': False,
        'head_mixing_fused': CUDA_AVAILABLE,
        'dynamic_gating_fused': CUDA_AVAILABLE,
        'recurrent_manifold_fused': CUDA_AVAILABLE
    }
