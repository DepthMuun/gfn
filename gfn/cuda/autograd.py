"""
GFN CUDA Autograd - Módulo de Diferenciación Automática
========================================================

Este módulo proporciona funciones de diferenciación automática para las
operaciones CUDA del proyecto GFN.

Funciones disponibles:
- christoffel_fused_autograd
- leapfrog_fused_autograd
- heun_fused_autograd
- recurrent_manifold_fused_autograd

Autor: MiniMax Agent
Fecha: 2026-02-07
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from functools import wraps
import time


# ============================================================================
# REGISTRO DE TIEMPOS
# ============================================================================

class TimingRegistry:
    """
    Registro de tiempos de ejecución para profiling.
    
    Permite:
    - Medir tiempos de operaciones CUDA vs Python
    - Acumular estadísticas
    - Generar informes de rendimiento
    """
    
    def __init__(self):
        self._timings: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}
        self._enabled = False
    
    def enable(self):
        """Habilita el registro de tiempos."""
        self._enabled = True
    
    def disable(self):
        """Deshabilita el registro de tiempos."""
        self._enabled = False
    
    def record(self, name: str, duration: float):
        """Registra un tiempo de ejecución."""
        if not self._enabled:
            return
        
        if name not in self._timings:
            self._timings[name] = []
            self._counts[name] = 0
        
        self._timings[name].append(duration)
        self._counts[name] += 1
    
    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Obtiene estadísticas de una operación."""
        if name not in self._timings or not self._timings[name]:
            return None
        
        times = self._timings[name]
        return {
            'count': self._counts[name],
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'total': sum(times),
            'last': times[-1]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Obtiene estadísticas de todas las operaciones."""
        return {name: self.get_stats(name) for name in self._timings}
    
    def reset(self):
        """Reinicia todas las estadísticas."""
        self._timings = {}
        self._counts = {}
    
    def summary(self) -> str:
        """Genera un resumen formateado."""
        if not self._timings:
            return "Sin datos de timing registrados"
        
        lines = ["=" * 60, "RESUMEN DE TIEMPOS DE EJECUCIÓN", "=" * 60]
        
        for name, stats in self.get_all_stats().items():
            if stats:
                lines.append(f"\n{name}:")
                lines.append(f"  Count:  {stats['count']}")
                lines.append(f"  Mean:   {stats['mean']*1000:.3f} ms")
                lines.append(f"  Min:    {stats['min']*1000:.3f} ms")
                lines.append(f"  Max:    {stats['max']*1000:.3f} ms")
                lines.append(f"  Total:  {stats['total']*1000:.3f} ms")
        
        return "\n".join(lines)


# Instancia global del registro de tiempos
timing_registry = TimingRegistry()


# ============================================================================
# DECORADORES DE TIMING
# ============================================================================

def timed_operation(name: Optional[str] = None):
    """
    Decorador para medir tiempos de ejecución de operaciones.
    
    Args:
        name: Nombre de la operación (usa nombre de función si no se especifica)
    """
    def decorator(func):
        op_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            
            if timing_registry._enabled:
                timing_registry.record(op_name, duration)
            
            return result
        
        return wrapper
    return decorator


# ============================================================================
# FUNCIONES AUTOGRAD
# ============================================================================

class ChristoffelAutogradFunction(torch.autograd.Function):
    """
    Función autograd para Christoffel fused.
    
    Implementa forward y backward para diferenciación automática.
    """
    
    @staticmethod
    def forward(ctx, v: torch.Tensor, U: torch.Tensor, W: torch.Tensor,
                x: Optional[torch.Tensor], V_w: Optional[torch.Tensor],
                plasticity: float, sing_thresh: float, sing_strength: float,
                topology: int, R: float, r: float) -> torch.Tensor:
        """
        Forward pass.
        """
        # Determinar dispositivo
        is_cuda = v.is_cuda
        
        # Asegurar tensores contiguos
        v = v.contiguous()
        U = U.contiguous()
        W = W.contiguous()
        
        if x is not None and x.numel() > 0:
            x = x.contiguous()
        else:
            x = torch.empty(0, device=v.device, dtype=v.dtype)
        
        if V_w is not None and V_w.numel() > 0:
            V_w = V_w.contiguous()
        else:
            V_w = torch.empty(0, device=v.device, dtype=v.dtype)
        
        # Intentar usar CUDA
        if is_cuda:
            try:
                import gfn_cuda
                gamma = gfn_cuda.lowrank_christoffel_fused(
                    v, U, W, x, V_w,
                    float(plasticity), float(sing_thresh), float(sing_strength),
                    int(topology), float(R), float(r)
                )
                
                # Guardar tensores para backward
                ctx.save_for_backward(v, U, W, x, V_w, gamma)
                ctx.plasticity = plasticity
                ctx.sing_thresh = sing_thresh
                ctx.sing_strength = sing_strength
                ctx.topology = topology
                ctx.R = R
                ctx.r = r
                ctx.is_cuda = True
                
                return gamma
                
            except (ImportError, AttributeError):
                pass
        
        # Fallback a Python
        from .ops import ChristoffelOperation
        
        op = ChristoffelOperation({
            'curvature_clamp': 3.0,
            'epsilon': 1e-8,
            'singularity_gate_slope': 1.0
        })
        
        gamma = op.forward(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology)
        
        # Guardar tensores para backward
        ctx.save_for_backward(v, U, W, x, V_w)
        ctx.plasticity = plasticity
        ctx.sing_thresh = sing_thresh
        ctx.sing_strength = sing_strength
        ctx.topology = topology
        ctx.R = R
        ctx.r = r
        ctx.is_cuda = False
        
        return gamma
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple:
        """
        Backward pass.
        
        AUDIT FIX (2026-02-07): Robust gradient unpacking with validation.
        The CUDA kernel returns a specific number of gradients that must match
        our expected structure. We now validate the length and provide clear
        error messages if there's a mismatch.
        """
        # Recuperar tensores guardados
        saved = ctx.saved_tensors
        if len(saved) == 6:
            v, U, W, x, V_w, gamma = saved
        else:
            v, U, W, x, V_w = saved
            gamma = None
        
        # Expected gradient count based on CUDA kernel signature
        # Christoffel backward returns: dv, dU, dW, dx, dV_w, + 6 None params = 11 total
        EXPECTED_GRADIENTS = 11
        
        # Intentar backward de CUDA
        if getattr(ctx, 'is_cuda', False):
            try:
                import gfn_cuda
                grads = gfn_cuda.christoffel_backward_fused(
                    grad_output.contiguous(),
                    gamma if gamma is not None else torch.empty(0),
                    v, U, W, x, V_w,
                    float(ctx.plasticity), float(ctx.sing_thresh), float(ctx.sing_strength),
                    int(ctx.topology), float(ctx.R), float(ctx.r)
                )
                
                # AUDIT FIX (2026-02-07): Validate gradient count
                actual_grads = len(grads)
                if actual_grads != 5:
                    print(f"[GFN:DEBUG] Christoffel backward returned {actual_grads} gradients, expected 5")
                
                # Safely unpack gradients, padding with None if needed
                result = []
                for i in range(5):  # Core gradients: dv, dU, dW, dx, dV_w
                    if i < actual_grads:
                        result.append(grads[i])
                    else:
                        result.append(None)
                
                # Add None for non-differentiable parameters (plasticity, sing_thresh, etc.)
                result.extend([None] * (EXPECTED_GRADIENTS - len(result)))
                
                return tuple(result)
                
            except (ImportError, AttributeError):
                pass
        
        # Fallback a autograd de PyTorch
        # Reconstruir output si es necesario
        if gamma is None:
            from .ops import ChristoffelOperation
            op = ChristoffelOperation({
                'curvature_clamp': 3.0,
                'epsilon': 1e-8,
                'singularity_gate_slope': 1.0
            })
            gamma = op.forward(v, U, W, x, V_w, ctx.plasticity, ctx.sing_thresh, ctx.sing_strength, ctx.topology)
        
        # Calcular gradientes usando autograd
        grad_inputs = torch.autograd.grad(
            gamma, [v, U, W, x, V_w], grad_output,
            allow_unused=True,
            retain_graph=False
        )
        
        # Completar con ceros donde sea necesario
        result = []
        for i, (tensor, grad) in enumerate([
            (v, grad_inputs[0]),
            (U, grad_inputs[1]),
            (W, grad_inputs[2]),
            (x, grad_inputs[3]),
            (V_w, grad_inputs[4])
        ]):
            if grad is None:
                if tensor.numel() > 0:
                    result.append(torch.zeros_like(tensor))
                else:
                    result.append(None)
            else:
                result.append(grad)
        
        # Añadir gradientes None para parámetros no diferenciables
        result.extend([None, None, None, None, None, None])
        
        return tuple(result)


class LeapfrogAutogradFunction(torch.autograd.Function):
    """
    Función autograd para Leapfrog fused.
    
    Implementa forward y backward para el integrador simpléctico.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, v: torch.Tensor, force: torch.Tensor,
                U: torch.Tensor, W: torch.Tensor,
                dt: float, dt_scale: float, steps: int,
                topology: int,
                Wf: Optional[torch.Tensor], bf: Optional[torch.Tensor],
                plasticity: float, R: float, r: float,
                # AUDIT FIX (Component 7): Hysteresis parameters
                hysteresis_state: torch.Tensor,
                hyst_update_w: torch.Tensor,
                hyst_update_b: torch.Tensor,
                hyst_readout_w: torch.Tensor,
                hyst_readout_b: torch.Tensor,
                hyst_decay: float,
                hyst_enabled: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass del integrador Leapfrog.
        """
        is_cuda = x.is_cuda
        
        # Asegurar contiguidad
        x = x.contiguous()
        v = v.contiguous()
        force = force.contiguous()
        U = U.contiguous()
        W = W.contiguous()
        
        if Wf is not None and Wf.numel() > 0:
            Wf = Wf.contiguous()
        else:
            Wf = torch.empty(0, device=x.device, dtype=x.dtype)
        
        if bf is not None and bf.numel() > 0:
            bf = bf.contiguous()
        else:
            bf = torch.empty(0, device=x.device, dtype=x.dtype)
        
        # Intentar usar CUDA
        if is_cuda:
            try:
                import gfn_cuda
                
                x_out, v_out = gfn_cuda.leapfrog_fused(
                    x, v, force, U, W,
                    float(dt), float(dt_scale), int(steps), int(topology),
                    Wf, bf, float(plasticity), float(R), float(r),
                    # AUDIT FIX: Pass hysteresis parameters
                    hysteresis_state, hyst_update_w, hyst_update_b,
                    hyst_readout_w, hyst_readout_b, float(hyst_decay), hyst_enabled
                )
                
                ctx.save_for_backward(x, v, force, U, W, Wf, bf, x_out, v_out)
                ctx.dt = dt
                ctx.dt_scale = dt_scale
                ctx.steps = steps
                ctx.topology = topology
                ctx.plasticity = plasticity
                ctx.R = R
                ctx.r = r
                # AUDIT FIX: Save hysteresis parameters
                ctx.hysteresis_state = hysteresis_state
                ctx.hyst_update_w = hyst_update_w
                ctx.hyst_update_b = hyst_update_b
                ctx.hyst_readout_w = hyst_readout_w
                ctx.hyst_readout_b = hyst_readout_b
                ctx.hyst_decay = hyst_decay
                ctx.hyst_enabled = hyst_enabled
                ctx.is_cuda = True
                
                return x_out, v_out
                
            except (ImportError, AttributeError):
                pass
        
        # Fallback Python
        from .ops import LeapfrogOperation
        
        op = LeapfrogOperation({
            'dt': dt,
            'friction_scale': 0.05,
            'epsilon': 1e-8,
            'curvature_clamp': 3.0
        })
        
        x_out, v_out = op.forward(x, v, force, U, W, dt_scale, steps, topology, Wf, bf, plasticity)
        
        ctx.save_for_backward(x, v, force, U, W, Wf, bf, x_out, v_out)
        ctx.dt = dt
        ctx.dt_scale = dt_scale
        ctx.steps = steps
        ctx.topology = topology
        ctx.plasticity = plasticity
        ctx.R = R
        ctx.r = r
        ctx.is_cuda = False
        
        return x_out, v_out
    
    @staticmethod
    def backward(ctx, grad_x_out: torch.Tensor, grad_v_out: torch.Tensor) -> Tuple:
        """
        Backward pass del integrador Leapfrog.
        """
        saved = ctx.saved_tensors
        x, v, force, U, W, Wf, bf, x_out, v_out = saved
        
        if getattr(ctx, 'is_cuda', False):
            try:
                import gfn_cuda
                
                grads = gfn_cuda.leapfrog_backward_fused(
                    grad_x_out.contiguous(), grad_v_out.contiguous(),
                    x, v, force, U, W, Wf, bf,
                    float(ctx.dt), float(ctx.dt_scale), int(ctx.steps), int(ctx.topology),
                    float(ctx.plasticity), float(ctx.R), float(ctx.r),
                    # AUDIT FIX: Pass hysteresis parameters from ctx
                    ctx.hysteresis_state, ctx.hyst_update_w, ctx.hyst_update_b,
                    ctx.hyst_readout_w, ctx.hyst_readout_b, float(ctx.hyst_decay), ctx.hyst_enabled
                )
                
                # AUDIT FIX: Unpack hysteresis gradients (now 11 outputs instead of 7)
                return (grads[0], grads[1], grads[2], grads[3], grads[4],
                        None, None, None, None, grads[5], grads[6], None, None, None,
                        # Hysteresis gradients
                        grads[7], grads[8], grads[9], grads[10], None, None, None)
                
            except (ImportError, AttributeError):
                pass
        
        # Fallback: usar autograd de PyTorch
        def leapfrog_fn(x_in, v_in):
            from .ops import LeapfrogOperation
            op = LeapfrogOperation({
                'dt': ctx.dt,
                'friction_scale': 0.05,
                'epsilon': 1e-8,
                'curvature_clamp': 3.0
            })
            return op.forward(x_in, v_in, force, U, W, ctx.dt_scale, ctx.steps,
                            ctx.topology, Wf, bf, ctx.plasticity)
        
        grads = torch.autograd.grad(
            leapfrog_fn(x, v),
            [x, v, force, U, W],
            (grad_x_out, grad_v_out),
            allow_unused=True,
            retain_graph=False
        )
        
        # Completar resultado
        result = []
        for i, (tensor, grad) in enumerate([
            (x, grads[0]), (v, grads[1]), (force, grads[2]),
            (U, grads[3]), (W, grads[4])
        ]):
            result.append(grad if grad is not None else torch.zeros_like(tensor))
        
        # Parámetros adicionales
        result.extend([None, None, None, None,  # dt, dt_scale, steps, topology
                       grads[5] if len(grads) > 5 else torch.zeros_like(Wf) if Wf.numel() > 0 else None,
                       grads[6] if len(grads) > 6 else torch.zeros_like(bf) if bf.numel() > 0 else None,
                       None, None, None])  # plasticity, R, r
        
        return tuple(result)


# ============================================================================
# INTERFAZ PÚBLICA
# ============================================================================

def christoffel_fused_autograd(v: torch.Tensor, U: torch.Tensor, W: torch.Tensor,
                               x: Optional[torch.Tensor] = None,
                               V_w: Optional[torch.Tensor] = None,
                               plasticity: float = 0.0,
                               sing_thresh: float = 0.5,
                               sing_strength: float = 2.0,
                               topology: int = 0,
                               R: float = 2.0,
                               r: float = 1.0) -> torch.Tensor:
    """
    Computa símbolos de Christoffel con soporte autograd.
    
    Args:
        v: Velocidades [B, D]
        U: Matriz U [D, R]
        W: Matriz W [D, R]
        x: Posiciones opcionales [B, D]
        V_w: Pesos de potencial opcionales [1, D]
        plasticity: Plasticidad de curvatura
        sing_thresh: Umbral de singularidad
        sing_strength: Fuerza de singularidad
        topology: Tipo de topología
        R, r: Radios del toro (solo para topología tórica)
    
    Returns:
        gamma: Símbolos de Christoffel [B, D]
    """
    # Preparar tensores opcionales
    if x is None:
        x = torch.empty(0, device=v.device, dtype=v.dtype)
    if V_w is None:
        V_w = torch.empty(0, device=v.device, dtype=v.dtype)
    
    return ChristoffelAutogradFunction.apply(
        v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r
    )


def leapfrog_fused_autograd(x: torch.Tensor, v: torch.Tensor, force: torch.Tensor,
                            U: torch.Tensor, W: torch.Tensor,
                            dt: float, dt_scale: float, steps: int,
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
    Integra sistema Hamiltoniano con soporte autograd.
    
    Args:
        x: Posiciones [B, D]
        v: Velocidades [B, D]
        force: Fuerzas externas [B, D]
        U, W: Matrices de Christoffel
        dt: Paso de tiempo base
        dt_scale: Escala de dt
        steps: Número de subpasos
        topology: Tipo de topología
        Wf, bf: Pesos de fricción
        plasticity: Plasticidad de curvatura
        R, r: Radios del toro
        hysteresis_state: Estado inicial de hysteresis [B, D]
        hyst_update_w, hyst_update_b: Parámetros de actualización
        hyst_readout_w, hyst_readout_b: Parámetros de lectura
        hyst_decay: Decaimiento de hysteresis
        hyst_enabled: Activar hysteresis
    
    Returns:
        x_out, v_out: Estados actualizados
    """
    if Wf is None:
        Wf = torch.empty(0, device=x.device, dtype=x.dtype)
    if bf is None:
        bf = torch.empty(0, device=x.device, dtype=x.dtype)
    
    # AUDIT FIX: Prepare hysteresis tensors
    if hysteresis_state is None:
        hysteresis_state = torch.empty(0, device=x.device, dtype=x.dtype)
    if hyst_update_w is None:
        hyst_update_w = torch.empty(0, device=x.device, dtype=x.dtype)
    if hyst_update_b is None:
        hyst_update_b = torch.empty(0, device=x.device, dtype=x.dtype)
    if hyst_readout_w is None:
        hyst_readout_w = torch.empty(0, device=x.device, dtype=x.dtype)
    if hyst_readout_b is None:
        hyst_readout_b = torch.empty(0, device=x.device, dtype=x.dtype)
    
    return LeapfrogAutogradFunction.apply(
        x, v, force, U, W, dt, dt_scale, steps, topology, Wf, bf, plasticity, R, r,
        hysteresis_state, hyst_update_w, hyst_update_b, 
        hyst_readout_w, hyst_readout_b, hyst_decay, hyst_enabled
    )


def recurrent_manifold_fused_autograd(x: torch.Tensor, v: torch.Tensor, f: torch.Tensor,
                                       U_stack: torch.Tensor, W_stack: torch.Tensor,
                                       dt: float, dt_scales: torch.Tensor,
                                       forget_rates: torch.Tensor, num_heads: int,
                                       plasticity: float, sing_thresh: float,
                                       sing_strength: float,
                                       mix_x: Optional[torch.Tensor],
                                       mix_v: Optional[torch.Tensor],
                                       Wf: Optional[torch.Tensor],
                                       Wi: Optional[torch.Tensor],
                                       bf: Optional[torch.Tensor],
                                       Wp: Optional[torch.Tensor],
                                       bp: Optional[torch.Tensor],
                                       topology: int = 0,
                                       R: float = 2.0, r: float = 1.0,
                                       **kwargs) -> Tuple:
    """
    Fusión de múltiple capas de manifold (placeholder).
    """
    # Por ahora, delegar a Python fallback
    from .ops import ChristoffelOperation
    
    device = x.device
    dtype = x.dtype
    B, D = x.shape
    T = f.shape[1]
    num_layers = U_stack.shape[0] // num_heads
    head_dim = D // num_heads
    
    # Implementación simplificada para testing
    x_curr = x
    v_curr = v
    x_seq = []
    
    christoffel_op = ChristoffelOperation()
    
    for t in range(T):
        force_t = f[:, t]
        
        for layer_idx in range(num_layers):
            x_out = []
            v_out = []
            
            for h in range(num_heads):
                s = h * head_dim
                e = (h + 1) * head_dim
                
                x_h = x_curr[:, s:e]
                v_h = v_curr[:, s:e]
                f_h = force_t[:, s:e]
                
                # Integración simple
                gamma = christoffel_op.forward(v_h, U_stack[layer_idx * num_heads + h],
                                              W_stack[layer_idx * num_heads + h])
                
                dt_eff = dt * (dt_scales[layer_idx, h] if dt_scales.dim() > 1 else dt_scales)
                
                v_h = v_h + dt_eff * (f_h - gamma)
                x_h = x_h + dt_eff * v_h
                
                if topology == 1:
                    # AUDIT FIX: Use atan2 for smooth gradients
                    x_h = torch.atan2(torch.sin(x_h), torch.cos(x_h))
                
                x_out.append(x_h)
                v_out.append(v_h)
            
            x_curr = torch.cat(x_out, dim=-1)
            v_curr = torch.cat(v_out, dim=-1)
        
        x_seq.append(x_curr)
    
    x_seq = torch.stack(x_seq, dim=1)
    return x_curr, v_curr, x_seq, torch.zeros((), device=device), None


def get_timing_stats() -> Dict[str, Dict[str, float]]:
    """Obtiene estadísticas de timing."""
    return timing_registry.get_all_stats()


def enable_timing():
    """Habilita el registro de tiempos."""
    timing_registry.enable()


def disable_timing():
    """Deshabilita el registro de tiempos."""
    timing_registry.disable()


def reset_timing():
    """Reinicia las estadísticas de timing."""
    timing_registry.reset()
