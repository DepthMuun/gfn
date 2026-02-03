#!/usr/bin/env python3
"""
Test de verificación numérica para el backward de LowRankChristoffelWithFrictionFunction.
Compara gradientes CUDA con gradientes numéricos calculados en Python.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from gfn.cuda.autograd import LowRankChristoffelWithFrictionFunction, CUDA_AVAILABLE
    from gfn.cuda.ops import christoffel_fused
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA no disponible, omitiendo test de verificación")

def compute_numerical_gradients(func, inputs, output_indices=None, eps=1e-5):
    """
    Calcula gradientes numéricos usando diferencias finitas.
    
    Args:
        func: Función a evaluar
        inputs: Lista de tensores de entrada
        output_indices: Índices de salida a considerar (None = todos)
        eps: Tamaño del paso para diferencias finitas
    
    Returns:
        Lista de gradientes numéricos para cada entrada
    """
    gradients = []
    
    # Evaluar función original
    outputs = func(*inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    
    # Seleccionar outputs a considerar
    if output_indices is not None:
        outputs = [outputs[i] for i in output_indices]
    
    # Para cada input
    for i, input_tensor in enumerate(inputs):
        if input_tensor.numel() == 0 or not input_tensor.requires_grad:
            gradients.append(None)
            continue
            
        grad = torch.zeros_like(input_tensor)
        original_data = input_tensor.data.clone()
        
        # Aplanar para facilitar iteración
        flat_input = input_tensor.view(-1)
        flat_grad = grad.view(-1)
        original_flat = original_data.view(-1)
        
        for j in range(flat_input.numel()):
            # Diferencia central para mayor precisión
            original_flat[j] += eps
            input_tensor.data = original_flat.view(input_tensor.shape)
            outputs_plus = func(*inputs)
            if isinstance(outputs_plus, torch.Tensor):
                outputs_plus = [outputs_plus]
            
            original_flat[j] -= 2 * eps
            input_tensor.data = original_flat.view(input_tensor.shape)
            outputs_minus = func(*inputs)
            if isinstance(outputs_minus, torch.Tensor):
                outputs_minus = [outputs_minus]
            
            # Restaurar original
            original_flat[j] += eps
            input_tensor.data = original_flat.view(input_tensor.shape)
            
            # Calcular gradientes para cada output seleccionado
            grad_sum = 0.0
            for k, (plus, minus) in enumerate(zip(outputs_plus, outputs_minus)):
                if plus.numel() > 0 and minus.numel() > 0:
                    diff = (plus - minus) / (2 * eps)
                    grad_sum += diff.sum().item()
            
            flat_grad[j] = grad_sum
        
        gradients.append(grad)
    
    return gradients

def test_backward_vs_numerical():
    """Test que compara backward CUDA con gradientes numéricos."""
    if not CUDA_AVAILABLE or not torch.cuda.is_available():
        print("CUDA no disponible, omitiendo test")
        return False
    
    print("=" * 70)
    print("TEST: Backward CUDA vs Gradientes Numéricos")
    print("=" * 70)
    
    # Parámetros de test
    batch_size = 3
    dim = 6
    rank = 4
    device = torch.device('cuda')
    
    # Semilla para reproducibilidad
    torch.manual_seed(42)
    
    print(f"Configuración: batch={batch_size}, dim={dim}, rank={rank}")
    
    # Crear inputs de test
    v = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    U = torch.randn(dim, rank, device=device, requires_grad=True, dtype=torch.float64)
    W = torch.randn(dim, rank, device=device, requires_grad=True, dtype=torch.float64)
    x = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    V_w = torch.randn(dim, device=device, requires_grad=True, dtype=torch.float64)
    force = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)  # Para topología Torus
    b_forget = torch.randn(dim, device=device, requires_grad=True, dtype=torch.float64)
    W_input = torch.randn(dim, dim, device=device, requires_grad=True, dtype=torch.float64)
    
    # Hiperparámetros
    plasticity = 0.1
    sing_thresh = 0.5
    sing_strength = 2.0
    topology = 1  # Torus
    R = 2.0
    r = 1.0
    
    print(f"Hiperparámetros: plasticity={plasticity}, sing_thresh={sing_thresh}, sing_strength={sing_strength}")
    print(f"Topología: {topology} (Torus), R={R}, r={r}")
    print()
    
    # Test 1: Verificar forward pass
    print("1. Verificando forward pass...")
    
    # Forward CUDA
    output_cuda = LowRankChristoffelWithFrictionFunction.apply(
        v, U, W, x, V_w, force, W_forget, b_forget, W_input,
        plasticity, sing_thresh, sing_strength, topology, R, r
    )
    
    # Forward Python (fallback)
    try:
        # Usar christoffel_fused para la parte de Christoffel
        gamma = christoffel_fused(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r)
        
        # Agregar fricción (simplificado)
        if force is not None and W_forget is not None and b_forget is not None:
            # Calcular coeficiente de fricción (simplificado)
            features = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) if topology == 1 else x
            mu = torch.sigmoid(torch.matmul(features, W_forget.t()) + b_forget) * 5.0  # FRICTION_SCALE
            friction_output = gamma + mu * v
        else:
            friction_output = gamma
        
        # Comparar
        forward_diff = torch.abs(output_cuda - friction_output).max().item()
        print(f"   Diferencia máxima forward: {forward_diff:.2e}")
        
        if forward_diff > 1e-4:
            print("   ⚠️  ADVERTENCIA: Diferencia significativa en forward")
        else:
            print("   ✅ Forward consistente")
            
    except Exception as e:
        print(f"   ⚠️  No se pudo comparar con Python: {e}")
    
    print()
    
    # Test 2: Comparar backward
    print("2. Verificando backward pass...")
    
    # Crear función de pérdida simple
    loss = output_cuda.sum()
    
    # Calcular gradientes con backward CUDA
    grad_output = torch.ones_like(output_cuda)
    grads_cuda = LowRankChristoffelWithFrictionFunction.backward(None, grad_output)
    
    # Función para gradientes numéricos
    def forward_func(v, U, W, x, V_w, force, W_forget, b_forget, W_input):
        return LowRankChristoffelWithFrictionFunction.apply(
            v, U, W, x, V_w, force, W_forget, b_forget, W_input,
            plasticity, sing_thresh, sing_strength, topology, R, r
        )
    
    inputs = [v, U, W, x, V_w, force, W_forget, b_forget, W_input]
    grads_numerical = compute_numerical_gradients(forward_func, inputs)
    
    # Comparar gradientes
    param_names = ['v', 'U', 'W', 'x', 'V_w', 'force', 'W_forget', 'b_forget', 'W_input']
    tolerances = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]  # Tolerancias por parámetro
    
    print("   Comparación de gradientes:")
    max_errors = []
    
    for i, (name, grad_cuda, grad_num, tol) in enumerate(zip(param_names, grads_cuda, grads_numerical, tolerances)):
        if grad_cuda is None or grad_num is None:
            print(f"   {name}: Omitido (None)")
            continue
            
        if grad_cuda.numel() == 0 or grad_num.numel() == 0:
            print(f"   {name}: Omitido (vacío)")
            continue
        
        # Verificar formas
        if grad_cuda.shape != grad_num.shape:
            print(f"   {name}: ❌ Forma inconsistente - CUDA: {grad_cuda.shape}, Numérico: {grad_num.shape}")
            continue
        
        # Calcular errores
        abs_diff = torch.abs(grad_cuda - grad_num)
        rel_diff = abs_diff / (torch.abs(grad_num) + 1e-8)
        
        max_abs_error = abs_diff.max().item()
        max_rel_error = rel_diff.max().item()
        mean_abs_error = abs_diff.mean().item()
        mean_rel_error = rel_diff.mean().item()
        
        max_errors.append(max_abs_error)
        
        print(f"   {name}:")
        print(f"     Error absoluto máximo: {max_abs_error:.2e}")
        print(f"     Error relativo máximo: {max_rel_error:.2e}")
        print(f"     Error absoluto medio: {mean_abs_error:.2e}")
        print(f"     Error relativo medio: {mean_rel_error:.2e}")
        
        # Verificar si está dentro de tolerancia
        if max_abs_error <= tol and max_rel_error <= 10 * tol:
            print(f"     ✅ DENTRO DE TOLERANCIA ({tol})")
        else:
            print(f"     ❌ FUERA DE TOLERANCIA ({tol})")
            
            # Mostrar algunos valores problemáticos
            if max_abs_error > tol:
                max_idx = torch.argmax(abs_diff)
                cuda_val = grad_cuda.view(-1)[max_idx].item()
                num_val = grad_num.view(-1)[max_idx].item()
                print(f"     Valor CUDA en índice {max_idx}: {cuda_val:.6e}")
                print(f"     Valor numérico en índice {max_idx}: {num_val:.6e}")
        
        print()
    
    # Resumen
    print("3. Resumen:")
    if max_errors:
        overall_max_error = max(max_errors)
        print(f"   Error máximo global: {overall_max_error:.2e}")
        
        if overall_max_error < 1e-3:
            print("   ✅ TEST APROBADO: Los gradientes CUDA coinciden con los numéricos")
            return True
        else:
            print("   ❌ TEST FALLIDO: Diferencias significativas detectadas")
            return False
    else:
        print("   ⚠️  No se pudieron comparar gradientes")
        return False

def test_gradient_checking_pytorch():
    """Test usando gradcheck de PyTorch."""
    if not CUDA_AVAILABLE or not torch.cuda.is_available():
        print("CUDA no disponible, omitiendo gradcheck")
        return
    
    print("\n" + "=" * 70)
    print("TEST: Gradient Checking con PyTorch")
    print("=" * 70)
    
    from torch.autograd import gradcheck
    
    # Crear inputs pequeños para gradcheck
    batch_size = 2
    dim = 4
    rank = 2
    device = torch.device('cuda')
    
    torch.manual_seed(123)
    
    # Crear inputs con requires_grad=True y dtype float64
    v = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    U = torch.randn(dim, rank, device=device, requires_grad=True, dtype=torch.float64)
    W = torch.randn(dim, rank, device=device, requires_grad=True, dtype=torch.float64)
    x = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    V_w = torch.randn(dim, device=device, requires_grad=True, dtype=torch.float64)
    force = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)
    b_forget = torch.randn(dim, device=device, requires_grad=True, dtype=torch.float64)
    W_input = torch.randn(dim, dim, device=device, requires_grad=True, dtype=torch.float64)
    
    # Hiperparámetros
    test_input = (v, U, W, x, V_w, force, W_forget, b_forget, W_input, 
                 0.1, 0.5, 2.0, 1, 2.0, 1.0)
    
    try:
        result = gradcheck(
            LowRankChristoffelWithFrictionFunction.apply,
            test_input,
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=False
        )
        
        if result:
            print("✅ Gradcheck aprobado!")
        else:
            print("❌ Gradcheck fallido!")
            
    except Exception as e:
        print(f"Error en gradcheck: {e}")
        print("Esto puede ser esperado si la función no es suficientemente diferenciable")

if __name__ == "__main__":
    print("INICIANDO TESTS DE VERIFICACIÓN CUDA")
    print("Objetivo: Verificar que el nuevo backward CUDA coincide con gradientes numéricos")
    print()
    
    try:
        # Test principal
        success = test_backward_vs_numerical()
        
        # Test adicional con gradcheck
        test_gradient_checking_pytorch()
        
        print("\n" + "=" * 70)
        if success:
            print("✅ TESTS COMPLETADOS: El backward CUDA es consistente")
        else:
            print("⚠️  TESTS COMPLETADOS: Se detectaron diferencias - revisar implementación")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error durante los tests: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)