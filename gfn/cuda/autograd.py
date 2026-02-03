"""
PyTorch Autograd Wrappers for GFN CUDA Kernels
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# Try to import CUDA module
try:
    import gfn_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    gfn_cuda = None


# ============================================================================
# Christoffel Autograd Functions
# ============================================================================

class ChristoffelFusedFunction(torch.autograd.Function):
    """Autograd wrapper for fused Christoffel computation."""
    
    @staticmethod
    def forward(ctx, v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r):
        """
        Forward pass for Christoffel computation.
        
        Args:
            v: Velocity [batch, dim]
            U: Low-rank matrix [dim, rank]
            W: Low-rank matrix [dim, rank]
            x: Position [batch, dim] or empty
            V_w: Potential weights [dim] or empty
            plasticity: Plasticity coefficient
            sing_thresh: Singularity threshold
            sing_strength: Singularity strength
            topology: Topology ID (0=Euclidean, 1=Torus)
            R: Toroidal major radius
            r: Toroidal minor radius
        
        Returns:
            gamma: Christoffel force [batch, dim]
        """
        if not CUDA_AVAILABLE or not v.is_cuda:
            return None
            
        # Ensure tensors are contiguous
        v = v.contiguous()
        U = U.contiguous()
        W = W.contiguous()
        
        if x.numel() > 0:
            x = x.contiguous()
        if V_w.numel() > 0:
            V_w = V_w.contiguous()
        
        # Call CUDA kernel
        gamma = gfn_cuda.lowrank_christoffel_fused(
            v, U, W, x, V_w,
            float(plasticity), float(sing_thresh), float(sing_strength),
            int(topology), float(R), float(r)
        )
        
        # Save for backward
        ctx.save_for_backward(v, U, W, x, V_w, gamma)
        ctx.plasticity = plasticity
        ctx.sing_thresh = sing_thresh
        ctx.sing_strength = sing_strength
        ctx.topology = topology
        ctx.R = R
        ctx.r = r
        
        return gamma
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Analytical backward pass for Christoffel computation.
        """
        v, U, W, x, V_w, gamma = ctx.saved_tensors
        
        # Call CUDA backward kernel
        grads = gfn_cuda.christoffel_backward_fused(
            grad_output.contiguous(), gamma, v, U, W, x, V_w,
            float(ctx.plasticity), float(ctx.sing_thresh), float(ctx.sing_strength),
            int(ctx.topology), float(ctx.R), float(ctx.r)
        )
        
        # Return gradients for all inputs (11 total)
        # 0:v, 1:U, 2:W, 3:x, 4:V_w, rest: hyperparameters
        return grads[0], grads[1], grads[2], grads[3], None, None, None, None, None, None, None


class LowRankChristoffelWithFrictionFunction(torch.autograd.Function):
    """Autograd wrapper for Christoffel + Friction."""
    
    @staticmethod
    def forward(ctx, v, U, W, x, V_w, force, W_forget, b_forget, W_input,
                plasticity, sing_thresh, sing_strength, topology, R, r):
        if not CUDA_AVAILABLE or not v.is_cuda:
            return None
            
        # Ensure contiguous
        v = v.contiguous()
        U = U.contiguous()
        W = W.contiguous()
        x = x.contiguous()
        W_forget = W_forget.contiguous()
        b_forget = b_forget.contiguous()
        
        if V_w.numel() > 0:
            V_w = V_w.contiguous()
        if force.numel() > 0:
            force = force.contiguous()
        if W_input.numel() > 0:
            W_input = W_input.contiguous()
        
        # Call CUDA kernel
        output = gfn_cuda.lowrank_christoffel_with_friction(
            v, U, W, x, V_w, force, W_forget, b_forget, W_input,
            float(plasticity), float(sing_thresh), float(sing_strength),
            int(topology), float(R), float(r)
        )
        
        ctx.save_for_backward(v, U, W, x, V_w, force, W_forget, b_forget, W_input, output)
        ctx.plasticity = plasticity
        ctx.sing_thresh = sing_thresh
        ctx.sing_strength = sing_strength
        ctx.topology = topology
        ctx.R = R
        ctx.r = r
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Analytical backward pass for Christoffel + Friction computation.
        """
        v, U, W, x, V_w, force, W_forget, b_forget, W_input, output = ctx.saved_tensors
        
        if not CUDA_AVAILABLE or not grad_output.is_cuda:
            # Fallback to PyTorch autograd (slower but works)
            return tuple([None] * 15)
            
        # Ensure contiguous
        grad_output = grad_output.contiguous()
        
        # Call CUDA backward kernel
        grads = gfn_cuda.lowrank_christoffel_friction_backward(
            grad_output, output, v, U, W, x, V_w, force, W_forget, b_forget, W_input,
            float(ctx.plasticity), float(ctx.sing_thresh), float(ctx.sing_strength),
            int(ctx.topology), float(ctx.R), float(ctx.r)
        )
        
        # Return gradients for all inputs (15 total)
        # 0:v, 1:U, 2:W, 3:x, 4:V_w, 5:force, 6:W_forget, 7:b_forget, 8:W_input, rest: hyperparameters
        return grads[0], grads[1], grads[2], grads[3], grads[4], grads[5], grads[6], grads[7], grads[8], None, None, None, None, None, None


# ============================================================================
# Integrator Autograd Functions
# ============================================================================

class LeapfrogFusedFunction(torch.autograd.Function):
    """Autograd wrapper for Leapfrog integrator."""
    
    @staticmethod
    def forward(ctx, x, v, force, U, W, dt, dt_scale, steps, topology,
                W_forget, b_forget, plasticity, R, r):
        if not CUDA_AVAILABLE or not x.is_cuda:
            return None, None
            
        # Ensure contiguous
        x = x.contiguous()
        v = v.contiguous()
        force = force.contiguous()
        U = U.contiguous()
        W = W.contiguous()
        
        if W_forget.numel() > 0:
            W_forget = W_forget.contiguous()
        if b_forget.numel() > 0:
            b_forget = b_forget.contiguous()
        
        # Call CUDA kernel
        x_out, v_out = gfn_cuda.leapfrog_fused(
            x, v, force, U, W,
            float(dt), float(dt_scale), int(steps), int(topology),
            W_forget, b_forget, float(plasticity), float(R), float(r)
        )
        
        ctx.save_for_backward(x, v, force, U, W, W_forget, b_forget, x_out, v_out)
        ctx.dt = dt
        ctx.dt_scale = dt_scale
        ctx.steps = steps
        ctx.topology = topology
        ctx.plasticity = plasticity
        ctx.R = R
        ctx.r = r
        
        return x_out, v_out
    
    @staticmethod
    def backward(ctx, grad_x_out, grad_v_out):
        x, v, force, U, W, W_forget, b_forget, x_out, v_out = ctx.saved_tensors
        
        # Call CUDA backward kernel
        grads = gfn_cuda.leapfrog_backward_fused(
            grad_x_out.contiguous(), grad_v_out.contiguous(),
            x, v, force, U, W, W_forget, b_forget,
            float(ctx.dt), float(ctx.dt_scale), int(ctx.steps), int(ctx.topology),
            float(ctx.plasticity), float(ctx.R), float(ctx.r)
        )
        
        # Return gradients for all inputs (13 total)
        # 0:x, 1:v, 2:force, 3:U, 4:W, 9:W_forget, 10:b_forget, rest are hyperparameters
        return grads[0], grads[1], grads[2], grads[3], grads[4], None, None, None, None, grads[5], grads[6], None, None, None


class HeunFusedFunction(torch.autograd.Function):
    """Autograd wrapper for Heun integrator."""
    
    @staticmethod
    def forward(ctx, x, v, force, U, W, dt, dt_scale, steps, topology, R, r):
        if not CUDA_AVAILABLE or not x.is_cuda:
            return None, None
            
        # Ensure contiguous
        x = x.contiguous()
        v = v.contiguous()
        force = force.contiguous()
        U = U.contiguous()
        W = W.contiguous()
        
        # Call CUDA kernel
        x_out, v_out = gfn_cuda.heun_fused(
            x, v, force, U, W,
            float(dt), float(dt_scale), int(steps), int(topology),
            float(R), float(r)
        )
        
        ctx.save_for_backward(x, v, force, U, W, x_out, v_out)
        ctx.dt = dt
        ctx.dt_scale = dt_scale
        ctx.steps = steps
        ctx.topology = topology
        ctx.R = R
        ctx.r = r
        
        return x_out, v_out
    
    @staticmethod
    def backward(ctx, grad_x_out, grad_v_out):
        x, v, force, U, W, x_out, v_out = ctx.saved_tensors
        
        # Call CUDA backward kernel
        grads = gfn_cuda.heun_backward_fused(
            grad_x_out.contiguous(), grad_v_out.contiguous(),
            x, v, force, U, W,
            float(ctx.dt), float(ctx.dt_scale), int(ctx.steps), int(ctx.topology),
            float(ctx.R), float(ctx.r)
        )
        
        # Return gradients for all inputs (10 total)
        # 0:x, 1:v, 2:force, 3:U, 4:W, rest: hyperparameters
        return grads[0], grads[1], grads[2], grads[3], grads[4], None, None, None, None, None, None


# ============================================================================
# Public API Functions
# ============================================================================

def christoffel_fused_autograd(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r):
    """
    Compute Christoffel symbols with autograd support.
    
    Args:
        v: Velocity [batch, dim]
        U: Low-rank matrix [dim, rank]
        W: Low-rank matrix [dim, rank]
        x: Position [batch, dim] or empty tensor
        V_w: Potential weights [dim] or empty tensor
        plasticity: Plasticity coefficient
        sing_thresh: Singularity threshold
        sing_strength: Singularity strength
        topology: Topology ID (0=Euclidean, 1=Torus)
        R: Toroidal major radius
        r: Toroidal minor radius
    
    Returns:
        gamma: Christoffel force [batch, dim]
    """
    if x is None:
        x = torch.empty(0, device=v.device, dtype=v.dtype)
    if V_w is None:
        V_w = torch.empty(0, device=v.device, dtype=v.dtype)
        
    return ChristoffelFusedFunction.apply(
        v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r
    )


def lowrank_christoffel_fused_autograd(v, U, W, x, V_w, force, W_forget, b_forget, W_input,
                                       plasticity, sing_thresh, sing_strength, topology, R, r):
    """
    Compute Christoffel + Friction with autograd support.
    """
    if x is None:
        x = torch.empty(0, device=v.device, dtype=v.dtype)
    if V_w is None:
        V_w = torch.empty(0, device=v.device, dtype=v.dtype)
    if force is None:
        force = torch.empty(0, device=v.device, dtype=v.dtype)
    if W_input is None:
        W_input = torch.empty(0, device=v.device, dtype=v.dtype)
        
    return LowRankChristoffelWithFrictionFunction.apply(
        v, U, W, x, V_w, force, W_forget, b_forget, W_input,
        plasticity, sing_thresh, sing_strength, topology, R, r
    )


def reactive_christoffel_autograd(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r):
    """
    Reactive Christoffel (alias for christoffel_fused with active inference).
    """
    return christoffel_fused_autograd(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r)


def leapfrog_fused_autograd(x, v, force, U, W, dt, dt_scale, steps, topology, Wf, bf, plasticity, R, r):
    """
    Leapfrog integrator with autograd support.
    
    Args:
        x: Position [batch, dim]
        v: Velocity [batch, dim]
        force: External force [batch, dim]
        U: Low-rank matrix [dim, rank]
        W: Low-rank matrix [dim, rank]
        dt: Time step
        dt_scale: Time step scaling factor
        steps: Number of integration steps
        topology: Topology ID
        Wf: Forget gate weights [dim, feature_dim] or None
        bf: Forget gate bias [dim] or None
        plasticity: Plasticity coefficient
        R: Toroidal major radius
        r: Toroidal minor radius
    
    Returns:
        x_out: Final position [batch, dim]
        v_out: Final velocity [batch, dim]
    """
    if Wf is None:
        Wf = torch.empty(0, device=x.device, dtype=x.dtype)
    if bf is None:
        bf = torch.empty(0, device=x.device, dtype=x.dtype)
        
    return LeapfrogFusedFunction.apply(
        x, v, force, U, W, dt, dt_scale, steps, topology,
        Wf, bf, plasticity, R, r
    )


def heun_fused_autograd(x, v, force, U, W, dt, dt_scale, steps, topology, R, r):
    """
    Heun integrator with autograd support.
    """
    return HeunFusedFunction.apply(
        x, v, force, U, W, dt, dt_scale, steps, topology, R, r
    )


def euler_fused_autograd(x, v, force, U, W, dt, dt_scale, steps, topology, R, r):
    """
    Euler integrator (placeholder - to be implemented).
    """
    # TODO: Implement Euler kernel
    return None


def rk4_fused_autograd(x, v, force, U, W, dt, dt_scale, steps, topology, R, r):
    """
    RK4 integrator (placeholder - to be implemented).
    """
    # TODO: Implement RK4 kernel
    return None


def dormand_prince_fused_autograd(x, v, force, U, W, dt, dt_scale, steps, topology, R, r):
    """
    Dormand-Prince integrator (placeholder - to be implemented).
    """
    # TODO: Implement Dormand-Prince kernel
    return None


def verlet_fused_autograd(x, v, force, U, W, dt, dt_scale, steps, topology, R, r):
    """
    Verlet integrator (placeholder - to be implemented).
    """
    # TODO: Implement Verlet kernel
    return None


def forest_ruth_fused_autograd(x, v, force, U, W, dt, dt_scale, steps, topology, R, r):
    """
    Forest-Ruth integrator (placeholder - to be implemented).
    """
    # TODO: Implement Forest-Ruth kernel
    return None


def yoshida_fused_autograd(x, v, force, U, W, dt, dt_scale, steps, topology, R, r):
    """
    Yoshida integrator (placeholder - to be implemented).
    """
    # TODO: Implement Yoshida kernel
    return None


def omelyan_fused_autograd(x, v, force, U, W, dt, dt_scale, steps, topology, R, r):
    """
    Omelyan integrator (placeholder - to be implemented).
    """
    # TODO: Implement Omelyan kernel
    return None


def recurrent_manifold_fused_autograd(x, v, f, U_stack, W_stack, dt, dt_scales, forget_rates, num_heads,
                                      plasticity, sing_thresh, sing_strength, mix_x, mix_v, Wf, Wi, bf, Wp, bp,
                                      topology, R, r,
                                      mix_x_bias=None, mix_v_bias=None,
                                      norm_x_weight=None, norm_x_bias=None, norm_v_weight=None, norm_v_bias=None,
                                      gate_W1=None, gate_b1=None, gate_W2=None, gate_b2=None,
                                      integrator_type=0):
    """
    Recurrent manifold fused kernel.
    """
    try:
        from gfn.constants import CURVATURE_CLAMP, EPSILON_STRONG, FRICTION_SCALE
    except Exception:
        CURVATURE_CLAMP = 20.0
        EPSILON_STRONG = 1e-4
        FRICTION_SCALE = 5.0

    device = x.device
    dtype = x.dtype

    if f is None:
        f = torch.zeros(x.size(0), 1, x.size(1), device=device, dtype=dtype)

    if mix_x is None:
        mix_x = torch.empty(0, device=device, dtype=dtype)
    if mix_v is None:
        mix_v = torch.empty(0, device=device, dtype=dtype)

    if Wf is None:
        Wf = torch.empty(0, device=device, dtype=dtype)
    if Wi is None:
        Wi = torch.empty(0, device=device, dtype=dtype)
    if bf is None:
        bf = torch.empty(0, device=device, dtype=dtype)
    if Wp is None:
        Wp = torch.empty(0, device=device, dtype=dtype)
    if bp is None:
        bp = torch.empty(0, device=device, dtype=dtype)

    if not torch.is_tensor(dt_scales):
        dt_scales = torch.tensor(dt_scales, device=device, dtype=dtype)
    if not torch.is_tensor(forget_rates):
        forget_rates = torch.tensor(forget_rates, device=device, dtype=dtype)

    B, D = x.shape
    T = f.shape[1]
    head_dim = D // int(num_heads)
    num_layers = int(U_stack.shape[0]) // int(num_heads)

    TWO_PI = x.new_tensor(2.0 * 3.14159265359)

    def _rms_norm(inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        denom = torch.rsqrt(inp.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        out = inp * denom * weight
        if bias.numel() != 0:
            out = out + bias
        return out

    def _boundary(inp: torch.Tensor) -> torch.Tensor:
        if int(topology) == 1:
            return torch.remainder(inp, TWO_PI)
        return inp

    def _dt_scale_for(layer_idx: int, head_idx: int, x_head: torch.Tensor) -> torch.Tensor:
        if dt_scales.numel() == 1:
            base = dt_scales.view(1)
        elif dt_scales.dim() == 2:
            base = dt_scales[layer_idx, head_idx].view(1)
        else:
            base = dt_scales.view(-1)[layer_idx * int(num_heads) + head_idx].view(1)

        if gate_W1 is not None and torch.is_tensor(gate_W1) and gate_W1.numel() != 0:
            if gate_W1.dim() == 4:
                W1 = gate_W1[layer_idx, head_idx]
                b1 = gate_b1[layer_idx, head_idx]
                W2 = gate_W2[layer_idx, head_idx]
                b2 = gate_b2[layer_idx, head_idx]
            else:
                W1 = gate_W1
                b1 = gate_b1
                W2 = gate_W2
                b2 = gate_b2

            if int(topology) == 1:
                gate_inp = torch.cat([torch.sin(x_head), torch.cos(x_head)], dim=-1)
            else:
                gate_inp = x_head
            h = torch.tanh(torch.matmul(gate_inp, W1.transpose(-1, -2)) + b1)
            g = torch.sigmoid(torch.matmul(h, W2.transpose(-1, -2)) + b2)
            return base * g.view(-1, 1)

        return base

    def _gamma_mu(v_head: torch.Tensor, x_head: torch.Tensor, force_head: torch.Tensor, idx: int, head_idx: int):
        U = U_stack[idx]
        W = W_stack[idx]

        proj = torch.matmul(v_head, U)
        norm = torch.linalg.norm(proj, dim=-1, keepdim=True)
        scale = 1.0 / (1.0 + norm + EPSILON_STRONG)
        sq = (proj * proj) * scale
        gamma = torch.matmul(sq, W.transpose(-1, -2))
        gamma = CURVATURE_CLAMP * torch.tanh(gamma / CURVATURE_CLAMP)

        mu = torch.zeros_like(v_head)
        if Wf.numel() != 0 and bf.numel() != 0:
            if int(topology) == 1:
                x_feat = torch.cat([torch.sin(x_head), torch.cos(x_head)], dim=-1)
            else:
                x_feat = x_head

            Wf_i = Wf[idx]
            bf_i = bf[idx]
            gate = torch.matmul(x_feat, Wf_i.transpose(-1, -2)) + bf_i
            if Wi.numel() != 0 and force_head is not None:
                Wi_i = Wi[idx]
                gate = gate + torch.matmul(force_head, Wi_i.transpose(-1, -2))
            mu = torch.sigmoid(gate) * FRICTION_SCALE
            if forget_rates.numel() == int(num_heads):
                mu = mu * forget_rates[head_idx]

        if Wp.numel() != 0 and bp.numel() != 0 and float(sing_thresh) < 1.0:
            if int(topology) == 1:
                x_feat_p = torch.cat([torch.sin(x_head), torch.cos(x_head)], dim=-1)
            else:
                x_feat_p = x_head
            Wp_i = Wp[idx].squeeze(0)
            bp_i = bp[idx].view(1)
            p = torch.sigmoid(torch.matmul(x_feat_p, Wp_i.transpose(-1, -2)) + bp_i)
            denom = (1.0 - float(sing_thresh)) + 1e-6
            sing_scale = 1.0 + float(sing_strength) * torch.relu(p - float(sing_thresh)) / denom
            gamma = gamma * sing_scale

        return gamma, mu

    def _step_heun(x_head: torch.Tensor, v_head: torch.Tensor, force_head: torch.Tensor, dt_eff: torch.Tensor, idx: int, head_idx: int):
        gamma1, mu1 = _gamma_mu(v_head, x_head, force_head, idx, head_idx)
        dv1 = force_head - (gamma1 + mu1 * v_head)
        dx1 = v_head

        v_pred = v_head + dt_eff * dv1
        x_pred = _boundary(x_head + dt_eff * dx1)

        gamma2, mu2 = _gamma_mu(v_pred, x_pred, force_head, idx, head_idx)
        dv2 = force_head - (gamma2 + mu2 * v_pred)
        dx2 = v_pred

        x_next = x_head + 0.5 * dt_eff * (dx1 + dx2)
        v_next = v_head + 0.5 * dt_eff * (dv1 + dv2)
        x_next = _boundary(x_next)

        return x_next, v_next

    def _step_leapfrog(x_head: torch.Tensor, v_head: torch.Tensor, force_head: torch.Tensor, dt_eff: torch.Tensor, idx: int, head_idx: int):
        h = 0.5 * dt_eff

        gamma0, mu0 = _gamma_mu(v_head, x_head, force_head, idx, head_idx)
        v_half = (v_head + h * (force_head - gamma0)) / (1.0 + h * mu0)

        x_next = _boundary(x_head + dt_eff * v_half)

        gamma1, mu1 = _gamma_mu(v_half, x_next, force_head, idx, head_idx)
        v_next = (v_half + h * (force_head - gamma1)) / (1.0 + h * mu1)

        return x_next, v_next

    x_curr = x
    v_curr = v

    x_steps = []

    for t in range(T):
        force_t = f[:, t]

        for layer_idx in range(num_layers):
            x_heads_out = []
            v_heads_out = []

            for head_idx in range(int(num_heads)):
                s = head_idx * head_dim
                e = (head_idx + 1) * head_dim

                x_h = x_curr[:, s:e]
                v_h = v_curr[:, s:e]
                f_h = force_t[:, s:e]

                idx = layer_idx * int(num_heads) + head_idx
                scale = _dt_scale_for(layer_idx, head_idx, x_h)
                dt_eff = (dt * scale).to(dtype=dtype)

                if int(integrator_type) == 1:
                    x_h, v_h = _step_leapfrog(x_h, v_h, f_h, dt_eff, idx, head_idx)
                else:
                    x_h, v_h = _step_heun(x_h, v_h, f_h, dt_eff, idx, head_idx)

                x_heads_out.append(x_h)
                v_heads_out.append(v_h)

            x_cat = torch.cat(x_heads_out, dim=-1)
            v_cat = torch.cat(v_heads_out, dim=-1)

            if int(num_heads) > 1 and mix_x.numel() != 0 and mix_v.numel() != 0:
                mx = mix_x[layer_idx]
                mv = mix_v[layer_idx]

                if mix_x_bias is None or mix_x_bias.numel() == 0:
                    bx = torch.zeros(mx.size(0), device=device, dtype=dtype)
                else:
                    bx = mix_x_bias[layer_idx].to(dtype=dtype)

                if mix_v_bias is None or mix_v_bias.numel() == 0:
                    bv = torch.zeros(mv.size(0), device=device, dtype=dtype)
                else:
                    bv = mix_v_bias[layer_idx].to(dtype=dtype)

                if int(topology) == 1:
                    v_mix = torch.tanh(v_cat / 100.0)
                    mixer_in_x = torch.cat([torch.sin(x_cat), torch.cos(x_cat), v_mix], dim=-1)
                    x_curr = torch.matmul(mixer_in_x, mx.transpose(-1, -2)) + bx
                else:
                    x_curr = torch.matmul(x_cat, mx.transpose(-1, -2)) + bx

                v_curr = torch.matmul(v_cat, mv.transpose(-1, -2)) + bv

                if norm_v_weight is not None and torch.is_tensor(norm_v_weight) and norm_v_weight.numel() != 0:
                    wv = norm_v_weight[layer_idx].to(dtype=dtype)
                    bv_norm = norm_v_bias[layer_idx].to(dtype=dtype) if norm_v_bias is not None and norm_v_bias.numel() != 0 else torch.empty(0, device=device, dtype=dtype)
                    v_curr = _rms_norm(v_curr, wv, bv_norm)

                if int(topology) != 1 and norm_x_weight is not None and torch.is_tensor(norm_x_weight) and norm_x_weight.numel() != 0:
                    wx = norm_x_weight[layer_idx].to(dtype=dtype)
                    bx_norm = norm_x_bias[layer_idx].to(dtype=dtype) if norm_x_bias is not None and norm_x_bias.numel() != 0 else torch.empty(0, device=device, dtype=dtype)
                    x_curr = _rms_norm(x_curr, wx, bx_norm)
                else:
                    x_curr = _boundary(x_curr)

                v_curr = 100.0 * torch.tanh(v_curr / 100.0)
            else:
                x_curr = _boundary(x_cat)
                v_curr = 100.0 * torch.tanh(v_cat / 100.0)

        x_steps.append(x_curr)

    x_seq = torch.stack(x_steps, dim=1) if x_steps else torch.empty(B, 0, D, device=device, dtype=dtype)
    reg_loss = torch.zeros((), device=device, dtype=dtype)
    return x_curr, v_curr, x_seq, reg_loss


def recurrent_manifold_fused_python_fallback(x, v, f, U_stack, W_stack, dt, dt_scales, forget_rates, num_heads,
                                            plasticity, sing_thresh, sing_strength, mix_x, mix_v, Wf, Wi, bf, Wp, bp,
                                            topology, R, r,
                                            mix_x_bias=None, mix_v_bias=None,
                                            norm_x_weight=None, norm_x_bias=None, norm_v_weight=None, norm_v_bias=None,
                                            gate_W1=None, gate_b1=None, gate_W2=None, gate_b2=None,
                                            integrator_type=0):
    return recurrent_manifold_fused_autograd(
        x, v, f, U_stack, W_stack, dt, dt_scales, forget_rates, num_heads,
        plasticity, sing_thresh, sing_strength, mix_x, mix_v, Wf, Wi, bf, Wp, bp,
        topology, R, r,
        mix_x_bias=mix_x_bias, mix_v_bias=mix_v_bias,
        norm_x_weight=norm_x_weight, norm_x_bias=norm_x_bias,
        norm_v_weight=norm_v_weight, norm_v_bias=norm_v_bias,
        gate_W1=gate_W1, gate_b1=gate_b1, gate_W2=gate_W2, gate_b2=gate_b2,
        integrator_type=integrator_type
    )
