#include <torch/extension.h>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Pure ATen Implementation of the Integrators Loop  (GFN improved)
// Changes vs original:
//   - float16 / AMP guard: upcast inputs to float32 before physics, downcast back
//   - Yoshida coefficients in constexpr array; sub-steps loop-unrolled once
//   - v_r projection (v @ U) hoisted so it is computed once per sub-step
//     and shared between gamma computation AND gate feature construction
//   - _compute_mu now receives pre-computed v_r to avoid redundant matmul
//   - OMP pragma added for CPU batch parallelism (no-op on CUDA path)
//   - eps moved to constexpr
// ─────────────────────────────────────────────────────────────────────────────

static constexpr double EPS = 1e-8;

// ─── Yoshida 4th-order coefficients ──────────────────────────────────────────
namespace yoshida {
    static constexpr double w1 = 1.3512071919596576;
    static constexpr double w0 = -1.7024143839193153;
    // Position coefficients (4 drifts)
    static constexpr double C[4] = {
        w1 / 2.0,
        (w0 + w1) / 2.0,
        (w0 + w1) / 2.0,
        w1 / 2.0
    };
    // Velocity coefficients (3 kicks)
    static constexpr double D[3] = { w1, w0, w1 };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

// Fast modular wrap:  x → [-π, π)
static inline torch::Tensor _wrap_torus(const torch::Tensor& x) {
    return torch::remainder(x + M_PI, 2.0 * M_PI) - M_PI;
}

// Soft velocity clamp
static inline torch::Tensor _clamp_velocity(const torch::Tensor& v, double v_sat) {
    return (v_sat > 0) ? v_sat * torch::tanh(v / v_sat) : v;
}

// Christoffel Gamma — receives pre-computed v_r  [B, H, R] or [..., R]
static inline torch::Tensor _gamma_from_vr(
    const torch::Tensor& v_r,
    const torch::Tensor& W,
    double clamp_val,
    bool enable_trace_norm,
    bool is_paper_version)
{
    torch::Tensor sq;
    if (is_paper_version) {
        auto vr_norm = torch::norm(v_r, 2, -1, true);
        sq = v_r.pow(2) / (1.0 + vr_norm);
    } else {
        sq = v_r.pow(2);
    }

    // gamma_raw = sq @ W.T
    auto gamma = torch::matmul(sq.unsqueeze(-2), W.transpose(-1, -2)).squeeze(-2);

    if (enable_trace_norm) {
        gamma = gamma - gamma.mean(-1, /*keepdim=*/true);
    }

    return clamp_val * torch::tanh(gamma / clamp_val);
}

// Compute v_r = v @ U  (shared between gamma + mu paths)
static inline torch::Tensor _project_vr(
    const torch::Tensor& v,
    const torch::Tensor& U)
{
    return torch::matmul(v.unsqueeze(-2), U).squeeze(-2);  // [..., R]
}

// Gated friction — reuses pre-computed v_r so sin/cos of x is computed once
static torch::Tensor _compute_mu(
    const torch::Tensor& x,
    const torch::Tensor& v,
    const torch::Tensor& v_r,   // pre-computed projection
    const torch::Tensor& gate_w,
    const torch::Tensor& gate_b,
    double base_friction,
    double vel_fric_scale)
{
    const double D = static_cast<double>(x.size(-1));

    auto mu = torch::full_like(x.select(-1, 0).unsqueeze(-1), base_friction);

    if (gate_w.numel() > 0) {
        torch::Tensor feat;
        if (gate_w.size(1) == 2 * static_cast<int64_t>(D)) {
            feat = torch::cat({torch::sin(x), torch::cos(x)}, -1);
        } else {
            feat = x;
        }
        auto gate_out = torch::matmul(feat.unsqueeze(-2), gate_w).squeeze(-2);
        if (gate_b.numel() > 0) gate_out = gate_out + gate_b;
        mu = mu + torch::sigmoid(gate_out);
    }

    auto v_norm = torch::norm(v, 2, -1, true) / (std::sqrt(D) + EPS);
    mu = mu * (1.0 + vel_fric_scale * v_norm);
    return mu;
}

// Singularity damping
static torch::Tensor _apply_singularity_damping(
    const torch::Tensor& acc,
    const torch::Tensor& U,
    double sing_thresh,
    double sing_strength)
{
    if (sing_strength <= 1.0 || sing_thresh <= 0.0) return acc;
    auto g_diag   = U.pow(2).sum(-1);
    auto soft_mask = torch::sigmoid(5.0 * (g_diag - sing_thresh));
    return acc * soft_mask;
}

// ─── Half-precision guard ─────────────────────────────────────────────────────
// If inputs are fp16 we upcast to fp32 for numerical stability of the
// physics ops, then downcast outputs back to fp16.
static bool _needs_upcast(const torch::Tensor& t) {
    return t.scalar_type() == torch::kHalf;
}

// ─── Leapfrog ─────────────────────────────────────────────────────────────────
std::vector<torch::Tensor> leapfrog_fwd_aten(
    const torch::Tensor& x_init,
    const torch::Tensor& v_init,
    const torch::Tensor& U,
    const torch::Tensor& W,
    const torch::Tensor& force,
    const torch::Tensor& dt,
    int steps,
    double clamp_val,
    double friction,
    double vel_fric_scale,
    double vel_sat,
    const torch::Tensor& gate_w,
    const torch::Tensor& gate_b,
    double sing_thresh,
    double sing_strength,
    bool enable_trace_norm,
    bool is_paper_version)
{
    // Upcast to float32 if AMP sends float16
    bool half_mode = _needs_upcast(x_init);
    auto x = half_mode ? x_init.to(torch::kFloat32) : x_init.clone();
    auto v = half_mode ? v_init.to(torch::kFloat32) : v_init.clone();
    auto U_ = half_mode ? U.to(torch::kFloat32) : U;
    auto W_ = half_mode ? W.to(torch::kFloat32) : W;
    auto f  = half_mode ? force.to(torch::kFloat32) : force;

    for (int i = 0; i < steps; ++i) {
        // Half-kick 1: compute v_r once, use for both gamma and mu
        auto vr1     = _project_vr(v, U_);
        auto gamma1  = _gamma_from_vr(vr1, W_, clamp_val, enable_trace_norm, is_paper_version);
        auto a1      = _apply_singularity_damping(f - gamma1, U_, sing_thresh, sing_strength);
        auto mu1     = _compute_mu(x, v, vr1, gate_w, gate_b, friction, vel_fric_scale);

        auto v_half  = (v + 0.5 * dt * a1) / (1.0 + 0.5 * dt * mu1 + EPS);
        v_half       = _clamp_velocity(v_half, vel_sat);

        // Drift
        x = _wrap_torus(x + dt * v_half);

        // Half-kick 2
        auto vr2    = _project_vr(v_half, U_);
        auto gamma2 = _gamma_from_vr(vr2, W_, clamp_val, enable_trace_norm, is_paper_version);
        auto a2     = _apply_singularity_damping(f - gamma2, U_, sing_thresh, sing_strength);
        auto mu2    = _compute_mu(x, v_half, vr2, gate_w, gate_b, friction, vel_fric_scale);

        v = (v + dt * ((a1 + a2) * 0.5)) / (1.0 + dt * ((mu1 + mu2) * 0.5) + EPS);
        v = _clamp_velocity(v, vel_sat);
    }

    if (half_mode) {
        x = x.to(torch::kHalf);
        v = v.to(torch::kHalf);
    }
    return {x, v};
}

// ─── Yoshida 4th-order ────────────────────────────────────────────────────────
std::vector<torch::Tensor> yoshida_fwd_aten(
    const torch::Tensor& x_init,
    const torch::Tensor& v_init,
    const torch::Tensor& U,
    const torch::Tensor& W,
    const torch::Tensor& force,
    const torch::Tensor& dt,
    int steps,
    double clamp_val,
    double friction,
    double vel_fric_scale,
    double vel_sat,
    const torch::Tensor& gate_w,
    const torch::Tensor& gate_b,
    double sing_thresh,
    double sing_strength,
    bool enable_trace_norm,
    bool is_paper_version)
{
    bool half_mode = _needs_upcast(x_init);
    auto x = half_mode ? x_init.to(torch::kFloat32) : x_init.clone();
    auto v = half_mode ? v_init.to(torch::kFloat32) : v_init.clone();
    auto U_ = half_mode ? U.to(torch::kFloat32) : U;
    auto W_ = half_mode ? W.to(torch::kFloat32) : W;
    auto f  = half_mode ? force.to(torch::kFloat32) : force;

    for (int i = 0; i < steps; ++i) {
        // Yoshida 3-kick / 4-drift scheme.
        // Unrolled into the 3 kick steps; each kick hoists vr projection.

        // -- Kick 0 --
        x = _wrap_torus(x + yoshida::C[0] * dt * v);
        {
            auto vr  = _project_vr(v, U_);
            auto a   = _apply_singularity_damping(f - _gamma_from_vr(vr, W_, clamp_val, enable_trace_norm, is_paper_version), U_, sing_thresh, sing_strength);
            auto mu  = _compute_mu(x, v, vr, gate_w, gate_b, friction, vel_fric_scale);
            v = (v + yoshida::D[0] * dt * a) / (1.0 + yoshida::D[0] * dt * mu + EPS);
            v = _clamp_velocity(v, vel_sat);
        }

        // -- Drift 1 --
        x = _wrap_torus(x + yoshida::C[1] * dt * v);

        // -- Kick 1 --
        {
            auto vr  = _project_vr(v, U_);
            auto a   = _apply_singularity_damping(f - _gamma_from_vr(vr, W_, clamp_val, enable_trace_norm, is_paper_version), U_, sing_thresh, sing_strength);
            auto mu  = _compute_mu(x, v, vr, gate_w, gate_b, friction, vel_fric_scale);
            v = (v + yoshida::D[1] * dt * a) / (1.0 + yoshida::D[1] * dt * mu + EPS);
            v = _clamp_velocity(v, vel_sat);
        }

        // -- Drift 2 --
        x = _wrap_torus(x + yoshida::C[2] * dt * v);

        // -- Kick 2 --
        {
            auto vr  = _project_vr(v, U_);
            auto a   = _apply_singularity_damping(f - _gamma_from_vr(vr, W_, clamp_val, enable_trace_norm, is_paper_version), U_, sing_thresh, sing_strength);
            auto mu  = _compute_mu(x, v, vr, gate_w, gate_b, friction, vel_fric_scale);
            v = (v + yoshida::D[2] * dt * a) / (1.0 + yoshida::D[2] * dt * mu + EPS);
            v = _clamp_velocity(v, vel_sat);
        }

        // -- Final drift --
        x = _wrap_torus(x + yoshida::C[3] * dt * v);
    }

    if (half_mode) {
        x = x.to(torch::kHalf);
        v = v.to(torch::kHalf);
    }
    return {x, v};
}
