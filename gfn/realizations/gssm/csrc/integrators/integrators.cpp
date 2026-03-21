#include <torch/extension.h>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------------------------------------------------------------
// Pure ATen Implementation of the Integrators Loop
// -------------------------------------------------------------
// We build this in standard C++ using ATen so it runs in a single
// Python call but is compiled by MSVC, averting NVCC OOM bugs.
// It performs `steps` loop using the exact GFN LowRank geometry.

// Helper to compute Christoffel Gamma
torch::Tensor _compute_gamma(
    const torch::Tensor& v, 
    const torch::Tensor& U, 
    const torch::Tensor& W,
    double clamp_val, 
    bool enable_trace_norm, 
    bool is_paper_version) 
{
    auto v_r = torch::matmul(v.unsqueeze(-2), U).squeeze(-2); // [..., R]
    torch::Tensor sq;
    if (is_paper_version) {
        auto vr_norm = torch::norm(v_r, 2, -1, true);
        sq = v_r.pow(2) / (1.0 + vr_norm);
    } else {
        sq = v_r.pow(2);
    }
    
    auto gamma = torch::matmul(sq.unsqueeze(-2), W.transpose(-1, -2)).squeeze(-2); // [..., D]
    
    if (enable_trace_norm) {
        auto mean_g = gamma.mean(-1, /*keepdim=*/true);
        gamma = gamma - mean_g;
    }
    
    return clamp_val * torch::tanh(gamma / clamp_val);
}

// Helper for velocity saturation (soft-clamp)
torch::Tensor _clamp_velocity(const torch::Tensor& v, double v_sat) {
    if (v_sat > 0) {
        return v_sat * torch::tanh(v / v_sat);
    }
    return v;
}

// Yoshida 4th order coefficients
const double w1 = 1.3512071919596576;
const double w0 = -1.7024143839193153;
const double y_c1 = w1 / 2.0;
const double y_c2 = (w0 + w1) / 2.0;
const double y_c3 = y_c2;
const double y_c4 = y_c1;
const double y_d1 = w1;
const double y_d2 = w0;
const double y_d3 = w1;

// Helper for Gated Friction (Active Inference)
torch::Tensor _compute_mu(
    const torch::Tensor& x,
    const torch::Tensor& v,
    const torch::Tensor& gate_w,
    const torch::Tensor& gate_b,
    double base_friction,
    double vel_fric_scale)
{
    const double eps = 1e-8;
    const double D = x.size(-1);
    
    // mu_base = base_friction
    torch::Tensor mu = torch::full_like(x.select(-1, 0).unsqueeze(-1), base_friction);

    // If gate weigths are provided, calculate learnable friction component
    if (gate_w.numel() > 0) {
        torch::Tensor feat;
        // Check if we need Torus features [sin, cos] (gate_w dim will be 2*D)
        if (gate_w.size(1) == 2 * D) {
            feat = torch::cat({torch::sin(x), torch::cos(x)}, -1); // [..., 2D]
        } else {
            feat = x; // Euclidean / Flat
        }
        
        // Linear gate: sigmoid(feat @ w + b)
        auto gate_out = torch::matmul(feat.unsqueeze(-2), gate_w).squeeze(-2); // [B, H, 1]
        if (gate_b.numel() > 0) {
            gate_out = gate_out + gate_b;
        }
        mu = mu + torch::sigmoid(gate_out);
    }

    // Velocity-dependent scaling: mu * (1 + scale * ||v||)
    auto v_norm = torch::norm(v, 2, -1, true) / (std::sqrt(D) + eps);
    mu = mu * (1.0 + vel_fric_scale * v_norm);
    
    return mu;
}

// Helper for Singularity Damping
torch::Tensor _apply_singularity_damping(
    const torch::Tensor& acc,
    const torch::Tensor& v,
    const torch::Tensor& U,
    double sing_thresh,
    double sing_strength)
{
    if (sing_strength <= 1.0 || sing_thresh <= 0.0) return acc;

    // Detect singularity: metrics are low near singular points.
    // In LowRank, g_diag = sum(U^2).
    auto g_diag = (U.pow(2)).sum(-1); // [H, D]
    
    // Potential = sigmoid(5.0 * (g - thresh))
    auto soft_mask = torch::sigmoid(5.0 * (g_diag - sing_thresh));
    
    // Scale acceleration by soft_mask (Damping Shield)
    // Near singularity: soft_mask -> 0, damping the forces.
    return acc * soft_mask;
}

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
    auto x = x_init.clone();
    auto v = v_init.clone();
    
    const double eps = 1e-8;

    for (int i = 0; i < steps; ++i) {
        // Sub-step 1
        x = x + y_c1 * dt * v;
        x = torch::remainder(x + M_PI, 2 * M_PI) - M_PI; // Toroidal resolve
        
        auto gamma1 = _compute_gamma(v, U, W, clamp_val, enable_trace_norm, is_paper_version);
        auto a1_nf = force - gamma1; 
        a1_nf = _apply_singularity_damping(a1_nf, v, U, sing_thresh, sing_strength);
        
        auto mu1 = _compute_mu(x, v, gate_w, gate_b, friction, vel_fric_scale);
        
        v = (v + y_d1 * dt * a1_nf) / (1.0 + y_d1 * dt * mu1 + eps);
        v = _clamp_velocity(v, vel_sat);

        // Sub-step 2
        x = x + y_c2 * dt * v;
        x = torch::remainder(x + M_PI, 2 * M_PI) - M_PI;
        
        auto gamma2 = _compute_gamma(v, U, W, clamp_val, enable_trace_norm, is_paper_version);
        auto a2_nf = force - gamma2;
        a2_nf = _apply_singularity_damping(a2_nf, v, U, sing_thresh, sing_strength);
        
        auto mu2 = _compute_mu(x, v, gate_w, gate_b, friction, vel_fric_scale);
        
        v = (v + y_d2 * dt * a2_nf) / (1.0 + y_d2 * dt * mu2 + eps);
        v = _clamp_velocity(v, vel_sat);

        // Sub-step 3
        x = x + y_c3 * dt * v;
        x = torch::remainder(x + M_PI, 2 * M_PI) - M_PI;
        
        auto gamma3 = _compute_gamma(v, U, W, clamp_val, enable_trace_norm, is_paper_version);
        auto a3_nf = force - gamma3;
        a3_nf = _apply_singularity_damping(a3_nf, v, U, sing_thresh, sing_strength);
        
        auto mu3 = _compute_mu(x, v, gate_w, gate_b, friction, vel_fric_scale);
        
        v = (v + y_d3 * dt * a3_nf) / (1.0 + y_d3 * dt * mu3 + eps);
        v = _clamp_velocity(v, vel_sat);

        // Final drift
        x = x + y_c4 * dt * v;
        x = torch::remainder(x + M_PI, 2 * M_PI) - M_PI;
    }
    
    return {x, v};
}

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
    auto x = x_init.clone();
    auto v = v_init.clone();
    
    const double eps = 1e-8;

    for (int i = 0; i < steps; ++i) {
        // Half-kick 1
        auto gamma1 = _compute_gamma(v, U, W, clamp_val, enable_trace_norm, is_paper_version);
        auto a1_nf = force - gamma1;
        a1_nf = _apply_singularity_damping(a1_nf, v, U, sing_thresh, sing_strength);
        
        auto mu1 = _compute_mu(x, v, gate_w, gate_b, friction, vel_fric_scale);
        
        auto v_half = (v + 0.5 * dt * a1_nf) / (1.0 + 0.5 * dt * mu1 + eps);
        v_half = _clamp_velocity(v_half, vel_sat);
        
        // Drift
        x = x + dt * v_half;
        x = torch::remainder(x + M_PI, 2 * M_PI) - M_PI;

        // Half-kick 2
        auto gamma2 = _compute_gamma(v_half, U, W, clamp_val, enable_trace_norm, is_paper_version);
        auto a2_nf = force - gamma2;
        a2_nf = _apply_singularity_damping(a2_nf, v_half, U, sing_thresh, sing_strength);
        
        auto mu2 = _compute_mu(x, v_half, gate_w, gate_b, friction, vel_fric_scale);
        
        auto a_avg = (a1_nf + a2_nf) / 2.0;
        auto mu_avg = (mu1 + mu2) / 2.0;
        
        v = (v + dt * a_avg) / (1.0 + dt * mu_avg + eps);
        v = _clamp_velocity(v, vel_sat);
    }
    
    return {x, v};
}
