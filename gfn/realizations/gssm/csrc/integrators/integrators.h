#include <torch/extension.h>
#include <vector>

// Forward declarations for the integrators
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
    bool is_paper_version);

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
    bool is_paper_version);
