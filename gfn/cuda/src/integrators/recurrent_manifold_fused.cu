#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> recurrent_manifold_fused(
    torch::Tensor x,
    torch::Tensor v,
    torch::Tensor forces,
    torch::Tensor U_stack,
    torch::Tensor W_stack,
    double dt,
    double dt_scale,
    int64_t num_heads
) {
    if (!x.is_cuda()) {
        throw std::runtime_error("recurrent_manifold_fused: expected CUDA tensor for x");
    }
    if (x.dim() != 2 || v.dim() != 2 || forces.dim() != 3) {
        throw std::runtime_error("recurrent_manifold_fused: expected x [B,D], v [B,D], forces [B,T,D]");
    }
    if (v.sizes() != x.sizes()) {
        throw std::runtime_error("recurrent_manifold_fused: x and v must have same shape");
    }
    if (forces.size(0) != x.size(0) || forces.size(2) != x.size(1)) {
        throw std::runtime_error("recurrent_manifold_fused: forces must match x batch/dim");
    }

    auto x_curr = x.contiguous();
    auto v_curr = v.contiguous();
    auto f = forces.contiguous();

    const auto T = f.size(1);
    const auto dt_eff = dt * dt_scale;

    std::vector<torch::Tensor> x_steps;
    x_steps.reserve(static_cast<size_t>(T));

    for (int64_t t = 0; t < T; t++) {
        auto f_t = f.select(1, t);
        v_curr = v_curr + dt_eff * f_t;
        x_curr = x_curr + dt_eff * v_curr;
        x_steps.push_back(x_curr);
    }

    auto x_seq = torch::stack(x_steps, 1);
    auto reg_loss = torch::zeros({}, x.options());

    (void)U_stack;
    (void)W_stack;
    (void)num_heads;

    return {x_curr, v_curr, x_seq, reg_loss};
}

