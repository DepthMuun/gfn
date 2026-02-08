#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions

// Geometry kernels
torch::Tensor lowrank_christoffel_fused(
    torch::Tensor v, torch::Tensor U, torch::Tensor W,
    torch::Tensor x, torch::Tensor V_w,
    float plasticity, float sing_thresh, float sing_strength,
    int topology, float R, float r
);

std::vector<torch::Tensor> christoffel_backward_cuda(
    torch::Tensor grad_out, torch::Tensor gamma, torch::Tensor v, torch::Tensor U, torch::Tensor W,
    torch::Tensor x, torch::Tensor V_w,
    double plasticity, double sing_thresh, double sing_strength,
    int topology, double R, double r
);

torch::Tensor lowrank_christoffel_with_friction(
    torch::Tensor v, torch::Tensor U, torch::Tensor W,
    torch::Tensor x, torch::Tensor V_w, torch::Tensor force,
    torch::Tensor W_forget, torch::Tensor b_forget, torch::Tensor W_input,
    float plasticity, float sing_thresh, float sing_strength,
    int topology, float R, float r
);

std::vector<torch::Tensor> lowrank_christoffel_friction_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor output,
    torch::Tensor v,
    torch::Tensor U,
    torch::Tensor W,
    torch::Tensor x,
    torch::Tensor V_w,
    torch::Tensor force,
    torch::Tensor W_forget,
    torch::Tensor b_forget,
    torch::Tensor W_input,
    double plasticity,
    double sing_thresh,
    double sing_strength,
    int topology,
    double R,
    double r
);

// Integrator kernels
std::vector<torch::Tensor> leapfrog_fused(
    torch::Tensor x, torch::Tensor v, torch::Tensor force,
    torch::Tensor U, torch::Tensor W,
    float dt, float dt_scale, int steps, int topology,
    torch::Tensor W_forget, torch::Tensor b_forget,
    float plasticity, float R, float r,
    
    torch::Tensor hysteresis_state,
    torch::Tensor hyst_update_w,
    torch::Tensor hyst_update_b,
    torch::Tensor hyst_readout_w,
    torch::Tensor hyst_readout_b,
    float hyst_decay,
    bool hyst_enabled
);

std::vector<torch::Tensor> heun_fused(
    torch::Tensor x, torch::Tensor v, torch::Tensor force,
    torch::Tensor U, torch::Tensor W,
    float dt, float dt_scale, int steps, int topology,
    torch::Tensor W_forget, torch::Tensor b_forget,
    float R, float r
);

std::vector<torch::Tensor> leapfrog_backward_cuda(
    torch::Tensor grad_x_out, torch::Tensor grad_v_out,
    torch::Tensor x_in, torch::Tensor v_in, torch::Tensor force,
    torch::Tensor U, torch::Tensor W, torch::Tensor W_forget, torch::Tensor b_forget,
    float dt, float dt_scale, int steps, int topology,
    float plasticity, float R, float r,
    // AUDIT FIX (Component 7): Hysteresis parameters
    torch::Tensor hysteresis_state_in,
    torch::Tensor hyst_update_w,
    torch::Tensor hyst_update_b,
    torch::Tensor hyst_readout_w,
    torch::Tensor hyst_readout_b,
    float hyst_decay,
    bool hyst_enabled
);

std::vector<torch::Tensor> heun_backward_cuda(
    torch::Tensor grad_x_out, torch::Tensor grad_v_out,
    torch::Tensor x_in, torch::Tensor v_in, torch::Tensor force,
    torch::Tensor U, torch::Tensor W,
    float dt, float dt_scale, int steps, int topology,
    float R, float r
);

std::vector<torch::Tensor> recurrent_manifold_fused(
    torch::Tensor x, torch::Tensor v, torch::Tensor forces,
    torch::Tensor U_stack, torch::Tensor W_stack,
    double dt, double dt_scale, int64_t num_heads
);

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GFN CUDA Kernels - High-performance manifold geometry and integration";
    
    // Geometry kernels
    m.def("lowrank_christoffel_fused", &lowrank_christoffel_fused,
          "Low-rank Christoffel symbol computation (CUDA)",
          py::arg("v"), py::arg("U"), py::arg("W"),
          py::arg("x"), py::arg("V_w"),
          py::arg("plasticity"), py::arg("sing_thresh"), py::arg("sing_strength"),
          py::arg("topology"), py::arg("R"), py::arg("r"));
    
    m.def("christoffel_backward_fused", &christoffel_backward_cuda,
          "Analytical backward pass for Christoffel symbols (CUDA)",
          py::arg("grad_out"), py::arg("gamma"), py::arg("v"), py::arg("U"), py::arg("W"),
          py::arg("x"), py::arg("V_w"),
          py::arg("plasticity"), py::arg("sing_thresh"), py::arg("sing_strength"),
          py::arg("topology"), py::arg("R"), py::arg("r"));
    
    m.def("lowrank_christoffel_with_friction", &lowrank_christoffel_with_friction,
          "Low-rank Christoffel with friction (CUDA)",
          py::arg("v"), py::arg("U"), py::arg("W"),
          py::arg("x"), py::arg("V_w"), py::arg("force"),
          py::arg("W_forget"), py::arg("b_forget"), py::arg("W_input"),
          py::arg("plasticity"), py::arg("sing_thresh"), py::arg("sing_strength"),
          py::arg("topology"), py::arg("R"), py::arg("r"));

    m.def("lowrank_christoffel_friction_backward", &lowrank_christoffel_friction_backward_cuda,
          "Analytical backward pass for Christoffel with friction (CUDA)",
          py::arg("grad_out"), py::arg("output"), py::arg("v"), py::arg("U"), py::arg("W"),
          py::arg("x"), py::arg("V_w"), py::arg("force"),
          py::arg("W_forget"), py::arg("b_forget"), py::arg("W_input"),
          py::arg("plasticity"), py::arg("sing_thresh"), py::arg("sing_strength"),
          py::arg("topology"), py::arg("R"), py::arg("r"));
    
    // Integrator kernels
    m.def("leapfrog_fused", &leapfrog_fused,
          "Leapfrog symplectic integrator (CUDA)",
          py::arg("x"), py::arg("v"), py::arg("force"),
          py::arg("U"), py::arg("W"),
          py::arg("dt"), py::arg("dt_scale"), py::arg("steps"), py::arg("topology"),
          py::arg("W_forget"), py::arg("b_forget"),
          py::arg("plasticity"), py::arg("R"), py::arg("r"),
          
          py::arg("hysteresis_state"),
          py::arg("hyst_update_w"),
          py::arg("hyst_update_b"),
          py::arg("hyst_readout_w"),
          py::arg("hyst_readout_b"),
          py::arg("hyst_decay"),
          py::arg("hyst_enabled"));
    
    m.def("heun_fused", &heun_fused,
          "Heun (RK2) integrator (CUDA)",
          py::arg("x"), py::arg("v"), py::arg("force"),
          py::arg("U"), py::arg("W"),
          py::arg("dt"), py::arg("dt_scale"), py::arg("steps"), py::arg("topology"),
          py::arg("W_forget"), py::arg("b_forget"),
          py::arg("R"), py::arg("r"));

    m.def("leapfrog_backward_fused", &leapfrog_backward_cuda,
          "Analytical backward pass for Leapfrog integrator (CUDA)",
          py::arg("grad_x_out"), py::arg("grad_v_out"),
          py::arg("x_in"), py::arg("v_in"), py::arg("force"),
          py::arg("U"), py::arg("W"), py::arg("W_forget"), py::arg("b_forget"),
          py::arg("dt"), py::arg("dt_scale"), py::arg("steps"), py::arg("topology"),
          py::arg("plasticity"), py::arg("R"), py::arg("r"),
          // AUDIT FIX (Component 7): Hysteresis parameters
          py::arg("hysteresis_state_in"),
          py::arg("hyst_update_w"),
          py::arg("hyst_update_b"),
          py::arg("hyst_readout_w"),
          py::arg("hyst_readout_b"),
          py::arg("hyst_decay"),
          py::arg("hyst_enabled"));

    m.def("heun_backward_fused", &heun_backward_cuda,
          "Analytical backward pass for Heun (RK2) integrator (CUDA)",
          py::arg("grad_x_out"), py::arg("grad_v_out"),
          py::arg("x_in"), py::arg("v_in"), py::arg("force"),
          py::arg("U"), py::arg("W"),
          py::arg("dt"), py::arg("dt_scale"), py::arg("steps"), py::arg("topology"),
          py::arg("R"), py::arg("r"));

    m.def("recurrent_manifold_fused", &recurrent_manifold_fused,
          "Recurrent manifold fused sequence step (CUDA)",
          py::arg("x"), py::arg("v"), py::arg("forces"),
          py::arg("U_stack"), py::arg("W_stack"),
          py::arg("dt"), py::arg("dt_scale"), py::arg("num_heads"));
}
