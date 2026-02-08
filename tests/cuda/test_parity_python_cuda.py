import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.cuda import ops as ops

def summarize(name, diff, tol):
    status = "PASS" if diff <= tol else "FAIL"
    print(f"{name}: max_diff={diff:.6g} tol={tol} => {status}")
    return status

def test_christoffel(topology):
    B, D, Rnk = 16, 64, 16
    v_cpu = torch.randn(B, D, dtype=torch.float32)
    U_cpu = torch.randn(D, Rnk, dtype=torch.float32)
    W_cpu = torch.randn(D, Rnk, dtype=torch.float32)
    x_cpu = torch.randn(B, D, dtype=torch.float32) if topology == 1 else None
    Vw_cpu = None
    gamma_cpu = ops.christoffel_fused(v_cpu, U_cpu, W_cpu, x_cpu, Vw_cpu, 0.0, 1.0, 1.0, topology, 2.0, 1.0)
    if not torch.cuda.is_available():
        print("SKIP: CUDA no disponible")
        return "SKIP"
    v_gpu = v_cpu.cuda()
    U_gpu = U_cpu.cuda()
    W_gpu = W_cpu.cuda()
    x_gpu = x_cpu.cuda() if x_cpu is not None else None
    Vw_gpu = None
    gamma_gpu = ops.christoffel_fused(v_gpu, U_gpu, W_gpu, x_gpu, Vw_gpu, 0.0, 1.0, 1.0, topology, 2.0, 1.0)
    diff = (gamma_cpu - gamma_gpu.cpu()).abs().max().item()
    return summarize(f"christoffel[topo={topology}]", diff, tol=3e-4)

def test_leapfrog(topology):
    B, D, Rnk = 8, 64, 16
    steps = 2
    dt = 0.05
    dt_scale = 1.0
    x_cpu = torch.randn(B, D, dtype=torch.float32)
    v_cpu = torch.randn(B, D, dtype=torch.float32)
    f_cpu = torch.randn(B, D, dtype=torch.float32)
    U_cpu = torch.randn(D, Rnk, dtype=torch.float32)
    W_cpu = torch.randn(D, Rnk, dtype=torch.float32)
    feat_dim = (2 * D) if topology == 1 else D
    Wf_cpu = torch.randn(D, feat_dim, dtype=torch.float32)
    bf_cpu = torch.zeros(D, dtype=torch.float32)
    x_out_cpu, v_out_cpu = ops.leapfrog_fused(x_cpu, v_cpu, f_cpu, U_cpu, W_cpu, dt, dt_scale, steps, topology=topology, Wf=Wf_cpu, bf=bf_cpu, plasticity=0.0, R=2.0, r=1.0)
    if not torch.cuda.is_available():
        print("SKIP: CUDA no disponible")
        return "SKIP"
    x_gpu = x_cpu.cuda()
    v_gpu = v_cpu.cuda()
    f_gpu = f_cpu.cuda()
    U_gpu = U_cpu.cuda()
    W_gpu = W_cpu.cuda()
    Wf_gpu = Wf_cpu.cuda()
    bf_gpu = bf_cpu.cuda()
    x_out_gpu, v_out_gpu = ops.leapfrog_fused(x_gpu, v_gpu, f_gpu, U_gpu, W_gpu, dt, dt_scale, steps, topology=topology, Wf=Wf_gpu, bf=bf_gpu, plasticity=0.0, R=2.0, r=1.0)
    dx = (x_out_cpu - x_out_gpu.cpu()).abs().max().item()
    dv = (v_out_cpu - v_out_gpu.cpu()).abs().max().item()
    s1 = summarize(f"leapfrog[topo={topology}]-x", dx, tol=5e-4)
    s2 = summarize(f"leapfrog[topo={topology}]-v", dv, tol=5e-4)
    return "PASS" if s1 == "PASS" and s2 == "PASS" else "FAIL"

def test_head_mixing():
    H, Dh, B = 4, 8, 16
    D = H * Dh
    x_heads_cpu = torch.randn(H, B, Dh, dtype=torch.float32)
    v_heads_cpu = torch.randn(H, B, Dh, dtype=torch.float32)
    W_x_cpu = torch.randn(D, D, dtype=torch.float32)
    W_v_cpu = torch.randn(D, D, dtype=torch.float32)
    x_out_cpu, v_out_cpu = ops.head_mixing_fused(x_heads_cpu, v_heads_cpu, W_x_cpu, W_v_cpu, topology=0)
    if not torch.cuda.is_available():
        print("SKIP: CUDA no disponible")
        return "SKIP"
    x_heads_gpu = x_heads_cpu.cuda()
    v_heads_gpu = v_heads_cpu.cuda()
    W_x_gpu = W_x_cpu.cuda()
    W_v_gpu = W_v_cpu.cuda()
    x_out_gpu, v_out_gpu = ops.head_mixing_fused(x_heads_gpu, v_heads_gpu, W_x_gpu, W_v_gpu, topology=0)
    dx = (x_out_cpu - x_out_gpu.cpu()).abs().max().item()
    dv = (v_out_cpu - v_out_gpu.cpu()).abs().max().item()
    s1 = summarize("head_mixing-x", dx, tol=1e-5)
    s2 = summarize("head_mixing-v", dv, tol=1e-5)
    return "PASS" if s1 == "PASS" and s2 == "PASS" else "FAIL"

def test_dynamic_gating():
    D = 64
    inp_dim = 2 * D
    x_cpu = torch.randn(32, inp_dim, dtype=torch.float32)
    W1_cpu = torch.randn(D // 4, inp_dim, dtype=torch.float32)
    b1_cpu = torch.randn(D // 4, dtype=torch.float32)
    W2_cpu = torch.randn(1, D // 4, dtype=torch.float32)
    b2_cpu = torch.randn(1, dtype=torch.float32)
    y_cpu = ops.dynamic_gating_fused(x_cpu, W1_cpu, b1_cpu, W2_cpu, b2_cpu)
    if not torch.cuda.is_available():
        print("SKIP: CUDA no disponible")
        return "SKIP"
    x_gpu = x_cpu.cuda()
    W1_gpu = W1_cpu.cuda()
    b1_gpu = b1_cpu.cuda()
    W2_gpu = W2_cpu.cuda()
    b2_gpu = b2_cpu.cuda()
    y_gpu = ops.dynamic_gating_fused(x_gpu, W1_gpu, b1_gpu, W2_gpu, b2_gpu)
    diff = (y_cpu - y_gpu.cpu()).abs().max().item()
    return summarize("dynamic_gating", diff, tol=1e-5)

def main():
    torch.manual_seed(123)
    print(f"CUDA_AVAILABLE={ops.CUDA_AVAILABLE}")
    statuses = []
    statuses.append(test_christoffel(topology=0))
    statuses.append(test_christoffel(topology=1))
    statuses.append(test_leapfrog(topology=0))
    statuses.append(test_leapfrog(topology=1))
    statuses.append(test_head_mixing())
    statuses.append(test_dynamic_gating())
    fails = [s for s in statuses if s == "FAIL"]
    if fails:
        print("Resultado: FAIL")
        sys.exit(2)
    print("Resultado: PASS")

if __name__ == "__main__":
    main()
