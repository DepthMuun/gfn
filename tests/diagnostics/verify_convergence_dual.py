import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.optim import RiemannianAdam
from gfn.losses import geodesic_regularization, hamiltonian_loss, ToroidalDistanceLoss


def run_experiment(mode_name, use_cuda_kernel):
    print(f"\n{'='*60}")
    print(f"Running Experiment: {mode_name}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = Manifold(
        vocab_size=2,
        dim=128,
        depth=2,
        heads=4,
        integrator_type="leapfrog",
    ).to(device)

    optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
    model.train()

    batch_size = 64
    seq_len = 16

    losses = []

    start_time = time.time()

    criterion = ToroidalDistanceLoss()

    for step in range(50):
        x = torch.randint(0, 2, (batch_size, seq_len), device=device)
        y_int = torch.cumsum(x, dim=1) % 2
        pi = 3.14159265359
        y_angle = (y_int.float() * 2.0 - 1.0) * (pi * 0.5)

        optimizer.zero_grad()

        if use_cuda_kernel:
            outputs = model(x)
        else:
            outputs = model(x, collect_christ=True)

        x_pred = outputs[0] if isinstance(outputs, tuple) else outputs
        y_expanded = y_angle.float().unsqueeze(-1).expand_as(x_pred)

        loss_val = criterion(x_pred, y_expanded)
        loss_phy = 0.0
        loss_ham = 0.0
        if isinstance(outputs, tuple) and len(outputs) >= 6:
            christoffels = outputs[2]
            v_seq = outputs[3]
            x_seq = outputs[4]
            all_forces = outputs[5]

            if christoffels:
                # AUDIT FIX: Correct signature
                loss_phy = geodesic_regularization(christoffels, velocities=None, lambda_g=0.001, mode='structural')
                def first_head_metric(x):
                    return model.layers[0].christoffels[0].get_metric(x) if hasattr(model.layers[0].christoffels[0], "get_metric") else torch.ones_like(x)
                loss_ham = hamiltonian_loss(v_seq, states=x_seq, metric_fn=first_head_metric, lambda_h=0.0, forces=all_forces)

        total_loss = loss_val + loss_phy + loss_ham

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(total_loss.item())

        if step % 10 == 0:
            two_pi = 2.0 * pi
            half_pi = pi * 0.5
            dist_pos = torch.min(torch.abs(x_pred - half_pi) % two_pi, two_pi - (torch.abs(x_pred - half_pi) % two_pi))
            dist_neg = torch.min(torch.abs(x_pred + half_pi) % two_pi, two_pi - (torch.abs(x_pred + half_pi) % two_pi))
            d_pos = dist_pos.mean(dim=-1)
            d_neg = dist_neg.mean(dim=-1)
            preds = (d_pos < d_neg).long()
            acc = (preds == y_int).float().mean().item()
            print(f"Step {step}: Loss={total_loss.item():.4f}, Acc={acc*100:.1f}%")

    duration = time.time() - start_time

    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Duration: {duration:.2f}s")

    return losses


if __name__ == "__main__":
    print("Verifying convergence for Leapfrog Integrator...")

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping dual-mode test.")
        sys.exit(0)

    losses_py = run_experiment("PYTHON Mode (Emulated)", use_cuda_kernel=False)
    losses_cuda = run_experiment("CUDA Mode (Fused Kernel)", use_cuda_kernel=True)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Python Final Loss: {losses_py[-1]:.4f}")
    print(f"CUDA Final Loss:   {losses_cuda[-1]:.4f}")

    converged_py = losses_py[-1] < losses_py[0] * 0.8
    converged_cuda = losses_cuda[-1] < losses_cuda[0] * 0.8

    if converged_py and converged_cuda:
        print("SUCCESS: Both modes are learning.")
    else:
        print("WARNING: One or both modes failed to converge significantly.")
