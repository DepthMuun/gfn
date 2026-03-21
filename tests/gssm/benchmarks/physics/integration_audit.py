import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

def run_audit(steps=2000, dt=0.5, mu=0.5, impulse_strength=30.0, noise_std=2.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Starting Physics Audit (Device: {device})")
    print(f"[*] Configuration: dt={dt}, friction={mu}, impulse={impulse_strength}")

    # 1. Initialization
    # We use a batch size of 1 to simulate a single particle
    x_expl = torch.tensor([[1.0]], device=device)
    v_expl = torch.tensor([[0.0]], device=device)
    
    x_impl = torch.tensor([[1.0]], device=device)
    v_impl = torch.tensor([[0.0]], device=device)
    
    # Trackers
    history = {
        'expl_x': [], 'expl_v': [], 'expl_h': [],
        'impl_x': [], 'impl_v': [], 'impl_h': [],
    }

    def compute_energy(x, v):
        # Hamiltonian: 0.5 * v^2 + 0.5 * x^2 (Spring potential)
        return 0.5 * (v**2 + x**2).sum().item()

    # 2. Simulation Loop
    for i in tqdm(range(steps)):
        # Common Force: Spring + Stochastic Noise
        # a = -k*x (harmonic oscillator)
        force = -x_expl * 1.0 + torch.randn_like(x_expl) * noise_std
        
        # Bombardment: Impulse Spikes every 200 ms
        if i % 200 == 0:
            force += (torch.rand_like(x_expl) * 2 - 1) * impulse_strength

        # --- EXPLICIT INTEGRATION ---
        # v = v + dt * (a - mu*v)
        # x = x + dt * v
        a_expl = -x_expl # Force field
        v_expl = v_expl + dt * (a_expl + force - mu * v_expl)
        x_expl = x_expl + dt * v_expl
        # Wrap to Torus [-pi, pi]
        x_expl = torch.atan2(torch.sin(x_expl), torch.cos(x_expl))

        # --- SEMI-IMPLICIT INTEGRATION ---
        # v = (v + dt * a) / (1 + dt * mu)
        # x = x + dt * v
        a_impl = -x_impl
        v_impl = (v_impl + dt * (a_impl + force)) / (1.0 + dt * mu)
        x_impl = x_impl + dt * v_impl
        # Wrap to Torus [-pi, pi]
        x_impl = torch.atan2(torch.sin(x_impl), torch.cos(x_impl))

        # Record
        history['expl_x'].append(x_expl.item())
        history['expl_v'].append(v_expl.item())
        history['expl_h'].append(compute_energy(x_expl, v_expl))
        
        history['impl_x'].append(x_impl.item())
        history['impl_v'].append(v_impl.item())
        history['impl_h'].append(compute_energy(x_impl, v_impl))

    # 3. Visualization
    plt.figure(figsize=(15, 10))
    
    # Trajectory Plot
    plt.subplot(2, 1, 1)
    plt.plot(history['expl_x'], label='Explicit (v = v + dt(a-mu*v))', color='#E76F51', alpha=0.7)
    plt.plot(history['impl_x'], label='Semi-Implicit (v = (v+dt*a)/(1+dt*mu))', color='#2A9D8F', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.title(f"Trajectory Comparison (dt={dt}, mu={mu})")
    plt.ylabel("Position (x)")
    plt.legend()
    plt.grid(alpha=0.3)

    # Energy Drift Plot
    plt.subplot(2, 1, 2)
    plt.plot(history['expl_h'], label='Explicit Energy', color='#E76F51', alpha=0.5)
    plt.plot(history['impl_h'], label='Implicit Energy', color='#264653', linewidth=2)
    plt.title("Hamiltonian Drift (Energy Stability)")
    plt.ylabel("Energy (H)")
    plt.xlabel("Step")
    plt.yscale('log') # Log scale to see the divergence clearly
    plt.legend()
    plt.grid(alpha=0.3, which='both')

    save_path = "tests/benchmarks/physics/audit_plots.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[*] Audit complete. Plots saved to {save_path}")
    
    # Print quantitative result
    expl_drift = np.std(history['expl_h'])
    impl_drift = np.std(history['impl_h'])
    print(f"\n[RESULTS]")
    print(f"Explicit Energy Variance: {expl_drift:.4f}")
    print(f"Implicit Energy Variance: {impl_drift:.4f}")
    print(f"Stability Ratio: {expl_drift / (impl_drift + 1e-8):.2f}x")

if __name__ == "__main__":
    run_audit()
