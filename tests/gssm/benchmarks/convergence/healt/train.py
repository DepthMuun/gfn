"""
train.py — ECG Signal Fitting (GFN V5 PROFESSIONAL)
MODIFIED: Option A (Closed-Loop / Autoregressive Training)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Path hack
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parents[3]
sys.path.append(str(PROJECT_ROOT))

import gfn
from ecg_dataset import ECGSignalDataset

def train_signal_fitting():
    # 1. Configuración Profesional
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HERE = Path(__file__).parent
    csv_path = HERE / "datasets" / "ECG Timeseries-20260303T021501Z-1-001" / "ECG Timeseries" / "ecg_timeseries.csv"
    
    dataset = ECGSignalDataset(str(csv_path), max_samples=42)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Construcción del Modelo GFN V5 (API Profesional: Sin Presets)
    dim_total = 64
    model = gfn.create(
        vocab_size=100, 
        dim=dim_total, 
        rank=16, 
        heads=4, 
        holographic=True, 
        topology_type='torus', 
        base_dt=0.2, 
        friction=0.03,
        adaptive=True,
        enable_trace_normalization=True,
        integrator='rk4',
        dynamics_type='mix',
        impulse_scale=50.0
    ).to(device)

    input_proj = nn.Linear(1, dim_total).to(device)
    output_proj = nn.Linear(dim_total, 1).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(input_proj.parameters()) + list(output_proj.parameters()),
        lr=2e-4, weight_decay=1e-5
    )
    
    criterion = nn.MSELoss()
    
    def calc_r2(pred, target):
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - pred) ** 2)
        if ss_tot == 0: return 0.0
        return (1 - ss_res / ss_tot).item()

    # 3. Training Loop Hybrid (Warmup + Closed-Loop)
    max_epochs = 1000
    pbar = tqdm(range(max_epochs), desc="Option A: Closed-Loop ECG")
    
    best_loss = float('inf')
    best_r2 = -float('inf')
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    
    warmup_steps = 20
    
    for epoch in pbar:
        epoch_loss = 0
        epoch_r2 = 0
        for signal, _ in dataloader:
            signal = signal.to(device) # [B, L, 1]
            batch_size = signal.shape[0]
            seq_len = signal.shape[1]
            
            optimizer.zero_grad()
            
            # --- CLOSED-LOOP EVOLUTION ---
            # 1. Warmup Vectorizado (Eficiente)
            current_input_warmup = signal[:, :warmup_steps, :] # [B, warmup, 1]
            force_warmup = input_proj(current_input_warmup)     # [B, warmup, dim]
            
            # Evolucionar todos los pasos del warmup en una sola llamada al modelo
            logits_w, (current_x, current_v), metrics_w = model(force_manual=force_warmup)
            current_state = (current_x, current_v)
            
            # Obtener predicciones del warmup
            # x_seq_total: [B, warmup, H, D]
            x_seq_w = metrics_w["x_seq"]
            preds_warmup = output_proj(x_seq_w.flatten(2)) # [B, warmup, 1]
            predictions = [preds_warmup[:, i:i+1, :] for i in range(warmup_steps)]
            
            # 2. Autoregressive loop (Sigue paso a paso, pero minimizamos overhead)
            for t in range(warmup_steps, seq_len - 1):
                # Detached predictions for the first few epochs to stabilize
                curr_pred = predictions[-1]
                current_input = curr_pred.detach() if epoch < 5 else curr_pred
                
                # Proyección y evolución de paso único (Manifold V5)
                force = input_proj(current_input)
                _, current_state, metrics = model(force_manual=force, state=current_state)
                
                # Extraer predicción del estado de la capa final
                # metrics["x_seq"] es [B, 1, H, D]
                x_next = metrics["x_seq"].flatten(2) # [B, D]
                pred = output_proj(x_next.unsqueeze(1)) # [B, 1, 1]
                predictions.append(pred)
            
            reconstructed = torch.cat(predictions, dim=1) # [B, L-1, 1]
            target = signal[:, 1:, :] # [B, L-1, 1]
            
            loss = criterion(reconstructed, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_r2 += calc_r2(reconstructed.detach(), target)
            
        avg_loss = epoch_loss / len(dataloader)
        avg_r2 = epoch_r2 / len(dataloader)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", r2_acc=f"{avg_r2*100:.1f}%")
        
        # --- MILESTONE SAVING SYSTEM ---
        save_dict = {
            'model': model.state_dict(),
            'input_proj': input_proj.state_dict(),
            'output_proj': output_proj.state_dict(),
            'epoch': epoch,
            'r2': avg_r2
        }
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(save_dict, results_dir / "ecg_manifold_best_loss.pt")
            print ("model best loss saved")
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            torch.save(save_dict, results_dir / "ecg_manifold_best_r2.pt")
            print ("model best r2 saved")
            
        # Milestones persistentes
        for milestone in [0.90, 0.95, 0.99]:
            m_path = results_dir / f"ecg_milestone_{milestone}.pt"
            if avg_r2 >= milestone and not m_path.exists():
                torch.save(save_dict, m_path)

    print(f"Entrenamiento completado. Mejor R2: {best_r2:.4f}. Modelos en {results_dir}")

if __name__ == "__main__":
    train_signal_fitting()
