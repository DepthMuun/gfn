"""
inference_ecg.py — Generative Simulation (Digital Twin)
Genera latidos sintéticos usando el modelo GFN V5 entrenado.
"""
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

# Path hack
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parents[3]
sys.path.append(str(PROJECT_ROOT))

import gfn

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = HERE / "results"
    model_path = results_dir / "ecg_manifold_best.pt"
    
    if not model_path.exists():
        print(f"Error: No se encontró el modelo en {model_path}")
        return

    # 1. Configuración (IDÉNTICA a train.py - Golden Path)
    dim_total = 64
    topo_cfg = gfn.TopologyConfig(type='torus', R=2.0, r=1.0, riemannian_rank=16)
    stab_cfg = gfn.StabilityConfig(base_dt=0.2, adaptive=True, enable_trace_normalization=True, friction=0.03)
    active_inf_cfg = gfn.ActiveInferenceConfig(enabled=True, holographic_geometry=True, 
                                              dynamic_time=gfn.DynamicTimeConfig(enabled=True))
    phys_cfg = gfn.PhysicsConfig(topology=topo_cfg, stability=stab_cfg, active_inference=active_inf_cfg,
                                embedding=gfn.EmbeddingConfig(type='functional', mode='linear'))
    
    # 2. Reconstruir Arquitectura Exacta
    model = gfn.create(
        vocab_size=100, 
        dim=dim_total, 
        rank=16, 
        heads=4, 
        holographic=True, 
        physics=phys_cfg,
        integrator='rk4',        # RK4 para precisión en señales cíclicas
        dynamics_type='mix',     # Mix para estabilidad de secuencia
        impulse_scale=50.0       # Escalado moderado para ECG
    ).to(device)
    input_proj = nn.Linear(1, dim_total).to(device)
    output_proj = nn.Linear(dim_total, 1).to(device)

    # 3. Cargar Pesos
    print(f"Cargando pesos desde {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    input_proj.load_state_dict(checkpoint['input_proj'])
    output_proj.load_state_dict(checkpoint['output_proj'])
    model.eval()
    input_proj.eval()
    output_proj.eval()

    # 4. Generación Autoregresiva (Persistencia de Estado)
    steps_to_generate = 1000
    generated_signal = []
    
    # Semilla inicial
    current_val = torch.tensor([[[0.1]]], device=device)
    current_state = None
    
    print(f"Iniciando generación de {steps_to_generate} timesteps...")
    
    with torch.no_grad():
        for i in range(steps_to_generate):
            force = input_proj(current_val)
            
            # Evolucionar pasando el estado acumulado
            _, state_next, metrics = model(force_manual=force, state=current_state)
            x_seq = metrics["x_seq"]
            
            x_agg = x_seq.flatten(2)
            next_val = output_proj(x_agg)
            
            generated_signal.append(next_val.item())
            
            current_val = next_val
            current_state = state_next

    # 5. Guardar Resultados
    output_csv = results_dir / "generated_ecg.csv"
    df = pd.DataFrame(generated_signal, columns=["voltaje"])
    df.to_csv(output_csv, index=False)
    print(f"Simulación completada. Resultado guardado en: {output_csv}")
    print("Muestra de los primeros 10 valores generados:")
    print(df.head(10))

if __name__ == "__main__":
    run_inference()
