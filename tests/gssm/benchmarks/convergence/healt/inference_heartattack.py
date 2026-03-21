import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Path hack para permitir import gfn desde cualquier lugar
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parents[3]
sys.path.append(str(PROJECT_ROOT))

import gfn
from train_heartattack import HeartAttackDataset

def evaluate_model():
    # 1. Configuración de Hardware y Rutas
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = HERE / "datasets" / "Heart Attack" / "heart_processed.csv"
    model_path = HERE / "results" / "heartattack_manifold_best.pt"
    
    if not model_path.exists():
        print(f"Error: No se encontro el modelo en {model_path}")
        print("Por favor entrena el modelo usando train_heartattack.py primero.")
        return

    # 2. Cargar Dataset (Misma división que en entrenamiento)
    print("Cargando dataset...")
    dataset = HeartAttackDataset(str(csv_path))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # IMPORTANTE: Usar la MISMA semilla (42) para evaluar exactamente en el Validation Set
    _, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 3. Recrear Arquitectura Exacta (Golden Path)
    dim_total = 64
    topo_cfg = gfn.TopologyConfig(type='torus', riemannian_rank=16)
    stab_cfg = gfn.StabilityConfig(base_dt=0.1, adaptive=True, enable_trace_normalization=True, friction=0.05)
    active_inf_cfg = gfn.ActiveInferenceConfig(
        enabled=True, holographic_geometry=True, dynamic_time=gfn.DynamicTimeConfig(enabled=True)
    )
    phys_cfg = gfn.PhysicsConfig(
        topology=topo_cfg, stability=stab_cfg, active_inference=active_inf_cfg,
        embedding=gfn.EmbeddingConfig(type='functional', mode='linear')
    )
    
    model = gfn.create(
        vocab_size=100,
        dim=dim_total,
        rank=16,
        heads=4,
        holographic=True,
        physics=phys_cfg,
        integrator='leapfrog',
        dynamics_type='residual',
        impulse_scale=80.0
    ).to(device)
    input_proj = nn.Linear(1, dim_total).to(device)
    output_proj = nn.Linear(dim_total, 1).to(device)
    
    # 4. Cargar Pesos Completos (Modelo + Proyectores)
    print(f"Cargando pesos desde {model_path.name}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    if 'model' not in checkpoint or 'input_proj' not in checkpoint:
        print("\n[!] ADVERTENCIA CRITICA: El archivo .pt no contiene los proyectores lineales.")
        print("El script train_heartattack.py anterior tenia un bug y no guardaba el input_proj ni output_proj.")
        print("Tratando de continuar asumiendo que el .pt solo tiene el model state...")
        # Fallback (rendimiento sera aleatorio)
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model'])
        input_proj.load_state_dict(checkpoint['input_proj'])
        output_proj.load_state_dict(checkpoint['output_proj'])
        print("Modelos y Proyectores cargados exitosamente.")
    
    model.eval()
    input_proj.eval()
    output_proj.eval()
    
    # 5. Evaluación Exhaustiva
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("\nEjecutando evaluacion en el Validation Set...")
    with torch.no_grad():
        for features, target in val_loader:
            features = features.to(device)
            target = target.to(device)
            
            forces = input_proj(features)
            _, _, metrics = model(force_manual=forces)
            
            x_seq = metrics["x_seq"]
            last_x = x_seq[:, -1, :, :]
            x_agg = last_x.flatten(1)
            
            logits = output_proj(x_agg)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            
    # 6. Calcular y Mostrar Métricas
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=["No Heart Attack (0)", "Heart Attack (1)"])
    
    print("\n" + "="*50)
    print(" RESULTADOS DE EVALUACION (MANIFOLD GFN V5)")
    print("="*50)
    print(f"\nAccuracy Absoluto: {acc*100:.2f}%\n")
    
    print("Matriz de Confusion:")
    print("-" * 20)
    print(f"[{cm[0,0]:4d}]  [{cm[0,1]:4d}]  <- Reales 0")
    print(f"[{cm[1,0]:4d}]  [{cm[1,1]:4d}]  <- Reales 1")
    print( "  ^       ^")
    print("Pred 0  Pred 1")
    print("-" * 20)
    
    print("\nReporte de Clasificacion:")
    print(report)
    print("="*50)

if __name__ == "__main__":
    evaluate_model()
