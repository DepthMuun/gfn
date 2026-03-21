"""
train_heartattack.py — Heart Attack Classification (GFN V5 PROFESSIONAL)

El modelo procesa los 15 features tabulares como una secuencia de pasos temporales
(flujo geodésico) y predice la probabilidad de ataque cardíaco en el estado final.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np

# Path hack para permitir import gfn desde cualquier lugar
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parents[3]
sys.path.append(str(PROJECT_ROOT))

import gfn

class HeartAttackDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        
        # El target es 'HeartDisease'
        self.targets = df['HeartDisease'].values.astype(np.float32)
        
        # Eliminar target de las features
        features_df = df.drop(columns=['HeartDisease'])
        
        # Convertir booleanos a float
        for col in features_df.columns:
            if features_df[col].dtype == bool:
                features_df[col] = features_df[col].astype(np.float32)
                
        self.features = features_df.values.astype(np.float32)
        
        # Normalización Z-score de las features
        mean = self.features.mean(axis=0)
        std = self.features.std(axis=0)
        std[std == 0] = 1e-6
        self.features = (self.features - mean) / std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Transormar a secuencia: [L=15, 1]
        x = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(-1)
        return x, y

def train_classification():
    # 1. Configuración Profesional
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HERE = Path(__file__).parent
    csv_path = HERE / "datasets" / "Heart Attack" / "heart_processed.csv"
    
    dataset = HeartAttackDataset(str(csv_path))
    # Dividir en train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # DataLoader se crea UNA vez fuera del loop
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 2. Construcción del Modelo GFN V5 (API Profesional: Sin Presets)
    dim_total = 64
    model = gfn.create(
        vocab_size=100,
        dim=dim_total,
        rank=16,
        heads=4,
        holographic=True,
        topology_type='torus',
        base_dt=0.1,
        friction=0.05,
        integrator='leapfrog',
        dynamics_type='residual',
        impulse_scale=80.0
    ).to(device)

    # Proyector de entrada: Valor escalar Feature [1] -> Espacio del manifold [D]
    input_proj = nn.Linear(1, dim_total).to(device)
    
    # Proyector de salida: Estado del manifold [D] -> Logit de ataque cardíaco [1]
    output_proj = nn.Linear(dim_total, 1).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(input_proj.parameters()) + list(output_proj.parameters()),
        lr=1e-3, weight_decay=1e-5
    )
    
    # Loss BCE para clasificación binaria
    criterion = nn.BCEWithLogitsLoss()
    
    def calc_accuracy(logits, targets):
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct = (preds == targets).sum().item()
        return correct / len(targets)

    # 3. Training Loop
    max_epochs = 1000
    pbar = tqdm(range(max_epochs), desc="Heart Attack Classification")
    
    best_loss = float('inf')
    best_acc = 0.0
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        for features, target in train_loader:
            features = features.to(device) # [B, L=15, 1]
            target = target.to(device)     # [B, 1]
            
            optimizer.zero_grad()
            
            # Proyectar señal a impulsos de fuerza
            forces = input_proj(features) # [B, L, D]
            
            # Forward: evolución del manifold forzada por la secuencia de features
            _, _, metrics = model(force_manual=forces)
            x_seq = metrics["x_seq"]  # [B, L, H, D]
            
            # Extraer el estado final de la secuencia [B, H, D]
            last_x = x_seq[:, -1, :, :]
            
            # Aplanar cabezas para clasificación escalar [B, H*HD = D]
            x_agg = last_x.flatten(1) # [B, D]
            
            # Reconstrucción: Manifold State -> Scalar Logit
            logits = output_proj(x_agg) # [B, 1]
            
            # Loss entre predicción y label
            loss = criterion(logits, target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += calc_accuracy(logits.detach(), target)
            
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_acc = epoch_acc / len(train_loader)
        
        # Validación
        model.eval()
        val_loss = 0
        val_acc = 0
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
                loss = criterion(logits, target)
                
                val_loss += loss.item()
                val_acc += calc_accuracy(logits, target)
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        pbar.set_postfix(
            t_loss=f"{avg_train_loss:.4f}", t_acc=f"{avg_train_acc*100:.1f}%",
            v_loss=f"{avg_val_loss:.4f}", v_acc=f"{avg_val_acc*100:.1f}%"
        )
        
        # Guardar mejor modelo (basado en Validation Loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_acc = avg_val_acc
            torch.save({
                'model': model.state_dict(),
                'input_proj': input_proj.state_dict(),
                'output_proj': output_proj.state_dict()
            }, results_dir / "heartattack_manifold_best.pt")

    # Guardar Resultados Finales
    torch.save({
        'model': model.state_dict(),
        'input_proj': input_proj.state_dict(),
        'output_proj': output_proj.state_dict()
    }, results_dir / "heartattack_manifold_final.pt")
    print(f"\nEntrenamiento completado. Mejor Val Loss: {best_loss:.4f} | Mejor Val Acc: {best_acc*100:.1f}%. Modelos guardados en {results_dir}")


if __name__ == "__main__":
    train_classification()
