import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Path hack
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gfn
from math_dataset import MathDataset

def train_math_benchmark(max_epochs: int = 50, batch_size: int = 64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = HERE / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset (AAopBB=CCC)
    dataset = MathDataset(num_samples=10000, operations=['+', '-'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    vocab_size = 16 # 0-9, +, -, =, PAD, *, /
    dim = 64
    
    # 2. Pure GFN V5 Model (New Flat API)
    model = gfn.create(
        vocab_size=vocab_size,
        dim=dim,
        rank=16,
        depth=2,
        heads=4,
        topology_type='torus',
        holographic=False,  # Requerido para CategoricalReadout nativo
        base_dt=0.1
    ).to(device)

    # 3. Pure GFN Loss
    # 'generative' mode='nll' handles CrossEntropy over logits sequence [B, L, V]
    criterion = gfn.loss('generative', mode='nll')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    print(f"\n[GFN Math Reasoning] Device: {device} | Tasks: + / -")

    # 4. Training Loop
    best_acc = 0.0
    for epoch in range(max_epochs):
        total_loss = 0
        correct_full = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            optimizer.zero_grad()
            
            # Forward: logits, (xf, vf), metrics
            logits, _, info = model(input_ids=inputs)
            
            # Pure GFN Loss usage
            loss = criterion(logits, targets, state_info=info)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # SYMBOLIC ACCURACY: ¿Es perfecto todo el resultado?
            # AAopBB=CCC
            # Predict after '=' means indices 6, 7, 8 in dataset (5, 6, 7 in targets)
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                # Evaluamos solo los últimos 3 tokens (la respuesta numérica)
                #AA op BB = C C C
                #01 2 34 5 6 7 8 (Dataset indices)
                #Targets son shift index-1: targets[index] = dataset[index+1]
                #index 5 targets es dataset 6.
                is_correct = (preds[:, 5:] == targets[:, 5:]).all(dim=-1)
                correct_full += is_correct.sum().item()
                total_samples += batch.size(0)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct_full/total_samples)*100:.1f}%")

        avg_loss = total_loss / len(dataloader)
        epoch_acc = correct_full / total_samples
        print(f"Epoch {epoch} Complete. Avg Loss: {avg_loss:.4f} | Final Acc: {epoch_acc*100:.2f}%")
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), output_dir / "math_best_model.pt")

        if epoch_acc > 0.99:
            print(f"[Success] Math Logic Converged at Epoch {epoch}!")
            break

if __name__ == "__main__":
    train_math_benchmark()
