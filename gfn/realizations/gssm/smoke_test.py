import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Path hack
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gfn
from .losses.factory import LossFactory
from .training.trainer import GFNTrainer
from .data.dataset import SequenceDataset

def run_smoke_test():
    print("=== GFN V5 SMOKE TEST ===")
    
    # 1. Config & Model (API Profesional)
    print("1. Building model via factory (flat config)...")
    vsize = 10
    model = gfn.create(
        vocab_size=vsize,
        dim=32,
        heads=4,
        depth=2,
        rank=8,
        topology_type='torus',
        holographic=False
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Data
    print("3. Preparing synthetic data...")
    batch_size = 8
    seq_len = 16
    x_train = torch.randint(1, vsize, (batch_size * 10, seq_len))
    y_train = torch.randint(1, vsize, (batch_size * 10, seq_len))
    dataset = SequenceDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 4. Loss & Optimizer
    print("4. Setting up loss and optimizer...")
    loss_fn = LossFactory.create({'type': 'generative'})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 5. Trainer
    print("5. Initializing trainer...")
    trainer = GFNTrainer(model, loss_fn, optimizer)
    
    # 6. Training Step
    print("6. Executing training epoch...")
    history = trainer.fit(loader, epochs=1)
    
    print("   Evolution complete. Final loss:", history["loss"][-1])
    
    # 7. Evaluation
    print("7. Evaluating...")
    acc = trainer.evaluate(loader)
    print(f"   Accuracy: {acc:.4f}")
    
    print("=== SMOKE TEST PASSED ===")

if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as e:
        print(f"!!! SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
