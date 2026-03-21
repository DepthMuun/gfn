"""
isn.Training Efficiency Benchmark
Compares convergence speed: Transformer vs SSM vs GFN.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_dir))
from gfn import isn

class CopyDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = torch.randint(2, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Target is the same as input
        return self.data[idx], self.data[idx]

def train_model(name, model, dataloader, num_steps=1000):
    print(f"\n--- Training {name} ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    start_time = time.time()
    losses = []
    
    for i, (input_ids, targets) in enumerate(dataloader):
        if i >= num_steps: break
        
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        output = model(input_ids)
        logits = output['logits']
        
        # Flatten for XE
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if (i+1) % 100 == 0:
            print(f"Step {i+1}/{num_steps} | Loss: {loss.item():.4f}")
            
    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f}s")
    print(f"Final Loss: {losses[-1]:.4f}")
    return losses

if __name__ == "__main__":
    V = 30
    L = 32
    D = 128
    BATCH = 16
    STEPS = 1000
    
    dataset = CopyDataset(V, L, 20000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True)
    
    # 1. Transformer
    m_trans = isn.create(vocab_size=V, d_model=D, scanner_cls=isn.TransformerScanner, emitter_cls=isn.ThresholdEmitter)
    l_trans = train_model("Transformer", m_trans, dataloader, STEPS)
    
    # 2. SSM
    m_ssm = isn.create(vocab_size=V, d_model=D, scanner_cls=isn.SSMScanner, emitter_cls=isn.SSMEmitter)
    l_ssm = train_model("SSM (O(1))", m_ssm, dataloader, STEPS)
    
    # 3. Pure GFN
    m_gfn = isn.create(vocab_size=V, d_model=D, 
                       scanner_cls=isn.GFNScanner, 
                       world_cls=isn.GFNWorld,
                       emitter_cls=isn.GFNEmitter)
    l_gfn = train_model("Pure GFN (O(1))", m_gfn, dataloader, STEPS)
    
    # Summary of convergence
    print("\n>>>> CONVERGENCE SUMMARY (Final Loss) <<<<")
    print(f"Transformer: {l_trans[-1]:.4f}")
    print(f"SSM:         {l_ssm[-1]:.4f}")
    print(f"Pure GFN:    {l_gfn[-1]:.4f}")
