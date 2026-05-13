"""
GFN Pure Flow Training Script for Tiny Shakespeare.
Uses GFNScanner, GFNWorld, and GFNEmitter.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gfn import isn
import time

class ShakespeareDataset(Dataset):
    def __init__(self, data_path, seq_len=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        self.data = [self.char_to_ix[ch] for ch in text]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk, dtype=torch.long)
        # Wrap for isn.Trainer expected format
        return {
            'input_ids': x,
            'output_ids': y
        }

def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved: {path}")

def run_shakespeare_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths (dynamic resolution for portability)
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    data_path = base_dir / "tests" / "isn. / "benchmarks" / "language" / "data" / "tinyshakespeare.txt"
    config_path = base_dir / "configs" / "shakespeare_config.json"
    checkpoint_dir = base_dir / "checkpoints" / "shakespeare"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dataset = ShakespeareDataset(data_path, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    print(f"Vocab size: {dataset.vocab_size} | Device: {device}")
    
    # Create Pure GFN Model
    model = isn.create(
        vocab_size=dataset.vocab_size,
        d_model=config['model']['d_model'],
        d_embedding=config['model']['d_embedding'],
        d_properties=config['model']['d_properties'],
        scanner_cls=isn.GFNScanner,
        world_cls=isn.GFNWorld,
        emitter_cls=isn.GFNEmitter
    ).to(device)
    
    criterion = isn.training.losses.coherence.MultiDimensionalLoss(
        vocab_size=dataset.vocab_size,
        **config['training']['curriculum']['phase1']['lambda_weights']
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    best_loss = float('inf')
    
    print("\n>>>> STARTING PURE GFN SHAKESPEARE TRAINING (MODULAR LOOP) <<<<")
    
    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0
        
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            output_ids = batch['output_ids'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(input_ids, return_world_state=True)
            
            # Inject vocab_basis for semantic loss
            if hasattr(model.emitter, 'emission'):
                outputs['vocab_basis'] = model.emitter.emission.weight
            
            # Loss
            targets = output_ids[:, 1:]
            loss_dict = criterion(targets=targets, **outputs)
            loss = loss_dict['loss']
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save Last
        save_checkpoint(model, optimizer, epoch, avg_loss, os.path.join(checkpoint_dir, "last_model.pt"))
        
        # Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"*** New Best Model! ***")

if __name__ == "__main__":
    run_shakespeare_training()

if __name__ == "__main__":
    run_shakespeare_training()
