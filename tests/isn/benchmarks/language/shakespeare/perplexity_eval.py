"""
isn.Perplexity Evaluation Script — GFN V4
Calculates PPL (exp(loss)) and World Coherence for Shakespeare models.
"""

import os
import torch
import torch.nn as nn
import json
import math
from pathlib import Path
from tqdm import tqdm

# Import project components (assumes PYTHONPATH is set to project root)
from gfn.realizations.isn.models.model import Model
from gfn.realizations.isn.components.scanners.gfn_scanner import GFNScanner
from gfn.realizations.isn.components.worlds.gfn_world import GFNWorld
from gfn.realizations.isn.components.emitters.gfn_emitter import GFNEmitter

def load_data(data_path: str):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    
    # Simple split: last 10% for validation
    n = len(text)
    val_data = text[int(n*0.9):]
    val_ids = torch.tensor([char_to_ix[c] for c in val_data], dtype=torch.long)
    
    return val_ids, vocab_size

def evaluate_ppl(model, val_ids, batch_size=64, seq_length=256, device='cpu'):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_coherence = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    n_batches = (len(val_ids) - 1) // (batch_size * seq_length)
    if n_batches == 0:
        n_batches = 1
        
    print(f"Evaluating {n_batches} batches...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(val_ids) - seq_length - 1, batch_size * seq_length)):
            # Prepare batch
            batch_inputs = []
            batch_targets = []
            
            for b in range(batch_size):
                start = i + b * seq_length
                if start + seq_length + 1 > len(val_ids):
                    break
                batch_inputs.append(val_ids[start : start + seq_length])
                batch_targets.append(val_ids[start + 1 : start + seq_length + 1])
            
            if not batch_inputs:
                break
                
            inputs = torch.stack(batch_inputs).to(device)
            targets = torch.stack(batch_targets).to(device)
            
            # Forward pass
            outputs = model(inputs)
            logits = outputs['logits']
            coherence = outputs['world_coherence']
            
            # Reshape for CrossEntropy
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
            total_coherence += coherence.mean().item() * inputs.size(0)
            total_samples += inputs.size(0)
            
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    avg_coherence = total_coherence / total_samples
    
    return avg_loss, ppl, avg_coherence

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths (dynamic resolution for portability)
    base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    checkpoint_dir = base_dir / "checkpoints" / "shakespeare"
    data_path = base_dir / "tests" / "isn" / "benchmarks" / "language" / "data" / "tinyshakespeare.txt"
    config_path = base_dir / "configs" / "isn" / "shakespeare_config.json"
    
    print(f"--- isn.Perplexity Evaluation ---")
    print(f"Device: {device}")
    
    # 1. Load Data
    val_ids, vocab_size = load_data(str(data_path))
    print(f"Vocab size: {vocab_size} | Val tokens: {len(val_ids)}")
    
    # 2. Load Config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 3. List Checkpoints
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} not found.")
        return
        
    checkpoints = sorted(list(checkpoint_dir.glob("*.pt")))
    if not checkpoints:
        print("No checkpoints found in directory.")
        return
        
    print(f"Found {len(checkpoints)} checkpoints.")
    
    results = []
    
    for ckpt_path in checkpoints:
        print(f"\nEvaluating: {ckpt_path.name}")
        
        # Build Model
        scanner = GFNScanner(vocab_size=vocab_size, d_model=config['model']['d_model'])
        world = GFNWorld(
            d_model=config['model']['d_model'], 
            d_embedding=config['model']['d_embedding'],
            d_properties=config['model']['d_properties']
        )
        emitter = GFNEmitter(d_embedding=config['model']['d_embedding'], vocab_size=vocab_size)
        
        model = Model(scanner=scanner, world=world, emitter=emitter).to(device)
        
        # Load State
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        
        # Evaluate
        loss, ppl, coherence = evaluate_ppl(model, val_ids, device=device)
        
        results.append({
            'name': ckpt_path.name,
            'epoch': ckpt.get('epoch', 'N/A'),
            'loss': loss,
            'ppl': ppl,
            'coherence': coherence
        })
        
        print(f"Result -> Loss: {loss:.4f} | PPL: {ppl:.4f} | Coherence: {coherence:.4f}")

    # Final Table
    print("\n" + "="*65)
    print(f"{'Checkpoint':<25} | {'Epoch':<6} | {'Loss':<8} | {'PPL':<8} | {'Coh':<6}")
    print("-" * 65)
    for res in results:
        print(f"{res['name']:<25} | {res['epoch']:<6} | {res['loss']:<8.4f} | {res['ppl']:<8.4f} | {res['coherence']:<6.4f}")
    print("="*65)

if __name__ == "__main__":
    main()
