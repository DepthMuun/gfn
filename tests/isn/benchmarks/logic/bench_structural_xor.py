import os
import sys
import torch
import torch.nn as nn
import yaml
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Add project root to path
base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(base_dir))

from gfn.realizations.isn.models.model import Model as RealityModel
from gfn.realizations.isn.training.coherence_loss import MultiDimensionalLoss

class XORDataGenerator:
    def __init__(self, n_bits=2):
        self.n_bits = n_bits
        self.vocab = ['0', '1', '^', '=', '<PAD>', '<START>', '<END>']
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def generate_sample(self):
        bits = [torch.randint(0, 2, (1,)).item() for _ in range(self.n_bits)]
        res = bits[0]
        for b in bits[1:]:
            res = res ^ b
        
        # Format: "1 ^ 0 ^ 1 ^ ..."
        input_str = " ^ ".join([str(b) for b in bits]) + " ="
        target_str = f"{res}"
        
        input_ids = [self.token_to_id[c] for c in input_str.split()]
        target_ids = [self.token_to_id['<START>']] + [self.token_to_id[target_str]] + [self.token_to_id['<END>']]
        
        return torch.tensor(input_ids), torch.tensor(target_ids)

class XORDataset(Dataset):
    def __init__(self, generator, n_samples=1000):
        self.samples = [generator.generate_sample() for _ in range(n_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    input_ids, target_ids = zip(*batch)
    return {
        'input_ids': torch.stack(input_ids),
        'target_ids': torch.stack(target_ids)
    }

def run_xor_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_bits = 20
    generator = XORDataGenerator(n_bits=train_bits)
    dataset = XORDataset(generator, n_samples=3000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Config (Minimal for XOR)
    d_model = 128
    model = RealityModel(
        vocab_size=generator.vocab_size,
        d_model=d_model,
        d_embedding=128,
        d_properties=64,
        encoder_layers=2,
        decoder_layers=2,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        max_entities=2500, # Handled 1000+ bits
        max_timesteps=1500,
        max_seq_length=2500,
        coherence_threshold=0.7
    ).to(device)

    criterion = MultiDimensionalLoss(vocab_size=generator.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    print(f"Starting XOR Parity Benchmark (L={train_bits}) on {device}...")
    model.train()
    epochs = 10
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, target_ids=target_ids[:, :-1], return_world_state=True)
            
            loss_dict = criterion(
                logits=outputs['logits'],
                targets=target_ids[:, 1:],
                world_coherence=outputs['world_coherence'],
                world_states=outputs.get('world_states')
            )
            
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Accuracy
            preds = outputs['logits'].argmax(dim=-1)
            correct += (preds[:, 0] == target_ids[:, 1]).sum().item()
            total += target_ids.size(0)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.2%}")

        avg_loss = epoch_loss / len(loader)
        accuracy = correct / total
        print(f"Summary Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}")
        
        if accuracy > 0.99:
            print("Target accuracy reached!")
            break

    # --- GENERALIZATION TEST ---
    print("\n" + "="*40)
    print("Testing Length Generalization")
    print("="*40)
    model.eval()
    
    test_lengths = [20, 50, 200, 1000]
    for n in test_lengths:
        gen_test = XORDataGenerator(n_bits=n)
        test_samples = 200
        correct_gen = 0
        
        with torch.no_grad():
            for _ in range(test_samples):
                input_ids, target_ids = gen_test.generate_sample()
                input_ids = input_ids.unsqueeze(0).to(device)
                target_ids = target_ids.to(device)
                
                outputs = model(input_ids=input_ids, target_ids=target_ids[:-1].unsqueeze(0), return_world_state=True)
                preds = outputs['logits'].argmax(dim=-1)
                
                # Check accuracy only on the prediction token
                if preds[0, 0] == target_ids[1]:
                    correct_gen += 1
        
        acc_gen = correct_gen / test_samples
        print(f"Bits: {n:2d} | Accuracy: {acc_gen:.2%}")

    # Save
    save_path = base_dir / "checkpoints" / "reality_xor.pt"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    run_xor_benchmark()
