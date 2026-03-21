"""
Reality Model - Addition Training Demo
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import time

# Add project root to path
base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(base_dir))

from gfn.realizations.isn.models.model import Model as RealityModel
from gfn.realizations.isn.training.coherence_loss import MultiDimensionalLoss
from gfn.realizations.isn.utils.data_generator import ArithmeticDataGenerator
from gfn.realizations.isn.training.trainer import ArithmeticDataset
from torch.utils.data import DataLoader

def load_mini_config():
    """Create a minimal config for the demo."""
    return {
        'model': {
            'encoder': {
                'd_model': 128,
                'n_layers': 2,
                'n_heads': 4,
                'd_ff': 512,
                'dropout': 0.1
            },
            'entity': {
                'd_embedding': 128,
                'd_properties': 64,
                'max_entities': 20
            },
            'world': {
                'max_timesteps': 10,
                'coherence_threshold': 0.7
            },
            'decoder': {
                'd_model': 128,
                'n_layers': 2,
                'n_heads': 4,
                'd_ff': 512,
                'dropout': 0.1
            }
        },
        'training': {
            'optimizer': {
                'lr': 1e-4,
                'betas': [0.9, 0.999],
                'weight_decay': 0.01
            },
            'batch_size': 16,
            'num_epochs': 1000
        },
        'loss': {
            'lambda': {
                'outcome': 1.0,
                'coherence': 0.5,
                'grounding': 0.2,
                'validity': 0.2,
                'emergence': 0.1,
                'efficiency': 0.05
            }
        },
        'data': {
            'arithmetic': {
                'min_digits': 1,
                'max_digits': 2,
                'operations': ['add']
            },
            'seed': 42,
            'train_size': 1500,
            'val_size': 200
        },
        'device': {'type': 'cuda'},
        'logging': {'checkpoint_frequency': 10}
    }

def run_addition_demo():
    print("="*60)
    print("Reality Model - Addition Training Demo")
    print("="*60)
    
    config = load_mini_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Generation
    generator = ArithmeticDataGenerator(
        min_digits=config['data']['arithmetic']['min_digits'],
        max_digits=config['data']['arithmetic']['max_digits'],
        operations=config['data']['arithmetic']['operations'],
        seed=config['data']['seed']
    )
    vocab_size = len(generator.vocab)
    
    train_data = generator.generate_dataset(n_samples=config['data']['train_size'])
    val_data = generator.generate_dataset(n_samples=config['data']['val_size'])
    
    train_loader = DataLoader(
        ArithmeticDataset(train_data),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=generator.collate_fn
    )
    
    val_loader = DataLoader(
        ArithmeticDataset(val_data),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=generator.collate_fn
    )
    
    # 2. Model Creation
    model = RealityModel(
        vocab_size=vocab_size,
        d_model=config['model']['encoder']['d_model'],
        d_embedding=config['model']['entity']['d_embedding'],
        d_properties=config['model']['entity']['d_properties'],
        encoder_layers=config['model']['encoder']['n_layers'],
        decoder_layers=config['model']['decoder']['n_layers'],
        n_heads=config['model']['encoder']['n_heads'],
        d_ff=config['model']['encoder']['d_ff'],
        dropout=config['model']['encoder']['dropout'],
        max_entities=config['model']['entity']['max_entities'],
        max_timesteps=config['model']['world']['max_timesteps'],
        coherence_threshold=config['model']['world']['coherence_threshold']
    ).to(device)
    
    # 3. Loss and Optimizer
    loss_weights = config['loss']['lambda']
    criterion = MultiDimensionalLoss(
        lambda_outcome=loss_weights['outcome'],
        lambda_coherence=loss_weights['coherence'],
        lambda_grounding=loss_weights['grounding'],
        lambda_validity=loss_weights['validity'],
        lambda_emergence=loss_weights['emergence'],
        lambda_efficiency=loss_weights['efficiency'],
        vocab_size=vocab_size
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr']
    )
    
    # 4. Training Loop
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(config['training']['num_epochs']):
        model.train()
        epoch_loss = 0
        epoch_coherence = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            output_ids = batch['output_ids'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, target_ids=output_ids[:, :-1], return_world_state=True)
            
            loss_dict = criterion(
                logits=outputs['logits'],
                targets=output_ids[:, 1:],
                world_coherence=outputs['world_coherence'],
                world_states=outputs.get('world_states')
            )
            
            loss_dict['loss'].backward()
            optimizer.step()
            
            epoch_loss += loss_dict['loss'].item()
            epoch_coherence += outputs['world_coherence'].mean().item()
            
        avg_loss = epoch_loss / len(train_loader)
        avg_coh = epoch_coherence / len(train_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Coherence={avg_coh:.3f}")
            
    print(f"\nTraining finished in {time.time() - start_time:.2f}s")
    
    # --- SAVE MODEL ---
    save_path = base_dir / "checkpoints"
    save_path.mkdir(exist_ok=True)
    model_file = save_path / "reality_model_addition.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab': generator.vocab
    }, model_file)
    print(f"✓ Model saved to {model_file}")
    
    # 5. Evaluation & Generalization
    model.eval()
    print("\n" + "="*30)
    print("Generalization Test (OOD)")
    print("="*30)
    
    def test_generalization(name, digits):
        print(f"\nTesting {name} ({digits} digits):")
        test_gen = ArithmeticDataGenerator(
            min_digits=digits,
            max_digits=digits,
            operations=['add'],
            seed=100  # Different seed
        )
        test_gen.vocab = generator.vocab  # Use same vocab
        test_gen.token_to_id = generator.token_to_id
        test_gen.id_to_token = generator.id_to_token
        
        test_samples = test_gen.generate_dataset(n_samples=5)
        
        with torch.no_grad():
            hits = 0
            for sample in test_samples:
                input_ids = torch.tensor([sample['input_ids']]).to(device)
                outputs = model(input_ids=input_ids, return_world_state=True)
                logits = outputs['logits']
                pred_ids = logits.argmax(dim=-1)
                
                # Detokenize (ignoring START/END)
                pred_tokens = [test_gen.id_to_token[tid.item()] for tid in pred_ids[0] 
                              if test_gen.id_to_token[tid.item()] not in ['<START>', '<END>', '<PAD>']]
                pred_text = "".join(pred_tokens).strip() 
                
                # The detokenize in generator adds spaces, let's be more precise
                raw_pred = test_gen.detokenize(pred_ids[0].tolist())
                
                is_correct = raw_pred.strip() == sample['output'].strip()
                if is_correct: hits += 1
                
                mark = "✓" if is_correct else "✗"
                print(f"  {mark} Input: {sample['input']} -> Target: {sample['output']} | Pred: {raw_pred}")
            print(f"Accuracy ({name}): {hits/5 * 100}%")

    # In-distribution (1 digit)
    test_generalization("In-Distribution", 1)
    # Out-of-distribution (2 digits)
    test_generalization("Generalization", 2)
    # Out-of-distribution (3 digits)
    test_generalization("Extreme Gen", 3)

if __name__ == "__main__":
    run_addition_demo()
