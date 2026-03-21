"""
Full training loop for Reality Model.

Implements:
- DataLoader with batching
- Training and validation loops
- Checkpoint saving/loading
- Curriculum learning
- Logging (console + optional wandb)
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..models import Model
from .losses.coherence import MultiDimensionalLoss


class GenericISNDataset(Dataset):
    """General purpose dataset for ISN training."""
    
    def __init__(self, data: list):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Ensure we return a dictionary with at least input_ids and output_ids
        if isinstance(item, dict):
            return item
        return {
            'input_ids': torch.tensor(item[0]),
            'output_ids': torch.tensor(item[1])
        }


class Trainer:
    """
    Complete trainer for Reality Model.
    """
    
    def __init__(
        self,
        model: Model,
        criterion: MultiDimensionalLoss,
        optimizer: torch.optim.Optimizer,
        config: dict,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Scheduler
        if config['training'].get('scheduler'):
            sched_config = config['training']['scheduler']
            if sched_config['type'] == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=sched_config['T_max'],
                    eta_min=sched_config['eta_min']
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
        
        # Gradient clipping
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)
        
        # Curriculum learning
        self.curriculum = config['training'].get('curriculum', {})
    
    def get_current_curriculum_phase(self, epoch: int) -> Optional[Dict]:
        """Get current curriculum learning phase."""
        for phase_name, phase_config in self.curriculum.items():
            epochs = phase_config['epochs']
            if epochs[0] <= epoch < epochs[1]:
                return phase_config
        return None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {
            'outcome': 0.0,
            'coherence': 0.0,
            'grounding': 0.0,
            'validity': 0.0,
            'emergence': 0.0,
            'efficiency': 0.0
        }
        
        # Get curriculum phase if applicable
        phase = self.get_current_curriculum_phase(epoch)
        if phase:
            # Update loss weights based on curriculum
            lambda_weights = phase['lambda_weights']
            self.criterion.lambda_outcome = lambda_weights['lambda_outcome']
            self.criterion.lambda_coherence = lambda_weights['lambda_coherence']
            self.criterion.lambda_grounding = lambda_weights['lambda_grounding']
            self.criterion.lambda_validity = lambda_weights['lambda_validity']
            self.criterion.lambda_emergence = lambda_weights['lambda_emergence']
            self.criterion.lambda_efficiency = lambda_weights['lambda_efficiency']
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            output_ids = batch['output_ids'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                target_ids=output_ids[:, :-1],
                return_world_state=True
            )
            
            # Inject vocab_basis for semantic loss
            if hasattr(self.model.emitter, 'emission'):
                outputs['vocab_basis'] = self.model.emitter.emission.weight
            
            # Compute loss
            targets = output_ids[:, 1:]
            
            loss_dict = self.criterion(
                targets=targets,
                **outputs
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict['loss'].item()
            for key in loss_components:
                if f'{key}_loss' in loss_dict:
                    loss_components[key] += loss_dict[f'{key}_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['loss'].item(),
                'coherence': outputs['world_coherence'].mean().item()
            })
        
        # Average losses
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        return {
            'loss': avg_loss,
            **avg_components
        }
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_coherence = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                output_ids = batch['output_ids'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    target_ids=output_ids[:, :-1],
                    return_world_state=False
                )
                
                targets = output_ids[:, 1:]
                loss_dict = self.criterion(
                    logits=outputs['logits'],
                    targets=targets,
                    world_coherence=outputs['world_coherence']
                )
                
                total_loss += loss_dict['loss'].item()
                total_coherence += outputs['world_coherence'].mean().item()
        
        n_batches = len(val_loader)
        
        return {
            'loss': total_loss / n_batches,
            'coherence': total_coherence / n_batches
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Clean old checkpoints (keep last N)
        keep_n = self.config['logging'].get('keep_n_checkpoints', 5)
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > keep_n:
            for old_ckpt in checkpoints[:-keep_n]:
                old_ckpt.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        resume_from: Optional[str] = None
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        # Resume if checkpoint provided
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_metrics['loss'])
            
            # Validation
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['loss'])
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Coherence: {val_metrics['coherence']:.3f}")
            
            if self.device.type == 'cuda':
                print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Save if frequency met OR if it's the best model
            if (epoch + 1) % self.config['logging'].get('checkpoint_frequency', 50) == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics['loss'], is_best)
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
