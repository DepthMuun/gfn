"""
Unified ISN trainer for backpropagation strategies and performance monitoring.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..registry import strategies
from .losses.coherence import MultiDimensionalLoss


class Trainer:
    """
    Unified trainer for any ISN Model and Strategy.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Baseline criteria (used for validation)
        self.criterion = MultiDimensionalLoss(
            vocab_size=config['model'].get('vocab_size', 65)
        ).to(device)
        
        # Strategy Initialization
        strategy_name = config['training'].get('backprop_strategy', 'full')
        strategy_kwargs = config['training'].get('strategy_kwargs', {})
        strategy_cls = strategies.get(strategy_name)
        self.strategy = strategy_cls(**strategy_kwargs)
        self.strategy.prepare_model(self.model)
        
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)
    
    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        last_batch_time = time.time()
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            output_ids = batch['output_ids'].to(self.device)
            targets = output_ids[:, 1:]
            
            self.optimizer.zero_grad()
            
            # 1. Forward pass (get impulses and possibly logits)
            outputs = self.model(input_ids, return_world_state=True)
            
            # 2. Strategy-specific loss computation
            # Removal of explicit logits=... to avoid TypeError: multiple values for keyword argument 'logits'
            loss_dict = self.strategy.compute_loss(
                targets=targets,
                model=self.model,
                **outputs
            )
            
            loss = loss_dict['loss']
            loss.backward()
            
            # 3. Optimization Step
            self.strategy.post_backward_hook(self.model)
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            
            # 4. Performance Metrics
            current_loss_val = loss.item()
            total_loss += current_loss_val
            batch_duration = time.time() - last_batch_time
            tps = (input_ids.size(0) * input_ids.size(1)) / batch_duration if batch_duration > 0 else 0
            last_batch_time = time.time()
            
            pbar.set_postfix({
                'loss': f"{current_loss_val:.4f}",
                'tok/s': f"{tps:.0f}"
            })
            
        return {'loss': total_loss / len(loader)}
    
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                output_ids = batch['output_ids'].to(self.device)
                outputs = self.model(input_ids)
                loss_dict = self.criterion(logits=outputs['logits'], targets=output_ids[:, 1:], **outputs)
                total_loss += loss_dict['loss'].item()
        return {'loss': total_loss / len(loader)}

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        print(f"\nStarting ISN Unified Training ({self.strategy.__class__.__name__})")
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            # Logging and Save
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best_model.pt")
                print(f"✓ New Best: {self.best_val_loss:.4f}")


class GenericISNDataset(torch.utils.data.Dataset):
    """Simple wrapper for list of dicts data."""
    def __init__(self, data: List[Dict]):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# Backward-compatible alias.
ArithmeticDataset = GenericISNDataset
