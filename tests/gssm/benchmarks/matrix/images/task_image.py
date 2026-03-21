import torch
import torch.nn as nn
import torch.optim as optim
import math
from pathlib import Path
from gfn.losses import ToroidalDistanceLoss
from tests.benchmarks.convergence.drone_detection.data import download_and_extract, get_pure_dataloader

def box_to_torus(boxes):
    """Maps boxes from [0, 1] to [-pi, pi] for Toroidal consistency."""
    boxes_clamped = torch.clamp(boxes, 0.0, 1.0)
    return (boxes_clamped * 2.0 * torch.pi) - torch.pi

def torus_to_box(angles):
    """Maps angles from anywhere to [0, 1] for metric evaluation."""
    wrapped = torch.atan2(torch.sin(angles), torch.cos(angles))
    return (wrapped + torch.pi) / (2.0 * torch.pi)

def ciou_loss(pred_boxes, target_boxes):
    """Box regression loss using Toroidal Distance."""
    return ToroidalDistanceLoss({'mode': 'mse'})(pred_boxes, target_boxes)

class ImageTaskAdapter(nn.Module):
    """
    Adapts a GFN manifold model to the drone detection task.
    Handles input projection andprediction head.
    """
    def __init__(self, manifold, img_size=32):
        super().__init__()
        self.manifold = manifold
        self.img_size = img_size
        self.dim_in = 3 * img_size * img_size
        
        # Determine manifold internal dim
        self.manifold_dim = manifold.config.dim
        self.heads = manifold.config.heads
        self.geometry_scope = getattr(manifold.config.physics.topology, 'geometry_scope', 'local')
        
        # Input projection if needed
        if self.dim_in != self.manifold_dim:
            self.input_proj = nn.Linear(self.dim_in, self.manifold_dim)
        else:
            self.input_proj = nn.Identity()
            
        # Total dimension for the linear head
        self.dim_total = self.manifold_dim * self.heads if self.geometry_scope == 'global' else self.manifold_dim
        
        # Deeper head for more capacity to map manifold state to boxes
        # We use 2 * dim_total because of the [sin, cos] projection in forward
        self.head = nn.Sequential(
            nn.LayerNorm(self.dim_total * 2),
            nn.Linear(self.dim_total * 2, self.manifold_dim),
            nn.GELU(),
            nn.Linear(self.manifold_dim, 5)
        )
        
    def forward(self, x):
        # x: [B, D_in]
        x_proj = self.input_proj(x)
        x_seq = x_proj.unsqueeze(1) # [B, 1, D_manifold]
        
        _, (_, _), telemetry = self.manifold(force_manual=x_seq)
        x_final = telemetry["x_final"] # [B, H, HD] or [B, D] 
        
        # Periodic Projection: Map manifold coordinates to [sin, cos]
        # This is CRITICAL for toroidal stability as it preserves circular continuity
        x_flat = x_final.flatten(1)
        x_periodic = torch.cat([torch.sin(x_flat), torch.cos(x_flat)], dim=-1)
        
        # The head now receives a periodic representation
        pred = self.head(x_periodic)
        
        # All outputs are angles in [-pi, pi]
        # Pred[0] is Objectness Angle (0 = BG, pi = Drone)
        # Pred[1:] are Box coordinates
        return torch.tanh(pred) * math.pi

def train_image(model, steps=100, dataset_dir="D:/ASAS/datasets/seraphim", img_size=32, batch_size=16, loss_mode='mse', lambda_geo=0.0, overrides=None):
    device = next(model.parameters()).device
    
    # Extract from overrides if present
    if overrides:
        loss_mode = overrides.get('loss_mode', loss_mode)
        lambda_geo = overrides.get('lambda_geo', lambda_geo)
    
    # 1. wrap model with adapter
    adapter = ImageTaskAdapter(model, img_size=img_size).to(device)
    
    # 2. Data
    root = download_and_extract(local_dir=dataset_dir)
    train_loader = get_pure_dataloader(
        root, split="train", batch_size=batch_size, img_size=img_size,
        max_samples=200, include_empty=True
    )
    
    # Stable LR for detection - increased stability with geometric objective
    optimizer = optim.AdamW(adapter.parameters(), lr=5e-4)
    
    # Configurable Toroidal Loss - Used for BOTH objectness and boxes
    toroidal_loss_fn = ToroidalDistanceLoss({'mode': loss_mode})
    
    # Optional Physics Loss
    from gfn.losses.physics import PhysicsLoss
    physics_loss_fn = PhysicsLoss({'lambda_geo': lambda_geo}) if lambda_geo > 0 else None
    
    adapter.train()
    last_loss = 0.0
    processed_steps = 0
    
    while processed_steps < steps:
        for x, y in train_loader:
            if processed_steps >= steps: break
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Forward adapter
            y_pred = adapter(x)
            
            # Target mapping:
            # y[:, 0] is objectness (0 or 1)
            # We map 0 -> 0.0 rad (Background)
            # We map 1 -> pi rad (Drone)
            obj_target = y[:, 0] * math.pi
            
            # 1. Objectness Loss (Toroidal)
            obj_loss = toroidal_loss_fn(y_pred[:, 0], obj_target)
            
            # 2. Box Loss (Toroidal) - only for drones
            mask = y[:, 0] == 1
            if mask.any():
                y_torus = box_to_torus(y[mask, 1:])
                box_loss = toroidal_loss_fn(y_pred[mask, 1:], y_torus)
            else:
                box_loss = torch.tensor(0.0, device=device)
                
            # Total Loss - purely geometric
            loss = obj_loss + 2.0 * box_loss
            
            # Physics Regularization (if enabled)
            if physics_loss_fn:
                # Need to get telemetry again or pass it through
                _, _, telemetry = adapter.manifold(force_manual=adapter.input_proj(x).unsqueeze(1))
                p_loss = physics_loss_fn(y_pred, y_pred, state_info=telemetry)
                loss = loss + p_loss
                
            loss.backward()
            optimizer.step()
            
            last_loss = loss.item()
            processed_steps += 1
            
            if processed_steps % 25 == 0:
                # Calculate "accuracy" for logging: 
                # drone if |dist(pred, pi)| < |dist(pred, 0)|
                # d_pi = pi - |pi - |pred - obj_target|| ... actually easier:
                # check if cos(pred) < 0 (closer to pi) vs cos(pred) > 0 (closer to 0)
                is_drone_pred = torch.cos(y_pred[:, 0]) < 0
                acc = (is_drone_pred == (y[:, 0] == 1)).float().mean().item()
                
                print(f"    [Step {processed_steps}] Loss: {last_loss:.4f} (Obj: {obj_loss.item():.3f}, Box: {box_loss.item():.3f}, Acc: {acc:.2%})", flush=True)

    # Store adapter for eval convenience
    return {"steps": processed_steps, "loss": last_loss, "adapter": adapter}

def eval_image(model, dataset_dir="D:/ASAS/datasets/seraphim", img_size=32, adapter=None):
    device = next(model.parameters()).device
    if adapter is None:
        # Fallback if not provided, though runner should pass it
        adapter = ImageTaskAdapter(model, img_size=img_size).to(device)
    
    adapter.eval()
    
    root = download_and_extract(local_dir=dataset_dir)
    test_loader = get_pure_dataloader(
        root, split="test", batch_size=8, img_size=img_size,
        max_samples=40, include_empty=True, shuffle=False
    )
    
    obj_correct = 0
    obj_total = 0
    box_loss_total = 0.0
    box_count = 0
    mse = nn.MSELoss(reduction="sum")
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = adapter(x)
            
            is_drone_pred = torch.cos(pred[:, 0]) < 0
            obj_true = (y[:, 0] == 1)
            obj_correct += (is_drone_pred == obj_true).sum().item()
            obj_total += obj_true.numel()
            
            mask = obj_true # True where it's a drone
            if mask.any():
                # pred[mask, 1:] are the box angles
                pred_boxes_01 = torus_to_box(pred[mask, 1:])
                box_loss_total += mse(pred_boxes_01, y[mask, 1:]).item()
                box_count += mask.sum().item()
                
    obj_acc = obj_correct / max(obj_total, 1)
    box_mse = box_loss_total / max(box_count, 1)
    
    # Return accuracy combined or objectness accuracy for simplicity
    return obj_acc
