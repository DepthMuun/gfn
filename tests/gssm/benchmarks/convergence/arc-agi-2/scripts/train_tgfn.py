"""
ARC-AGI-2 Training Script for tgfn (UnifiedGFN)
===============================================
Entrenamiento few-shot de la arquitectura experimental tgfn en ARC-AGI-2.
Utiliza Cross-Entropy enmascarada para estabilidad numérica y precisión.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

# Paths
BENCHMARK_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = BENCHMARK_ROOT.parent.parent.parent.parent.parent  # Points to dev/dev/gfn

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(BENCHMARK_ROOT))

# Expose the nested tgfn realizations package under gfn.realizations.
import gfn
import gfn.realizations
gfn_tgfn_path = str(PROJECT_ROOT / "gfn" / "realizations" / "tgfn" / "gfn" / "realizations")
if gfn_tgfn_path not in gfn.realizations.__path__:
    gfn.realizations.__path__.append(gfn_tgfn_path)

from src.data.arc_dataset import create_arc_dataloader
from src.evaluation.metrics import ARCMetrics
from gfn.realizations.unified_gfn.models import UnifiedGFN


class DictWrapper:
    def __init__(self, d):
        self._d = d
    def __getattr__(self, name):
        val = self._d.get(name)
        if isinstance(val, dict):
            return DictWrapper(val)
        return val
    def get(self, name, default=None):
        return self._d.get(name, default)


class UnifiedGFNARCAdapter(nn.Module):
    """
    Adapter wrapper to project 30x30 grids (900 dims) to the embedding space
    of the modular UnifiedGFN model.
    """
    def __init__(self, config, d_embedding=64):
        super().__init__()
        self.config = config
        self.d_embedding = d_embedding
        # Proyección por celda: 10 colores → d_embedding
        self.cell_embedding = nn.Embedding(10, d_embedding)
        # Proyector espacial: 30x30*d_embedding -> d_embedding (para el GRUCell)
        # Reduce cada grid embebido a un solo vector de d_embedding dims
        self.grid_proj = nn.Linear(30 * 30 * d_embedding, d_embedding)
        self.model = UnifiedGFN(config)

    def embed_grids(self, x_seq):
        """Embed raw grid sequence into compact per-grid vectors."""
        B, L, _ = x_seq.shape
        x_indices = x_seq.round().clamp(0, 9).long()
        x_2d = x_indices.view(B, L, 30, 30)
        emb = self.cell_embedding(x_2d)
        emb = emb.view(B, L, 30 * 30 * self.d_embedding)
        emb = self.grid_proj(emb)
        return emb

    def forward(self, x_seq, targets=None, compute_loss=False):
        """
        Args:
            x_seq: [B, L, 900] float grids
            targets: same shape as x_seq (next-step supervision for JEPA)
            compute_loss: enable internal losses (policy, predictor)
        """
        emb = self.embed_grids(x_seq)

        # Embed los targets ANTES de pasar al modelo para que la JEPA loss
        # pueda calcularse sobre el espacio latente correcto.
        target_emb = self.embed_grids(targets) if (compute_loss and targets is not None) else None

        # Forward con compute_loss. El UnifiedGFN.forward manejará la JEPA loss
        # si los targets están en el espacio latente correcto (post-encoder).
        # Como target_emb ya pasó por el encoder en el forward del modelo,
        # necesitamos asegurar que tenga la forma correcta.
        # NOTA: pasamos emb sin targets, y calculamos la JEPA loss manualmente
        outputs = self.model(emb, targets=None, compute_loss=False)
        return outputs, emb, target_emb


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_raw_sequence(train_pairs, test_input, device='cpu'):
    """
    Build sequence of raw grids: [input_1, output_1, input_2, output_2, ..., test_input]
    """
    grids = []
    prediction_timesteps = []
    target_grids = []

    for i, pair in enumerate(train_pairs):
        inp = pair['input'].flatten().to(device)
        out = pair['output'].flatten().to(device)
        
        grids.append(inp)
        prediction_timesteps.append(len(grids) - 1)
        target_grids.append(out)
        
        grids.append(out)

    test_in = test_input.flatten().to(device)
    grids.append(test_in)
    prediction_timesteps.append(len(grids) - 1)

    # Stack to [B, L, 900] donde B=1
    x_seq = torch.stack(grids, dim=0).unsqueeze(0)
    return x_seq, prediction_timesteps, target_grids


def masked_cross_entropy_loss(logits, targets, height, width):
    """
    Calcula Cross Entropy únicamente sobre la región activa (no padded) del grid.
    """
    # logits: [H*W*10] -> reshape a [H, W, 10]
    # Primero reshape a [30, 30, 10] para luego recortar
    logits_3d = logits.view(30, 30, 10)
    # Recortar a tamaño real (usar reshape para evitar problemas de contigüidad)
    logits_cropped = logits_3d[:height, :width, :].reshape(-1, 10)  # [H*W, 10]

    # targets: [900] -> reshape a [30, 30] y recortar
    targets_2d = targets.view(30, 30).long()
    targets_cropped = targets_2d[:height, :width].reshape(-1)  # [H*W]

    # Loss por cada pixel en región activa
    loss_all = F.cross_entropy(
        logits_cropped,
        targets_cropped,
        reduction='none'
    )

    # Promediar sobre el área real
    return loss_all.mean()


def train_epoch(model, dataloader, optimizer, device, epoch, auxiliary_weight=0.5, jepa_weight=0.1, writer=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    running_pixel_acc = 0.0
    running_task_acc = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        optimizer.zero_grad()

        train_pairs = batch['train_pairs']
        if not train_pairs:
            continue

        test_input = batch['test_input'].to(device)
        test_output = batch.get('test_output', None)
        if test_output is not None:
            test_output = test_output.to(device)

        # Build raw sequence of grids
        x_seq, pred_timesteps, target_grids = build_raw_sequence(train_pairs, test_input, device)

        # Forward pass: pasamos el siguiente grid como target para JEPA autorregresivo
        # targets[t] = grids[t+1] (last se duplica)
        targets_seq = torch.cat([x_seq[:, 1:, :], x_seq[:, -1:, :]], dim=1)
        outputs, emb, target_emb = model(x_seq, targets=targets_seq, compute_loss=True)
        logits = outputs['policy']  # [1, L, 9000]

        # Compute JEPA loss manualmente: predice el siguiente embedding
        # Usamos la salida del encoder como current_state
        jepa_loss = torch.tensor(0.0, device=device)
        if jepa_weight > 0 and target_emb is not None and 'encoded' in outputs:
            # encoded: [B, L, state_dim] (post-encoder)
            # target_emb: [B, L, d_embedding]
            # Necesitamos target en el mismo espacio. Usamos la diferencia L2.
            current_state = outputs['encoded']  # [B, L, state_dim]
            # Predicción del predictor (sin targets, sin loss)
            predictor_out = model.model.predictor(current_state)
            predicted_next = predictor_out['prediction']  # [B, L, state_dim]

            # Para comparar, necesitamos codificar target_emb con el mismo encoder
            with torch.no_grad():
                target_encoded, _ = model.model.encoder(target_emb, None)

            jepa_loss = F.mse_loss(predicted_next, target_encoded.detach())

        # Extract predictions
        predictions = [logits[0, t, :] for t in pred_timesteps]
        train_preds = predictions[:-1]
        test_pred = predictions[-1]

        # Compute loss
        if test_output is not None and len(target_grids) > 0:
            # Primary loss (test output)
            test_target_flat = test_output.flatten()
            out_h, out_w = batch['test_output_size']
            primary_loss = masked_cross_entropy_loss(test_pred, test_target_flat, out_h, out_w)

            # Auxiliary loss (train outputs)
            if train_preds and auxiliary_weight > 0:
                aux_losses = []
                for pred, target, pair in zip(train_preds, target_grids, train_pairs):
                    pair_h, pair_w = pair['output_size']
                    aux_losses.append(masked_cross_entropy_loss(pred, target, pair_h, pair_w))
                aux_loss = torch.stack(aux_losses).mean()
            else:
                aux_loss = torch.tensor(0.0, device=device)

            # JEPA loss ya calculada arriba con peso bajo
            loss = primary_loss + auxiliary_weight * aux_loss + jepa_weight * jepa_loss

            # Real-time metrics (only on active region)
            with torch.no_grad():
                # Reshape to classes per pixel (reshape para evitar contigüidad)
                pred_classes = test_pred.view(30, 30, 10).argmax(dim=-1).cpu().numpy()
                pred_grid_2d = pred_classes[:out_h, :out_w]

                gt_grid = test_output.cpu().numpy().round().astype(np.int64)
                gt_grid_2d = gt_grid.reshape(30, 30)[:out_h, :out_w]

                correct_pixels = (pred_grid_2d == gt_grid_2d)
                task_correct = np.all(correct_pixels)
                p_acc = correct_pixels.mean()
                t_acc = 1.0 if task_correct else 0.0

                running_pixel_acc = (running_pixel_acc * num_batches + p_acc) / (num_batches + 1)
                running_task_acc = (running_task_acc * num_batches + t_acc) / (num_batches + 1)
        else:
            if train_preds and target_grids:
                aux_losses = []
                for pred, target, pair in zip(train_preds, target_grids, train_pairs):
                    pair_h, pair_w = pair['output_size']
                    aux_losses.append(masked_cross_entropy_loss(pred, target, pair_h, pair_w))
                loss = torch.stack(aux_losses).mean()
            else:
                continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        
        pbar_dict = {'loss': f'{loss.item():.4f}'}
        if test_output is not None:
            pbar_dict.update({
                'pix_acc': f'{running_pixel_acc:.2%}',
                'task_acc': f'{running_task_acc:.2%}'
            })
        pbar.set_postfix(pbar_dict)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    if writer:
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        if test_output is not None:
            writer.add_scalar('Train/PixelAccuracy', running_pixel_acc, epoch)
            writer.add_scalar('Train/TaskAccuracy', running_task_acc, epoch)
    return avg_loss


def validate(model, dataloader, device, epoch, writer=None):
    model.eval()
    task_results = []
    total_loss = 0.0
    num_tasks = 0

    with torch.no_grad():
        for batch in dataloader:
            if 'test_output' not in batch:
                continue

            train_pairs = batch['train_pairs']
            if not train_pairs:
                continue

            test_input = batch['test_input'].to(device)
            test_output = batch['test_output'].to(device)
            test_output_size = batch.get('test_output_size', None)

            # Build sequence
            x_seq, pred_timesteps, target_grids = build_raw_sequence(train_pairs, test_input, device)

            # Forward (mismo flujo que train para consistencia)
            targets_seq = torch.cat([x_seq[:, 1:, :], x_seq[:, -1:, :]], dim=1)
            outputs, _, _ = model(x_seq, targets=targets_seq, compute_loss=True)
            logits = outputs['policy']
            test_pred = logits[0, pred_timesteps[-1], :]

            # Compute loss
            test_target_flat = test_output.flatten()
            out_h, out_w = test_output_size if test_output_size is not None else (30, 30)
            loss = masked_cross_entropy_loss(test_pred, test_target_flat, out_h, out_w)
            total_loss += loss.item()

            # Reshape classes and crop prediction to original size
            pred_classes = test_pred.view(30, 30, 10).argmax(dim=-1).cpu().numpy()
            pred_2d = pred_classes[:out_h, :out_w]

            # Ground truth
            gt_np = test_output.cpu().numpy().round().astype(np.int64)
            gt_2d = gt_np.reshape(30, 30)[:out_h, :out_w]

            metrics = ARCMetrics.evaluate_task(
                pred_2d, gt_2d,
                pred_size=(out_h, out_w),
                true_size=(out_h, out_w)
            )
            task_results.append(metrics)
            num_tasks += 1

    aggregated = ARCMetrics.aggregate_metrics(task_results)
    avg_loss = total_loss / num_tasks if num_tasks > 0 else 0.0

    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/TaskAccuracy', aggregated['task_accuracy'], epoch)
        writer.add_scalar('Val/PixelAccuracy', aggregated['mean_pixel_accuracy'], epoch)

    return avg_loss, aggregated


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train tgfn on ARC-AGI-2")
    parser.add_argument("--data_path", type=str, default="data/processed/splits", help="Path to processed ARC data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jepa_weight", type=float, default=0.1, help="Weight for JEPA loss")
    parser.add_argument("--aux_weight", type=float, default=0.5, help="Weight for auxiliary loss")
    args = parser.parse_args()

    set_seed(args.seed)

    # Setup directories
    output_dir = Path("results_tgfn")
    output_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir / "logs")

    # Modular unified_gfn (tgfn) config
    # d_embedding = dimensión del embedding por celda (32, 64, 128...)
    # vocab_size = número de símbolos (colores ARC: 10)
    tgfn_config = {
        'encoder_type': 'hierarchical',
        'encoder_config': {
            'd_embedding': 64,   # tamaño del embedding por celda
            'd_world': 64,
            'd_local': 64,
            'vocab_size': 10,   # 10 colores ARC
            'dropout': 0.1
        },
        'attention_type': 'geodesic',
        'attention_config': DictWrapper({
            'd_model': 128,
            'num_heads': 4,
            'dropout': 0.1,
            'temperature': 1.0,
            'use_bias': True,
            'curvature': 1.0,
            'custom_params': {'dim': 128}
        }),
        'predictor_type': 'jepa',
        'predictor_config': {
            'state_dim': 128,  # d_world + d_local
            'hidden_dim': 128,
            'dropout': 0.1,
            'momentum': 0.99
        },
        'heads_config': {
            'policy': {
                'type': 'linear',
                'output_dim': 9000  # 900 pixels * 10 colors
            }
        },
        'model_config': {
            'persistent_state': False
        }
    }

    print("=" * 60)
    print("TGFN (UnifiedGFN) ARC-AGI-2 Training (Cross-Entropy)")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    # Instantiate model
    model = UnifiedGFNARCAdapter(tgfn_config, d_embedding=64).to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load data
    train_loader = create_arc_dataloader(args.data_path, split="train", batch_size=1)
    val_loader = create_arc_dataloader(args.data_path, split="val", batch_size=1)

    # Optimizer con scheduler cosine para estabilizar el loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_acc = 0.0
    best_epoch = 0

    # Loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, args.device, epoch,
            auxiliary_weight=args.aux_weight, jepa_weight=args.jepa_weight, writer=writer
        )
        val_loss, metrics = validate(model, val_loader, args.device, epoch, writer=writer)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | LR = {current_lr:.2e}")
        print(f"         Val Acc = {metrics['task_accuracy']:.2%} | Pixel Acc = {metrics['mean_pixel_accuracy']:.2%}")

        # Update JEPA target network (EMA)
        if model.model.predictor is not None:
            model.model.predictor.update_target()

        # Save best model
        val_acc = metrics['task_accuracy']
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "tgfn_arc_best.pt")
            print(f"         → New best model saved (val_acc={val_acc:.2%})")

    print(f"\nBest epoch: {best_epoch} with val_acc={best_val_acc:.2%}")

    # Save final model
    torch.save(model.state_dict(), output_dir / "tgfn_arc_final.pt")
    print(f"Model saved to {output_dir / 'tgfn_arc_final.pt'}")


if __name__ == "__main__":
    main()
