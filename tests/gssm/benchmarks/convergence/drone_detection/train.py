"""
Drone Detection — GFN V5 Training
===================================
Post-audit v4 — usa framework imports, sin duplicación de código.

Framework imports:
    from gfn.losses import GIoULoss, ToroidalDistanceLoss
    from gfn.realizations.gssm.utils.coords import box_to_torus, torus_to_box
    from gfn.training.optimizer import make_gfn_optimizer, all_parameters
    from gfn.training.checkpoint import save_checkpoint
    from gfn.realizations.gssm.physics.monitor import PhysicsMonitorPlugin

Run:
    $env:PYTHONPATH="d:\\ASAS\\manifold_mini\\manifold_working"
    python train.py [--epochs 5] [--batch-size 4]
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[0].parent  # benchmarks/convergence/drone_detection -> benchmarks/convergence -> convergence -> tests -> ROOT (which contains gfn/)
# Direct check: we need the directory that CONTAINS the 'gfn' folder.
# Structure: D:\ASAS\manifold_mini\gfn_2.6.6\gfn\tests\gssm\benchmarks\convergence\drone_detection\train.py
# gfn/ is at: D:\ASAS\manifold_mini\gfn_2.6.6\gfn\gfn
# So we need PROJECT_ROOT = D:\ASAS\manifold_mini\gfn_2.6.6\gfn
# That is 5 levels up from train.py: drone_detection (0), convergence (1), benchmarks (2), gssm (3), tests (4), gfn (5)
PROJECT_ROOT = HERE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Benchmark-local ───────────────────────────────────────────────────────────
from model import build_detector
from data  import download_and_extract, get_pure_dataloader

# ── Framework imports — lo que antes vivía en este archivo ───────────────────
from gfn.realizations.gssm.losses.detection import GIoULoss
from gfn.realizations.gssm.losses.toroidal import ToroidalDistanceLoss
from gfn.realizations.gssm.losses.physics import PhysicsLoss
from gfn.realizations.gssm.utils.coords import box_to_torus, torus_to_box
from gfn.realizations.gssm.training.optimizer import make_gfn_optimizer, all_parameters
from gfn.realizations.gssm.training.checkpoint import save_checkpoint
from gfn.realizations.gssm.physics.monitor import PhysicsMonitorPlugin

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(projector, manifold, det_head, loader, device):
    projector.eval(); manifold.eval(); det_head.eval()
    obj_correct = obj_total = 0
    box_mse_total = box_count = iou_total = 0

    giou = GIoULoss()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        forces = projector(imgs)
        _, (x_f, _), _ = manifold(force_manual=forces)
        pred = det_head(x_f)

        is_pred_drone = torch.cos(pred[:, 0]) < 0
        is_true_drone = labels[:, 0] == 1
        obj_correct  += (is_pred_drone == is_true_drone).sum().item()
        obj_total    += labels.shape[0]

        mask = is_true_drone
        if mask.any():
            boxes_01 = torus_to_box(pred[mask, 1:])
            box_mse_total += nn.functional.mse_loss(
                boxes_01, labels[mask, 1:], reduction='sum'
            ).item()
            iou_val = 1.0 - giou(boxes_01.clamp(0, 1), labels[mask, 1:].clamp(0, 1))
            iou_total += iou_val.item() * mask.sum().item()
            box_count += mask.sum().item()

    projector.train(); manifold.train(); det_head.train()
    return (
        obj_correct  / max(obj_total, 1),
        box_mse_total / max(box_count, 1),
        iou_total    / max(box_count, 1),
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_drone(
    epochs:             int   = 10,
    img_size:           int   = 64,
    patch_size:         int   = 16,
    dim:                int   = 64,
    batch_size:         int   = 4,
    lr:                 float = 2e-4,
    max_lr:             float = 5e-4,
    dataset_dir:        str   = "D:/ASAS/datasets/seraphim",
    max_train_samples:  int   = 2000000,
    max_test_samples:   int   = 4000,
    max_steps:          int   = None,
    monitor_energy:     bool  = True,
    iou_weight:         float = 2.0,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[GFN Drone Detection] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    root = download_and_extract(local_dir=dataset_dir)
    train_loader = get_pure_dataloader(
        root, split='train', batch_size=batch_size, img_size=img_size,
        max_samples=max_train_samples, include_empty=True,
    )
    test_loader = get_pure_dataloader(
        root, split='test', batch_size=batch_size, img_size=img_size,
        max_samples=max_test_samples, include_empty=True, shuffle=False,
    )

    # ── Modelo ────────────────────────────────────────────────────────────────
    projector, manifold, det_head = build_detector(
        img_size=img_size, patch_size=patch_size, dim=dim, depth=2, heads=4, rank=64,
    )
    projector = projector.to(device)
    manifold  = manifold.to(device)
    det_head  = det_head.to(device)

    n_patches = (img_size // patch_size) ** 2
    n_params  = sum(p.numel() for p in all_parameters(projector, manifold, det_head))
    print(f"  Params: {n_params:,} | img={img_size} | patches={n_patches} | dim={dim}")

    # ── PhysicsMonitor ────────────────────────────────────────────────────────
    physics_monitor = None
    if monitor_energy:
        try:
            geometry = manifold.layers[0].integrator.physics_engine.geometry
            physics_monitor = PhysicsMonitorPlugin(geometry=geometry, enabled=True, window=32)
            physics_monitor.register_hooks(manifold.hooks)
            manifold.add_module('physics_monitor', physics_monitor)
            print(f"  PhysicsMonitor: activo ({type(geometry).__name__})")
        except Exception as e:
            print(f"  PhysicsMonitor: no disponible ({e})")

    # ── Optimizer (framework) ─────────────────────────────────────────────────
    optimizer = make_gfn_optimizer(
        manifold,
        lr=lr,
        extra_modules=[projector, det_head],
    )

    total_steps = max(max_steps or (len(train_loader) * epochs), 1)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.2,
    )

    # ── Losses (framework) ────────────────────────────────────────────────────
    # hybrid mode connects d^2 with (1-cos d) for better global-local gradients
    toro_loss  = ToroidalDistanceLoss(config={'mode': 'hybrid'})
    giou_crit  = GIoULoss(weight=1.0) # weight=1.0 because we scale it by 2.0 later
    phys_reg   = PhysicsLoss({'lambda_geo': 0.01, 'lambda_ham': 0.005})

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    ckpt_dir = Path("tests/benchmarks/convergence/drone_detection/checkpoints")
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step  = 0
    best_score   = -float('inf')
    all_params   = all_parameters(projector, manifold, det_head)

    projector.train(); manifold.train(); det_head.train()

    for epoch in range(epochs):
        total_loss    = 0.0
        total_box_mse = 0.0
        box_batches   = 0
        epoch_correct = 0
        epoch_total   = 0
        drift_vals    = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            forces = projector(imgs)
            _, (x_f, _), state_info = manifold(force_manual=forces)
            pred = det_head(x_f)

            if physics_monitor is not None:
                drift_vals.append(physics_monitor.energy_drift)

            # Objectness loss (toroidal)
            obj_target = labels[:, 0] * math.pi
            loss_obj   = toro_loss(pred[:, 0], obj_target)

            # Box loss (toroidal + GIoU)
            drone_mask = labels[:, 0] == 1
            if drone_mask.any():
                y_torus   = box_to_torus(labels[drone_mask, 1:])
                loss_toro = toro_loss(pred[drone_mask, 1:], y_torus)
                pred_01   = torus_to_box(pred[drone_mask, 1:])
                loss_giou = giou_crit(pred_01, labels[drone_mask, 1:])
                loss_box  = loss_toro + loss_giou
                with torch.no_grad():
                    total_box_mse += nn.functional.mse_loss(
                        pred_01, labels[drone_mask, 1:]
                    ).item()
                box_batches += 1
            else:
                loss_box = torch.zeros((), device=device)

            # Optional physics regularization
            loss_phys = phys_reg(pred, obj_target, state_info=state_info)

            loss = loss_obj + 2.0 * loss_box + loss_phys
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            with torch.no_grad():
                is_pred_drone  = torch.cos(pred[:, 0]) < 0
                epoch_correct += (is_pred_drone == (labels[:, 0] == 1)).sum().item()
                epoch_total   += labels.shape[0]

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{epoch_correct/max(epoch_total,1):.1%}",
                box=f"{total_box_mse/max(box_batches,1):.4f}",
                drift=f"{sum(drift_vals)/len(drift_vals) if drift_vals else 0:.2f}",
            )

            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break

        # ── Validation ────────────────────────────────────────────────────────
        val_acc, val_mse, val_iou = evaluate(projector, manifold, det_head, test_loader, device)
        train_acc  = epoch_correct / max(epoch_total, 1)
        avg_loss   = total_loss / max(len(train_loader), 1)
        score      = val_acc + val_iou - val_mse
        mean_drift = sum(drift_vals) / len(drift_vals) if drift_vals else 0.0

        is_best = score > best_score
        if is_best:
            best_score = score
            # ── save_checkpoint (framework) ───────────────────────────────────
            save_checkpoint(
                ckpt_dir / "drone_best.pt",
                modules={'projector': projector, 'manifold': manifold, 'det_head': det_head},
                metadata={'epoch': epoch, 'score': score,
                          'img_size': img_size, 'patch_size': patch_size, 'dim': dim},
            )

        save_checkpoint(
            ckpt_dir / "drone_latest.pt",
            modules={'projector': projector, 'manifold': manifold, 'det_head': det_head},
            metadata={'img_size': img_size, 'patch_size': patch_size, 'dim': dim},
        )

        drift_warn = " ⚠ HIGH DRIFT" if mean_drift > 100 else ""
        print(
            f"\nEpoch {epoch:02d} | Loss: {avg_loss:.4f} | "
            f"Train: {train_acc:.1%} | Val: {val_acc:.1%} | "
            f"MSE: {val_mse:.4f} | IoU: {val_iou:.3f} | "
            f"Drift: {mean_drift:.1f}{drift_warn} {'[BEST]' if is_best else ''}",
            flush=True,
        )

        if max_steps is not None and global_step >= max_steps:
            break

    print(f"\nTraining done. Best score: {best_score:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GFN Drone Detection Training')
    parser.add_argument('--epochs',            type=int,   default=10)
    parser.add_argument('--img-size',          type=int,   default=64)
    parser.add_argument('--patch-size',        type=int,   default=16)
    parser.add_argument('--dim',               type=int,   default=64)
    parser.add_argument('--batch-size',        type=int,   default=4)
    parser.add_argument('--lr',                type=float, default=2e-4)
    parser.add_argument('--max-lr',            type=float, default=5e-4)
    parser.add_argument('--max-steps',         type=int,   default=None)
    parser.add_argument('--max-train-samples', type=int,   default=2000000)
    parser.add_argument('--max-test-samples',  type=int,   default=4000)
    parser.add_argument('--iou-weight',        type=float, default=2.0)
    parser.add_argument('--no-monitor',        action='store_true')
    args = parser.parse_args()
    train_drone(
        epochs=args.epochs,
        img_size=args.img_size,
        patch_size=args.patch_size,
        dim=args.dim,
        batch_size=args.batch_size,
        lr=args.lr,
        max_lr=args.max_lr,
        max_steps=args.max_steps,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        monitor_energy=not args.no_monitor,
        iou_weight=args.iou_weight,
    )
