"""
Drone Detection — GFN V5 Prediction Script
===========================================
Usa los mismos 3 componentes que train.py:
  ContinuousImageProjector → GFN ManifoldModel → DetectionHead

El checkpoint es un dict con claves: 'projector', 'manifold', 'det_head'
generado por train.py al guardar drone_best.pt / drone_latest.pt.

Run:
    $env:PYTHONPATH="d:\\ASAS\\manifold_mini\\manifold_working"
    python predict.py [--ckpt path/to/drone_best.pt] [--img-size 32] [--num-samples 10]
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw

import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import build_detector
from data import download_and_extract, get_pure_dataloader
from gfn.realizations.gssm.utils.coords import torus_to_box


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — idénticos a train.py
# ══════════════════════════════════════════════════════════════════════════════

# torus_to_box is now imported from gfn.realizations.gssm.utils.coords


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def run_prediction(
    model_path: str,
    img_size: int = 32,
    dim: int = 64,
    num_samples: int = 10,
    dataset_dir: str = "D:/ASAS/datasets/seraphim",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[GFN Drone Detection — predict] Device: {device}")

    ckpt = torch.load(model_path, map_location=device)

    # ── 1. Construir arquitectura ──────────────────────────────────────────────
    projector, manifold, det_head = build_detector(
        img_size=img_size, dim=dim, depth=2, heads=4, rank=64,
    )
    projector = projector.to(device)
    manifold  = manifold.to(device)
    det_head  = det_head.to(device)

    # ── 2. Cargar projector y manifold (tienen parametros desde __init__) ──────
    projector.load_state_dict(ckpt['projector'])

    # Filtrar PhysicsMonitorPlugin registrado durante training vía add_module()
    manifold_sd = {
        k: v for k, v in ckpt['manifold'].items()
        if not k.startswith('physics_monitor.')
    }
    manifold.load_state_dict(manifold_sd, strict=True)

    # ── 3. Warm-up del DetectionHead (lazy init) ───────────────────────────────
    # DetectionHead.head se construye dentro de forward() con self.head = nn.Sequential(...).
    # PyTorch solo registra submodulos asignados en __init__ o via add_module().
    # Si llamamos load_state_dict antes del forward, head no existe → "Unexpected keys".
    # Solución: forward dummy con pesos ya cargados, luego sobreescribir con checkpoint.
    with torch.no_grad():
        _f = projector(torch.zeros(1, 3 * img_size * img_size, device=device))
        _, (_xf, _), _ = manifold(force_manual=_f)
        det_head(_xf)   # construye self.head con dimensiones correctas

    det_head.load_state_dict(ckpt['det_head'])

    projector.eval()
    manifold.eval()
    det_head.eval()

    epoch_info = ckpt.get('epoch', '?')
    score_info = ckpt.get('score')
    score_str  = f"{score_info:.4f}" if score_info is not None else '?'
    print(f"  Checkpoint cargado: {model_path}  (epoch={epoch_info}, score={score_str})")

    # ── 4. Dataset ─────────────────────────────────────────────────────────────
    root = download_and_extract(local_dir=dataset_dir)
    test_loader = get_pure_dataloader(
        root, split='test', batch_size=1, img_size=img_size,
        max_samples=num_samples, include_empty=True, shuffle=False,
    )

    # ── 5. Output dir ──────────────────────────────────────────────────────────
    out_dir = Path("tests/benchmarks/convergence/drone_detection/output")
    out_dir.mkdir(exist_ok=True, parents=True)

    correct_obj = 0
    total       = 0

    for i, (imgs, labels) in enumerate(test_loader):
        if i >= num_samples:
            break

        imgs   = imgs.to(device)    # [1, 3*H*W]
        labels = labels.to(device)  # [1, 5]

        with torch.no_grad():
            forces          = projector(imgs)
            _, (x_f, _), _  = manifold(force_manual=forces)
            pred            = det_head(x_f)               # [1, 5] en [-pi, pi]

        raw_pred  = pred[0].cpu()
        raw_label = labels[0].cpu()

        # Objectness: cos(angle) < 0 → Drone (cerca de pi)
        is_drone_pred = torch.cos(raw_pred[0]).item() < 0
        is_drone_true = raw_label[0].item() > 0.5

        if is_drone_pred == is_drone_true:
            correct_obj += 1
        total += 1

        # ── Visualización ──────────────────────────────────────────────────────
        img_np = (imgs[0].cpu().reshape(3, img_size, img_size)
                         .permute(1, 2, 0)
                         .clamp(0, 1)
                         .numpy() * 255).astype("uint8")
        pil_img = Image.fromarray(img_np).resize((256, 256))
        draw    = ImageDraw.Draw(pil_img)
        W, H    = pil_img.size

        # Caja predicha (rojo)
        if is_drone_pred:
            boxes_01       = torus_to_box(raw_pred[1:].unsqueeze(0)).squeeze(0)
            px, py, pw, ph = boxes_01.tolist()
            x1, y1 = (px - pw / 2) * W, (py - ph / 2) * H
            x2, y2 = (px + pw / 2) * W, (py + ph / 2) * H
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            draw.text((x1 + 2, y1 + 2), f"PRED (th={raw_pred[0]:.2f})", fill="red")

        # Caja ground-truth (verde)
        if is_drone_true:
            tx, ty, tw, th = raw_label[1:].tolist()
            tx1, ty1 = (tx - tw / 2) * W, (ty - th / 2) * H
            tx2, ty2 = (tx + tw / 2) * W, (ty + th / 2) * H
            draw.rectangle([tx1, ty1, tx2, ty2], outline="#00FF00", width=2)
            draw.text((tx1 + 2, ty1 + 15), "TARGET", fill="#00FF00")

        out_path = out_dir / f"result_{i:04d}.jpg"
        pil_img.save(out_path)
        print(
            f"  [{i:04d}] Pred: {'Drone' if is_drone_pred else 'BG':5s} | "
            f"True: {'Drone' if is_drone_true else 'BG':5s} | "
            f"{'OK' if is_drone_pred == is_drone_true else 'X '} -> {out_path}"
        )

    # ── Resumen ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 40)
    print(f"PREDICTION SUMMARY ({total} muestras)")
    print(f"Objectness Accuracy: {100.0 * correct_obj / max(total, 1):.1f}%")
    print(f"Resultados en: {out_dir}")
    print("=" * 40)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GFN Drone Detection — prediccion')
    parser.add_argument(
        '--ckpt', type=str,
        default='tests/benchmarks/convergence/drone_detection/checkpoints/drone_best.pt',
    )
    parser.add_argument('--img-size',    type=int, default=32)
    parser.add_argument('--dim',         type=int, default=64,
                        help='Debe coincidir con el dim usado en train.py')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--dataset-dir', type=str, default='D:/ASAS/datasets/seraphim')
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint no encontrado: {ckpt_path}")
        print("        Ejecuta train.py primero para generar el checkpoint.")
    else:
        run_prediction(
            model_path=str(ckpt_path),
            img_size=args.img_size,
            dim=args.dim,
            num_samples=args.num_samples,
            dataset_dir=args.dataset_dir,
        )
