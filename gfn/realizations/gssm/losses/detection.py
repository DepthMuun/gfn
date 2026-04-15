"""
gfn/losses/detection.py
=======================
Losses for object detection — generic, not coupled to any domain.

Design: GIoU and IoU work on boxes [cx, cy, w, h] in coordinates [0,1].
They are manifold-independent losses — used together with ToroidalDistanceLoss
for bounding box regression from toroidal space.

Typical usage:
    from ..losses import GIoULoss
    criterion = GIoULoss()
    loss = criterion(pred_01, target_01)    # [B, 4] in [cx, cy, w, h] ∈ [0,1]
"""

import torch
import torch.nn as nn

__all__ = ['GIoULoss', 'IoULoss', 'giou_loss', 'iou_loss']


# ══════════════════════════════════════════════════════════════════════════════
# PURE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """[cx, cy, w, h] → [x1, y1, x2, y2]"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([cx - w / 2, cy - h / 2,
                        cx + w / 2, cy + h / 2], dim=-1)


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU Loss for boxes [cx, cy, w, h] ∈ [0, 1].

    GIoU = IoU - |C \\ (A∪B)| / |C|
    Loss = 1 - GIoU  ∈ [0, 2]

    Args:
        pred:   [B, 4]  — predicted boxes  [cx, cy, w, h] in [0,1]
        target: [B, 4]  — target boxes   [cx, cy, w, h] in [0,1]

    Returns:
        Scalar — mean GIoU loss over the batch.
    """
    pred_xyxy   = _cxcywh_to_xyxy(pred)
    target_xyxy = _cxcywh_to_xyxy(target.clamp(0, 1))

    # ── Intersection ──────────────────────────────────────────────────────────
    inter_x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # ── Union ─────────────────────────────────────────────────────────────────
    pred_area   = pred[:, 2].clamp(0)   * pred[:, 3].clamp(0)
    target_area = target[:, 2].clamp(0) * target[:, 3].clamp(0)
    union_area  = pred_area + target_area - inter_area + 1e-7

    iou = inter_area / union_area

    # ── Enclosing box (for GIoU additional term) ──────────────────────────────
    enc_x1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
    enc_y1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
    enc_x2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
    enc_y2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
    enc_area = ((enc_x2 - enc_x1) * (enc_y2 - enc_y1)).clamp(1e-7)

    giou = iou - (enc_area - union_area) / enc_area
    return (1.0 - giou).mean()


def iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Standard IoU Loss (without GIoU enclosing box term).
    Loss = 1 - IoU  ∈ [0, 1]

    Args:
        pred:   [B, 4]  — predicted boxes  [cx, cy, w, h] in [0,1]
        target: [B, 4]  — target boxes   [cx, cy, w, h] in [0,1]
    """
    pred_xyxy   = _cxcywh_to_xyxy(pred)
    target_xyxy = _cxcywh_to_xyxy(target.clamp(0, 1))

    inter_x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    pred_area   = pred[:, 2].clamp(0)   * pred[:, 3].clamp(0)
    target_area = target[:, 2].clamp(0) * target[:, 3].clamp(0)
    union_area  = pred_area + target_area - inter_area + 1e-7

    iou = inter_area / union_area
    return (1.0 - iou).mean()


# ══════════════════════════════════════════════════════════════════════════════
# nn.Module CLASSES — for use inside standard training loops
# ══════════════════════════════════════════════════════════════════════════════

class GIoULoss(nn.Module):
    """
    Generalized IoU Loss as nn.Module.

    Usage:
        criterion = GIoULoss(weight=2.0)
        loss = criterion(pred_boxes_01, target_boxes_01)
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   [B, 4]  [cx, cy, w, h] ∈ [0,1]
        target: [B, 4]  [cx, cy, w, h] ∈ [0,1]
        """
        return giou_loss(pred, target) * self.weight


class IoULoss(nn.Module):
    """
    Standard IoU Loss as nn.Module.

    Usage:
        criterion = IoULoss()
        loss = criterion(pred_boxes_01, target_boxes_01)
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return iou_loss(pred, target) * self.weight
