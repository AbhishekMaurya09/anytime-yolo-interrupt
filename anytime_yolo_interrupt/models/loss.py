"""
models/loss.py
----------------
From-scratch anchor-free detection loss, applied INDEPENDENTLY to every
anytime tap (brief section 21: "Total Loss = L_head1 + lambda2*L_head2
+ lambda3*L_head3 + lambda_final*L_final", weights configurable).

Target assignment (deliberately simplified vs. real YOLOv9's TAL/SimOTA,
so it can be implemented correctly from scratch without a live GPU to
debug against):
  1. Each GT box is assigned to exactly ONE FPN level, by size
     (small -> P3/stride8, medium -> P4/stride16, large -> P5/stride32),
     the same idea as FCOS/YOLOv3's scale-to-level assignment.
  2. Within that level, it is assigned to the grid cell whose
     center contains the box's center, plus any of its 8 neighbours whose
     own centre lies inside the box (centre sampling, up to 9 per GT).
  3. If two GT boxes land on the same cell at the same level, the
     smaller box wins (standard practice -- prevents a big box from
     starving small-object supervision).

Per scale, per tap:
  - obj loss:  BCE-with-logits, target=1 at assigned cells, 0 elsewhere,
               over ALL cells (this is what teaches the network where
               there is/isn't an object).
  - cls loss:  BCE-with-logits, one-hot target, ONLY at assigned cells
               (background cells contribute no classification gradient --
               objectness already handles "is there anything here").
  - box loss:  CIoU loss between the decoded box and the assigned GT
               box, ONLY at assigned cells.

This is intentionally close to a simplified YOLOX loss. It is a
legitimate "slight modification" of the anytime-YOLO idea: multiple
independently-supervised anytime taps, trained end-to-end, with
per-head accuracy measurable independently (evaluation/evaluate_heads.py
already built earlier handles that part).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .detect_head import make_grid


def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """pred_boxes, target_boxes: (N, 4) xyxy. Returns (N,) per-box CIoU loss (1 - CIoU)."""
    px1, py1, px2, py2 = pred_boxes.unbind(-1)
    tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)

    # Intersection
    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    # Enclosing box
    cx1 = torch.minimum(px1, tx1)
    cy1 = torch.minimum(py1, ty1)
    cx2 = torch.maximum(px2, tx2)
    cy2 = torch.maximum(py2, ty2)
    c_diag_sq = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2) + eps

    p_cx, p_cy = (px1 + px2) / 2, (py1 + py2) / 2
    t_cx, t_cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    center_dist_sq = (p_cx - t_cx).pow(2) + (p_cy - t_cy).pow(2)

    w_p, h_p = (px2 - px1).clamp(min=eps), (py2 - py1).clamp(min=eps)
    w_t, h_t = (tx2 - tx1).clamp(min=eps), (ty2 - ty1).clamp(min=eps)
    v = (4 / (3.14159265 ** 2)) * (torch.atan(w_t / h_t) - torch.atan(w_p / h_p)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_dist_sq / c_diag_sq) - alpha * v
    return 1.0 - ciou


@dataclass
class LevelTargets:
    pos_mask: torch.Tensor      # (B, H, W) bool
    target_boxes: torch.Tensor  # (B, H, W, 4) valid only where pos_mask
    target_cls: torch.Tensor    # (B, H, W) long, valid only where pos_mask


def _level_for_box(w: float, h: float, small_max: float = 64.0, medium_max: float = 128.0) -> int:
    m = max(w, h)
    if m < small_max:
        return 0  # P3
    elif m < medium_max:
        return 1  # P4
    return 2  # P5


def build_targets(
    gt_per_image: Sequence[torch.Tensor],  # list of (N_i, 5) tensors: [cls, x1, y1, x2, y2] in pixels
    feat_shapes: Sequence[Tuple[int, int]],  # [(H,W) for P3, P4, P5]
    strides: Sequence[int],
    device,
) -> List[LevelTargets]:
    # Built on the CPU in numpy and moved to `device` once at the end: writing
    # element by element into GPU tensors cost ~150 ms per call.
    batch_size = len(gt_per_image)
    level_targets = []
    for (h, w), stride in zip(feat_shapes, strides):
        pos_mask = np.zeros((batch_size, h, w), dtype=bool)
        target_boxes = np.zeros((batch_size, h, w, 4), dtype=np.float32)
        target_cls = np.zeros((batch_size, h, w), dtype=np.int64)
        # track assigned box area to resolve collisions (smaller box wins)
        assigned_area = np.full((batch_size, h, w), np.inf)
        level_targets.append((pos_mask, target_boxes, target_cls, assigned_area, stride))

    for b, gts in enumerate(gt_per_image):
        if gts.numel() == 0:
            continue
        for row in gts.detach().cpu().tolist():
            cls_id, x1, y1, x2, y2 = row
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            level_idx = _level_for_box(bw, bh)
            pos_mask, target_boxes, target_cls, assigned_area, stride = level_targets[level_idx]
            h, w = pos_mask.shape[1], pos_mask.shape[2]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            gx = min(max(int(cx // stride), 0), w - 1)
            gy = min(max(int(cy // stride), 0), h - 1)

            # Centre sampling: the centre cell plus any of its 3x3
            # neighbours whose own centre falls inside the GT box. With a
            # single positive per GT the model sees ~20 positives vs ~8400
            # negatives per image and learns to predict "no object" everywhere.
            area = bw * bh
            for ny in range(max(gy - 1, 0), min(gy + 2, h)):
                for nx in range(max(gx - 1, 0), min(gx + 2, w)):
                    ccx, ccy = (nx + 0.5) * stride, (ny + 0.5) * stride
                    is_centre = nx == gx and ny == gy
                    if not is_centre and not (x1 < ccx < x2 and y1 < ccy < y2):
                        continue
                    if area < assigned_area[b, ny, nx]:
                        pos_mask[b, ny, nx] = True
                        target_boxes[b, ny, nx] = (x1, y1, x2, y2)
                        target_cls[b, ny, nx] = int(cls_id)
                        assigned_area[b, ny, nx] = area

    return [
        LevelTargets(pos_mask=torch.from_numpy(lt[0]).to(device),
                     target_boxes=torch.from_numpy(lt[1]).to(device),
                     target_cls=torch.from_numpy(lt[2]).to(device))
        for lt in level_targets
    ]


def decode_boxes_for_loss(reg_raw: torch.Tensor, stride: int, reg_max_ratio: float = 8.0) -> torch.Tensor:
    """Differentiable decode (mirrors detect_head.decode_scale's box math)."""
    b, c, h, w = reg_raw.shape
    cx, cy = make_grid(h, w, stride, reg_raw.device, reg_raw.dtype)
    max_exp = float(torch.log(torch.tensor(reg_max_ratio)).item() + 2.0)
    reg_clamped = torch.clamp(reg_raw, min=-4.0, max=max_exp)
    ltrb = torch.exp(reg_clamped) * stride
    l, t, r, btm = ltrb[:, 0], ltrb[:, 1], ltrb[:, 2], ltrb[:, 3]
    x1 = cx.unsqueeze(0) - l
    y1 = cy.unsqueeze(0) - t
    x2 = cx.unsqueeze(0) + r
    y2 = cy.unsqueeze(0) + btm
    return torch.stack([x1, y1, x2, y2], dim=-1)  # (B, H, W, 4)


@dataclass
class HeadLossOutput:
    total: torch.Tensor
    box_loss: float
    obj_loss: float
    cls_loss: float
    num_positives: int


def compute_head_loss(
    scale_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    gt_per_image: Sequence[torch.Tensor],
    strides: Sequence[int],
    num_classes: int,
    box_weight: float = 5.0,
    obj_weight: float = 1.0,
    cls_weight: float = 1.0,
    targets: List[LevelTargets] = None,
) -> HeadLossOutput:
    """`targets`: precomputed build_targets() output; built here if None."""
    device = scale_outputs[0][0].device
    if targets is None:
        feat_shapes = [(o[0].shape[2], o[0].shape[3]) for o in scale_outputs]
        targets = build_targets(gt_per_image, feat_shapes, strides, device)

    total_box, total_obj, total_cls = 0.0, 0.0, 0.0
    total_positives = 0
    loss = torch.zeros((), device=device)

    # Objectness is summed over all cells and normalised by the number of
    # positives across all scales (YOLOX-style), not averaged over cells --
    # a plain mean drowns the few positive cells in background.
    num_pos_all = max(sum(int(lt.pos_mask.sum().item()) for lt in targets), 1)

    for (cls_logits, obj_logit, reg_raw), lt, stride in zip(scale_outputs, targets, strides):
        pos_mask = lt.pos_mask
        n_pos = int(pos_mask.sum().item())

        # --- objectness: over ALL cells ---
        obj_target = pos_mask.float().unsqueeze(1)  # (B,1,H,W)
        obj_loss_map = F.binary_cross_entropy_with_logits(obj_logit, obj_target, reduction="sum") / num_pos_all

        if n_pos > 0:
            pred_boxes_full = decode_boxes_for_loss(reg_raw, stride)  # (B,H,W,4)
            pred_boxes_pos = pred_boxes_full[pos_mask]                # (n_pos, 4)
            target_boxes_pos = lt.target_boxes[pos_mask]              # (n_pos, 4)
            box_loss_val = ciou_loss(pred_boxes_pos, target_boxes_pos).mean()

            cls_logits_pos = cls_logits.permute(0, 2, 3, 1)[pos_mask]  # (n_pos, num_classes)
            cls_target_pos = F.one_hot(lt.target_cls[pos_mask], num_classes=num_classes).float()
            cls_loss_val = F.binary_cross_entropy_with_logits(cls_logits_pos, cls_target_pos, reduction="mean")
        else:
            box_loss_val = torch.zeros((), device=device)
            cls_loss_val = torch.zeros((), device=device)

        loss = loss + box_weight * box_loss_val + obj_weight * obj_loss_map + cls_weight * cls_loss_val
        total_box += float(box_loss_val.item())
        total_obj += float(obj_loss_map.item())
        total_cls += float(cls_loss_val.item())
        total_positives += n_pos

    return HeadLossOutput(
        total=loss, box_loss=total_box, obj_loss=total_obj, cls_loss=total_cls,
        num_positives=total_positives,
    )


def compute_multihead_loss(
    all_head_outputs,  # List[HeadOutput] from GelanAnytimeModel.forward
    gt_per_image: Sequence[torch.Tensor],
    strides: Sequence[int],
    num_classes: int,
    head_weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, HeadLossOutput]]:
    """Sums lambda_head * L_head across all taps (brief section 21)."""
    total_loss = None
    per_head: Dict[str, HeadLossOutput] = {}

    # Every head sees the same P3/P4/P5 grid sizes, so the targets are built once.
    first = all_head_outputs[0].scale_outputs
    targets = build_targets(gt_per_image, [(o[0].shape[2], o[0].shape[3]) for o in first],
                            strides, first[0][0].device)

    for head_output in all_head_outputs:
        head_name = head_output.head_name
        weight = head_weights.get(head_name, 1.0)
        head_loss = compute_head_loss(
            head_output.scale_outputs, gt_per_image, strides, num_classes, targets=targets,
        )
        per_head[head_name] = head_loss
        weighted = weight * head_loss.total
        total_loss = weighted if total_loss is None else total_loss + weighted

    return total_loss, per_head
