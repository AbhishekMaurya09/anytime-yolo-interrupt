"""
models/postprocess.py
------------------------
Turns one tap's raw (cls_logits, obj_logit, reg_raw) per scale into final,
NMS'd, per-image detections in MODEL (letterboxed) pixel coordinates.
Coordinate remap to the original image is a separate step
(data.yolo_dataset.unletterbox_boxes) done by the caller, since that
step needs per-image letterbox metadata this function doesn't have.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from .detect_head import decode_scale, greedy_nms


def postprocess_head(
    scale_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    strides: Sequence[int],
    conf_thres: float = 0.3,
    iou_thres: float = 0.5,
    max_detections: int = 100,
) -> List[dict]:
    """Returns a list (length = batch size) of dicts:
        {"boxes": (M,4) tensor, "scores": (M,) tensor, "cls_ids": (M,) tensor}
    in MODEL/letterboxed pixel coordinates.
    """
    batch_size = scale_outputs[0][0].shape[0]
    per_image_boxes = [[] for _ in range(batch_size)]
    per_image_scores = [[] for _ in range(batch_size)]
    per_image_cls = [[] for _ in range(batch_size)]

    for (cls_logits, obj_logit, reg_raw), stride in zip(scale_outputs, strides):
        boxes, scores, cls_ids = decode_scale(cls_logits, obj_logit, reg_raw, stride)
        b, h, w = scores.shape
        boxes_flat = boxes.reshape(b, h * w, 4)
        scores_flat = scores.reshape(b, h * w)
        cls_flat = cls_ids.reshape(b, h * w)

        for i in range(b):
            keep = scores_flat[i] >= conf_thres
            if keep.any():
                per_image_boxes[i].append(boxes_flat[i][keep])
                per_image_scores[i].append(scores_flat[i][keep])
                per_image_cls[i].append(cls_flat[i][keep])

    results = []
    for i in range(batch_size):
        if len(per_image_boxes[i]) == 0:
            results.append({
                "boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)),
                "cls_ids": torch.zeros((0,), dtype=torch.long),
            })
            continue

        boxes_i = torch.cat(per_image_boxes[i], dim=0)
        scores_i = torch.cat(per_image_scores[i], dim=0)
        cls_i = torch.cat(per_image_cls[i], dim=0)

        # per-class NMS
        keep_indices = []
        for c in torch.unique(cls_i):
            mask = cls_i == c
            idxs = torch.nonzero(mask, as_tuple=True)[0]
            keep = greedy_nms(boxes_i[idxs], scores_i[idxs], iou_threshold=iou_thres)
            keep_indices.append(idxs[keep])
        if keep_indices:
            keep_all = torch.cat(keep_indices)
        else:
            keep_all = torch.zeros((0,), dtype=torch.long)

        # cap total detections, keep highest-scoring
        if keep_all.numel() > max_detections:
            top = torch.topk(scores_i[keep_all], max_detections).indices
            keep_all = keep_all[top]

        results.append({
            "boxes": boxes_i[keep_all],
            "scores": scores_i[keep_all],
            "cls_ids": cls_i[keep_all],
        })

    return results
