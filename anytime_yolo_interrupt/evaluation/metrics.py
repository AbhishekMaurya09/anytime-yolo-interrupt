"""
metrics.py
----------
Precision / Recall / mAP50 / mAP50-95 computation for one head's
predictions over a test set, given ground-truth boxes per image.

This is a standard, dependency-free (numpy-only) implementation of
COCO-style mAP so it works without pycocotools/torchvision. If your
real repo already has an evaluation pipeline (brief section 42 asks
you to reuse it), prefer that -- this module exists so the framework
here is runnable/testable standalone and so there is a clear, auditable
reference implementation.

IMPORTANT (brief section 25): all functions here operate over an
entire test set's predictions, not a single image. `evaluate_predictions`
also reports how many images were actually processed so that a run
interrupted partway through is never mislabeled as a full test-set
metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = np.clip((ax2 - ax1) * (ay2 - ay1), 1e-9, None)
    area_b = np.clip((bx2 - bx1) * (by2 - by1), 1e-9, None)
    union = area_a + area_b - inter
    return inter / union


@dataclass
class ImagePrediction:
    boxes: np.ndarray   # (N, 4)
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,) string or int labels


@dataclass
class ImageGroundTruth:
    boxes: np.ndarray   # (M, 4)
    classes: np.ndarray  # (M,) string or int labels


def _ap_for_class(
    preds: List[ImagePrediction],
    gts: List[ImageGroundTruth],
    class_label,
    iou_threshold: float,
) -> float:
    """Standard 101-point interpolated AP for a single class at a single IoU threshold."""
    all_scores = []
    all_tp = []
    n_gt_total = 0

    gt_matched = [np.zeros(len(g.classes), dtype=bool) for g in gts]

    for img_idx, (pred, gt) in enumerate(zip(preds, gts)):
        cls_mask_gt = gt.classes == class_label
        n_gt_total += int(cls_mask_gt.sum())

        cls_mask_pred = pred.classes == class_label
        if not np.any(cls_mask_pred):
            continue

        p_boxes = pred.boxes[cls_mask_pred]
        p_scores = pred.scores[cls_mask_pred]
        g_boxes = gt.boxes[cls_mask_gt]
        g_indices = np.where(cls_mask_gt)[0]

        order = np.argsort(-p_scores)
        ious = _iou_matrix(p_boxes[order], g_boxes) if len(g_boxes) else None

        for rank, orig_idx in enumerate(order):
            all_scores.append(p_scores[orig_idx])
            if ious is None or ious.shape[1] == 0:
                all_tp.append(0)
                continue
            row = ious[rank]
            best_j = int(np.argmax(row)) if row.size else -1
            if best_j >= 0 and row[best_j] >= iou_threshold and not gt_matched[img_idx][g_indices[best_j]]:
                gt_matched[img_idx][g_indices[best_j]] = True
                all_tp.append(1)
            else:
                all_tp.append(0)

    if n_gt_total == 0:
        return float("nan")  # class not present in this test set
    if len(all_scores) == 0:
        return 0.0

    order = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[order]
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / n_gt_total
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)

    # 101-point interpolation (COCO-style)
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_thresholds:
        mask = recall >= r
        p = precision[mask].max() if np.any(mask) else 0.0
        ap += p
    return ap / 101.0


@dataclass
class EvalResult:
    precision: float
    recall: float
    map50: float
    map50_95: float
    n_images_processed: int
    n_images_expected: int
    per_class_ap50: Dict[str, float]

    @property
    def is_partial(self) -> bool:
        return self.n_images_processed < self.n_images_expected


def evaluate_predictions(
    predictions: List[ImagePrediction],
    ground_truths: List[ImageGroundTruth],
    class_names: Sequence[str],
    n_images_expected: int,
    iou_thresholds_50_95=tuple(np.arange(0.5, 1.0, 0.05)),
    score_threshold_for_pr: float = 0.5,
) -> EvalResult:
    """Compute precision/recall (at a fixed score threshold, for a
    simple summary figure) and mAP50 / mAP50-95 (threshold-independent,
    the standard research metrics) over whatever images have been
    supplied so far. Clearly reports partial-evaluation status."""
    n_processed = len(predictions)

    per_class_ap50 = {}
    aps_50_95_per_class = {}
    for cls in class_names:
        per_class_ap50[cls] = _ap_for_class(predictions, ground_truths, cls, 0.5)
        aps_at_thresholds = [
            _ap_for_class(predictions, ground_truths, cls, thr) for thr in iou_thresholds_50_95
        ]
        aps_50_95_per_class[cls] = np.nanmean(aps_at_thresholds)

    map50 = float(np.nanmean(list(per_class_ap50.values())))
    map50_95 = float(np.nanmean(list(aps_50_95_per_class.values())))

    # Simple aggregate precision/recall at a fixed score threshold,
    # matched at IoU 0.5, for an easy-to-read summary number.
    tp = fp = fn = 0
    for pred, gt in zip(predictions, ground_truths):
        keep = pred.scores >= score_threshold_for_pr
        p_boxes, p_classes = pred.boxes[keep], pred.classes[keep]
        matched_gt = np.zeros(len(gt.classes), dtype=bool)
        ious = _iou_matrix(p_boxes, gt.boxes)
        for i in range(len(p_boxes)):
            if ious.shape[1] == 0:
                fp += 1
                continue
            candidates = np.where((gt.classes == p_classes[i]) & (~matched_gt))[0]
            if len(candidates) == 0:
                fp += 1
                continue
            best = candidates[np.argmax(ious[i, candidates])]
            if ious[i, best] >= 0.5:
                matched_gt[best] = True
                tp += 1
            else:
                fp += 1
        fn += int((~matched_gt).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return EvalResult(
        precision=precision,
        recall=recall,
        map50=map50,
        map50_95=map50_95,
        n_images_processed=n_processed,
        n_images_expected=n_images_expected,
        per_class_ap50=per_class_ap50,
    )
