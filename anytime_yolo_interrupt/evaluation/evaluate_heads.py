"""
evaluate_heads.py
------------------
Runs the AnytimeDetector over every image in a labeled test set and
reports Precision / Recall / mAP50 / mAP50-95 PER HEAD, independently
(brief sections 24-25).

Usage (with the real model, once GelanTAnytimeAdapter is wired up):

    python scripts/evaluate_heads.py \
        --data-yaml "<path>/archive (2)/data.yaml" \
        --split test \
        --weights runs/train/weights/best.pt \
        --out runs/anytime_interrupt/summary.json

With this deliverable's MockAnytimeAdapter and no real dataset on this
machine, `run_evaluation` accepts any iterable of (frame, ground_truth)
pairs, so it is fully testable now and only needs a real BDD100K-YOLO
format data loader (glob the `test/images` + `test/labels` folders
named in your data.yaml) swapped in to run on your machine.
"""

from __future__ import annotations

import json
from typing import Iterable, List, Tuple

import numpy as np

from ..inference.anytime_detector import AnytimeDetector
from ..inference.result_manager import ResultManager
from .metrics import EvalResult, ImageGroundTruth, ImagePrediction, evaluate_predictions

CLASS_NAMES = [
    "pedestrian", "rider", "car", "bus", "truck", "bicycle",
    "motorcycle", "traffic light", "traffic sign", "train",
]


def run_evaluation(
    detector: AnytimeDetector,
    result_manager: ResultManager,
    dataset: Iterable[Tuple[dict, ImageGroundTruth]],
    n_images_expected: int,
    class_names=CLASS_NAMES,
) -> dict:
    """Returns {head_name: EvalResult} plus timing info per head."""
    per_head_predictions: dict = {h.head_name: [] for h in detector.adapter.heads}
    per_head_gt: dict = {h.head_name: [] for h in detector.adapter.heads}
    per_head_latency_ms: dict = {h.head_name: [] for h in detector.adapter.heads}

    n_processed = 0
    try:
        for frame_id, (frame, gt) in enumerate(dataset):
            results = detector.process_frame(frame, frame_id)
            for r in results:
                per_head_predictions[r.head_name].append(
                    ImagePrediction(
                        boxes=np.array(r.boxes, dtype=float).reshape(-1, 4),
                        scores=np.array(r.scores, dtype=float),
                        classes=np.array(r.classes, dtype=object),
                    )
                )
                per_head_gt[r.head_name].append(gt)
                per_head_latency_ms[r.head_name].append(r.elapsed_ms)
            n_processed += 1
    except KeyboardInterrupt:
        pass  # allow graceful partial evaluation; reported honestly below

    report = {}
    for head_name in per_head_predictions:
        eval_result: EvalResult = evaluate_predictions(
            per_head_predictions[head_name],
            per_head_gt[head_name],
            class_names,
            n_images_expected=n_images_expected,
        )
        report[head_name] = {
            "precision": eval_result.precision,
            "recall": eval_result.recall,
            "mAP50": eval_result.map50,
            "mAP50_95": eval_result.map50_95,
            "n_images_processed": eval_result.n_images_processed,
            "n_images_expected": eval_result.n_images_expected,
            "is_partial": eval_result.is_partial,
            "mean_time_to_result_ms": float(np.mean(per_head_latency_ms[head_name]))
            if per_head_latency_ms[head_name] else None,
        }
    return report


def save_report(report: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
