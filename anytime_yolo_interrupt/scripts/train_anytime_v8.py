"""
scripts/train_anytime_v8.py
-----------------------------
Trains the anytime detector built on pretrained YOLOv8n (models/anytime_v8.py)
with Ultralytics' trainer: its augmentation, task-aligned assignment and
DFL loss, summed over the three heads. Settings match
scripts/train_yolov8n_baseline.py, so the two runs compare directly.

    python -m anytime_yolo_interrupt.scripts.train_anytime_v8 \
        --data "C:/Users/abhis/Downloads/bdd1500/data.yaml" --epochs 50
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anytime_yolo_interrupt.models.anytime_v8 import AnytimeV8Trainer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--weights", default="runs/yolov8n_compare/yolov8n.pt", help="COCO-pretrained YOLOv8n")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--project", default="runs/anytime_v8")
    p.add_argument("--name", default="bdd1500")
    p.add_argument("--freeze-base", action="store_true",
                   help="freeze all YOLOv8n layers (0-22, BatchNorm statistics too); train only the early heads")
    p.add_argument("--depthwise", action="store_true", help="depthwise 3x3 + 1x1 instead of a full 3x3 in early heads")
    p.add_argument("--head-width", type=int, default=32)
    args = p.parse_args()

    anytime_kwargs = dict(head_width=args.head_width, depthwise=args.depthwise)
    if args.freeze_base:
        anytime_kwargs["head_weights"] = (1.0, 1.0, 0.0)   # the frozen final head gets no gradient anyway

    trainer = AnytimeV8Trainer(anytime_kwargs=anytime_kwargs, overrides=dict(
        model=args.weights, data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=0,
        workers=args.workers, project=os.path.abspath(args.project), name=args.name, exist_ok=True, seed=0,
        deterministic=False, plots=False, freeze=23 if args.freeze_base else None))
    trainer.train()


if __name__ == "__main__":
    main()
