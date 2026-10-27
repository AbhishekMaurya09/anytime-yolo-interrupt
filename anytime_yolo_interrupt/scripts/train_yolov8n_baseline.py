"""
scripts/train_yolov8n_baseline.py
-----------------------------------
Fine-tunes the COCO-pretrained YOLOv8n (Ultralytics) on a YOLO-format dataset,
as a baseline for the anytime model. Needs `pip install ultralytics`.

    python -m anytime_yolo_interrupt.scripts.train_yolov8n_baseline \
        --data "C:/Users/abhis/Downloads/bdd1500/data.yaml" --epochs 50

The weights file yolov8n.pt is downloaded by Ultralytics on first use.
"""
from __future__ import annotations

import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="data.yaml with train/val/test image folders")
    p.add_argument("--weights", default="runs/yolov8n_compare/yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--project", default="runs/yolov8n_compare")
    p.add_argument("--name", default="bdd1500_ft")
    args = p.parse_args()

    from ultralytics import YOLO

    os.makedirs(os.path.dirname(args.weights) or ".", exist_ok=True)
    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=0,
                workers=args.workers, project=os.path.abspath(args.project), name=args.name, exist_ok=True,
                seed=0, deterministic=False, plots=False)


if __name__ == "__main__":
    main()
