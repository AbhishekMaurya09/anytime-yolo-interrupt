#!/usr/bin/env python3
"""
train.py
----------
Trains the from-scratch GelanAnytimeModel end-to-end: ONE forward pass
per batch, ALL FOUR taps supervised simultaneously (brief section 21),
weighted sum loss. Every `--val-every` epochs, runs a full validation
pass and reports precision/recall/mAP50/mAP50-95 PER HEAD independently
(reusing evaluation/metrics.py built earlier), so you can literally see
"how good is Head 1 vs Head 2 vs Head 3 vs Final" as training progresses.

Usage:
    python train.py \
        --data-root "C:\\Users\\abhis\\Downloads\\archive (2)" \
        --data-yaml "C:\\Users\\abhis\\Downloads\\archive (2)\\data.yaml" \
        --epochs 100 --batch-size 4 --img-size 640 --device cuda \
        --log-dir runs/anytime_interrupt/train1

Notes on the GTX 1650 / 4GB VRAM target (brief section 38):
  - default batch size is small (4); reduce further (2, then 1) if you
    hit CUDA OOM before reducing img-size.
  - torch.no_grad() is used during validation; detach()+.cpu() before
    any persistent logging (never holds GPU tensors across iterations).
  - mixed precision (torch.cuda.amp) is used by default on CUDA to help
    fit the small VRAM budget; disable with --no-amp if you hit issues.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from anytime_yolo_interrupt.models import build_model, compute_multihead_loss, postprocess_head, STRIDES
from anytime_yolo_interrupt.data import YoloAnytimeDataset, collate_fn, load_data_yaml, unletterbox_boxes
from anytime_yolo_interrupt.evaluation.metrics import evaluate_predictions, ImagePrediction, ImageGroundTruth


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True, help='e.g. "C:\\Users\\abhis\\Downloads\\archive (2)"')
    p.add_argument("--data-yaml", type=str, required=True)
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="calibration")  # brief section 18: calibration -> validation
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--log-dir", type=str, default="runs/anytime_interrupt/train")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--lambda-head1", type=float, default=0.3)
    p.add_argument("--lambda-head2", type=float, default=0.5)
    p.add_argument("--lambda-head3", type=float, default=0.7)
    p.add_argument("--lambda-final", type=float, default=1.0)
    return p.parse_args()


def build_gt_tensors(labels_list, device):
    """labels_list: list of (N_i, 5) CPU tensors [cls, x1,y1,x2,y2] -> moved to device."""
    return [lab.to(device) for lab in labels_list]


def run_validation(model, val_loader, class_names, strides, device, log_dir, epoch):
    model.eval()
    per_head_preds = {name: [] for name in model.head_names}
    per_head_gts = {name: [] for name in model.head_names}

    n_images = 0
    with torch.no_grad():
        for images, labels_list, metas in val_loader:
            images = images.to(device)
            all_head_outputs = model(images)

            for head_output in all_head_outputs:
                dets = postprocess_head(head_output.scale_outputs, strides)
                for i, det in enumerate(dets):
                    boxes = det["boxes"].cpu().numpy()
                    scores = det["scores"].cpu().numpy()
                    cls_ids = det["cls_ids"].cpu().numpy()
                    boxes_orig = unletterbox_boxes(boxes, metas[i])
                    classes_str = np.array([class_names[c] for c in cls_ids], dtype=object)

                    per_head_preds[head_output.head_name].append(
                        ImagePrediction(boxes=boxes_orig, scores=scores, classes=classes_str)
                    )

                    gt = labels_list[i].numpy()
                    gt_boxes = unletterbox_boxes(gt[:, 1:5], metas[i]) if gt.shape[0] else gt[:, 1:5]
                    gt_classes = np.array([class_names[int(c)] for c in gt[:, 0]], dtype=object) if gt.shape[0] else np.array([], dtype=object)
                    per_head_gts[head_output.head_name].append(
                        ImageGroundTruth(boxes=gt_boxes, classes=gt_classes)
                    )
            n_images += images.shape[0]

    report = {}
    for head_name in per_head_preds:
        result = evaluate_predictions(
            per_head_preds[head_name], per_head_gts[head_name], class_names, n_images_expected=len(val_loader.dataset),
        )
        report[head_name] = {
            "precision": result.precision, "recall": result.recall,
            "mAP50": result.map50, "mAP50_95": result.map50_95,
            "n_images_processed": result.n_images_processed,
            "n_images_expected": result.n_images_expected,
            "is_partial": result.is_partial,
        }
        print(f"  [val][epoch {epoch}] {head_name:>14}: "
              f"P={result.precision:.3f} R={result.recall:.3f} "
              f"mAP50={result.map50:.3f} mAP50-95={result.map50_95:.3f}"
              f"{'  (PARTIAL)' if result.is_partial else ''}")

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"val_epoch_{epoch:04d}.json"), "w") as f:
        json.dump(report, f, indent=2)
    model.train()
    return report


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("WARNING: CUDA requested but not available -- falling back to CPU (will be slow).")

    data_cfg = load_data_yaml(args.data_yaml)
    if "names" not in data_cfg:
        raise ValueError("data.yaml must contain a `names` list/dict of class names.")
    names_field = data_cfg["names"]
    class_names = list(names_field.values()) if isinstance(names_field, dict) else list(names_field)
    num_classes = len(class_names)

    train_ds = YoloAnytimeDataset(args.data_root, args.train_split, class_names, img_size=args.img_size)
    val_ds = YoloAnytimeDataset(args.data_root, args.val_split, class_names, img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = build_model(num_classes).to(device)

    head_weights = {
        "early_head_1": args.lambda_head1,
        "early_head_2": args.lambda_head2,
        "early_head_3": args.lambda_head3,
        "final": args.lambda_final,
    }
    with open(os.path.join(args.log_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "head_weights": head_weights, "class_names": class_names}, f, indent=2)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.937, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        running = {"early_head_1": 0.0, "early_head_2": 0.0, "early_head_3": 0.0, "final": 0.0, "total": 0.0}
        n_batches = 0

        for images, labels_list, metas in train_loader:
            images = images.to(device)
            gt_per_image = build_gt_tensors(labels_list, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                all_head_outputs = model(images)
                total_loss, per_head = compute_multihead_loss(
                    all_head_outputs, gt_per_image, STRIDES, num_classes, head_weights,
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running["total"] += float(total_loss.item())
            for name, hl in per_head.items():
                running[name] += hl.box_loss + hl.obj_loss + hl.cls_loss
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - epoch_start
        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        print(f"[epoch {epoch}] total={avg['total']:.4f} "
              f"head1={avg['early_head_1']:.4f} head2={avg['early_head_2']:.4f} "
              f"head3={avg['early_head_3']:.4f} final={avg['final']:.4f} "
              f"({elapsed:.1f}s)")

        with open(os.path.join(args.log_dir, "train_loss_log.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch, **avg, "elapsed_s": elapsed}) + "\n")

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            run_validation(model, val_loader, class_names, STRIDES, device, args.log_dir, epoch)

        ckpt_path = os.path.join(args.log_dir, "last.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "class_names": class_names}, ckpt_path)

    print(f"\nTraining complete. Checkpoints + logs in {args.log_dir}")


if __name__ == "__main__":
    main()
