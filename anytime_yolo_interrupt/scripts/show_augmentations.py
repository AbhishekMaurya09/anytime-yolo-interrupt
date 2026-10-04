"""
scripts/show_augmentations.py
-------------------------------
Saves a grid of augmented training images with their boxes drawn, to check
that labels still line up after mosaic / zoom / flip / colour jitter.

    python -m anytime_yolo_interrupt.scripts.show_augmentations \
        --data-root "C:\\Users\\abhis\\Downloads\\archive (2)" --out runs/anytime_interrupt/augment_preview.jpg
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anytime_yolo_interrupt.data import AugmentConfig, YoloAnytimeDataset, load_data_yaml  # noqa: E402

COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
          (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/anytime_interrupt/augment_preview.jpg")
    args = p.parse_args()

    names = load_data_yaml(os.path.join(args.data_root, "data.yaml"))["names"]
    class_names = list(names.values()) if isinstance(names, dict) else list(names)
    ds = YoloAnytimeDataset(args.data_root, args.split, class_names, augment=AugmentConfig())
    random.seed(args.seed)

    tiles = []
    for _ in range(args.n):
        image, labels, _ = ds[random.randrange(len(ds))]
        tile = np.ascontiguousarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1])
        for cls, x1, y1, x2, y2 in labels.numpy():
            color = COLORS[int(cls) % len(COLORS)][::-1]
            cv2.rectangle(tile, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(tile, class_names[int(cls)], (int(x1), max(int(y1) - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        tiles.append(tile)

    cols = 4
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 114))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
