"""
data/yolo_dataset.py
-----------------------
Loads your BDD100K-derived dataset (brief section 18):

    archive (2)/
    |-- calibration/{images,labels}
    |-- test/{images,labels}
    |-- train/{images,labels}
    `-- data.yaml

Standard YOLO txt-label format per image: one line per object,
`class cx cy w h` all normalized to [0, 1] relative to image width/height.

Images are letterboxed (aspect-ratio-preserving resize + padding) to a
fixed square input size, and labels are transformed into the same
letterboxed pixel coordinate system as (x1, y1, x2, y2, class) so the
model always trains/evaluates in one consistent coordinate frame. The
letterbox parameters (scale, pad) are also returned so predictions can
be mapped back to ORIGINAL image coordinates for logging/visualization
(brief section 13: "boxes converted back to the original image
coordinate system").
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import yaml
except ImportError:
    yaml = None

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object  # allows this module to at least import without torch


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class LetterboxInfo:
    scale: float
    pad_x: float
    pad_y: float
    orig_width: int
    orig_height: int


def load_data_yaml(path: str) -> Dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read data.yaml (`pip install pyyaml`).")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def letterbox(image: np.ndarray, size: int) -> Tuple[np.ndarray, LetterboxInfo]:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    if _HAS_CV2:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        # Nearest-neighbor fallback with pure numpy so this still runs
        # without opencv installed (useful for quick sanity checks).
        y_idx = (np.arange(new_h) / scale).astype(int).clip(0, h - 1)
        x_idx = (np.arange(new_w) / scale).astype(int).clip(0, w - 1)
        resized = image[y_idx][:, x_idx]

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_y = (size - new_h) // 2
    pad_x = (size - new_w) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, LetterboxInfo(scale=scale, pad_x=pad_x, pad_y=pad_y, orig_width=w, orig_height=h)


def read_yolo_labels(label_path: str, img_w: int, img_h: int) -> np.ndarray:
    """Returns (N, 5): [class_id, x1, y1, x2, y2] in ORIGINAL pixel coords."""
    if not os.path.exists(label_path):
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, w, h = map(float, parts)
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            rows.append([cls_id, x1, y1, x2, y2])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def transform_labels_to_letterbox(labels: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Applies the same scale+pad used on the image to (x1,y1,x2,y2)."""
    if labels.shape[0] == 0:
        return labels
    out = labels.copy()
    out[:, 1] = labels[:, 1] * info.scale + info.pad_x
    out[:, 2] = labels[:, 2] * info.scale + info.pad_y
    out[:, 3] = labels[:, 3] * info.scale + info.pad_x
    out[:, 4] = labels[:, 4] * info.scale + info.pad_y
    return out


def find_image_label_pairs(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    pairs = []
    for ext in IMAGE_EXTS:
        for img_path in sorted(glob.glob(os.path.join(images_dir, f"*{ext}"))):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
            pairs.append((img_path, label_path))
    return pairs


class YoloAnytimeDataset(Dataset):
    """One dataset split (train/calibration/test).

    __getitem__ returns:
        image: float32 tensor (3, size, size), normalized to [0, 1]
        labels: float32 tensor (N, 5): [class_id, x1, y1, x2, y2] in the
                LETTERBOXED pixel coordinate frame (i.e. matches what the
                model sees -- this is what build_targets expects).
        meta: dict with letterbox info + original image path, needed to
              map predictions back to original coordinates for logging/
              visualization/mAP-on-original-image evaluation.
    """

    def __init__(self, root: str, split: str, class_names: List[str], img_size: int = 640):
        if torch is None:
            raise RuntimeError("PyTorch is required to use YoloAnytimeDataset for training/eval.")
        self.img_size = img_size
        self.class_names = class_names
        images_dir = os.path.join(root, split, "images")
        labels_dir = os.path.join(root, split, "labels")
        self.pairs = find_image_label_pairs(images_dir, labels_dir)
        if not self.pairs:
            raise FileNotFoundError(
                f"No images found under {images_dir}. Check `root`/`split` in your config."
            )

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path: str) -> np.ndarray:
        if _HAS_CV2:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            from PIL import Image
            img = np.array(Image.open(path).convert("RGB"))
        return img

    def __getitem__(self, idx: int):
        img_path, label_path = self.pairs[idx]
        image = self._load_image(img_path)
        h, w = image.shape[:2]

        canvas, info = letterbox(image, self.img_size)
        labels = read_yolo_labels(label_path, w, h)
        labels = transform_labels_to_letterbox(labels, info)

        img_tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1)
        labels_tensor = torch.from_numpy(labels)

        meta = {
            "path": img_path,
            "scale": info.scale, "pad_x": info.pad_x, "pad_y": info.pad_y,
            "orig_width": info.orig_width, "orig_height": info.orig_height,
        }
        return img_tensor, labels_tensor, meta


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = [b[1] for b in batch]   # keep as a LIST -- variable N per image
    metas = [b[2] for b in batch]
    return images, labels, metas


def unletterbox_boxes(boxes: np.ndarray, info: dict) -> np.ndarray:
    """Inverse of transform_labels_to_letterbox -- maps model-space boxes
    (letterboxed pixel coords) back to ORIGINAL image pixel coordinates."""
    if boxes.shape[0] == 0:
        return boxes
    out = boxes.copy()
    out[:, 0] = (boxes[:, 0] - info["pad_x"]) / info["scale"]
    out[:, 1] = (boxes[:, 1] - info["pad_y"]) / info["scale"]
    out[:, 2] = (boxes[:, 2] - info["pad_x"]) / info["scale"]
    out[:, 3] = (boxes[:, 3] - info["pad_y"]) / info["scale"]
    out[:, 0] = np.clip(out[:, 0], 0, info["orig_width"])
    out[:, 1] = np.clip(out[:, 1], 0, info["orig_height"])
    out[:, 2] = np.clip(out[:, 2], 0, info["orig_width"])
    out[:, 3] = np.clip(out[:, 3], 0, info["orig_height"])
    return out
