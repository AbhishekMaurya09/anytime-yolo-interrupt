"""
scripts/build_bdd_subset.py
-----------------------------
Builds a random BDD100K subset in this project's layout:

    <out>/train/{images,labels}  <out>/calibration/{images,labels}  <out>/test/{images,labels}  <out>/data.yaml

Source, either:
  --images-dir + --labels-json   official BDD100K: images/100k/train and a detection
                                 label JSON (det_20 "det_train.json" or the 2018
                                 "bdd100k_labels_images_train.json")
  --yolo-dir                     a YOLO-format copy (images/ + labels/ + data.yaml,
                                 optionally nested in split folders)

    python -m anytime_yolo_interrupt.scripts.build_bdd_subset \
        --images-dir "D:\\bdd100k\\images\\100k\\train" \
        --labels-json "D:\\bdd100k\\labels\\det_20\\det_train.json" \
        --out "C:\\Users\\abhis\\Downloads\\bdd1500" --n-train 1200 --n-val 150 --n-test 150 \
        --exclude-dir "C:\\Users\\abhis\\Downloads\\archive (2)"

Images already in --exclude-dir (e.g. the old 200-image set) are skipped, so the
old test images can still be used as an untouched extra test set.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import shutil
import sys

import yaml
from PIL import Image

# Class ids as they are actually used in the original 200-image subset (checked
# by looking at cropped boxes on 2026-09-30); that subset's data.yaml lists the
# names in a different, wrong order. Keeping these ids lets new data mix with
# the old labels and the existing checkpoints.
CLASS_NAMES = ["pedestrian", "rider", "car", "bus", "truck", "bicycle",
               "motorcycle", "traffic light", "traffic sign", "train"]
# 2018 label names -> det_20 names; categories not listed here are dropped
# (lane, drivable area, other vehicle, other person, trailer).
ALIASES = {"person": "pedestrian", "bike": "bicycle", "motor": "motorcycle"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def from_bdd_json(images_dir, labels_json):
    """-> {image_path: [(class_id, x1, y1, x2, y2) in pixels]}"""
    with open(labels_json, "r") as f:
        frames = json.load(f)
    items = {}
    for frame in frames:
        path = os.path.join(images_dir, frame["name"])
        if not os.path.exists(path):
            continue
        boxes = []
        for label in frame.get("labels") or []:
            name = ALIASES.get(label.get("category"), label.get("category"))
            box = label.get("box2d")
            if box is None or name not in CLASS_NAMES:
                continue
            boxes.append((CLASS_NAMES.index(name), box["x1"], box["y1"], box["x2"], box["y2"]))
        items[path] = boxes
    return items


def from_yolo_dir(yolo_dir):
    """Reads YOLO txt labels (normalised) and maps their class ids to CLASS_NAMES by name."""
    yamls = glob.glob(os.path.join(yolo_dir, "*.yaml"))
    if not yamls:
        sys.exit(f"No data.yaml in {yolo_dir}")
    names = yaml.safe_load(open(yamls[0]))["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    remap = {i: CLASS_NAMES.index(ALIASES.get(n, n)) for i, n in enumerate(names) if ALIASES.get(n, n) in CLASS_NAMES}

    items = {}
    for img in glob.glob(os.path.join(yolo_dir, "**", "images", "*"), recursive=True):
        if not img.lower().endswith(IMAGE_EXTS):
            continue
        lab = os.path.splitext(img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))[0] + ".txt"
        if not os.path.exists(lab):
            continue
        w, h = Image.open(img).size
        boxes = []
        for line in open(lab):
            parts = line.split()
            if len(parts) != 5 or int(float(parts[0])) not in remap:
                continue
            c, cx, cy, bw, bh = int(float(parts[0])), *map(float, parts[1:])
            boxes.append((remap[c], (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        items[img] = boxes
    return items


def write_split(out_dir, split, paths, items):
    img_dir, lab_dir = os.path.join(out_dir, split, "images"), os.path.join(out_dir, split, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    counts = collections.Counter()
    for path in paths:
        w, h = Image.open(path).size
        shutil.copy2(path, os.path.join(img_dir, os.path.basename(path)))
        lines = []
        for c, x1, y1, x2, y2 in items[path]:
            x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            counts[CLASS_NAMES[c]] += 1
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(lab_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir")
    p.add_argument("--labels-json")
    p.add_argument("--yolo-dir")
    p.add_argument("--out", required=True)
    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--n-val", type=int, default=150)
    p.add_argument("--n-test", type=int, default=150)
    p.add_argument("--exclude-dir", action="append", default=[], help="skip images whose file name appears here")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.yolo_dir:
        items = from_yolo_dir(args.yolo_dir)
    elif args.images_dir and args.labels_json:
        items = from_bdd_json(args.images_dir, args.labels_json)
    else:
        sys.exit("Give --yolo-dir, or both --images-dir and --labels-json.")

    excluded = {os.path.basename(f) for d in args.exclude_dir
                for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)}
    candidates = sorted(p for p, boxes in items.items() if boxes and os.path.basename(p) not in excluded)
    need = args.n_train + args.n_val + args.n_test
    print(f"{len(items)} labelled images found, {len(candidates)} usable after removing empty/excluded ones")
    if len(candidates) < need:
        sys.exit(f"Need {need} images but only {len(candidates)} are usable.")
    if os.path.exists(args.out) and os.listdir(args.out):
        sys.exit(f"{args.out} already exists and is not empty; choose a new --out.")

    random.Random(args.seed).shuffle(candidates)
    splits = {"train": candidates[:args.n_train],
              "calibration": candidates[args.n_train:args.n_train + args.n_val],
              "test": candidates[args.n_train + args.n_val:need]}

    summary = {}
    for split, paths in splits.items():
        counts = write_split(args.out, split, paths, items)
        summary[split] = {"images": len(paths), "boxes": sum(counts.values()), "per_class": dict(counts)}
        print(f"{split:<12} {len(paths):>5} images {sum(counts.values()):>7} boxes  "
              + ", ".join(f"{k} {counts[k]}" for k in CLASS_NAMES if counts[k]))

    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        yaml.safe_dump({"path": args.out, "train": "train/images", "val": "calibration/images",
                        "test": "test/images", "names": dict(enumerate(CLASS_NAMES))}, f, sort_keys=False)
    with open(os.path.join(args.out, "subset_info.json"), "w") as f:
        json.dump({"source": args.labels_json or args.yolo_dir, "seed": args.seed,
                   "excluded_dirs": args.exclude_dir, "splits": summary}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
