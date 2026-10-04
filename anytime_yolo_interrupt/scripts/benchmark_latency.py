"""
scripts/benchmark_latency.py
------------------------------
Measures when each head's result becomes available (published to the
ResultManager) with the real trained model, for the sequential and the
pipelined detector.

    python -m anytime_yolo_interrupt.scripts.benchmark_latency \
        --checkpoint runs/anytime_interrupt/train2/last.pt \
        --data-root "C:\\Users\\abhis\\Downloads\\archive (2)" --split test

Each image is run twice; only the second (warm) pass is reported.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anytime_yolo_interrupt.data.yolo_dataset import IMAGE_EXTS, load_data_yaml  # noqa: E402
from anytime_yolo_interrupt.inference import AnytimeDetector, ResultManager  # noqa: E402
from anytime_yolo_interrupt.inference.gelan_adapter import GelanAnytimeAdapter  # noqa: E402


def load_rgb(path):
    import cv2
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def measure(adapter, images, pipelined):
    detector = AnytimeDetector(adapter, ResultManager(log_dir=None), pipelined=pipelined)
    times = []
    for rep in range(2):
        for frame_id, image in enumerate(images):
            results = detector.process_frame(adapter.prepare_frame(image), frame_id)
            if rep == 1:
                times.append([r.elapsed_ms for r in results])
    detector.close()
    return np.array(times)  # (n_images, n_heads)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--data-yaml", default=None, help="defaults to <data-root>/data.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--max-images", type=int, default=40)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="runs/anytime_interrupt/latency_benchmark.json")
    args = p.parse_args()

    names = load_data_yaml(args.data_yaml or os.path.join(args.data_root, "data.yaml"))["names"]
    class_names = list(names.values()) if isinstance(names, dict) else list(names)
    adapter = GelanAnytimeAdapter(args.checkpoint, class_names, device=args.device)

    paths = sorted(p for p in glob.glob(os.path.join(args.data_root, args.split, "images", "*"))
                   if p.lower().endswith(IMAGE_EXTS))[: args.max_images]
    images = [load_rgb(path) for path in paths]
    head_names = [h.head_name for h in adapter.heads]

    report = {"checkpoint": args.checkpoint, "n_images": len(images), "modes": {}}
    print(f"{len(images)} images, warm pass, median ms until each head's result is available\n")
    print(f"{'mode':<12}" + "".join(f"{n:>14}" for n in head_names))
    for mode, pipelined in (("sequential", False), ("pipelined", True)):
        t = measure(adapter, images, pipelined)
        med = np.median(t, axis=0)
        p90 = np.percentile(t, 90, axis=0)
        report["modes"][mode] = {n: {"median_ms": float(m), "p90_ms": float(q)}
                                 for n, m, q in zip(head_names, med, p90)}
        print(f"{mode:<12}" + "".join(f"{m:>14.1f}" for m in med))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
