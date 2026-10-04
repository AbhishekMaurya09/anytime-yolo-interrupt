"""
scripts/profile_heads.py
--------------------------
Times every step of one frame separately (GPU-synchronised): preprocessing,
each neck stage, each head's network, and each head's post-processing, then
reports when each head's result becomes available.

    python -m anytime_yolo_interrupt.scripts.profile_heads \
        --checkpoint runs/anytime_interrupt/train6_bdd70k/last.pt \
        --data-root "C:\\Users\\abhis\\Downloads\\bdd1500" --split test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anytime_yolo_interrupt.data.yolo_dataset import IMAGE_EXTS, letterbox, load_data_yaml  # noqa: E402
from anytime_yolo_interrupt.inference.gelan_adapter import GelanAnytimeAdapter  # noqa: E402

STEPS = ["preprocess", "backbone", "head1_net", "head1_post", "fpn", "head2_net", "head2_post",
         "pan", "head3_net", "head3_post", "refine", "final_net", "final_post"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max-images", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="runs/anytime_interrupt/head_timing.json")
    args = p.parse_args()

    import cv2
    names = load_data_yaml(os.path.join(args.data_root, "data.yaml"))["names"]
    class_names = list(names.values()) if isinstance(names, dict) else list(names)
    adapter = GelanAnytimeAdapter(args.checkpoint, class_names, device=args.device)
    model, dev = adapter.model, adapter.device
    paths = sorted(q for q in glob.glob(os.path.join(args.data_root, args.split, "images", "*"))
                   if q.lower().endswith(IMAGE_EXTS))[: args.max_images]
    images = [cv2.cvtColor(cv2.imread(q), cv2.COLOR_BGR2RGB) for q in paths]

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    times = {s: [] for s in STEPS}

    def timed(step, fn):
        sync()
        t = time.perf_counter()
        out = fn()
        sync()
        times[step].append((time.perf_counter() - t) * 1000.0)
        return out

    def preprocess(image):
        canvas, info = letterbox(image, adapter.img_size)
        adapter._current_letterbox_info = {"scale": info.scale, "pad_x": info.pad_x, "pad_y": info.pad_y,
                                           "orig_width": info.orig_width, "orig_height": info.orig_height}
        return torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)

    with torch.no_grad():
        for rep in range(2):  # first pass is warm-up
            if rep == 1:
                times = {s: [] for s in STEPS}
            for image in images:
                h, w = image.shape[:2]
                x = timed("preprocess", lambda: preprocess(image))
                bb = timed("backbone", lambda: model.forward_backbone(x))
                r = timed("head1_net", lambda: model.head1(list(bb)))
                timed("head1_post", lambda: adapter.postprocess(r, 0, w, h))
                td = timed("fpn", lambda: model.forward_topdown(*bb))
                r = timed("head2_net", lambda: model.head2(list(td)))
                timed("head2_post", lambda: adapter.postprocess(r, 1, w, h))
                bu = timed("pan", lambda: model.forward_bottomup(*td))
                r = timed("head3_net", lambda: model.head3(list(bu)))
                timed("head3_post", lambda: adapter.postprocess(r, 2, w, h))
                fr = timed("refine", lambda: model.forward_final_refine(*bu))
                r = timed("final_net", lambda: model.head_final(list(fr)))
                timed("final_post", lambda: adapter.postprocess(r, 3, w, h))

    t = {s: np.array(v) for s, v in times.items()}
    med = {s: float(np.median(v)) for s, v in t.items()}
    p90 = {s: float(np.percentile(v, 90)) for s, v in t.items()}
    print(f"{len(images)} images, device {dev}, warm pass. Milliseconds per step (median / 90th percentile):\n")
    for s in STEPS:
        print(f"  {s:<12} {med[s]:6.2f} / {p90[s]:6.2f}")

    cum = np.zeros(len(images))
    ready = {}
    for s in STEPS:
        cum = cum + t[s]
        if s.endswith("_post"):
            ready[s[:-5]] = (float(np.median(cum)), float(np.percentile(cum, 90)))
    print("\nPer head: own time = network + post-processing; ready at = time since the frame arrived")
    print(f"  {'head':<7}{'network':>9}{'post':>8}{'own total':>11}{'ready at (median / p90)':>26}")
    for hname in ("head1", "head2", "head3", "final"):
        n, q = med[f"{hname}_net"], med[f"{hname}_post"]
        print(f"  {hname:<7}{n:>9.2f}{q:>8.2f}{n + q:>11.2f}{ready[hname][0]:>14.1f} / {ready[hname][1]:.1f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "device": str(dev), "n_images": len(images),
                   "median_ms": med, "p90_ms": p90,
                   "ready_at_ms": {k: {"median": v[0], "p90": v[1]} for k, v in ready.items()}}, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
