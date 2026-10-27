"""
scripts/evaluate_anytime_v8.py
--------------------------------
Measures the anytime YOLOv8n model (models/anytime_v8.py) head by head, and
plain YOLOv8n through the exact same pipeline, on the same GPU and images:

  accuracy   every head, scored by this project's metric code (confidence >= 0.001)
  pipeline   image in -> boxes out, for each head as it becomes ready:
             rectangular letterbox (Ultralytics LetterBox, auto=True), network
             on the GPU, NMS on the GPU (Ultralytics non_max_suppression)
  network    the network alone up to each head, at 640 x 640 and 640 x 384
  FPS        frames processed back to back through the pipeline
  lazy       the same, but an early head is computed only when an interrupt asks
             for it (its input feature maps are kept anyway for later layers), so
             a frame with no interrupt costs exactly the YOLOv8 pass; also the
             interrupt response: time from the tap being ready to boxes out
  size       parameters and multiply-adds (counted with hooks on every conv)

    python -m anytime_yolo_interrupt.scripts.evaluate_anytime_v8 \
        --anytime runs/anytime_v8/bdd1500/weights/best.pt \
        --yolov8 runs/yolov8n_compare/bdd1500_ft/weights/best.pt \
        --data-root "C:/Users/abhis/Downloads/bdd1500"
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
import torch.nn as nn
import torchvision  # noqa: F401  Ultralytics' NMS uses torchvision's fast kernel only if it is already imported

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anytime_yolo_interrupt.data.yolo_dataset import (  # noqa: E402
    IMAGE_EXTS, YoloAnytimeDataset, load_data_yaml, unletterbox_boxes)
from anytime_yolo_interrupt.evaluation.metrics import (  # noqa: E402
    ImageGroundTruth, ImagePrediction, evaluate_predictions)
from anytime_yolo_interrupt.models.anytime_v8 import HEAD_NAMES, TAPS  # noqa: E402
from anytime_yolo_interrupt.scripts.quantize_compare import on_battery  # noqa: E402

TAP_DONE = {9: 0, 15: 1}   # after this YOLOv8 layer, early head k has all its inputs


def sync():
    torch.cuda.synchronize()


def load(path, dev, fuse=True):
    """Load a checkpoint; with fuse, fold every BatchNorm into its conv (as Ultralytics' predictor does),
    including the early heads, which DetectionModel.fuse() does not reach."""
    from ultralytics.nn.modules import Conv
    from ultralytics.utils.torch_utils import fuse_conv_and_bn

    m = torch.load(path, map_location="cpu", weights_only=False)
    m = (m.get("ema") or m["model"]).float().eval()
    if fuse:
        m.fuse(verbose=False)
        for c in getattr(m, "early", nn.ModuleList()).modules():
            if isinstance(c, Conv) and hasattr(c, "bn"):
                c.conv = fuse_conv_and_bn(c.conv, c.bn)
                delattr(c, "bn")
                c.forward = c.forward_fuse
    return m.to(dev)


def stream(model, x, on_answer, last_head=None, lazy=False, on_tap=None):
    """Run one frame layer by layer; call on_answer(k, decoded) the moment head k is computed.

    Works for AnytimeV8 (three heads) and plain YOLOv8 (final head only). Stops after
    `last_head` when given (for timing the network up to a head). With lazy, an early
    head is computed only if it is `last_head` (the head an interrupt asks for);
    on_tap(k) is called when its inputs are ready, before it is computed."""
    early = getattr(model, "early", None)
    y = []
    for m in model.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x)
        if early is not None and m.i in TAP_DONE:
            k = TAP_DONE[m.i]
            if lazy and last_head != k:
                continue
            if on_tap is not None:
                on_tap(k)
            on_answer(k, early[k]([y[i] for i in TAPS[k]])[0])
            if last_head == k:
                return
    on_answer(len(early) if early is not None else 0, x[0])


def count(model, x):
    n = {"m": 0}

    def hook(c, i, o):
        n["m"] += o.numel() * (c.in_channels // c.groups) * c.kernel_size[0] * c.kernel_size[1]

    hs = [c.register_forward_hook(hook) for c in model.modules() if isinstance(c, nn.Conv2d)]
    macs = []
    with torch.no_grad():
        stream(model, x, lambda k, d: macs.append(n["m"]))
    for h in hs:
        h.remove()
    return sum(p.numel() for p in model.parameters()), macs


def summary(ts):
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anytime", required=True)
    p.add_argument("--yolov8", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max-images", type=int, default=150)
    p.add_argument("--out", default="runs/anytime_v8/evaluation.json")
    args = p.parse_args()

    import cv2
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes

    if on_battery():
        print("WARNING: the laptop is on battery; timings will be slower and unstable.")
    dev = torch.device("cuda")
    names = load_data_yaml(os.path.join(args.data_root, "data.yaml"))["names"]
    class_names = list(names.values()) if isinstance(names, dict) else list(names)
    paths = sorted(q for q in glob.glob(os.path.join(args.data_root, args.split, "images", "*"))
                   if q.lower().endswith(IMAGE_EXTS))[: args.max_images]
    bgr = [cv2.imread(q) for q in paths]
    ds = YoloAnytimeDataset(args.data_root, args.split, class_names, 640)
    ds.pairs = ds.pairs[: len(paths)]
    assert [os.path.basename(q) for q in paths] == [os.path.basename(a) for a, _ in ds.pairs]

    models = {"anytime_v8": load(args.anytime, dev), "yolov8n": load(args.yolov8, dev)}
    heads = {"anytime_v8": HEAD_NAMES, "yolov8n": ["final"]}
    letterbox = LetterBox(new_shape=(640, 640), auto=True, stride=32)

    def prep(im):
        a = letterbox(image=im)[..., ::-1].transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev).float().div_(255.0).unsqueeze(0)

    res = {"n_images": len(paths), "split": args.split, "data_root": args.data_root, "on_battery": on_battery(),
           "gpu": torch.cuda.get_device_name(0), "checkpoints": {"anytime_v8": args.anytime, "yolov8n": args.yolov8},
           "input_shape_example": list(prep(bgr[0]).shape)}

    # ------------------------------------------------------------ size
    res["size"] = {}
    for name, m in models.items():
        n_params, macs = count(m, torch.zeros(1, 3, 640, 640, device=dev))
        res["size"][name] = {"parameters": n_params, "macs_640x640_upto_head": macs}
    print("size:", res["size"], flush=True)

    # ------------------------------------------------------------ accuracy, every head
    res["accuracy"] = {}
    for name, m in models.items():
        print(f"accuracy: {name} ...", flush=True)
        preds = [[] for _ in heads[name]]
        gts = []
        for (image, labels, meta), im in zip(ds, bgr):
            g = labels.numpy()
            gts.append(ImageGroundTruth(boxes=unletterbox_boxes(g[:, 1:5], meta) if len(g) else g[:, 1:5],
                                        classes=np.array([class_names[int(c)] for c in g[:, 0]], dtype=object)))
            x = prep(im)
            out = {}
            with torch.no_grad():
                stream(m, x, lambda k, d: out.__setitem__(k, d))
            for k in range(len(heads[name])):
                det = non_max_suppression(out[k], conf_thres=0.001, iou_thres=0.7, max_det=300)[0]
                boxes = scale_boxes(x.shape[2:], det[:, :4].clone(), im.shape[:2]).cpu().numpy()
                preds[k].append(ImagePrediction(boxes=boxes, scores=det[:, 4].cpu().numpy(),
                                                classes=np.array([class_names[int(c)] for c in det[:, 5].cpu()],
                                                                 dtype=object)))
        res["accuracy"][name] = {}
        for k, h in enumerate(heads[name]):
            e = evaluate_predictions(preds[k], gts, class_names, n_images_expected=len(ds))
            res["accuracy"][name][h] = {"mAP50": e.map50, "mAP50_95": e.map50_95, "precision": e.precision,
                                        "recall": e.recall}
        print(res["accuracy"][name], flush=True)

    # ------------------------------------------------------------ network only, up to each head
    print("network-only timing ...", flush=True)
    res["network_only"] = {}
    for shape in ((640, 640), (640, 384)):
        x = torch.rand(1, 3, shape[1], shape[0], device=dev)
        key = f"{shape[0]}x{shape[1]}"
        res["network_only"][key] = {}
        for name, m in models.items():
            lasts = list(range(len(heads[name]) - 1)) + [None]
            times = {h: [] for h in heads[name]}
            for _ in range(30):  # warm-up
                with torch.no_grad():
                    stream(m, x, lambda k, d: None)
            for rep in range(150):
                for h, last in zip(heads[name], lasts):
                    sync()
                    t = time.perf_counter()
                    with torch.no_grad():
                        stream(m, x, lambda k, d: None, last_head=last)
                    sync()
                    times[h].append((time.perf_counter() - t) * 1000.0)
            res["network_only"][key][name] = {h: summary(v) for h, v in times.items()}
    print(res["network_only"], flush=True)

    # ------------------------------------------------------------ full pipeline, each head as it is ready
    print("pipeline timing ...", flush=True)

    def frame(m, im, marks):
        t0 = time.perf_counter()
        x = prep(im)

        def answer(k, d):
            det = non_max_suppression(d, conf_thres=0.25, iou_thres=0.7, max_det=300)[0]
            scale_boxes(x.shape[2:], det[:, :4], im.shape[:2]).cpu()   # boxes on the CPU = answer delivered
            marks.append((time.perf_counter() - t0) * 1000.0)

        with torch.no_grad():
            stream(m, x, answer)

    for _ in range(2):  # warm-up
        for im in bgr:
            for m in models.values():
                frame(m, im, [])
    pipe = {name: [] for name in models}
    for im in bgr:  # interleave the two models frame by frame
        for name, m in models.items():
            marks = []
            frame(m, im, marks)
            pipe[name].append(marks)
    res["pipeline"] = {name: {h: summary(np.array(v)[:, k]) for k, h in enumerate(heads[name])}
                       for name, v in pipe.items()}
    print(res["pipeline"], flush=True)

    # ------------------------------------------------------------ FPS
    res["fps"] = {}
    for name, m in models.items():
        runs = []
        for rep in range(3):
            t = time.perf_counter()
            for im in bgr:
                frame(m, im, [])
            runs.append(len(bgr) / (time.perf_counter() - t))
        res["fps"][name] = {"per_repeat": runs, "median": float(np.median(runs))}
    print(res["fps"], flush=True)

    # ------------------------------------------------------------ lazy early heads
    print("lazy timing ...", flush=True)
    m = models["anytime_v8"]
    lasts = list(range(len(HEAD_NAMES) - 1)) + [None]
    lazy = {"network_only_640x384": {}, "answer_ms": {}, "interrupt_response_ms": {}}
    x = torch.rand(1, 3, 384, 640, device=dev)
    times = {h: [] for h in HEAD_NAMES}
    for _ in range(30):
        with torch.no_grad():
            stream(m, x, lambda k, d: None, lazy=True)
    for rep in range(150):
        for h, last in zip(HEAD_NAMES, lasts):
            sync()
            t = time.perf_counter()
            with torch.no_grad():
                stream(m, x, lambda k, d: None, last_head=last, lazy=True)
            sync()
            times[h].append((time.perf_counter() - t) * 1000.0)
    lazy["network_only_640x384"] = {h: summary(v) for h, v in times.items()}

    def lazy_frame(im, last, marks, tap_marks=None):
        t0 = time.perf_counter()
        x = prep(im)

        def tap(k):   # the interrupt arrives the moment this head's inputs are ready
            sync()
            tap_marks.append((time.perf_counter() - t0) * 1000.0)

        def answer(k, d):
            det = non_max_suppression(d, conf_thres=0.25, iou_thres=0.7, max_det=300)[0]
            scale_boxes(x.shape[2:], det[:, :4], im.shape[:2]).cpu()
            marks.append((time.perf_counter() - t0) * 1000.0)

        with torch.no_grad():
            stream(m, x, answer, last_head=last, lazy=True, on_tap=tap if tap_marks is not None else None)

    for im in bgr:  # warm-up
        for last in lasts:
            lazy_frame(im, last, [])
    ans = {h: [] for h in HEAD_NAMES}
    resp = {h: [] for h in HEAD_NAMES[:-1]}
    for im in bgr:
        for h, last in zip(HEAD_NAMES, lasts):
            marks = []
            lazy_frame(im, last, marks)
            ans[h].append(marks[0])
        for h, last in zip(HEAD_NAMES[:-1], lasts[:-1]):
            marks, taps = [], []
            lazy_frame(im, last, marks, taps)
            resp[h].append(marks[0] - taps[0])
    lazy["answer_ms"] = {h: summary(v) for h, v in ans.items()}
    lazy["interrupt_response_ms"] = {h: summary(v) for h, v in resp.items()}
    runs = []
    for rep in range(3):
        t = time.perf_counter()
        for im in bgr:
            lazy_frame(im, None, [])
        runs.append(len(bgr) / (time.perf_counter() - t))
    lazy["fps"] = {"per_repeat": runs, "median": float(np.median(runs))}
    res["lazy"] = lazy
    print(res["lazy"], flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
