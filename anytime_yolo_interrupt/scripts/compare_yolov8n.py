"""
scripts/compare_yolov8n.py
----------------------------
Measures the anytime model and a fine-tuned YOLOv8n the same way, on the same
GPU and the same test images:

  accuracy   both scored by this project's own metric code (evaluation/metrics.py)
             at confidence >= 0.001; YOLOv8n is also scored by Ultralytics' own
             validator for reference
  latency    (a) full pipeline per frame: image in -> boxes out, each model's
                 own pre- and post-processing
             (b) network only on a 640 x 640 input
  FPS        frames processed back to back through the full pipeline
  size       parameters and multiply-adds (counted with hooks on every conv)

    python -m anytime_yolo_interrupt.scripts.compare_yolov8n \
        --anytime runs/anytime_interrupt/train6_bdd70k/last.pt \
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anytime_yolo_interrupt.data.yolo_dataset import (  # noqa: E402
    IMAGE_EXTS, YoloAnytimeDataset, load_data_yaml, unletterbox_boxes)
from anytime_yolo_interrupt.evaluation.metrics import (  # noqa: E402
    ImageGroundTruth, ImagePrediction, evaluate_predictions)
from anytime_yolo_interrupt.inference.anytime_detector import AnytimeDetector  # noqa: E402
from anytime_yolo_interrupt.inference.gelan_adapter import GelanAnytimeAdapter  # noqa: E402
from anytime_yolo_interrupt.inference.result_manager import ResultManager  # noqa: E402
from anytime_yolo_interrupt.scripts.quantize_compare import (  # noqa: E402
    HEADS, evaluate, on_battery, torch_raw_fn)


def sync():
    torch.cuda.synchronize()


def count(model: nn.Module, x: torch.Tensor):
    macs = {"n": 0}

    def hook(mod, inp, out):
        macs["n"] += out.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return sum(p.numel() for p in model.parameters()), macs["n"]


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t) * 1000.0)
    return ts


def summary(ts):
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anytime", required=True)
    p.add_argument("--yolov8", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max-images", type=int, default=150)
    p.add_argument("--out", default="runs/yolov8n_compare/comparison.json")
    args = p.parse_args()

    import cv2
    from ultralytics import YOLO

    if on_battery():
        print("WARNING: the laptop is on battery; timings will be slower and unstable.")
    dev = torch.device("cuda")
    names = load_data_yaml(os.path.join(args.data_root, "data.yaml"))["names"]
    class_names = list(names.values()) if isinstance(names, dict) else list(names)
    paths = sorted(q for q in glob.glob(os.path.join(args.data_root, args.split, "images", "*"))
                   if q.lower().endswith(IMAGE_EXTS))[: args.max_images]
    rgb = [cv2.cvtColor(cv2.imread(q), cv2.COLOR_BGR2RGB) for q in paths]
    bgr = [cv2.imread(q) for q in paths]
    ds = YoloAnytimeDataset(args.data_root, args.split, class_names, 640)
    ds.pairs = ds.pairs[: len(paths)]
    assert [os.path.basename(q) for q in paths] == [os.path.basename(a) for a, _ in ds.pairs], \
        "image order differs between the dataset and the file list"
    res = {"n_images": len(paths), "split": args.split, "data_root": args.data_root, "on_battery": on_battery(),
           "gpu": torch.cuda.get_device_name(0)}

    # ---------------------------------------------------------------- models
    from anytime_yolo_interrupt.models.gelan_anytime import build_model
    ckpt = torch.load(args.anytime, map_location="cpu", weights_only=False)
    anytime = build_model(len(class_names))
    anytime.load_state_dict(ckpt["model"])
    anytime = anytime.to(dev).eval()
    yolo = YOLO(args.yolov8)
    ynet = yolo.model.to(dev).eval()
    x = torch.zeros(1, 3, 640, 640, device=dev)
    res["size"] = {"anytime": dict(zip(("parameters", "macs_640"), count(anytime, x))),
                   "yolov8n": dict(zip(("parameters", "macs_640"), count(ynet, x)))}
    print("size:", res["size"], flush=True)

    # ---------------------------------------------------------------- accuracy (same metric code for both)
    print("accuracy: anytime ...", flush=True)
    res["accuracy_anytime"] = evaluate(torch_raw_fn(anytime, dev, False), ds, class_names)
    print("accuracy: yolov8n ...", flush=True)
    preds, gts = [], []
    for (image, labels, meta), im in zip(ds, bgr):
        g = labels.numpy()
        gts.append(ImageGroundTruth(boxes=unletterbox_boxes(g[:, 1:5], meta) if len(g) else g[:, 1:5],
                                    classes=np.array([class_names[int(c)] for c in g[:, 0]], dtype=object)))
        r = yolo.predict(im, imgsz=640, conf=0.001, iou=0.7, max_det=300, device=0, verbose=False)[0]
        preds.append(ImagePrediction(boxes=r.boxes.xyxy.cpu().numpy(), scores=r.boxes.conf.cpu().numpy(),
                                     classes=np.array([class_names[int(c)] for c in r.boxes.cls.cpu().numpy()],
                                                      dtype=object)))
    e = evaluate_predictions(preds, gts, class_names, n_images_expected=len(ds))
    res["accuracy_yolov8n"] = {"mAP50": e.map50, "mAP50_95": e.map50_95, "precision": e.precision,
                               "recall": e.recall}
    try:
        v = yolo.val(data=os.path.join(args.data_root, "data.yaml"), split=args.split, imgsz=640, batch=1,
                     device=0, verbose=False, plots=False)
        res["accuracy_yolov8n_ultralytics_val"] = {"mAP50": float(v.box.map50), "mAP50_95": float(v.box.map)}
    except Exception as exc:  # reference only
        res["accuracy_yolov8n_ultralytics_val"] = {"error": str(exc)}

    # ---------------------------------------------------------------- (b) network only, 640 x 640
    print("latency: network only ...", flush=True)
    xr = torch.rand(1, 3, 640, 640, device=dev)

    def anytime_upto(k):
        def f():
            with torch.no_grad():
                bb = anytime.forward_backbone(xr)
                anytime.head1(list(bb))
                if k == 1:
                    return
                td = anytime.forward_topdown(*bb)
                anytime.head2(list(td))
                if k == 2:
                    return
                bu = anytime.forward_bottomup(*td)
                anytime.head3(list(bu))
                if k == 3:
                    return
                fr = anytime.forward_final_refine(*bu)
                anytime.head_final(list(fr))
        return f

    def ynet_fwd():
        with torch.no_grad():
            ynet(xr)

    for f in [anytime_upto(4), ynet_fwd]:
        timed(f, 50)  # warm-up
    net = {"anytime": {}, "yolov8n": None}
    for rep in range(3):  # interleave so drift affects both equally
        for k, h in enumerate(HEADS, start=1):
            net["anytime"].setdefault(h, []).extend(timed(anytime_upto(k), 50))
        net.setdefault("_y", []).extend(timed(ynet_fwd, 50))
    res["network_only_640"] = {"anytime": {h: summary(v) for h, v in net["anytime"].items()},
                               "yolov8n": summary(net["_y"])}

    # ---------------------------------------------------------------- (a) full pipeline per frame
    print("latency: full pipeline ...", flush=True)
    del anytime
    torch.cuda.empty_cache()
    adapter = GelanAnytimeAdapter(args.anytime, class_names, device="cuda")
    detector = AnytimeDetector(adapter, ResultManager(log_dir=None))

    def yolo_frame(im):
        yolo.predict(im, imgsz=640, conf=0.25, iou=0.7, device=0, verbose=False)

    for _ in range(2):  # warm-up
        for i, (a, b) in enumerate(zip(rgb, bgr)):
            detector.process_frame(adapter.prepare_frame(a), i)
            yolo_frame(b)
    rows, ytimes = [], []
    for i, (a, b) in enumerate(zip(rgb, bgr)):
        rows.append([r.elapsed_ms for r in detector.process_frame(adapter.prepare_frame(a), i)])
        sync()
        t = time.perf_counter()
        yolo_frame(b)
        sync()
        ytimes.append((time.perf_counter() - t) * 1000.0)
    rows = np.array(rows)
    res["full_pipeline"] = {"anytime": {h: summary(rows[:, k]) for k, h in enumerate(HEADS)},
                            "yolov8n": summary(ytimes)}

    # ---------------------------------------------------------------- FPS, frames back to back
    print("FPS ...", flush=True)
    fps = {"anytime": [], "yolov8n": []}
    for rep in range(3):
        t = time.perf_counter()
        for i, a in enumerate(rgb):
            detector.process_frame(adapter.prepare_frame(a), i)
        sync()
        fps["anytime"].append(len(rgb) / (time.perf_counter() - t))
        t = time.perf_counter()
        for b in bgr:
            yolo_frame(b)
        sync()
        fps["yolov8n"].append(len(bgr) / (time.perf_counter() - t))
    res["fps"] = {k: {"per_repeat": v, "median": float(np.median(v))} for k, v in fps.items()}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
