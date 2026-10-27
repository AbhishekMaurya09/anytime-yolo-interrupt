"""
scripts/compare_all_models.py
-------------------------------
Measures five detectors in one run, on the same GPU and the same test images:

  train6            the four-head GELAN anytime model (models/gelan_anytime.py), every head on every frame
  anytime_v8        anytime heads on YOLOv8n, trained together with it, every head on every frame
  anytime_v8_fast   frozen YOLOv8n + depthwise early heads, an early head only when an interrupt asks
  yolov8n, yolov8m  plain Ultralytics models fine-tuned on the same data

  accuracy   every head, this project's metric code, confidence >= 0.001, NMS IoU 0.7, 300 boxes
  pipeline   image in -> boxes on the CPU, for every head (median, 90th percentile); the models are
             interleaved frame by frame so clock drift affects all of them equally
  network    the network alone up to each head, 640 x 640 input
  FPS        frames back to back with no interrupt (3 repeats, median)
  size       parameters and multiply-adds up to each head, 640 x 640 (hooks on every convolution)

train6 runs through this project's own inference code (AnytimeDetector, square 640 x 640 input,
its own post-processing), as in scripts/compare_yolov8n.py. The four Ultralytics-based models share
one pipeline: rectangular letterbox (640 x 384 for 1280 x 720 frames), BatchNorm folded, GPU NMS.

    python -m anytime_yolo_interrupt.scripts.compare_all_models --data-root "C:/Users/abhis/Downloads/bdd1500"
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
from anytime_yolo_interrupt.inference.anytime_detector import AnytimeDetector  # noqa: E402
from anytime_yolo_interrupt.inference.gelan_adapter import GelanAnytimeAdapter  # noqa: E402
from anytime_yolo_interrupt.inference.result_manager import ResultManager  # noqa: E402
from anytime_yolo_interrupt.models.anytime_v8 import HEAD_NAMES  # noqa: E402
from anytime_yolo_interrupt.scripts.evaluate_anytime_v8 import load, stream, summary, sync  # noqa: E402
from anytime_yolo_interrupt.scripts.quantize_compare import HEADS as GELAN_HEADS  # noqa: E402
from anytime_yolo_interrupt.scripts.quantize_compare import evaluate, on_battery, torch_raw_fn  # noqa: E402

V8_MODELS = {  # name: (checkpoint, heads, lazy)
    "anytime_v8": ("runs/anytime_v8/bdd1500/weights/best.pt", HEAD_NAMES, False),
    "anytime_v8_fast": ("runs/anytime_v8/bdd1500_fast/weights/last.pt", HEAD_NAMES, True),
    "yolov8n": ("runs/yolov8n_compare/bdd1500_ft/weights/best.pt", ["final"], False),
    "yolov8m": ("runs/yolov8n_compare/bdd1500_ft_m/weights/best.pt", ["final"], False),
}
TRAIN6 = "runs/anytime_interrupt/train6_bdd70k/last.pt"


def macs_hooks(model):
    n = {"m": 0}

    def hook(c, i, o):
        n["m"] += o.numel() * (c.in_channels // c.groups) * c.kernel_size[0] * c.kernel_size[1]

    return n, [c.register_forward_hook(hook) for c in model.modules() if isinstance(c, nn.Conv2d)]


def v8_macs(model, heads, lazy, x):
    """Multiply-adds up to each head, as the model runs (lazy: only the layers that head needs)."""
    n, hs = macs_hooks(model)
    out = []
    with torch.no_grad():
        if lazy:
            for k in range(len(heads)):
                n["m"] = 0
                stream(model, x, lambda k_, d: None, last_head=k if k < len(heads) - 1 else None, lazy=True)
                out.append(n["m"])
        else:
            stream(model, x, lambda k_, d: out.append(n["m"]))
    for h in hs:
        h.remove()
    return out


def gelan_parts(model):
    """train6 up to head k (1..4), for network-only timing and multiply-adds."""
    def upto(x, k):
        bb = model.forward_backbone(x)
        model.head1(list(bb))
        if k == 1:
            return
        td = model.forward_topdown(*bb)
        model.head2(list(td))
        if k == 2:
            return
        bu = model.forward_bottomup(*td)
        model.head3(list(bu))
        if k == 3:
            return
        model.head_final(list(model.forward_final_refine(*bu)))
    return upto


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max-images", type=int, default=150)
    p.add_argument("--out", default="runs/model_comparison/results.json")
    args = p.parse_args()

    import cv2
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes

    from anytime_yolo_interrupt.models.gelan_anytime import build_model

    if on_battery():
        print("WARNING: the laptop is on battery; timings will be slower and unstable.")
    dev = torch.device("cuda")
    names = load_data_yaml(os.path.join(args.data_root, "data.yaml"))["names"]
    class_names = list(names.values()) if isinstance(names, dict) else list(names)
    paths = sorted(q for q in glob.glob(os.path.join(args.data_root, args.split, "images", "*"))
                   if q.lower().endswith(IMAGE_EXTS))[: args.max_images]
    bgr = [cv2.imread(q) for q in paths]
    rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in bgr]
    ds = YoloAnytimeDataset(args.data_root, args.split, class_names, 640)
    ds.pairs = ds.pairs[: len(paths)]
    assert [os.path.basename(q) for q in paths] == [os.path.basename(a) for a, _ in ds.pairs]
    letterbox = LetterBox(new_shape=(640, 640), auto=True, stride=32)

    def prep(im):
        a = letterbox(image=im)[..., ::-1].transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev).float().div_(255.0).unsqueeze(0)

    models = {name: load(ckpt, dev) for name, (ckpt, _, _) in V8_MODELS.items()}
    g = build_model(len(class_names))
    g.load_state_dict(torch.load(TRAIN6, map_location="cpu", weights_only=False)["model"])
    gelan = g.to(dev).eval()
    upto = gelan_parts(gelan)
    heads = {"train6": GELAN_HEADS, **{n: h for n, (_, h, _) in V8_MODELS.items()}}
    res = {"n_images": len(paths), "split": args.split, "on_battery": on_battery(),
           "gpu": torch.cuda.get_device_name(0), "heads": heads,
           "checkpoints": {"train6": TRAIN6, **{n: c for n, (c, _, _) in V8_MODELS.items()}},
           "lazy": {n: lz for n, (_, _, lz) in V8_MODELS.items()}, "size": {}, "accuracy": {}}

    # ------------------------------------------------------------ size
    x640 = torch.zeros(1, 3, 640, 640, device=dev)
    n, hs = macs_hooks(gelan)
    gm = []
    with torch.no_grad():
        for k in range(1, 5):
            n["m"] = 0
            upto(x640, k)
            gm.append(n["m"])
    for h in hs:
        h.remove()
    res["size"]["train6"] = {"parameters": sum(q.numel() for q in gelan.parameters()), "macs_640_upto_head": gm}
    for name, (_, hd, lazy) in V8_MODELS.items():
        res["size"][name] = {"parameters": sum(q.numel() for q in models[name].parameters()),
                             "macs_640_upto_head": v8_macs(models[name], hd, lazy, x640)}
    print("size:", res["size"], flush=True)

    # ------------------------------------------------------------ accuracy
    print("accuracy: train6 ...", flush=True)
    res["accuracy"]["train6"] = evaluate(torch_raw_fn(gelan, dev, False), ds, class_names)
    for name, m in models.items():
        print(f"accuracy: {name} ...", flush=True)
        preds, gts = [[] for _ in heads[name]], []
        for (image, labels, meta), im in zip(ds, bgr):
            gl = labels.numpy()
            gts.append(ImageGroundTruth(boxes=unletterbox_boxes(gl[:, 1:5], meta) if len(gl) else gl[:, 1:5],
                                        classes=np.array([class_names[int(c)] for c in gl[:, 0]], dtype=object)))
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
                                        "recall": e.recall,
                                        "per_class_ap50": {c: (None if np.isnan(v) else float(v))
                                                           for c, v in e.per_class_ap50.items()}}
        print({h: round(v["mAP50"], 4) for h, v in res["accuracy"][name].items()}, flush=True)

    # ------------------------------------------------------------ network only, 640 x 640
    print("network-only timing ...", flush=True)
    xr = torch.rand(1, 3, 640, 640, device=dev)
    jobs = {"train6": [(h, lambda k=k: upto(xr, k)) for k, h in enumerate(GELAN_HEADS, start=1)]}
    for name, (_, hd, lazy) in V8_MODELS.items():
        lasts = list(range(len(hd) - 1)) + [None]
        jobs[name] = [(h, lambda m=models[name], l=last, lz=lazy: stream(m, xr, lambda k, d: None, last_head=l,
                                                                         lazy=lz)) for h, last in zip(hd, lasts)]
    with torch.no_grad():
        for js in jobs.values():
            for _ in range(30):
                js[-1][1]()
        times = {name: {h: [] for h, _ in js} for name, js in jobs.items()}
        for rep in range(100):
            for name, js in jobs.items():
                for h, f in js:
                    sync()
                    t = time.perf_counter()
                    f()
                    sync()
                    times[name][h].append((time.perf_counter() - t) * 1000.0)
    res["network_only_640"] = {name: {h: summary(v) for h, v in t.items()} for name, t in times.items()}
    print(res["network_only_640"], flush=True)

    # ------------------------------------------------------------ pipeline
    print("pipeline timing ...", flush=True)
    del gelan
    torch.cuda.empty_cache()
    adapter = GelanAnytimeAdapter(TRAIN6, class_names, device="cuda")
    detector = AnytimeDetector(adapter, ResultManager(log_dir=None))

    def v8_frame(m, im, marks, last=None, lazy=False):
        t0 = time.perf_counter()
        x = prep(im)

        def answer(k, d):
            det = non_max_suppression(d, conf_thres=0.25, iou_thres=0.7, max_det=300)[0]
            scale_boxes(x.shape[2:], det[:, :4], im.shape[:2]).cpu()   # boxes on the CPU = answer delivered
            marks.append((time.perf_counter() - t0) * 1000.0)

        with torch.no_grad():
            stream(m, x, answer, last_head=last, lazy=lazy)

    def frame_all(i, record):
        r = [e.elapsed_ms for e in detector.process_frame(adapter.prepare_frame(rgb[i]), i)]
        if record is not None:
            record["train6"].append(r)
        for name, (_, hd, lazy) in V8_MODELS.items():
            if lazy:   # one pass per head an interrupt could ask for
                row = []
                for k in range(len(hd)):
                    marks = []
                    v8_frame(models[name], bgr[i], marks, last=k if k < len(hd) - 1 else None, lazy=True)
                    row.append(marks[0])
            else:
                row = []
                v8_frame(models[name], bgr[i], row)
            if record is not None:
                record[name].append(row)

    for _ in range(2):  # warm-up
        for i in range(len(bgr)):
            frame_all(i, None)
    rec = {name: [] for name in heads}
    for i in range(len(bgr)):
        frame_all(i, rec)
    res["pipeline"] = {name: {h: summary(np.array(v)[:, k]) for k, h in enumerate(heads[name])}
                       for name, v in rec.items()}
    print(res["pipeline"], flush=True)

    # ------------------------------------------------------------ FPS, no interrupt
    print("FPS ...", flush=True)
    fps = {name: [] for name in heads}
    for rep in range(3):
        t = time.perf_counter()
        for i, a in enumerate(rgb):
            detector.process_frame(adapter.prepare_frame(a), i)
        sync()
        fps["train6"].append(len(rgb) / (time.perf_counter() - t))
        for name, (_, _, lazy) in V8_MODELS.items():
            t = time.perf_counter()
            for im in bgr:
                v8_frame(models[name], im, [], lazy=lazy)
            fps[name].append(len(bgr) / (time.perf_counter() - t))
    res["fps"] = {k: {"per_repeat": v, "median": float(np.median(v))} for k, v in fps.items()}
    print(res["fps"], flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
