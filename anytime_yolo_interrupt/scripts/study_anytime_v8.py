"""
scripts/study_anytime_v8.py
-----------------------------
The same study as scripts/quantize_compare.py + scripts/measure_model_sizes.py,
for the anytime model on YOLOv8n (models/anytime_v8.py), with lazy early heads:

  accuracy   every head, with per-class AP50, in four versions:
               gpu_fp32, gpu_fp16 (PyTorch)  and  cpu_fp32, cpu_int8 (ONNX Runtime)
  latency    GPU: image in -> boxes on the CPU, when an interrupt asks for that head
                  (lazy: only the layers that head needs run), FP32 and FP16
             CPU: network only, on an ONNX graph cut to exactly the layers that head
                  needs, FP32 and INT8, three interleaved repeats, median of medians
  FPS        frames back to back with no interrupt (final head only), GPU FP32 and FP16
  size       ONNX file of each cut graph (FP32, FP16, INT8), PyTorch state_dict sizes
  compute    multiply-adds and parameters of every stage, counted with hooks

INT8: ONNX Runtime static QDQ quantisation, 8-bit weights per output channel, 8-bit
activations, MinMax ranges from 100 calibration images. The ONNX graphs end at the
raw head convolutions; box decoding and NMS run in float outside, as for train6.

    python -m anytime_yolo_interrupt.scripts.study_anytime_v8 \
        --checkpoint runs/anytime_v8/bdd1500_fast/weights/last.pt \
        --data-root "C:/Users/abhis/Downloads/bdd1500"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anytime_yolo_interrupt.data.yolo_dataset import (  # noqa: E402
    IMAGE_EXTS, YoloAnytimeDataset, load_data_yaml, unletterbox_boxes)
from anytime_yolo_interrupt.evaluation.metrics import (  # noqa: E402
    ImageGroundTruth, ImagePrediction, evaluate_predictions)
from anytime_yolo_interrupt.models.anytime_v8 import HEAD_NAMES, TAPS  # noqa: E402
from anytime_yolo_interrupt.scripts.evaluate_anytime_v8 import load, stream, summary, sync  # noqa: E402
from anytime_yolo_interrupt.scripts.quantize_compare import on_battery  # noqa: E402

TAP_LAYER = {0: 9, 1: 15}          # the YOLOv8 layer after which early head k has its inputs
STAGES = {"backbone": range(0, 10), "fpn": range(10, 16), "pan": range(16, 22)}


def heads_of(model):
    return list(model.early) + [model.model[-1]]


def raw_outputs(model, x, wanted=(0, 1, 2)):
    """Run the layers head max(wanted) needs; return {k: [box_s0, cls_s0, box_s1, cls_s1, box_s2, cls_s2]}
    for every wanted head: the raw maps of the last convolutions, before decoding."""
    last = max(wanted)
    stop = TAP_LAYER[last] if last < 2 else len(model.model) - 2
    y, outs = [], {}
    for m in model.model[:-1]:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x)
        for k in (0, 1):
            if k in wanted and m.i == TAP_LAYER[k]:
                outs[k] = branch(model.early[k], [y[i] for i in TAPS[k]])
        if m.i == stop:
            break
    if 2 in wanted:
        final = model.model[-1]
        outs[2] = branch(final, [y[j] for j in final.f])
    return outs


def branch(head, feats):
    return [t for i in range(head.nl) for t in (head.cv2[i](feats[i]), head.cv3[i](feats[i]))]


def decode(head, maps):
    """Raw maps of one head -> decoded (1, 4 + nc, anchors), exactly as Detect does it."""
    maps = [torch.as_tensor(np.asarray(t)).float() if not torch.is_tensor(t) else t.float() for t in maps]
    n = head.nl
    preds = {"boxes": torch.cat([maps[2 * i].view(1, 4 * head.reg_max, -1) for i in range(n)], -1),
             "scores": torch.cat([maps[2 * i + 1].view(1, head.nc, -1) for i in range(n)], -1),
             "feats": [maps[2 * i] for i in range(n)]}
    head.shape = None   # rebuild anchors for this device and size
    return head._inference(preds)


class RawWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        outs = raw_outputs(self.m, x)
        return tuple(t for k in (0, 1, 2) for t in outs[k])


def out_names(k):
    return [f"h{k + 1}_{kind}{s}" for s in range(3) for kind in ("box", "cls")]


ALL_OUT = [n for k in range(3) for n in out_names(k)]


def accuracy(raw_fn, dec_heads, images, xs, ds, class_names):
    """raw_fn(x) -> {k: maps}. Accuracy of every head, NMS as in evaluate_anytime_v8."""
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    preds, gts = [[] for _ in range(3)], []
    for (image, labels, meta), im, x in zip(ds, images, xs):
        g = labels.numpy()
        gts.append(ImageGroundTruth(boxes=unletterbox_boxes(g[:, 1:5], meta) if len(g) else g[:, 1:5],
                                    classes=np.array([class_names[int(c)] for c in g[:, 0]], dtype=object)))
        outs = raw_fn(x)
        for k in range(3):
            d = decode(dec_heads[k], [torch.as_tensor(np.asarray(t)) if not torch.is_tensor(t) else t.cpu()
                                      for t in outs[k]])
            det = non_max_suppression(d, conf_thres=0.001, iou_thres=0.7, max_det=300)[0]
            boxes = scale_boxes(x.shape[2:], det[:, :4].clone(), im.shape[:2]).numpy()
            preds[k].append(ImagePrediction(boxes=boxes, scores=det[:, 4].numpy(),
                                            classes=np.array([class_names[int(c)] for c in det[:, 5]], dtype=object)))
    rep = {}
    for k, h in enumerate(HEAD_NAMES):
        e = evaluate_predictions(preds[k], gts, class_names, n_images_expected=len(ds))
        rep[h] = {"mAP50": e.map50, "mAP50_95": e.map50_95, "precision": e.precision, "recall": e.recall,
                  "per_class_ap50": {c: (None if np.isnan(v) else float(v)) for c, v in e.per_class_ap50.items()}}
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--max-images", type=int, default=150)
    p.add_argument("--latency-images", type=int, default=60)
    p.add_argument("--calib-images", type=int, default=100)
    p.add_argument("--work-dir", default="runs/anytime_v8/study_fast")
    p.add_argument("--eager", action="store_true",
                   help="every head runs on every frame (anytime_v8): time each head inside one full pass, and "
                        "cut the CPU graphs after each head with all earlier heads included")
    args = p.parse_args()

    import cv2
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes

    os.makedirs(args.work_dir, exist_ok=True)
    if on_battery():
        print("WARNING: the laptop is on battery; timings will be slower and unstable.")
    dev = torch.device("cuda")
    names = load_data_yaml(os.path.join(args.data_root, "data.yaml"))["names"]
    class_names = list(names.values()) if isinstance(names, dict) else list(names)

    def image_list(split, n):
        return sorted(q for q in glob.glob(os.path.join(args.data_root, split, "images", "*"))
                      if q.lower().endswith(IMAGE_EXTS))[:n]

    letterbox = LetterBox(new_shape=(640, 640), auto=True, stride=32)

    def prep(im):
        a = letterbox(image=im)[..., ::-1].transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(a)).float().div_(255.0).unsqueeze(0)

    paths = image_list("test", args.max_images)
    bgr = [cv2.imread(q) for q in paths]
    xs = [prep(im) for im in bgr]
    shape = tuple(xs[0].shape)
    assert all(tuple(x.shape) == shape for x in xs), "ONNX graphs are exported for one input size"
    ds = YoloAnytimeDataset(args.data_root, "test", class_names, 640)
    ds.pairs = ds.pairs[: len(paths)]
    calib = [prep(cv2.imread(q)) for q in image_list("calibration", args.calib_images)]
    assert all(tuple(x.shape) == shape for x in calib)

    m32 = load(args.checkpoint, dev)                  # BatchNorm folded, as the predictor runs it
    m16 = load(args.checkpoint, dev).half()
    cpu_model = load(args.checkpoint, "cpu")          # its heads decode the ONNX outputs
    res = {"checkpoint": args.checkpoint, "n_images": len(xs), "input_shape": list(shape), "on_battery": on_battery(),
           "gpu": torch.cuda.get_device_name(0), "calibration_images": len(calib),
           "mode": "eager" if args.eager else "lazy", "accuracy": {}, "latency": {},
           "fps": {}, "size_mb": {}, "compute": {}}

    # ------------------------------------------------------------ compute: multiply-adds and parameters per stage
    macs = {}
    cur = {"k": None}

    def hook(c, i, o):
        macs[cur["k"]] = macs.get(cur["k"], 0) + o.numel() * (c.in_channels // c.groups) * c.kernel_size[0] * c.kernel_size[1]

    def stage_of(i):
        return next(s for s, r in STAGES.items() if i in r) if i < 22 else "final_head"

    hs = [c.register_forward_hook(hook) for c in cpu_model.modules() if isinstance(c, nn.Conv2d)]
    params = {}
    with torch.no_grad():
        y, x = [], torch.zeros(1, 3, 640, 640)
        for m in cpu_model.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            cur["k"] = stage_of(m.i)
            x = m(x)
            y.append(x)
            params[cur["k"]] = params.get(cur["k"], 0) + sum(q.numel() for q in m.parameters())
        for k, h in enumerate(cpu_model.early):
            cur["k"] = f"early_head_{k + 1}"
            h([y[i] for i in TAPS[k]])
            params[cur["k"]] = sum(q.numel() for q in h.parameters())
    for h in hs:
        h.remove()
    bb, fpn, pan = macs["backbone"], macs["fpn"], macs["pan"]
    res["compute"] = {
        "image": "1 x 3 x 640 x 640 (multiply-adds of convolutions)", "macs_per_stage": macs,
        "parameters_per_stage": params, "parameters_total": sum(params.values()),
        "macs_upto_head_lazy": [bb + macs["early_head_1"], bb + fpn + macs["early_head_2"],
                                bb + fpn + pan + macs["final_head"]],
        "macs_upto_head_eager": [bb + macs["early_head_1"], bb + macs["early_head_1"] + fpn + macs["early_head_2"],
                                 bb + fpn + pan + macs["final_head"] + macs["early_head_1"] + macs["early_head_2"]]}
    print("compute:", res["compute"], flush=True)

    # ------------------------------------------------------------ ONNX export and INT8 quantisation
    fp32_path = os.path.join(args.work_dir, "model_fp32.onnx")
    fp16_path = os.path.join(args.work_dir, "model_fp16.onnx")
    int8_path = os.path.join(args.work_dir, "model_int8.onnx")
    with torch.no_grad():
        torch.onnx.export(RawWrapper(cpu_model).eval(), torch.zeros(shape), fp32_path, input_names=["images"],
                          output_names=ALL_OUT, opset_version=17, dynamo=False)
        torch.onnx.export(RawWrapper(m16).eval(), torch.zeros(shape, device=dev, dtype=torch.half), fp16_path,
                          input_names=["images"], output_names=ALL_OUT, opset_version=17, dynamo=False)
    for m in (cpu_model, m16):   # export restores the wrapper's (training) mode on every child
        m.eval()
    prep_path = os.path.join(args.work_dir, "model_fp32_prep.onnx")
    try:
        quant_pre_process(fp32_path, prep_path)
    except Exception as exc:  # optional clean-up step
        print(f"quant_pre_process skipped: {exc}")
        prep_path = fp32_path

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"images": x.numpy()}

    print(f"int8: calibrating on {len(calib)} images ...", flush=True)
    quantize_static(prep_path, int8_path, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)

    # lazy: graphs cut to exactly what each head needs (head 1 = backbone + head 1, final = no early heads)
    # eager: graphs cut after each head, every earlier head included (they always run first)
    cut = {}
    for fmt, whole in (("fp32", fp32_path), ("fp16", fp16_path), ("int8", int8_path)):
        cut[fmt] = []
        for k in range(3):
            if args.eager:
                sub = os.path.join(args.work_dir, f"{fmt}_upto_head{k + 1}.onnx")
                outs = [n for j in range(k + 1) for n in out_names(j)]
            else:
                sub = os.path.join(args.work_dir, f"{fmt}_head{k + 1}_only.onnx")
                outs = out_names(k)
            onnx.utils.extract_model(whole, sub, ["images"], outs)
            cut[fmt].append(sub)
        res["size_mb"][fmt] = {"whole_model": os.path.getsize(whole) / 1e6,
                               "graph_for_head": [os.path.getsize(s) / 1e6 for s in cut[fmt]]}
    with tempfile.TemporaryDirectory() as tmp:
        sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = (sd.get("ema") or sd["model"]).float().state_dict()
        torch.save(sd, os.path.join(tmp, "a.pt"))
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}, os.path.join(tmp, "b.pt"))
        res["size_mb"]["pytorch_state_dict"] = {"fp32": os.path.getsize(os.path.join(tmp, "a.pt")) / 1e6,
                                                "fp16": os.path.getsize(os.path.join(tmp, "b.pt")) / 1e6}
    res["size_mb"]["checkpoint_file"] = os.path.getsize(args.checkpoint) / 1e6
    print("size:", res["size_mb"], flush=True)

    # ------------------------------------------------------------ accuracy, four versions
    dec_heads = heads_of(cpu_model)
    sessions = {fmt: ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                for fmt, path in (("fp32", fp32_path), ("int8", int8_path))}

    def torch_fn(model, half):
        def fn(x):
            with torch.no_grad():
                outs = raw_outputs(model, x.to(dev).half() if half else x.to(dev))
            return {k: [t.float().cpu() for t in v] for k, v in outs.items()}
        return fn

    def ort_fn(sess):
        def fn(x):
            o = sess.run(ALL_OUT, {"images": x.numpy()})
            return {k: o[6 * k: 6 * k + 6] for k in range(3)}
        return fn

    for tag, fn in (("gpu_fp32", torch_fn(m32, False)), ("gpu_fp16", torch_fn(m16, True)),
                    ("cpu_fp32", ort_fn(sessions["fp32"])), ("cpu_int8", ort_fn(sessions["int8"]))):
        print(f"accuracy: {tag} ...", flush=True)
        res["accuracy"][tag] = accuracy(fn, dec_heads, bgr, xs, ds, class_names)
        print({h: round(v["mAP50"], 4) for h, v in res["accuracy"][tag].items()}, flush=True)

    # ------------------------------------------------------------ GPU latency, FP32 and FP16
    print("gpu latency ...", flush=True)
    lat_bgr = bgr[: args.latency_images]
    lasts = [0, 1, None]
    gpu_models = {"gpu_fp32": (m32, False), "gpu_fp16": (m16, True)}

    def frame(model, half, im, last, marks):
        t0 = time.perf_counter()
        a = letterbox(image=im)[..., ::-1].transpose(2, 0, 1)   # 8-bit image to the GPU, then float there
        x = torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        x = (x.half() if half else x.float()).div_(255.0).unsqueeze(0)

        def answer(k, d):
            det = non_max_suppression(d.float(), conf_thres=0.25, iou_thres=0.7, max_det=300)[0]
            scale_boxes(x.shape[2:], det[:, :4], im.shape[:2]).cpu()
            marks.append((time.perf_counter() - t0) * 1000.0)

        with torch.no_grad():
            stream(model, x, answer, last_head=last, lazy=not args.eager)

    if args.eager:   # one full pass gives the time of every head
        lasts = [None]
    for _ in range(2):  # warm-up: the laptop GPU needs a few seconds of work to leave its low-power state
        for im in lat_bgr:
            for model, half in gpu_models.values():
                for last in lasts:
                    frame(model, half, im, last, [])
    times = {tag: {h: [] for h in HEAD_NAMES} for tag in gpu_models}
    for im in lat_bgr:  # interleave formats and heads frame by frame
        for tag, (model, half) in gpu_models.items():
            if args.eager:
                marks = []
                frame(model, half, im, None, marks)
                for h, t in zip(HEAD_NAMES, marks):
                    times[tag][h].append(t)
                continue
            for h, last in zip(HEAD_NAMES, lasts):
                marks = []
                frame(model, half, im, last, marks)
                times[tag][h].append(marks[0])
    for tag in gpu_models:
        res["latency"][tag] = {h: summary(v) for h, v in times[tag].items()}
        runs = []
        model, half = gpu_models[tag]
        for _ in range(3):
            t = time.perf_counter()
            for im in lat_bgr:
                frame(model, half, im, None, [])
            runs.append(len(lat_bgr) / (time.perf_counter() - t))
        res["fps"][tag] = {"per_repeat": runs, "median": float(np.median(runs))}
    print(res["latency"], res["fps"], flush=True)

    # ------------------------------------------------------------ CPU latency, network only, FP32 and INT8
    print("cpu latency ...", flush=True)
    inputs = [x.numpy() for x in xs[: args.latency_images]]
    subs = {fmt: [ort.InferenceSession(s, providers=["CPUExecutionProvider"]) for s in cut[fmt]]
            for fmt in ("fp32", "int8")}
    for sess_list in subs.values():
        for s in sess_list:
            for x in inputs[:15]:
                s.run(None, {"images": x})
    medians = {fmt: {h: [] for h in HEAD_NAMES} for fmt in subs}
    for rep in range(3):
        for k, h in enumerate(HEAD_NAMES):
            for fmt in subs:
                t_list = []
                for x in inputs:
                    t0 = time.perf_counter()
                    subs[fmt][k].run(None, {"images": x})
                    t_list.append((time.perf_counter() - t0) * 1000.0)
                medians[fmt][h].append(float(np.median(t_list)))
    for fmt in subs:
        res["latency"][f"cpu_{fmt}"] = {h: {"median_ms": float(np.median(v)), "repeat_medians_ms": v}
                                        for h, v in medians[fmt].items()}
    print({k: v for k, v in res["latency"].items() if k.startswith("cpu")}, flush=True)

    out = os.path.join(args.work_dir, "results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
