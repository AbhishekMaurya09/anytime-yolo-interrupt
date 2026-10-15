"""
scripts/quantize_compare.py
-----------------------------
Compares the trained model in three number formats:

  FP32  32-bit floats, the model as trained
  FP16  16-bit floats (GPU, PyTorch)
  INT8  8-bit integers (CPU, ONNX Runtime, post-training static quantisation
        calibrated on the dataset's calibration split)

For each it measures per-head accuracy on the test split, and the time until
each head's answer is ready.

    python -m anytime_yolo_interrupt.scripts.quantize_compare \
        --checkpoint runs/anytime_interrupt/train6_bdd70k/last.pt \
        --data-root "C:/Users/abhis/Downloads/bdd1500"

INT8 cannot run on the GPU here (that needs TensorRT), so FP16 is compared
with FP32 on the GPU and INT8 is compared with FP32 on the CPU.
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

from anytime_yolo_interrupt.data.yolo_dataset import (  # noqa: E402
    IMAGE_EXTS, YoloAnytimeDataset, load_data_yaml, unletterbox_boxes)
from anytime_yolo_interrupt.evaluation.metrics import (  # noqa: E402
    ImageGroundTruth, ImagePrediction, evaluate_predictions)
from anytime_yolo_interrupt.inference import AnytimeDetector, ResultManager  # noqa: E402
from anytime_yolo_interrupt.inference.gelan_adapter import GelanAnytimeAdapter  # noqa: E402
from anytime_yolo_interrupt.models.gelan_anytime import STRIDES, build_model  # noqa: E402
from anytime_yolo_interrupt.models.postprocess import postprocess_head  # noqa: E402
from anytime_yolo_interrupt.scripts.export_onnx import SCALES, ExportWrapper  # noqa: E402

HEADS = ["early_head_1", "early_head_2", "early_head_3", "final"]
NUM_CLASSES = 10


def on_battery():
    """True if a Windows laptop is running on battery (GPU and CPU are then throttled), None if unknown."""
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        status = Status()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return None
        return {0: True, 1: False}.get(status.ACLineStatus)
    except Exception:
        return None


# --------------------------------------------------------------------------- accuracy
def evaluate(raw_fn, dataset, class_names):
    """raw_fn(image tensor (1,3,H,W) float32 on CPU) -> 12 arrays/tensors (head x scale), each (1,15,H,W)."""
    preds = [[] for _ in HEADS]
    gts = [[] for _ in HEADS]
    for image, labels, meta in dataset:
        outs = raw_fn(image.unsqueeze(0))
        g = labels.numpy()
        gt = ImageGroundTruth(boxes=unletterbox_boxes(g[:, 1:5], meta) if len(g) else g[:, 1:5],
                              classes=np.array([class_names[int(c)] for c in g[:, 0]], dtype=object))
        for h in range(len(HEADS)):
            scale_outputs = []
            for s in range(3):
                t = torch.as_tensor(np.asarray(outs[h * 3 + s]), dtype=torch.float32)
                scale_outputs.append((t[:, :NUM_CLASSES], t[:, NUM_CLASSES:NUM_CLASSES + 1], t[:, NUM_CLASSES + 1:]))
            det = postprocess_head(scale_outputs, STRIDES, conf_thres=0.001, max_detections=300)[0]
            preds[h].append(ImagePrediction(
                boxes=unletterbox_boxes(det["boxes"].numpy(), meta), scores=det["scores"].numpy(),
                classes=np.array([class_names[c] for c in det["cls_ids"].numpy()], dtype=object)))
            gts[h].append(gt)
    report = {}
    for h, name in enumerate(HEADS):
        r = evaluate_predictions(preds[h], gts[h], class_names, n_images_expected=len(dataset))
        report[name] = {"mAP50": r.map50, "mAP50_95": r.map50_95, "precision": r.precision, "recall": r.recall}
    return report


def torch_raw_fn(model, device, half):
    wrapper = ExportWrapper(model)

    def fn(image):
        x = image.to(device)
        with torch.no_grad():
            outs = wrapper(x.half() if half else x)
        return [o.float().cpu().numpy() for o in outs]
    return fn


# --------------------------------------------------------------------------- latency
def gpu_latency(checkpoint, class_names, image_paths, half):
    import cv2
    adapter = GelanAnytimeAdapter(checkpoint, class_names, device="cuda", half=half)
    detector = AnytimeDetector(adapter, ResultManager(log_dir=None))
    images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in image_paths]
    # Two untimed passes first: the accuracy pass before this leaves the GPU
    # mostly idle, and a laptop GPU needs a few seconds of work to leave its
    # low-power state.
    for _ in range(2):
        for image in images:
            detector.process_frame(adapter.prepare_frame(image), 0)
    rows = [[r.elapsed_ms for r in detector.process_frame(adapter.prepare_frame(image), i)]
            for i, image in enumerate(images)]
    t = np.array(rows)
    return {n: {"median_ms": float(np.median(t[:, i])), "p90_ms": float(np.percentile(t[:, i], 90))}
            for i, n in enumerate(HEADS)}


def ort_session(path, simulate=False):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    if simulate:  # keep Quantize/Dequantize as separate float steps: quantisation effect without integer kernels
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def cpu_latency(model_path, work_dir, tag, inputs):
    """Time to each head's raw output on the CPU: head k is timed on a graph cut to heads 1..k."""
    import onnx
    out_names = [f"{h}_{s}" for h in HEADS for s in SCALES]
    result = {}
    for k, name in enumerate(HEADS, start=1):
        sub = os.path.join(work_dir, f"{tag}_upto_head{k}.onnx")
        onnx.utils.extract_model(model_path, sub, ["images"], out_names[: 3 * k])
        sess = ort_session(sub)
        for x in inputs[:15]:
            sess.run(None, {"images": x})
        times = []
        for x in inputs:
            t0 = time.perf_counter()
            sess.run(None, {"images": x})
            times.append((time.perf_counter() - t0) * 1000.0)
        result[name] = {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90))}
    return result


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--test-split", default="test")
    p.add_argument("--calib-split", default="calibration")
    p.add_argument("--calib-images", type=int, default=100)
    p.add_argument("--latency-images", type=int, default=60)
    p.add_argument("--calib-method", default="minmax", choices=["minmax", "percentile", "entropy"])
    p.add_argument("--per-tensor", action="store_true", help="one weight scale per layer instead of per output channel")
    p.add_argument("--exclude-head-outputs", action="store_true",
                   help="leave the 36 final 1x1 head output convolutions in float")
    p.add_argument("--work-dir", default="runs/anytime_interrupt/quantization")
    p.add_argument("--skip-gpu", action="store_true")
    args = p.parse_args()

    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    os.makedirs(args.work_dir, exist_ok=True)
    names = load_data_yaml(os.path.join(args.data_root, "data.yaml"))["names"]
    class_names = list(names.values()) if isinstance(names, dict) else list(names)
    test_ds = YoloAnytimeDataset(args.data_root, args.test_split, class_names, 640)
    calib_ds = YoloAnytimeDataset(args.data_root, args.calib_split, class_names, 640)
    test_paths = sorted(q for q in glob.glob(os.path.join(args.data_root, args.test_split, "images", "*"))
                        if q.lower().endswith(IMAGE_EXTS))[: args.latency_images]

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(q.numel() for q in model.parameters())
    battery = on_battery()
    if battery:
        print("WARNING: running on battery. The GPU and CPU are throttled, so every time below is slower and less "
              "stable than on mains power. Accuracy is not affected.\n", flush=True)
    results = {"checkpoint": args.checkpoint, "test_images": len(test_ds), "calibration_images": args.calib_images,
               "on_battery": battery,
               "settings": {"calib_method": args.calib_method, "per_channel": not args.per_tensor,
                            "exclude_head_outputs": args.exclude_head_outputs},
               "accuracy": {}, "latency": {}, "size_mb": {}}

    # ---- GPU: FP32 vs FP16
    if not args.skip_gpu and torch.cuda.is_available():
        dev = torch.device("cuda")
        for tag, half in (("gpu_fp32", False), ("gpu_fp16", True)):
            m = build_model(NUM_CLASSES).to(dev)
            m.load_state_dict(ckpt["model"])
            m.eval()
            if half:
                m.half()
            print(f"[{tag}] accuracy on {len(test_ds)} images ...", flush=True)
            results["accuracy"][tag] = evaluate(torch_raw_fn(m, dev, half), test_ds, class_names)
            del m
            torch.cuda.empty_cache()
            print(f"[{tag}] latency ...", flush=True)
            results["latency"][tag] = gpu_latency(args.checkpoint, class_names, test_paths, half)
        results["size_mb"]["fp32"] = n_params * 4 / 1e6
        results["size_mb"]["fp16"] = n_params * 2 / 1e6

    # ---- export + INT8 quantisation
    fp32_path = os.path.join(args.work_dir, "model_fp32.onnx")
    out_names = [f"{h}_{s}" for h in HEADS for s in SCALES]
    torch.onnx.export(ExportWrapper(model), torch.zeros(1, 3, 640, 640), fp32_path, input_names=["images"],
                      output_names=out_names, opset_version=17, dynamo=False)
    prep_path = os.path.join(args.work_dir, "model_fp32_prep.onnx")
    try:
        quant_pre_process(fp32_path, prep_path)
    except Exception as exc:  # optional clean-up step
        print(f"quant_pre_process skipped: {exc}")
        prep_path = fp32_path

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.i = 0

        def get_next(self):
            if self.i >= min(args.calib_images, len(calib_ds)):
                return None
            image = calib_ds[self.i][0].unsqueeze(0).numpy()
            self.i += 1
            return {"images": image}

    exclude = []
    if args.exclude_head_outputs:
        exclude = [n.name for n in onnx.load(prep_path).graph.node
                   if n.op_type == "Conv" and any(k in n.name for k in ("cls_pred", "reg_pred", "obj_pred"))]
    method = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
              "entropy": CalibrationMethod.Entropy}[args.calib_method]
    int8_path = os.path.join(args.work_dir, "model_int8.onnx")
    print(f"[int8] calibrating on {min(args.calib_images, len(calib_ds))} images ({args.calib_method}) ...", flush=True)
    quantize_static(prep_path, int8_path, Reader(), quant_format=QuantFormat.QDQ, per_channel=not args.per_tensor,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=method,
                    nodes_to_exclude=exclude)
    results["size_mb"]["onnx_fp32"] = os.path.getsize(fp32_path) / 1e6
    results["size_mb"]["onnx_int8"] = os.path.getsize(int8_path) / 1e6
    results["int8_excluded_nodes"] = len(exclude)

    # ---- CPU: FP32 vs INT8
    def ort_raw_fn(sess):
        return lambda image: sess.run(None, {"images": image.numpy()})

    for tag, path, simulate in (("cpu_fp32", fp32_path, False), ("cpu_int8", int8_path, False),
                                ("cpu_int8_simulated", int8_path, True)):
        print(f"[{tag}] accuracy on {len(test_ds)} images ...", flush=True)
        results["accuracy"][tag] = evaluate(ort_raw_fn(ort_session(path, simulate)), test_ds, class_names)

    inputs = [test_ds[i][0].unsqueeze(0).numpy() for i in range(min(args.latency_images, len(test_ds)))]
    for tag, path in (("cpu_fp32", fp32_path), ("cpu_int8", int8_path)):
        print(f"[{tag}] latency ...", flush=True)
        results["latency"][tag] = cpu_latency(path, args.work_dir, tag, inputs)

    with open(os.path.join(args.work_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- report
    print("\nACCURACY (mAP50 / mAP50-95) on the test split")
    print(f"{'version':<22}" + "".join(f"{h:>18}" for h in HEADS))
    for tag, rep in results["accuracy"].items():
        print(f"{tag:<22}" + "".join(f"{rep[h]['mAP50']:>10.3f} /{rep[h]['mAP50_95']:>6.3f}" for h in HEADS))
    print("\nTIME until each head's output (median ms)")
    print(f"{'version':<22}" + "".join(f"{h:>18}" for h in HEADS))
    for tag, rep in results["latency"].items():
        print(f"{tag:<22}" + "".join(f"{rep[h]['median_ms']:>18.1f}" for h in HEADS))
    print("\nSIZE (MB):", {k: round(v, 1) for k, v in results["size_mb"].items()})
    if battery:
        print("TIMES WERE MEASURED ON BATTERY: compare versions with each other, not with mains-power results.")
    print("GPU times are the full pipeline (resize + network + post-processing).")
    print("CPU times are the network only, on a graph cut to heads 1..k.")
    print(f"\nSaved: {os.path.join(args.work_dir, 'results.json')}")


if __name__ == "__main__":
    main()
