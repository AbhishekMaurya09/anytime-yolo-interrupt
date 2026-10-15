"""
scripts/profile_fp16_layers.py
--------------------------------
Times every layer of the network on the GPU in FP32 and in FP16, to show
where FP16 gains or loses time. Each layer is run alone on a random input of
the shape it sees in the real network, with the GPU synchronised around
every call. The whole network is also timed end to end in both formats.

    python -m anytime_yolo_interrupt.scripts.profile_fp16_layers \
        --checkpoint runs/anytime_interrupt/train6_bdd70k/last.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anytime_yolo_interrupt.inference.gelan_adapter import GelanAnytimeAdapter  # noqa: E402
from anytime_yolo_interrupt.scripts.quantize_compare import on_battery  # noqa: E402

STAGES = [("head1.", "head1"), ("head2.", "head2"), ("head3.", "head3"), ("head_final.", "head_final"),
          ("_td", "fpn"), ("reduce_", "fpn"), ("up", "fpn"), ("_bu", "pan"), ("_final", "refine")]


def stage_of(name: str) -> str:
    for key, stage in STAGES:
        if name.startswith(key) or key in name.split(".")[0]:
            return stage
    return "backbone"


def kind_of(m: nn.Module) -> str:
    if isinstance(m, nn.Conv2d):
        k, s = m.kernel_size[0], m.stride[0]
        return f"conv {k}x{k}" + (" stride 2" if s == 2 else "")
    return {nn.BatchNorm2d: "batchnorm", nn.SiLU: "silu", nn.MaxPool2d: "maxpool", nn.Upsample: "upsample"}.get(
        type(m), type(m).__name__.lower())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--out", default="runs/anytime_interrupt/fp16_layer_timing.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("This script needs a CUDA GPU.")
    if on_battery():
        print("WARNING: the laptop is on battery; timings will be slower and unstable.")
    dev = torch.device("cuda")
    model32 = GelanAnytimeAdapter(args.checkpoint, [str(i) for i in range(10)], device="cuda").model.eval()
    model16 = copy.deepcopy(model32).half()
    x32 = torch.rand(1, 3, 640, 640, device=dev)

    # one pass to record the input shape of every leaf layer, in execution order
    calls, hooks = [], []
    for name, m in model32.named_modules():
        if len(list(m.children())) == 0 and not isinstance(m, nn.Identity):
            hooks.append(m.register_forward_hook(lambda mod, inp, out, name=name: calls.append((name, tuple(inp[0].shape)))))
    with torch.no_grad():
        model32(x32)
    for h in hooks:
        h.remove()
    mods32, mods16 = dict(model32.named_modules()), dict(model16.named_modules())

    def run(fn, reps):
        for _ in range(5):
            fn()
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000.0)
        return float(np.median(ts))

    rows = []
    with torch.no_grad():
        # cost of one empty synchronised call: the floor under every per-layer number
        floor = run(lambda: None, 200)
        for name, shape in calls:
            a32 = torch.rand(shape, device=dev)
            a16 = a32.half()
            t32 = run(lambda: mods32[name](a32), args.reps)
            t16 = run(lambda: mods16[name](a16), args.reps)
            rows.append({"layer": name, "kind": kind_of(mods32[name]), "stage": stage_of(name),
                         "input_shape": list(shape), "fp32_ms": t32, "fp16_ms": t16})
        x16 = x32.half()
        whole = {"fp32_ms": run(lambda: model32(x32), args.reps), "fp16_ms": run(lambda: model16(x16), args.reps)}
        torch.backends.cudnn.benchmark = True      # let cuDNN try every algorithm and keep the fastest
        whole_bench = {"fp32_ms": run(lambda: model32(x32), args.reps), "fp16_ms": run(lambda: model16(x16), args.reps)}

    def table(key, order=None):
        groups = {}
        for r in rows:
            g = groups.setdefault(r[key], [0, 0.0, 0.0])
            g[0] += 1
            g[1] += r["fp32_ms"]
            g[2] += r["fp16_ms"]
        names = order or sorted(groups, key=lambda n: groups[n][2] - groups[n][1], reverse=True)
        print(f"  {key:<18}{'layers':>7}{'FP32 ms':>10}{'FP16 ms':>10}{'FP16-FP32':>11}{'ratio':>7}")
        for n in names:
            c, a, b = groups[n]
            print(f"  {n:<18}{c:>7}{a:>10.2f}{b:>10.2f}{b - a:>+11.2f}{b / a:>7.2f}")
        return {n: {"layers": groups[n][0], "fp32_ms": groups[n][1], "fp16_ms": groups[n][2]} for n in names}

    s32, s16 = sum(r["fp32_ms"] for r in rows), sum(r["fp16_ms"] for r in rows)
    print(f"GPU: {torch.cuda.get_device_name(0)}   layers timed: {len(rows)}   repeats: {args.reps}")
    print(f"Empty synchronised call: {floor * 1000:.1f} microseconds\n")
    print("Whole network, one image, end to end (median ms):")
    print(f"  default settings        FP32 {whole['fp32_ms']:6.2f}   FP16 {whole['fp16_ms']:6.2f}   ratio {whole['fp16_ms'] / whole['fp32_ms']:.2f}")
    print(f"  cudnn.benchmark = True  FP32 {whole_bench['fp32_ms']:6.2f}   FP16 {whole_bench['fp16_ms']:6.2f}   ratio {whole_bench['fp16_ms'] / whole_bench['fp32_ms']:.2f}")
    print(f"  sum of single layers    FP32 {s32:6.2f}   FP16 {s16:6.2f}   ratio {s16 / s32:.2f}\n")
    print("By layer type (sorted by time lost in FP16):")
    by_kind = table("kind")
    print("\nBy stage:")
    by_stage = table("stage", ["backbone", "head1", "fpn", "head2", "pan", "head3", "refine", "head_final"])
    print("\nThe 12 layers that lose the most time in FP16:")
    print(f"  {'layer':<38}{'kind':<18}{'input':<20}{'FP32':>7}{'FP16':>7}{'ratio':>7}")
    for r in sorted(rows, key=lambda r: r["fp16_ms"] - r["fp32_ms"], reverse=True)[:12]:
        shape = "x".join(str(v) for v in r["input_shape"][1:])
        print(f"  {r['layer']:<38}{r['kind']:<18}{shape:<20}{r['fp32_ms']:>7.3f}{r['fp16_ms']:>7.3f}{r['fp16_ms'] / r['fp32_ms']:>7.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "gpu": torch.cuda.get_device_name(0), "reps": args.reps,
                   "on_battery": on_battery(), "empty_call_ms": floor, "whole_network": whole,
                   "whole_network_cudnn_benchmark": whole_bench, "by_kind": by_kind, "by_stage": by_stage,
                   "layers": rows}, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
