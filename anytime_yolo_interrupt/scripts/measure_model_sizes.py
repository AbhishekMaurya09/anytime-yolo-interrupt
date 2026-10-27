"""
scripts/measure_model_sizes.py
--------------------------------
Measures, rather than calculates, two things for each head:

1. File size: the ONNX graph cut to heads 1..k (everything that has run when
   head k answers), in FP32, FP16 and INT8. FP32 and INT8 graphs come from
   scripts/quantize_compare.py; the FP16 graph is exported here from the same
   checkpoint. Also records the size of PyTorch state_dict files.
2. Multiply-adds: counted on the real model with forward hooks on every
   convolution while one 640 x 640 image runs through it.

    python -m anytime_yolo_interrupt.scripts.measure_model_sizes \
        --checkpoint runs/anytime_interrupt/train6_bdd70k/last.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anytime_yolo_interrupt.models.gelan_anytime import build_model  # noqa: E402
from anytime_yolo_interrupt.scripts.export_onnx import ExportWrapper  # noqa: E402

HEADS = ["early_head_1", "early_head_2", "early_head_3", "final"]
SCALES = ["p3", "p4", "p5"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--quant-dir", default="runs/anytime_interrupt/quantization",
                   help="folder with model_fp32.onnx and the *_upto_head*.onnx files from quantize_compare")
    p.add_argument("--out", default="runs/anytime_interrupt/model_sizes.json")
    args = p.parse_args()
    import onnx

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(10)
    model.load_state_dict(ckpt["model"])
    model.eval()
    out_names = [f"{h}_{s}" for h in HEADS for s in SCALES]
    mb = lambda path: os.path.getsize(path) / 1e6

    # ---- FP16 ONNX export (on the GPU, where half precision convolutions are supported)
    fp16_path = os.path.join(args.quant_dir, "model_fp16.onnx")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        sys.exit("FP16 export needs a CUDA GPU.")
    m16 = build_model(10)
    m16.load_state_dict(ckpt["model"])
    m16 = m16.to(dev).half().eval()
    torch.onnx.export(ExportWrapper(m16), torch.zeros(1, 3, 640, 640, device=dev, dtype=torch.half), fp16_path,
                      input_names=["images"], output_names=out_names, opset_version=17, dynamo=False)
    del m16

    files, missing = {}, []
    for fmt, whole in (("fp32", "model_fp32.onnx"), ("fp16", "model_fp16.onnx"), ("int8", "model_int8.onnx")):
        whole_path = os.path.join(args.quant_dir, whole)
        sizes = []
        for k in range(1, 5):
            sub = os.path.join(args.quant_dir, f"{'cpu_' if fmt != 'fp16' else ''}{fmt}_upto_head{k}.onnx")
            if fmt == "fp16" or not os.path.exists(sub):
                if not os.path.exists(whole_path):
                    missing.append(whole_path)
                    break
                onnx.utils.extract_model(whole_path, sub, ["images"], out_names[: 3 * k])
            sizes.append(mb(sub))
        files[fmt] = {"whole_model_mb": mb(whole_path) if os.path.exists(whole_path) else None,
                      "upto_head_mb": sizes}
    if missing:
        sys.exit(f"Missing: {missing}. Run scripts/quantize_compare.py first.")

    # ---- PyTorch files: weights only, and the training checkpoint as saved
    with tempfile.TemporaryDirectory() as tmp:
        sd32, sd16 = os.path.join(tmp, "fp32.pt"), os.path.join(tmp, "fp16.pt")
        torch.save(model.state_dict(), sd32)
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}, sd16)
        torch_files = {"state_dict_fp32_mb": mb(sd32), "state_dict_fp16_mb": mb(sd16),
                       "training_checkpoint_mb": mb(args.checkpoint)}

    # ---- multiply-adds counted on the real model
    counter = {"macs": 0}

    def hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1]
        counter["macs"] += out.numel() * (mod.in_channels // mod.groups) * k

    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    macs_upto = []
    with torch.no_grad():
        x = torch.zeros(1, 3, 640, 640)
        bb = model.forward_backbone(x)
        model.head1(list(bb))
        macs_upto.append(counter["macs"])
        td = model.forward_topdown(*bb)
        model.head2(list(td))
        macs_upto.append(counter["macs"])
        bu = model.forward_bottomup(*td)
        model.head3(list(bu))
        macs_upto.append(counter["macs"])
        fr = model.forward_final_refine(*bu)
        model.head_final(list(fr))
        macs_upto.append(counter["macs"])
    for h in hooks:
        h.remove()
    n_params = sum(p.numel() for p in model.parameters())

    result = {"checkpoint": args.checkpoint, "onnx_files": files, "pytorch_files": torch_files,
              "parameters": n_params, "conv_layers": len(hooks), "macs_upto_head": macs_upto}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
