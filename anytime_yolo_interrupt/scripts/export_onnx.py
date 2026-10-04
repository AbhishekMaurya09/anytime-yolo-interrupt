"""
scripts/export_onnx.py
------------------------
Exports a training checkpoint (last.pt holds only weights, no graph) to
ONNX so the full network can be viewed in Netron or run elsewhere.

    python -m anytime_yolo_interrupt.scripts.export_onnx \
        --checkpoint runs/anytime_interrupt/train2/last.pt

Writes <checkpoint dir>/model.onnx unless --out is given. Outputs are one
tensor per head per scale, named e.g. early_head_1_P3, each shaped
(batch, 10 class + 1 objectness + 4 box, H, W), raw (before sigmoid/decode).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anytime_yolo_interrupt.models.gelan_anytime import build_model  # noqa: E402

SCALES = ("P3", "P4", "P5")


class ExportWrapper(nn.Module):
    """forward() returns plain tensors (ONNX can't return HeadOutput dataclasses)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = []
        for head in self.model(images):
            for cls_logits, obj_logit, reg_raw in head.scale_outputs:
                outputs.append(torch.cat([cls_logits, obj_logit, reg_raw], dim=1))
        return tuple(outputs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_classes = len(ckpt["class_names"]) if "class_names" in ckpt else 10
    model = build_model(num_classes)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "model.onnx")
    output_names = [f"{h}_{s}" for h in model.head_names for s in SCALES]
    dummy = torch.zeros(1, 3, args.img_size, args.img_size)

    torch.onnx.export(
        ExportWrapper(model), dummy, out_path,
        input_names=["images"], output_names=output_names,
        dynamic_axes={"images": {0: "batch"}, **{n: {0: "batch"} for n in output_names}},
        opset_version=args.opset, dynamo=False,
    )

    import onnx
    onnx.checker.check_model(onnx.load(out_path))
    print(f"Saved {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB) with outputs: {', '.join(output_names)}")


if __name__ == "__main__":
    main()
