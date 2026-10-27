"""Writes the weight-value distribution of every convolution layer of the trained model.

Run from the workspace root:
    .venv/Scripts/python docs/make_weight_distribution.py
    .venv/Scripts/python docs/make_weight_distribution.py --model anytime_v8         --checkpoint runs/anytime_v8/bdd1500_fast/weights/last.pt --out docs/anytime_v8_fast/weight_distribution

Outputs, in docs/weight_distribution/ (or --out):
  layer_weight_stats.csv      one row per convolution: statistics of its weights, both as trained
                              and with BatchNorm folded in (what an accelerator or INT8 model runs)
  layer_weight_histogram.csv  64-bin histogram of every layer's weights, both versions

Layers are listed in execution order, the same order as docs/accelerator_workload.csv.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from anytime_yolo_interrupt.models.gelan_anytime import build_model  # noqa: E402

CHECKPOINT = os.path.join(ROOT, "runs", "anytime_interrupt", "train6_bdd70k", "last.pt")
OUT = os.path.join(HERE, "weight_distribution")
BINS = 64
PCTS = [1, 5, 25, 50, 75, 95, 99]


def stage_of_v8(name: str) -> str:
    """Stage of a layer of models/anytime_v8.py (YOLOv8n layers 0-22 plus early.0 / early.1)."""
    top, idx = name.split(".")[:2]
    if top == "early":
        return f"head{int(idx) + 1}"
    i = int(idx)
    return "backbone" if i <= 9 else "fpn" if i <= 15 else "pan" if i <= 21 else "head_final"


def load_v8(checkpoint):
    """The anytime YOLOv8n model unfused (BatchNorm separate), and a function running all three heads."""
    from anytime_yolo_interrupt.scripts.evaluate_anytime_v8 import stream
    m = torch.load(checkpoint, map_location="cpu", weights_only=False)
    m = (m.get("ema") or m["model"]).float().eval()
    return m, lambda x: stream(m, x, lambda k, d: None)


def stage_of(name: str) -> str:
    top = name.split(".")[0]
    if top in ("head1", "head2", "head3", "head_final"):
        return top
    if top == "stem" or top.endswith("_bb") or top in ("down_p3", "down_p4", "down_p5", "sppf_p5"):
        return "backbone"
    if top.endswith("_td") or top.startswith("reduce"):
        return "fpn"
    if top.endswith("_bu"):
        return "pan"
    if top.endswith("_final"):
        return "refine"
    raise ValueError(f"unknown stage for {name}")


def stats(w: np.ndarray, prefix: str) -> dict:
    a = np.abs(w)
    p = np.percentile(w, PCTS)
    row = {f"{prefix}_min": w.min(), f"{prefix}_max": w.max(), f"{prefix}_mean": w.mean(), f"{prefix}_std": w.std(),
           f"{prefix}_abs_mean": a.mean(), f"{prefix}_abs_max": a.max()}
    row.update({f"{prefix}_p{q}": v for q, v in zip(PCTS, p)})
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gelan", choices=["gelan", "anytime_v8"])
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()
    if args.model == "gelan":
        model = build_model(10).eval()
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"])
        run, stage = model, stage_of
    else:
        model, run = load_v8(args.checkpoint)
        stage = stage_of_v8
    bn_of = {m.conv: m.bn for m in model.modules() if hasattr(m, "conv") and isinstance(getattr(m, "bn", None),
                                                                                        nn.BatchNorm2d)}

    order = []  # execution order, recorded by running one image
    hooks = [m.register_forward_hook(lambda mod, i, o, name=n: order.append((name, mod)))
             for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        run(torch.zeros(1, 3, 640, 640))
    for h in hooks:
        h.remove()

    stat_rows, hist_rows = [], []
    for idx, (name, mod) in enumerate(order):
        w = mod.weight.detach().double()
        if mod in bn_of:  # fold BatchNorm: w' = w * g / sqrt(var + eps), b' = beta - mean * g / sqrt(var + eps)
            bn = bn_of[mod]
            scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
            wf = w * scale.view(-1, 1, 1, 1)
            bf = bn.bias.detach().double() - bn.running_mean.double() * scale
        else:
            wf = w   # no BatchNorm after it (head output convs, and DFL, whose fixed weights are 0..15)
            bf = mod.bias.detach().double() if mod.bias is not None else torch.zeros(mod.out_channels).double()
        raw, folded = w.numpy().ravel(), wf.numpy().ravel()
        ch_absmax = wf.abs().amax(dim=(1, 2, 3)).numpy()  # per output channel, the range INT8 per-channel uses

        row = dict(order=idx, layer=name, stage=stage(name), kernel=mod.kernel_size[0], stride=mod.stride[0],
                   cin=mod.in_channels, cout=mod.out_channels, weights=raw.size, batchnorm_folded=mod in bn_of)
        row.update(stats(raw, "trained"))
        row.update(stats(folded, "folded"))
        row.update(folded_bias_min=bf.min().item(), folded_bias_max=bf.max().item(),
                   folded_channel_absmax_min=ch_absmax.min(), folded_channel_absmax_max=ch_absmax.max(),
                   folded_int8_step_per_tensor=np.abs(folded).max() / 127,
                   folded_int8_step_per_channel_mean=(ch_absmax / 127).mean(),
                   folded_share_below_1pct_of_absmax=(np.abs(folded) < 0.01 * np.abs(folded).max()).mean())
        stat_rows.append(row)

        for version, values in (("trained", raw), ("folded", folded)):
            counts, edges = np.histogram(values, bins=BINS)
            for b in range(BINS):
                hist_rows.append(dict(order=idx, layer=name, version=version, bin=b, bin_low=edges[b],
                                      bin_high=edges[b + 1], count=int(counts[b])))

    os.makedirs(args.out, exist_ok=True)
    for fname, rows in (("layer_weight_stats.csv", stat_rows), ("layer_weight_histogram.csv", hist_rows)):
        with open(os.path.join(args.out, fname), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            for r in rows:
                writer.writerow({k: (f"{float(v):.6g}" if isinstance(v, (float, np.floating)) else v)
                                 for k, v in r.items()})
        print(f"saved {fname}: {len(rows)} rows")


if __name__ == "__main__":
    main()
