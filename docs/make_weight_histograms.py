"""Plots the weight-value histogram of every convolution layer.

Reads docs/weight_distribution/layer_weight_histogram.csv and layer_weight_stats.csv (written by
docs/make_weight_distribution.py), so the plots show exactly the values in those files.

Run from the workspace root:
    .venv/Scripts/python docs/make_weight_histograms.py
    .venv/Scripts/python docs/make_weight_histograms.py --src docs/anytime_v8_fast/weight_distribution         --model-name "anytime YOLOv8n (fast)"

Outputs, in docs/weight_distribution/histograms/:
  overview_trained_page_<n>.png   24 layers per page, weights as trained
  overview_folded_page_<n>.png    24 layers per page, BatchNorm folded in
  all_layers.pdf                  every overview page in one file
  layers/<order>_<layer>.png      one picture per layer: trained and folded side by side
"""
import argparse
import collections
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "weight_distribution")
OUT = os.path.join(SRC, "histograms")
LAYERS_OUT = os.path.join(OUT, "layers")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"trained": "#2a78d6", "folded": "#eb6834"}   # validated categorical slots 1 and 2
LABEL = {"trained": "Weights as trained", "folded": "BatchNorm folded in"}
PER_PAGE, NCOLS = 24, 6
MODEL_NAME = "train6"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def load():
    with open(os.path.join(SRC, "layer_weight_stats.csv")) as f:
        stats = list(csv.DictReader(f))
    bins = collections.defaultdict(list)
    with open(os.path.join(SRC, "layer_weight_histogram.csv")) as f:
        for r in csv.DictReader(f):
            bins[(r["layer"], r["version"])].append((float(r["bin_low"]), float(r["bin_high"]), int(r["count"])))
    return stats, bins


def short(name):
    return name.replace(".scale_heads.", ".s").replace(".conv", "").replace("head_final", "final")


def draw(ax, rows, color, fontsize):
    lows = [b[0] for b in rows]
    widths = [b[1] - b[0] for b in rows]
    counts = [b[2] for b in rows]
    ax.bar(lows, counts, width=widths, align="edge", color=color, edgecolor=SURFACE, linewidth=0.3, zorder=3)
    lo, hi = rows[0][0], rows[-1][1]
    if lo < 0 < hi:
        ax.axvline(0, color=INK2, lw=0.7, zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=fontsize, labelcolor=INK2, pad=2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def overview(stats, bins, version, pdf):
    pages = [stats[i:i + PER_PAGE] for i in range(0, len(stats), PER_PAGE)]
    for p, page in enumerate(pages, start=1):
        nrows = (len(page) + NCOLS - 1) // NCOLS
        fig, axes = plt.subplots(4, NCOLS, figsize=(16, 10.5))
        fig.patch.set_facecolor(SURFACE)
        fig.subplots_adjust(left=0.035, right=0.99, top=0.885, bottom=0.04, hspace=0.62, wspace=0.28)
        for ax in axes.flat[len(page):]:
            ax.axis("off")
        for ax, s in zip(axes.flat, page):
            draw(ax, bins[(s["layer"], version)], COLOR[version], 6.5)
            ax.set_title(f"{s['order']} · {short(s['layer'])}", fontsize=7.5, color=INK, loc="left", pad=11)
            ax.text(0, 1.015, f"{s['kernel']}×{s['kernel']}, {int(s['weights']):,} weights · "
                              f"std {float(s[version + '_std']):.3g}",
                    transform=ax.transAxes, fontsize=6.5, color=INK2, va="bottom")
        first, last = page[0]["order"], page[-1]["order"]
        fig.text(0.035, 0.965, f"Weight value histogram of every layer · {LABEL[version]} · "
                               f"page {p} of {len(pages)} (layers {first} to {last})",
                 fontsize=13, weight="semibold", color=INK, va="center")
        fig.text(0.035, 0.935, f"{MODEL_NAME} · 64 equal bins from each layer's own minimum to maximum, so the x-axes differ "
                               "between panels · count of weights on the y-axis · dark line = 0 · layer number = "
                               "execution order",
                 fontsize=9, color=INK2, va="center")
        name = f"overview_{version}_page_{p}"
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=150, facecolor=SURFACE)
        pdf.savefig(fig, facecolor=SURFACE)
        plt.close(fig)
        print("saved", name)


def per_layer(stats, bins):
    os.makedirs(LAYERS_OUT, exist_ok=True)
    for s in stats:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        fig.patch.set_facecolor(SURFACE)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.70, bottom=0.12, wspace=0.18)
        for ax, version in zip(axes, ("trained", "folded")):
            draw(ax, bins[(s["layer"], version)], COLOR[version], 8.5)
            g = lambda k: float(s[f"{version}_{k}"])
            ax.set_title(LABEL[version], fontsize=10.5, color=INK, loc="left", pad=26)
            ax.text(0, 1.03, f"min {g('min'):.4g} · max {g('max'):.4g} · mean {g('mean'):.3g} · std {g('std'):.3g}\n"
                             f"1% below {g('p1'):.4g} · 99% below {g('p99'):.4g}",
                    transform=ax.transAxes, fontsize=8, color=INK2, va="bottom", linespacing=1.4)
        axes[0].set_ylabel("Number of weights", fontsize=9, color=INK2)
        same = " (no BatchNorm, so both are the same)" if s["batchnorm_folded"] == "False" else ""
        fig.text(0.07, 0.94, f"Layer {s['order']}: {s['layer']}", fontsize=12.5, weight="semibold", color=INK,
                 va="center")
        fig.text(0.07, 0.885, f"{s['stage']} · {s['kernel']}×{s['kernel']} stride {s['stride']} · "
                              f"{s['cin']} → {s['cout']} channels · {int(s['weights']):,} weights{same}",
                 fontsize=9, color=INK2, va="center")
        name = f"{int(s['order']):03d}_{s['layer'].replace('.', '_')}"
        fig.savefig(os.path.join(LAYERS_OUT, name + ".png"), dpi=110, facecolor=SURFACE)
        plt.close(fig)
    print(f"saved {len(stats)} per-layer pictures")


def main():
    global SRC, OUT, LAYERS_OUT, MODEL_NAME
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC, help="folder with the two CSV files; pictures go to <src>/histograms")
    p.add_argument("--model-name", default=MODEL_NAME)
    args = p.parse_args()
    SRC, MODEL_NAME = args.src, args.model_name
    OUT = os.path.join(SRC, "histograms")
    LAYERS_OUT = os.path.join(OUT, "layers")
    os.makedirs(OUT, exist_ok=True)
    stats, bins = load()
    with PdfPages(os.path.join(OUT, "all_layers.pdf")) as pdf:
        for version in ("trained", "folded"):
            overview(stats, bins, version, pdf)
    print("saved all_layers.pdf")
    per_layer(stats, bins)


if __name__ == "__main__":
    main()
