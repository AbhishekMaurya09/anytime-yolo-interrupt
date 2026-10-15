"""Draws the result charts in docs/results/.

Run from the workspace root:
    .venv/Scripts/python docs/make_result_charts.py

Data comes from the run folders (validation results, quantisation results),
docs/accelerator_workload.csv, and docs/results/measured_results.json for
numbers that were measured in one-off evaluations.
"""
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results")
RUNS = os.path.join(ROOT, "runs", "anytime_interrupt")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"       # categorical slots 1-3 (validated)
HEAD_RAMP = ["#86b6ef", "#5598e7", "#256abf", "#104281"]    # ordinal ramp for Head 1 .. Final (validated)
GREY_BAR = "#c3c2b7"
HEADS = ["Head 1", "Head 2", "Head 3", "Final"]
HEAD_KEYS = ["early_head_1", "early_head_2", "early_head_3", "final"]
BAR_IN = 0.22      # bar thickness in inches
GAP_IN = 0.02      # surface gap between touching bars
RADIUS_IN = 0.04   # rounded data end

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
with open(os.path.join(OUT, "measured_results.json")) as f:
    MEASURED = json.load(f)


# --------------------------------------------------------------------------- helpers
def new_fig(w=9.0, h=5.4, ncols=1, left=0.085, right=0.97, top=0.78, bottom=0.13, wspace=0.10, sharey=True):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), sharey=sharey)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)
    return fig, axes


def titles(fig, title, subtitle, x=0.085):
    fig.text(x, 0.955, title, fontsize=14.5, weight="semibold", color=INK, va="center")
    fig.text(x, 0.905, subtitle, fontsize=10, color=INK2, va="top", linespacing=1.35)


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom" if grid == "y" else "left"].set_visible(True)
    ax.spines["bottom" if grid == "y" else "left"].set_color(AXIS)
    ax.tick_params(length=0, pad=6, labelsize=10, labelcolor=INK2)
    ax.grid(axis=grid, color=GRID, lw=1)
    ax.set_axisbelow(True)


def _rounded(x0, y0, x1, y1, r, end):
    """Rectangle with two rounded corners at the data end ('top' or 'right'); square at the baseline."""
    M, L, C = Path.MOVETO, Path.LINETO, Path.CURVE3
    if end == "top":
        verts = [(x0, y0), (x1, y0), (x1, y1 - r), (x1, y1), (x1 - r, y1), (x0 + r, y1), (x0, y1), (x0, y1 - r), (x0, y0)]
    else:
        verts = [(x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + r), (x1, y1 - r), (x1, y1), (x1 - r, y1), (x0, y1), (x0, y0)]
        return Path(verts, [M, L, C, C, L, C, C, L, L])
    return Path(verts, [M, L, L, C, C, L, C, C, L])


def vbar(ax, xc, value, color, offset_in=0.0):
    """Vertical bar from the baseline, fixed thickness in inches, rounded top. Call after limits are set."""
    fig = ax.figure
    bx, by0 = ax.transData.transform((xc, 0)) / fig.dpi
    _, by1 = ax.transData.transform((xc, value)) / fig.dpi
    x0 = bx + offset_in - BAR_IN / 2
    r = max(0.0, min(RADIUS_IN, (by1 - by0)))
    ax.add_patch(PathPatch(_rounded(x0, by0, x0 + BAR_IN, max(by1, by0 + 0.012), r, "top"), fc=color, ec="none",
                           transform=fig.dpi_scale_trans, clip_on=False, zorder=3))


def hbar(ax, yc, value, color):
    fig = ax.figure
    bx0, by = ax.transData.transform((0, yc)) / fig.dpi
    bx1, _ = ax.transData.transform((value, yc)) / fig.dpi
    r = max(0.0, min(RADIUS_IN, (bx1 - bx0)))
    ax.add_patch(PathPatch(_rounded(bx0, by - BAR_IN / 2, max(bx1, bx0 + 0.012), by + BAR_IN / 2, r, "right"),
                           fc=color, ec="none", transform=fig.dpi_scale_trans, clip_on=False, zorder=3))


def label(ax, x, y, text, dx_in=0.0, dy_pt=4, ha="center", va="bottom", color=INK2, size=9.5, weight="normal"):
    ax.annotate(text, xy=(x, y), xytext=(dx_in * 72, dy_pt), textcoords="offset points", ha=ha, va=va,
                fontsize=size, color=color, weight=weight, zorder=5)


def legend(fig, handles, x=0.078, y=0.845, ncol=4):
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y), ncol=ncol, frameon=False, fontsize=10,
                     handlelength=1.6, columnspacing=1.8, handletextpad=0.6)
    for t in leg.get_texts():
        t.set_color(INK2)


def line_key(color, text):
    return Line2D([0], [0], color=color, lw=2, marker="o", markersize=6.5, markeredgecolor=SURFACE,
                  markeredgewidth=1.5, label=text)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("saved", name)


def val_curve(run, key):
    """[(epoch number starting at 1, mAP50)] from a run's val_epoch_XXXX.json files."""
    points = []
    for path in sorted(glob.glob(os.path.join(RUNS, run, "val_epoch_*.json"))):
        with open(path) as f:
            points.append((int(os.path.basename(path)[10:14]) + 1, json.load(f)[key]["mAP50"]))
    return points


def offsets(n):
    return [(i - (n - 1) / 2) * (BAR_IN + GAP_IN) for i in range(n)]


# --------------------------------------------------------------------------- 1. accuracy by run
def chart_accuracy_by_run():
    runs = MEASURED["accuracy_by_run"]["runs"]
    fig, ax = new_fig(top=0.82, bottom=0.19)
    style(ax)
    ax.set_xlim(-0.6, len(runs) - 0.4)
    ax.set_ylim(0, 0.42)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{r['name']}\n{r['training_images']}\n{r['change']}" for r in runs], linespacing=1.4)
    ax.set_ylabel("mAP50", fontsize=10.5, color=INK2, labelpad=8)
    for i, r in enumerate(runs):
        vbar(ax, i, r["map50"], BLUE)
        label(ax, i, r["map50"], f"{r['map50']:.3f}", weight="semibold", color=INK)
    titles(fig, "More training data gave the largest gains",
           "Final head · mAP50 on the same 40 test images, which no run trained on")
    save(fig, "accuracy_by_run")


# --------------------------------------------------------------------------- 2. learning curves, small data
def chart_learning_small():
    series = [("train2", "train2 · no augmentation", BLUE), ("train3", "train3 · learning rate ×10", ORANGE),
              ("train4", "train4 · with augmentation", AQUA)]
    fig, ax = new_fig(right=0.90)
    style(ax)
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 0.04)
    ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Epoch", fontsize=10.5, color=INK2, labelpad=8)
    ax.set_ylabel("Validation mAP50", fontsize=10.5, color=INK2, labelpad=8)
    for run, text, color in series:
        pts = val_curve(run, "final")
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.scatter(xs[-1:], ys[-1:], s=55, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        label(ax, xs[-1], ys[-1], f"{ys[-1]:.3f}", dx_in=0.12, dy_pt=0, ha="left", va="center")
    legend(fig, [line_key(c, t) for _, t, c in series])
    titles(fig, "With augmentation the model kept improving; without it, it levelled off",
           "Final head · validation mAP50 every 5 epochs · 120 training images, 40 validation images")
    save(fig, "learning_curves_small_data")


# --------------------------------------------------------------------------- 3. learning curves, train6 heads
def chart_learning_train6():
    fig, ax = new_fig(right=0.86)
    style(ax)
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 0.36)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("Epoch", fontsize=10.5, color=INK2, labelpad=8)
    ax.set_ylabel("Validation mAP50", fontsize=10.5, color=INK2, labelpad=8)
    ends = []
    for name, key, color in zip(HEADS, HEAD_KEYS, HEAD_RAMP):
        pts = val_curve("train6_bdd70k", key)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.scatter(xs[-1:], ys[-1:], s=55, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        ends.append((ys[-1], xs[-1], name))
    # end labels: spread vertically with thin leaders where the lines finish close together
    ends.sort()
    placed, min_gap = [], 0.026
    for y, x, name in ends:
        ly = y if not placed else max(y, placed[-1] + min_gap)
        placed.append(ly)
        ax.plot([x + 0.35, x + 1.5], [y, ly], color=AXIS, lw=1, zorder=2, clip_on=False)
        ax.text(x + 1.7, ly, f"{name}  {y:.3f}", fontsize=10, color=INK2, va="center", ha="left", clip_on=False)
    legend(fig, [line_key(c, n) for n, c in zip(HEADS, HEAD_RAMP)])
    titles(fig, "All four heads improve together and stay in order",
           "train6 · validation mAP50 every 2 epochs · 65,000 training images, 1,500 validation images")
    save(fig, "learning_curves_train6_heads")


# --------------------------------------------------------------------------- 4. per head
def chart_per_head():
    d = MEASURED["train6_per_head"]
    fig, ax = new_fig()
    style(ax)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 0.36)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xticks(range(4))
    ax.set_xticklabels(d["heads"])
    ax.set_ylabel("Accuracy", fontsize=10.5, color=INK2, labelpad=8)
    off = offsets(2)
    for i in range(4):
        for j, (key, color) in enumerate((("map50", BLUE), ("map50_95", ORANGE))):
            vbar(ax, i, d[key][i], color, off[j])
            label(ax, i, d[key][i], f"{d[key][i]:.3f}", dx_in=off[j] + (0.09 if j else -0.03), size=9)
    legend(fig, [Patch(fc=BLUE, label="mAP50 (box overlap ≥ 50%)"), Patch(fc=ORANGE, label="mAP50-95 (stricter overlap)")])
    titles(fig, "Accuracy rises from Head 1 to Head 3; the Final head matches Head 3",
           "train6 · 1,500 unseen test images")
    save(fig, "per_head_accuracy")


# --------------------------------------------------------------------------- 5. per class
def chart_per_class():
    classes = MEASURED["train6_per_class"]["classes"]
    fig, ax = new_fig(h=5.6, left=0.15, right=0.80, top=0.80, bottom=0.11)
    style(ax, grid="x")
    n = len(classes)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlim(0, 0.72)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_yticks(range(n))
    ax.set_yticklabels([c["name"] for c in classes], fontsize=10.5)
    ax.set_xlabel("AP50", fontsize=10.5, color=INK2, labelpad=8)
    for i, c in enumerate(classes):
        hbar(ax, i, c["ap50"], BLUE)
        label(ax, c["ap50"], i, f"{c['ap50']:.3f}", dx_in=0.07, dy_pt=0, ha="left", va="center", weight="semibold", color=INK)
        ax.text(1.03, i, f"{c['training_boxes']:,}", transform=ax.get_yaxis_transform(), fontsize=10, color=MUTED,
                va="center", ha="left")
    ax.text(1.03, -0.85, "training boxes", transform=ax.get_yaxis_transform(), fontsize=10, color=MUTED, va="center", ha="left")
    titles(fig, "Accuracy follows the amount of training data",
           "train6, final head · AP50 for each class on 1,500 unseen test images", x=0.15)
    save(fig, "per_class_accuracy")


# --------------------------------------------------------------------------- 6-7. quantisation
def chart_quantization():
    with open(os.path.join(RUNS, "quantization", "results.json")) as f:
        q = json.load(f)
    with open(os.path.join(RUNS, "quantization", "cpu_latency_repeat.json")) as f:
        cpu = json.load(f)
    acc = {tag: [q["accuracy"][tag][k]["mAP50"] for k in HEAD_KEYS] for tag in ("gpu_fp32", "gpu_fp16", "cpu_int8")}

    # accuracy
    fig, ax = new_fig()
    style(ax)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 0.42)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xticks(range(4))
    ax.set_xticklabels(HEADS)
    ax.set_ylabel("mAP50", fontsize=10.5, color=INK2, labelpad=8)
    off = offsets(3)
    for i in range(4):
        for j, (tag, color) in enumerate((("gpu_fp32", BLUE), ("gpu_fp16", ORANGE), ("cpu_int8", AQUA))):
            vbar(ax, i, acc[tag][i], color, off[j])
        label(ax, i, acc["gpu_fp32"][i], f"{acc['gpu_fp32'][i]:.3f}", dx_in=off[0] - 0.07, size=9)
        label(ax, i, acc["cpu_int8"][i], f"{acc['cpu_int8'][i]:.3f}", dx_in=off[2] + 0.07, size=9)
    legend(fig, [Patch(fc=BLUE, label="FP32 (as trained)"), Patch(fc=ORANGE, label="FP16"), Patch(fc=AQUA, label="INT8")])
    drop = 100 * (1 - acc["cpu_int8"][3] / acc["gpu_fp32"][3])
    titles(fig, f"FP16 keeps full accuracy; INT8 costs about {drop:.0f}%",
           "train6 · mAP50 on 150 unseen test images · no retraining · FP16 is within 0.001 of FP32 on every head")
    save(fig, "quantization_accuracy")

    # latency: two panels, one shared millisecond axis
    gpu32 = [q["latency"]["gpu_fp32"][k]["median_ms"] for k in HEAD_KEYS]
    gpu16 = [q["latency"]["gpu_fp16"][k]["median_ms"] for k in HEAD_KEYS]
    fig, (a1, a2) = new_fig(w=10.5, ncols=2, left=0.075, right=0.975, top=0.70)
    off = offsets(2)
    for ax, panel, rows in ((a1, "GPU (GTX 1650) · resize + network + post-processing", ((gpu32, BLUE), (gpu16, ORANGE))),
                            (a2, "CPU · network only", ((cpu["fp32"], BLUE), (cpu["int8"], AQUA)))):
        style(ax)
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(0, 130)
        ax.set_yticks([0, 25, 50, 75, 100, 125])
        ax.set_xticks(range(4))
        ax.set_xticklabels(HEADS)
        ax.set_title(panel, fontsize=10.5, color=INK2, loc="left", pad=10)
        for i in range(4):
            for j, (vals, color) in enumerate(rows):
                vbar(ax, i, vals[i], color, off[j])
                label(ax, i, vals[i], f"{vals[i]:.0f}", dx_in=off[j], size=9)
    a1.set_ylabel("Time until the head's output (ms)", fontsize=10.5, color=INK2, labelpad=8)
    legend(fig, [Patch(fc=BLUE, label="FP32"), Patch(fc=ORANGE, label="FP16"), Patch(fc=AQUA, label="INT8")], x=0.068, y=0.835)
    s = q["size_mb"]
    titles(fig, "FP16 is slower on this GPU; INT8 is faster on the CPU",
           "train6 · median time, laptop on mains power · "
           f"model size: FP32 {s['fp32']:.1f} MB, FP16 {s['fp16']:.1f} MB, INT8 {s['onnx_int8']:.1f} MB", x=0.075)
    save(fig, "quantization_latency")


# --------------------------------------------------------------------------- 8. workload by stage
def chart_workload():
    order = [("backbone", "Stem + backbone"), ("head1", "Head 1"), ("fpn", "FPN"), ("head2", "Head 2"),
             ("pan", "PAN"), ("head3", "Head 3"), ("refine", "Final refine"), ("head_final", "Final head")]
    macs = {k: 0 for k, _ in order}
    with open(os.path.join(HERE, "accelerator_workload.csv")) as f:
        for row in csv.DictReader(f):
            macs[row["stage"]] += int(row["macs"])
    total = sum(macs.values())
    heads_share = 100 * sum(v for k, v in macs.items() if k.startswith("head")) / total
    fig, ax = new_fig(h=5.0, left=0.16, right=0.93, top=0.74, bottom=0.13)
    style(ax, grid="x")
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlim(0, 2.0)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([name for _, name in order], fontsize=10.5)
    ax.set_xlabel("Multiply-adds per image (billions)", fontsize=10.5, color=INK2, labelpad=8)
    for i, (key, _) in enumerate(order):
        g = macs[key] / 1e9
        hbar(ax, i, g, BLUE if key.startswith("head") else GREY_BAR)
        label(ax, g, i, f"{g:.2f}  ({100 * macs[key] / total:.0f}%)", dx_in=0.07, dy_pt=0, ha="left", va="center")
    legend(fig, [Patch(fc=BLUE, label="Detection heads"), Patch(fc=GREY_BAR, label="Body stages")], x=0.152, y=0.83)
    titles(fig, f"The four heads are {heads_share:.0f}% of the computation",
           f"Stages in the order they run · 640 × 640 input · total {total / 1e9:.2f} billion multiply-adds", x=0.16)
    save(fig, "workload_by_stage")


# --------------------------------------------------------------------------- 9. post-processing speed-up
def chart_speedup():
    d = MEASURED["postprocess_speedup"]
    light, dark = HEAD_RAMP[0], HEAD_RAMP[2]
    fig, ax = new_fig(h=4.6, left=0.11, right=0.95, top=0.72, bottom=0.16)
    style(ax, grid="x")
    ax.set_ylim(3.5, -0.5)
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks(range(4))
    ax.set_yticklabels(d["heads"], fontsize=10.5)
    ax.set_xlabel("Time until the head's answer is ready (ms)", fontsize=10.5, color=INK2, labelpad=8)
    for i, (b, a) in enumerate(zip(d["before_ms"], d["after_ms"])):
        ax.plot([a, b], [i, i], color=AXIS, lw=2, zorder=2, solid_capstyle="round")
        ax.scatter([b], [i], s=75, color=light, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.scatter([a], [i], s=75, color=dark, edgecolor=SURFACE, linewidth=2, zorder=4)
        label(ax, b, i, f"{b:.1f}", dx_in=0.12, dy_pt=0, ha="left", va="center")
        label(ax, a, i, f"{a:.1f}", dx_in=-0.12, dy_pt=0, ha="right", va="center", weight="semibold", color=INK)
    dot = lambda c, t: Line2D([0], [0], color="none", marker="o", markersize=8, markerfacecolor=c,
                              markeredgecolor=SURFACE, markeredgewidth=1.5, label=t)
    legend(fig, [dot(dark, "After the rewrite"), dot(light, "Before")], x=0.103, y=0.83)
    cuts = [100 * (1 - a / b) for a, b in zip(d["after_ms"], d["before_ms"])]
    titles(fig, f"Faster post-processing cut the wait for an answer by {min(cuts):.0f}% to {max(cuts):.0f}%",
           "Same model and 40 images, GTX 1650 · post-processing went from about 13.6 ms to 2.9 ms per head", x=0.11)
    save(fig, "postprocess_speedup")


if __name__ == "__main__":
    chart_accuracy_by_run()
    chart_learning_small()
    chart_learning_train6()
    chart_per_head()
    chart_per_class()
    chart_quantization()
    chart_workload()
    chart_speedup()
