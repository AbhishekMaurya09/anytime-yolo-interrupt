"""Draws the result charts of the fast anytime model on YOLOv8n into docs/anytime_v8_fast/.

The same set as docs/make_result_charts.py draws for train6, for the model trained with
    train_anytime_v8 --freeze-base --depthwise --name bdd1500_fast

Run from the workspace root:
    .venv/Scripts/python docs/make_anytime_v8_fast_charts.py

Every value comes from a measured file:
  runs/anytime_v8/study_fast/results.json   formats, per-class AP50, latency, sizes, multiply-adds
                                            (scripts/study_anytime_v8.py)
  runs/anytime_v8/evaluation_fast.json      lazy answer times and FPS against YOLOv8n in one run
  runs/anytime_v8/evaluation.json           the earlier anytime-v8 model (heads always computed)
  runs/anytime_v8/bdd1500_fast/results.csv  training log
  <data-root>/train/labels                  boxes per class in the training images
"""
import csv
import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from make_result_charts import (AQUA, AXIS, BLUE, DASH, FORMATS, GREY_BAR, HEAD_RAMP, INK, INK2, MUTED,  # noqa: E402
                                ORANGE, SURFACE, hbar, label, legend, line_key, new_fig, offsets, save, style,
                                titles, vbar)

OUT = os.path.join(HERE, "anytime_v8_fast")
RESULTS = os.path.join(OUT, "results")
STUDY = os.path.join(OUT, "parameter_study")
V8 = os.path.join(ROOT, "runs", "anytime_v8")
DATA_ROOT = "C:/Users/abhis/Downloads/bdd1500"

HEADS = ["Head 1", "Head 2", "Final"]
KEYS = ["early_head_1", "early_head_2", "final"]
RAMP = [HEAD_RAMP[0], HEAD_RAMP[2], HEAD_RAMP[3]]
MODEL = "Fast anytime YOLOv8n"
N = 3
LAZY = True        # early heads only on interrupt; docs/make_anytime_v8_charts.py sets False for anytime_v8
TRAIN_RUN = "bdd1500_fast"


def run_note():
    return "only the layers the asked head needs run" if LAZY else "every earlier head runs first"

with open(os.path.join(V8, "study_fast", "results.json")) as f:
    S = json.load(f)
with open(os.path.join(V8, "evaluation_fast.json")) as f:
    E = json.load(f)
with open(os.path.join(V8, "evaluation.json")) as f:
    P = json.load(f)
CLASSES = list(S["accuracy"]["gpu_fp32"]["final"]["per_class_ap50"])


def acc(tag, metric="mAP50"):
    return [S["accuracy"][tag][k][metric] for k in KEYS]


def head_axis(ax, xlabel=True):
    ax.set_xlim(-0.6, N - 0.4)
    ax.set_xticks(range(N))
    ax.set_xticklabels(HEADS)
    if xlabel:
        ax.set_xlabel("Head that answers", fontsize=10.5, color=INK2, labelpad=8)


# --------------------------------------------------------------------------- 1. per head
def chart_per_head():
    m50, m95 = acc("gpu_fp32"), acc("gpu_fp32", "mAP50_95")
    y50, y95 = E["accuracy"]["yolov8n"]["final"]["mAP50"], E["accuracy"]["yolov8n"]["final"]["mAP50_95"]
    fig, ax = new_fig()
    style(ax)
    ax.set_xlim(-0.6, N + 0.6)
    ax.set_ylim(0, 0.36)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xticks(range(N + 1))
    ax.set_xticklabels(HEADS + ["YOLOv8n"])
    ax.set_ylabel("Accuracy", fontsize=10.5, color=INK2, labelpad=8)
    off = offsets(2)
    for i, (a, b) in enumerate(zip(m50 + [y50], m95 + [y95])):
        for j, (v, color) in enumerate(((a, BLUE), (b, ORANGE))):
            vbar(ax, i, v, color if i < N else (GREY_BAR if j else MUTED), off[j])
            label(ax, i, v, f"{v:.3f}", dx_in=off[j] + (0.09 if j else -0.03), size=9)
    legend(fig, [Patch(fc=BLUE, label="mAP50 (box overlap ≥ 50%)"), Patch(fc=ORANGE, label="mAP50-95 (stricter)"),
                 Patch(fc=MUTED, label="YOLOv8n, the same two scores")], ncol=3)
    d = m50[2] - y50
    vs = "the Final head equals YOLOv8n" if abs(d) < 5e-5 else \
        f"Final is {abs(d):.3f} {'above' if d > 0 else 'below'} YOLOv8n"
    titles(fig, f"Head 1 reaches {100 * m50[0] / m50[2]:.0f}% and Head 2 {100 * m50[1] / m50[2]:.0f}% of Final; {vs}",
           f"{MODEL} · {S['n_images']} unseen test images · GPU, FP32")
    save(fig, "per_head_accuracy", RESULTS)


# --------------------------------------------------------------------------- 2. time vs accuracy
def chart_time_vs_accuracy():
    fast_t = [E["lazy"]["answer_ms"][k]["median_ms"] for k in KEYS]
    fast_a = [E["accuracy"]["anytime_v8"][k]["mAP50"] for k in KEYS]
    prev_t = [P["pipeline"]["anytime_v8"][k]["median_ms"] for k in KEYS]
    prev_a = [P["accuracy"]["anytime_v8"][k]["mAP50"] for k in KEYS]
    yt, ya = E["pipeline"]["yolov8n"]["final"]["median_ms"], E["accuracy"]["yolov8n"]["final"]["mAP50"]
    fig, ax = new_fig(right=0.95, top=0.76)
    style(ax)
    ax.grid(axis="x", color="#e1e0d9", lw=1)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 0.34)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel("Time from image to boxes (ms, median)", fontsize=10.5, color=INK2, labelpad=8)
    ax.set_ylabel("mAP50", fontsize=10.5, color=INK2, labelpad=8)
    for ts, as_, color, dash in ((prev_t, prev_a, ORANGE, (0, (3.5, 2.5))), (fast_t, fast_a, BLUE, "-")):
        ax.plot(ts, as_, color=color, lw=2, linestyle=dash, zorder=3)
        ax.scatter(ts, as_, s=55, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        for i, (t, a, h) in enumerate(zip(ts, as_, HEADS)):
            left = color == ORANGE and i == 0
            label(ax, t, a, f"{h}  {t:.1f} ms · {a:.3f}", dx_in=-0.1 if left else 0.1,
                  dy_pt=0 if left else -3 if color == ORANGE else 4, ha="right" if left else "left",
                  va="center" if left else "top" if color == ORANGE else "bottom", size=8.5)
    ax.scatter([yt], [ya], s=80, marker="D", color=INK2, edgecolor=SURFACE, linewidth=2, zorder=5)
    label(ax, yt, ya, f"YOLOv8n  {yt:.1f} ms · {ya:.3f}", dx_in=-0.1, dy_pt=6, ha="right", size=8.5, color=INK)
    legend(fig, [line_key(BLUE, "Fast model: frozen YOLOv8n, depthwise heads, heads only on interrupt"),
                 Line2D([0], [0], color=ORANGE, lw=2, linestyle=(0, (3.5, 2.5)), marker="o", markersize=6.5,
                        markeredgecolor=SURFACE, label="Earlier model: every head computed on every frame"),
                 Line2D([0], [0], color="none", marker="D", markersize=7, markerfacecolor=INK2,
                        markeredgecolor=SURFACE, label="YOLOv8n")], ncol=1, y=0.86)
    titles(fig, f"The Final answer now comes at {fast_t[2]:.1f} ms instead of {prev_t[2]:.1f} ms",
           f"GTX 1650 · resize + network + NMS, boxes on the CPU · {E['n_images']} test images · "
           "each point: the head an interrupt asked for")
    save(fig, "time_vs_accuracy", RESULTS)


# --------------------------------------------------------------------------- 3. per class
def train_boxes():
    counts = [0] * len(CLASSES)
    for path in glob.glob(os.path.join(DATA_ROOT, "train", "labels", "*.txt")):
        with open(path) as f:
            for line in f:
                if line.strip():
                    counts[int(line.split()[0])] += 1
    return counts


def chart_per_class():
    pc = [S["accuracy"]["gpu_fp32"][k]["per_class_ap50"] for k in KEYS]
    boxes = train_boxes()
    fig, ax = new_fig(h=7.6, left=0.15, right=0.80, top=0.81, bottom=0.08)
    style(ax, grid="x")
    n = len(CLASSES)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlim(0, 0.8)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks(range(n))
    ax.set_yticklabels(CLASSES, fontsize=10.5)
    ax.set_xlabel("AP50", fontsize=10.5, color=INK2, labelpad=8)
    step = 0.27
    for i, c in enumerate(CLASSES):
        for j in range(N):
            v = pc[j][c]
            y = i + (j - 1) * step
            if v is None:
                if j == 1:
                    ax.text(0.01, i, "not in the test images", fontsize=9, color=MUTED, va="center")
                continue
            hbar_thin(ax, y, v, RAMP[j])
            label(ax, v, y, f"{v:.3f}", dx_in=0.05, dy_pt=0, ha="left", va="center", size=8.5,
                  color=INK if j == 2 else INK2)
        ax.text(1.03, i, f"{boxes[i]:,}", transform=ax.get_yaxis_transform(), fontsize=10, color=MUTED,
                va="center", ha="left")
    ax.text(1.03, -0.95, "training boxes", transform=ax.get_yaxis_transform(), fontsize=10, color=MUTED,
            va="center", ha="left")
    legend(fig, [Patch(fc=c, label=h) for c, h in zip(RAMP, HEADS)], x=0.143, y=0.862, ncol=3)
    zero = [c for c in CLASSES if pc[0][c] == 0]
    best = all(pc[2][c] >= max(pc[0][c], pc[1][c]) for c in CLASSES if pc[2][c] is not None)
    titles(fig, "The Final head is the most accurate for every class" if best else "Accuracy for each class",
           f"{MODEL} · AP50 for each class on {S['n_images']} unseen test images · GPU, FP32"
           + (f"\nHead 1 finds no {', '.join(zero[:-1])} or {zero[-1]}: the classes with the fewest training boxes"
              if len(zero) > 1 else ""), x=0.15)
    save(fig, "per_class_accuracy", RESULTS)


def hbar_thin(ax, y, v, color):
    ax.barh(y, v, height=0.24, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)


# --------------------------------------------------------------------------- 4. training curves
def chart_learning():
    with open(os.path.join(V8, TRAIN_RUN, "results.csv")) as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    ep = [int(r["epoch"]) for r in rows]
    series = [("train/box_loss", "Box loss (CIoU)", BLUE), ("train/cls_loss", "Class loss (BCE)", ORANGE),
              ("train/dfl_loss", "Box-edge loss (DFL)", AQUA)]
    fig, ax = new_fig(right=0.84)
    style(ax)
    ax.set_xlim(0, len(ep) + 1)
    top = max(float(r[k]) for r in rows for k, _, _ in series)
    ax.set_ylim(0, math.ceil(top + 0.5))
    ax.set_xlabel("Epoch", fontsize=10.5, color=INK2, labelpad=8)
    ax.set_ylabel("Training loss, Head 1 + Head 2" if LAZY else "Training loss, weighted sum of all heads",
                  fontsize=10.5, color=INK2, labelpad=8)
    ends = [float(rows[-1][k]) for k, _, _ in series]
    for (key, text, color), end in zip(series, ends):
        ys = [float(r[key]) for r in rows]
        ax.plot(ep, ys, color=color, lw=2, solid_capstyle="round", zorder=3)
        ax.scatter(ep[-1:], ys[-1:], s=55, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        # push apart end labels that would touch (closer than 0.25 in loss units)
        near = [e for e in ends if e != end and abs(e - end) < 0.25]
        dy = 0 if not near else (7 if end > near[0] else -7)
        label(ax, ep[-1], ys[-1], f"{ys[0]:.2f} → {ys[-1]:.2f}", dx_in=0.12, dy_pt=dy, ha="left", va="center")
    legend(fig, [line_key(c, t) for _, t, c in series], ncol=3)
    fin = float(rows[-1]["metrics/mAP50(B)"])
    if LAZY:
        titles(fig, "The early heads' training losses fell throughout the 50 epochs",
               f"Sum of the two early heads' losses · 1,200 training images\nYOLOv8n is frozen, so Ultralytics' "
               f"validation (Final head only) stayed at mAP50 {fin:.3f} in every epoch")
    else:
        first = float(rows[0]["metrics/mAP50(B)"])
        titles(fig, "All three heads' training losses fell throughout the 50 epochs",
               "Losses of Head 1, Head 2 and Final added with weights 0.3, 0.6 and 1.0 · 1,200 training images\n"
               f"Ultralytics' validation (Final head only, 150 calibration images): mAP50 {first:.3f} after epoch 1, "
               f"{fin:.3f} after epoch {len(rows)}")
    save(fig, "learning_curves", RESULTS)


# --------------------------------------------------------------------------- 5-6. number formats
def chart_formats():
    a = {t: acc(t) for t in ("gpu_fp32", "gpu_fp16", "cpu_int8")}
    fig, ax = new_fig()
    style(ax)
    head_axis(ax, False)
    ax.set_ylim(0, 0.34)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_ylabel("mAP50", fontsize=10.5, color=INK2, labelpad=8)
    off = offsets(3)
    for i in range(N):
        for j, (tag, color) in enumerate((("gpu_fp32", BLUE), ("gpu_fp16", ORANGE), ("cpu_int8", AQUA))):
            vbar(ax, i, a[tag][i], color, off[j])
        label(ax, i, a["gpu_fp32"][i], f"{a['gpu_fp32'][i]:.3f}", dx_in=off[0] - 0.07, size=9)
        label(ax, i, a["cpu_int8"][i], f"{a['cpu_int8'][i]:.3f}", dx_in=off[2] + 0.07, size=9)
    legend(fig, [Patch(fc=BLUE, label="FP32 (as trained)"), Patch(fc=ORANGE, label="FP16"),
                 Patch(fc=AQUA, label="INT8")])
    drops = [100 * (1 - i8 / f32) for i8, f32 in zip(a["cpu_int8"], acc("cpu_fp32"))]
    d16 = max(abs(x - y) for x, y in zip(a["gpu_fp16"], a["gpu_fp32"]))
    titles(fig, f"FP16 keeps full accuracy; INT8 costs {min(drops):.0f}% to {max(drops):.0f}%",
           f"{MODEL} · mAP50 on {S['n_images']} unseen test images · no retraining · "
           f"FP16 is within {d16:.4f} of FP32 on every head")
    save(fig, "quantization_accuracy", RESULTS)

    lat = {t: [S["latency"][t][k]["median_ms"] for k in KEYS] for t in ("gpu_fp32", "gpu_fp16", "cpu_fp32", "cpu_int8")}
    ymax = max(max(v) for v in lat.values())
    fig, (a1, a2) = new_fig(w=10.5, ncols=2, left=0.075, right=0.975, top=0.70)
    off = offsets(2)
    for ax, panel, rows in ((a1, "GPU (GTX 1650) · resize + network + NMS", (("gpu_fp32", BLUE), ("gpu_fp16", ORANGE))),
                            (a2, "CPU · network only", (("cpu_fp32", BLUE), ("cpu_int8", AQUA)))):
        style(ax)
        head_axis(ax, False)
        ax.set_ylim(0, ymax * 1.15)
        ax.set_title(panel, fontsize=10.5, color=INK2, loc="left", pad=10)
        for i in range(N):
            for j, (tag, color) in enumerate(rows):
                vbar(ax, i, lat[tag][i], color, off[j])
                label(ax, i, lat[tag][i], f"{lat[tag][i]:.1f}", dx_in=off[j], size=9)
    a1.set_ylabel("Time until the answer (ms)", fontsize=10.5, color=INK2, labelpad=8)
    legend(fig, [Patch(fc=BLUE, label="FP32"), Patch(fc=ORANGE, label="FP16"), Patch(fc=AQUA, label="INT8")],
           x=0.068, y=0.835)
    mb = {f: S["size_mb"][f]["whole_model"] for f in ("fp32", "fp16", "int8")}
    g = lat["gpu_fp16"][2] / lat["gpu_fp32"][2]
    c = lat["cpu_fp32"][2] / lat["cpu_int8"][2]
    titles(fig, f"FP16 is {g:.2f}× the FP32 time on this GPU; INT8 is {c:.2f}× faster on the CPU (Final head)",
           f"{MODEL} · median, mains power · {run_note()} · ONNX file: "
           f"FP32 {mb['fp32']:.1f} MB, FP16 {mb['fp16']:.1f} MB, INT8 {mb['int8']:.1f} MB", x=0.075)
    save(fig, "quantization_latency", RESULTS)


# --------------------------------------------------------------------------- 7. workload by stage
def chart_workload():
    m = S["compute"]["macs_per_stage"]
    when = "on interrupt" if LAZY else "every frame"
    order = [("backbone", "Backbone (layers 0-9)"), ("early_head_1", f"Head 1 ({when})"),
             ("fpn", "FPN (layers 10-15)"), ("early_head_2", f"Head 2 ({when})"),
             ("pan", "PAN (layers 16-21)"), ("final_head", "Final head (layer 22)")]
    lazy_total = m["backbone"] + m["fpn"] + m["pan"] + m["final_head"]
    if not LAZY:
        lazy_total += m["early_head_1"] + m["early_head_2"]   # every head runs on every frame
    fig, ax = new_fig(h=4.8, left=0.20, right=0.93, top=0.74, bottom=0.13)
    style(ax, grid="x")
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlim(0, 2.3)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([n for _, n in order], fontsize=10.5)
    ax.set_xlabel("Multiply-adds per image (billions)", fontsize=10.5, color=INK2, labelpad=8)
    for i, (key, _) in enumerate(order):
        g = m[key] / 1e9
        early = key.startswith("early")
        hbar(ax, i, g, ORANGE if early else BLUE if key == "final_head" else GREY_BAR)
        text = f"{g:.3f}" + ("" if early and LAZY else f"  ({100 * m[key] / lazy_total:.0f}% of a frame)")
        label(ax, g, i, text, dx_in=0.07, dy_pt=0, ha="left", va="center")
    legend(fig, [Patch(fc=GREY_BAR, label="Body stages"), Patch(fc=BLUE, label="Final head"),
                 Patch(fc=ORANGE, label="Early heads, run only when an interrupt asks" if LAZY
                       else "Early heads, run on every frame")], x=0.192, y=0.83, ncol=3)
    if LAZY:
        titles(fig, f"A frame with no interrupt costs {lazy_total / 1e9:.2f} billion multiply-adds, as YOLOv8n",
               f"{MODEL} · stages in the order they run · 640 × 640 input\n"
               f"Each early head adds {m['early_head_1'] / 1e9:.3f} billion, only when an interrupt asks for it",
               x=0.20)
    else:
        share = 100 * (m["early_head_1"] + m["early_head_2"]) / lazy_total
        titles(fig, f"Every frame costs {lazy_total / 1e9:.2f} billion multiply-adds; the two early heads are "
                    f"{share:.0f}% of it",
               f"{MODEL} · stages in the order they run · 640 × 640 input\n"
               f"Each early head costs {m['early_head_1'] / 1e9:.3f} billion and runs on every frame", x=0.20)
    save(fig, "workload_by_stage", RESULTS)


# --------------------------------------------------------------------------- 8. before / after
def chart_before_after():
    before = [P["pipeline"]["anytime_v8"][k]["median_ms"] for k in KEYS]
    after = [E["lazy"]["answer_ms"][k]["median_ms"] for k in KEYS]
    yt = E["pipeline"]["yolov8n"]["final"]["median_ms"]
    light, dark = HEAD_RAMP[0], HEAD_RAMP[2]
    fig, ax = new_fig(h=4.4, left=0.11, right=0.95, top=0.70, bottom=0.17)
    style(ax, grid="x")
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks(range(N))
    ax.set_yticklabels(HEADS, fontsize=10.5)
    ax.set_xlabel("Time from image to boxes (ms, median)", fontsize=10.5, color=INK2, labelpad=8)
    ax.axvline(yt, color=MUTED, lw=1.2, linestyle=(0, (3, 3)), zorder=2)
    ax.text(yt, -0.62, f"YOLOv8n {yt:.1f} ms", fontsize=9, color=INK2, ha="center", va="bottom")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.plot([a, b], [i, i], color=AXIS, lw=2, zorder=2, solid_capstyle="round")
        ax.scatter([b], [i], s=75, color=light, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.scatter([a], [i], s=75, color=dark, edgecolor=SURFACE, linewidth=2, zorder=4)
        label(ax, a, i, f"{a:.1f}", dy_pt=9, weight="semibold", color=INK)
        label(ax, b, i, f"{b:.1f}", dx_in=0.12 if b > a else -0.12, dy_pt=0, ha="left" if b > a else "right",
              va="center")
    dot = lambda c, t: Line2D([0], [0], color="none", marker="o", markersize=8, markerfacecolor=c,
                              markeredgecolor=SURFACE, markeredgewidth=1.5, label=t)
    legend(fig, [dot(dark, "Fast model (heads only on interrupt)"), dot(light, "Earlier model (every head, every frame)")],
           x=0.103, y=0.83, ncol=2)
    fps_b, fps_a = P["fps"]["anytime_v8"]["median"], E["lazy"]["fps"]["median"]
    titles(fig, f"Final answer {before[2]:.1f} → {after[2]:.1f} ms; frames per second {fps_b:.1f} → {fps_a:.1f}",
           "GTX 1650, FP32 · Head 1 is slower now, because it is computed only after the interrupt arrives", x=0.11)
    save(fig, "latency_before_after", RESULTS)


# --------------------------------------------------------------------------- 9-13. three inputs, five outputs
DEVICES = {"GPU (GTX 1650)": {"FP32": "gpu_fp32", "FP16": "gpu_fp16"}, "CPU": {"FP32": "cpu_fp32", "INT8": "cpu_int8"}}


def lines(ax, series, fmt, same=0.0):
    for name, ys in series.items():
        ax.plot(range(N), ys, color=FORMATS[name], lw=2, linestyle=DASH.get(name, "-"), dash_capstyle="round",
                zorder=3)
        ax.scatter(range(N), ys, s=55, color=FORMATS[name], edgecolor=SURFACE, linewidth=2, zorder=4)
    order = [n for n in FORMATS if n in series]
    for i in range(N):
        groups = []
        for n in sorted(order, key=lambda n: series[n][i], reverse=True):
            if groups and series[groups[-1][-1]][i] - series[n][i] <= same:
                groups[-1].append(n)
            else:
                groups.append([n])
        for k, g in enumerate(groups):
            texts = []
            for n in sorted(g, key=order.index):
                t = fmt.format(series[n][i])
                if t not in texts:
                    texts.append(t)
            below = len(groups) > 1 and k == len(groups) - 1
            y = min(series[n][i] for n in g) if below else max(series[n][i] for n in g)
            label(ax, i, y, " / ".join(texts), dy_pt=-8 if below else 8, va="top" if below else "bottom", size=9)


def format_key(f):
    return Line2D([0], [0], color=FORMATS[f], lw=2, linestyle=DASH.get(f, "-"), marker="o", markersize=6.5,
                  markeredgecolor=SURFACE, markeredgewidth=1.5, label=f)


def device_panels(name, title, subtitle, get, ylabel, ylim, fmt, same=0.0):
    fig, axes = new_fig(w=10.5, h=5.7, ncols=2, left=0.075, right=0.975, top=0.67, bottom=0.13)
    for ax, (device, formats) in zip(axes, DEVICES.items()):
        style(ax)
        head_axis(ax)
        ax.set_xlim(-0.35, N - 0.65)
        ax.set_ylim(*ylim)
        ax.set_title(device, fontsize=10.5, color=INK2, loc="left", pad=10)
        lines(ax, {f: get(key) for f, key in formats.items()}, fmt, same)
    axes[0].set_ylabel(ylabel, fontsize=10.5, color=INK2, labelpad=8)
    legend(fig, [format_key(f) for f in FORMATS], x=0.068, y=0.815)
    titles(fig, title, subtitle, x=0.075)
    save(fig, name, STUDY)


def chart_parameter_study():
    base = (f"{MODEL} · {S['n_images']} unseen test images · FP16 was run only on the GPU, INT8 only on the CPU\n"
            "Where FP32 and FP16 are equal the lines overlap; labels read FP32 / FP16 where they differ")
    for metric, name, text, ylim in (("mAP50", "params_map50", "mAP50", (0, 0.34)),
                                     ("mAP50_95", "params_map50_95", "mAP50-95", (0, 0.19))):
        get = lambda key, metric=metric: acc(key, metric)
        f32, i8 = get("cpu_fp32"), get("cpu_int8")
        device_panels(name, f"Output: {text} {f32[0]:.3f} at Head 1, {f32[2]:.3f} at Final (FP32); "
                            f"INT8 is {f32[2] - i8[2]:.3f} lower at Final",
                      base, get, text, ylim, "{:.3f}", same=0.0015)

    lat = lambda key: [S["latency"][key][k]["median_ms"] for k in KEYS]
    ymax = max(max(lat(k)) for k in ("gpu_fp32", "gpu_fp16", "cpu_fp32", "cpu_int8"))
    device_panels("params_latency", "Output: time until the answer, when an interrupt asks for that head" if LAZY
                  else "Output: time until each head's answer, every head computed in order",
                  f"{MODEL} · median, mains power · {run_note()}\n"
                  "GPU: resize + network + NMS · CPU: network only, so compare formats within a panel",
                  lat, "Time until the answer (ms)", (0, ymax * 1.2), "{:.1f}")

    sizes = {f.upper(): S["size_mb"][f]["graph_for_head"] for f in ("fp32", "fp16", "int8")}
    fig, ax = new_fig(right=0.95, top=0.74)
    style(ax)
    head_axis(ax)
    ax.set_xlim(-0.35, N - 0.65)
    ax.set_ylim(0, max(sizes["FP32"]) * 1.2)
    ax.set_ylabel("Model file size (MB)", fontsize=10.5, color=INK2, labelpad=8)
    lines(ax, sizes, "{:.2f}")
    legend(fig, [format_key(f) for f in FORMATS], y=0.81)
    titles(fig, f"Output: model file size, {sizes['FP32'][2]:.2f} / {sizes['FP16'][2]:.2f} / "
                f"{sizes['INT8'][2]:.2f} MB for the Final head",
           ("Measured size of the ONNX file holding exactly the layers that head needs · the same on GPU and CPU\n"
            "The Final head's file has no early heads, because they run only on interrupt") if LAZY else
           ("Measured size of the ONNX file holding every layer that has run by that head · the same on GPU and CPU\n"
            "Each file is the model cut after that head, so earlier heads are included"))
    save(fig, "params_file_size", STUDY)

    lazy = [v / 1e9 for v in S["compute"]["macs_upto_head_lazy"]]
    eager = [v / 1e9 for v in S["compute"]["macs_upto_head_eager"]]
    main_text = "Only the layers the asked head needs (how the fast model runs)"
    alt_text = "Every earlier head also computed"
    if not LAZY:   # the model runs every head; show the lazy alternative as the dashed line
        lazy, eager = eager, lazy
        main_text, alt_text = "Every earlier head computed (how this model runs)", \
            "If early heads ran only on interrupt"
    fig, ax = new_fig(right=0.95, top=0.74)
    style(ax)
    head_axis(ax)
    ax.set_xlim(-0.35, N - 0.65)
    ax.set_ylim(0, math.ceil(max(lazy + eager) * 1.15))
    ax.set_ylabel("Multiply-adds (billions)", fontsize=10.5, color=INK2, labelpad=8)
    ax.plot(range(N), eager, color=MUTED, lw=1.5, linestyle=(0, (3, 3)), zorder=2)
    ax.scatter(range(N), eager, s=35, color=MUTED, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.plot(range(N), lazy, color=INK2, lw=2, zorder=3)
    ax.scatter(range(N), lazy, s=55, color=INK2, edgecolor=SURFACE, linewidth=2, zorder=4)
    for i, (v, e) in enumerate(zip(lazy, eager)):
        label(ax, i, v, f"{v:.3f}", dy_pt=-10 if e > v else 8, va="top" if e > v else "bottom", size=9.5,
              color=INK)
        if abs(e - v) > 1e-9:
            label(ax, i, e, f"{e:.3f}", dy_pt=8 if e > v else -10, va="bottom" if e > v else "top", size=8.5,
                  color=MUTED)
    legend(fig, [line_key(INK2, main_text),
                 Line2D([0], [0], color=MUTED, lw=1.5, linestyle=(0, (3, 3)), marker="o", markersize=5,
                        label=alt_text)], ncol=1, y=0.85)
    titles(fig, f"Output: multiply-adds {lazy[0]:.2f}, {lazy[1]:.2f} and {lazy[2]:.2f} billion",
           "Counted on the model for one 640 × 640 image · the same for every number format and on GPU and CPU")
    save(fig, "params_compute", STUDY)


if __name__ == "__main__":
    chart_per_head()
    chart_time_vs_accuracy()
    chart_per_class()
    chart_learning()
    chart_formats()
    chart_workload()
    chart_before_after()
    chart_parameter_study()
