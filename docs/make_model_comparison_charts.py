"""Draws the charts comparing all five detectors into docs/model_comparison/.

Run from the workspace root (after scripts/compare_all_models.py has written its results.json):
    .venv/Scripts/python docs/make_model_comparison_charts.py

Data:
    runs/model_comparison/results.json          accuracy, pipeline, network-only time, FPS, size (one run)
    runs/anytime_interrupt/quantization/results.json, runs/anytime_v8/study_v8/results.json,
    runs/anytime_v8/study_fast/results.json      FP32 and INT8 accuracy of the three anytime models
    docs/weight_distribution, docs/anytime_v8/weight_distribution,
    docs/anytime_v8_fast/weight_distribution     layer_weight_stats.csv of the three anytime models
"""
import csv
import json
import os

import matplotlib.pyplot as plt
from make_result_charts import (AQUA, AXIS, BLUE, GREY_BAR, HEAD_RAMP, INK, INK2, MUTED, ORANGE, SURFACE, label,
                                legend, line_key, new_fig, offsets, save, style, titles, vbar, hbar)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "model_comparison")
with open(os.path.join(ROOT, "runs", "model_comparison", "results.json")) as f:
    R = json.load(f)

YELLOW, PINK = "#eda100", "#e87ba4"   # categorical slots 4-5 of the validated palette
MODELS = ["train6", "anytime_v8", "anytime_v8_fast", "yolov8n", "yolov8m"]
NAME = {"train6": "train6 (GELAN, 4 heads)", "anytime_v8": "anytime_v8", "anytime_v8_fast": "anytime_v8_fast",
        "yolov8n": "YOLOv8n", "yolov8m": "YOLOv8m"}
SHORT = {"train6": "train6", "anytime_v8": "anytime_v8", "anytime_v8_fast": "v8_fast", "yolov8n": "YOLOv8n",
         "yolov8m": "YOLOv8m"}
COLOR = {"train6": BLUE, "anytime_v8": ORANGE, "anytime_v8_fast": AQUA, "yolov8n": YELLOW, "yolov8m": PINK}
HEAD_NAME = {"early_head_1": "Head 1", "early_head_2": "Head 2", "early_head_3": "Head 3", "final": "Final"}
HEAD_SHADE = {"early_head_1": HEAD_RAMP[0], "early_head_2": HEAD_RAMP[1], "early_head_3": HEAD_RAMP[2],
              "final": HEAD_RAMP[3]}
ANYTIME = ["train6", "anytime_v8", "anytime_v8_fast"]
N_IMG = R["n_images"]
TRAIN_NOTE = "train6 was trained on 65,000 images, the other four on the same 1,200"


def acc(m, k, key="mAP50"):
    return R["accuracy"][m][k][key]


# --------------------------------------------------------------------------- 1. time vs accuracy
def chart_time_vs_accuracy():
    fig, ax = new_fig(h=5.8, right=0.95, top=0.74)
    style(ax)
    ax.grid(axis="x", color="#e1e0d9", lw=1)
    ax.set_xlim(0, 42)
    ax.set_ylim(0, 0.48)
    ax.set_xlabel("Time from image to boxes (ms, median)", fontsize=10.5, color=INK2, labelpad=8)
    ax.set_ylabel("mAP50", fontsize=10.5, color=INK2, labelpad=8)
    for m in MODELS:
        ks = R["heads"][m]
        ts = [R["pipeline"][m][k]["median_ms"] for k in ks]
        as_ = [acc(m, k) for k in ks]
        if len(ks) > 1:
            ax.plot(ts, as_, color=COLOR[m], lw=2, zorder=3)
        ax.scatter(ts, as_, s=60 if len(ks) > 1 else 85, marker="o" if len(ks) > 1 else "D", color=COLOR[m],
                   edgecolor=SURFACE, linewidth=2, zorder=4)
        t, a = ts[-1], as_[-1]
        if m == "anytime_v8_fast":
            continue   # its Final answer sits on YOLOv8n's point; the YOLOv8n label names both
        text = f"{t:.1f} ms · {a:.3f}"
        if m == "yolov8n":
            fast_t = R["pipeline"]["anytime_v8_fast"]["final"]["median_ms"]
            text = f"YOLOv8n {t:.2f} ms\nanytime_v8_fast Final {fast_t:.2f} ms\nboth {a:.3f}"
        place = {"train6": (0.1, 0, "left", "center"), "anytime_v8": (0.1, -2, "left", "top"),
                 "yolov8n": (-0.14, 2, "right", "bottom"), "yolov8m": (0.1, 0, "left", "center")}[m]
        label(ax, t, a, text, dx_in=place[0], dy_pt=place[1], ha=place[2], va=place[3], size=8.5, color=INK)
    legend(fig, [line_key(COLOR[m], NAME[m]) if len(R["heads"][m]) > 1 else
                 Line2D([0], [0], color="none", marker="D", markersize=7, markerfacecolor=COLOR[m],
                        markeredgecolor=SURFACE, label=NAME[m]) for m in MODELS], y=0.835, ncol=3)
    ym_t, ym_a = R["pipeline"]["yolov8m"]["final"]["median_ms"], acc("yolov8m", "final")
    titles(fig, f"YOLOv8m is the most accurate ({ym_a:.3f} at {ym_t:.1f} ms); anytime_v8_fast's Final = YOLOv8n",
           f"GTX 1650 · resize + network + NMS, boxes on the CPU · {N_IMG} test images · each point: when that "
           f"head's answer is ready\n{TRAIN_NOTE}; train6 runs at 640 × 640, the others at 640 × 384")
    save(fig, "time_vs_accuracy", OUT)


# --------------------------------------------------------------------------- 2-3. grouped bars per model
def grouped_by_head(name, value, ymax, yticks, ylabel, fmt, title, subtitle):
    fig, ax = new_fig(w=11, top=0.74)
    style(ax)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(0, ymax)
    ax.set_yticks(yticks)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([NAME[m] for m in MODELS], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10.5, color=INK2, labelpad=8)
    for i, m in enumerate(MODELS):
        ks = R["heads"][m]
        off = [(j - (len(ks) - 1) / 2) * 0.33 for j in range(len(ks))]   # wider pitch so labels clear
        for j, k in enumerate(ks):
            v = value(m, k)
            vbar(ax, i, v, HEAD_SHADE[k], off[j])
            label(ax, i, v, fmt.format(v), dx_in=off[j], size=8 if len(ks) == 4 else 8.5,
                  color=INK if k == "final" else INK2)
    legend(fig, [Patch(fc=HEAD_SHADE[k], label=HEAD_NAME[k]) for k in HEAD_SHADE], y=0.835, ncol=4)
    titles(fig, title, subtitle)
    save(fig, name, OUT)


def chart_accuracy():
    grouped_by_head("accuracy_by_head", lambda m, k: acc(m, k), 0.48, [0, 0.1, 0.2, 0.3, 0.4], "mAP50", "{:.3f}",
                    f"YOLOv8m's single answer ({acc('yolov8m', 'final'):.3f}) beats every head of the 1,200-image "
                    "models",
                    f"mAP50 of every head · {N_IMG} unseen test images · GPU, FP32\n{TRAIN_NOTE}")
    grouped_by_head("accuracy_by_head_map50_95", lambda m, k: acc(m, k, "mAP50_95"), 0.26,
                    [0, 0.05, 0.1, 0.15, 0.2, 0.25], "mAP50-95", "{:.3f}",
                    "The stricter score gives the same order",
                    f"mAP50-95 of every head · {N_IMG} unseen test images · GPU, FP32\n{TRAIN_NOTE}")


def chart_network_latency():
    nl = R["network_only_640"]
    grouped_by_head("network_latency", lambda m, k: nl[m][k]["median_ms"], 40, [0, 10, 20, 30, 40],
                    "Network time (ms, median)", "{:.1f}",
                    f"anytime_v8_fast answers in {nl['anytime_v8_fast']['final']['median_ms']:.1f} ms, the same "
                    f"as YOLOv8n ({nl['yolov8n']['final']['median_ms']:.1f} ms)",
                    "GTX 1650 · network only, the same 640 × 640 input for every model · median · up to the head "
                    "that answers\nanytime_v8_fast: an early head runs only when an interrupt asks for it")


# --------------------------------------------------------------------------- 4. frames per second
def hbars(ax, values, fmt, xmax):
    n = len(MODELS)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlim(0, xmax)
    ax.set_yticks(range(n))
    ax.set_yticklabels([NAME[m] for m in MODELS], fontsize=10.5)
    for i, m in enumerate(MODELS):
        hbar(ax, i, values[m], COLOR[m])
        label(ax, values[m], i, fmt.format(values[m]), dx_in=0.06, dy_pt=0, ha="left", va="center", size=9.5,
              color=INK)


def chart_fps():
    fps = {m: R["fps"][m]["median"] for m in MODELS}
    fig, ax = new_fig(h=4.6, left=0.22, right=0.93, top=0.77)
    style(ax, grid="x")
    hbars(ax, fps, "{:.1f}", 100)
    ax.set_xlabel("Frames per second (median of 3 repeats)", fontsize=10.5, color=INK2, labelpad=8)
    titles(fig, f"With no interrupt, anytime_v8_fast runs as fast as YOLOv8n ({fps['anytime_v8_fast']:.1f} "
                f"against {fps['yolov8n']:.1f} FPS)",
           f"GTX 1650 · frames back to back, image in to boxes · {N_IMG} test images\nanytime_v8 and train6 "
           "compute every head on every frame", x=0.03)
    save(fig, "fps", OUT)


# --------------------------------------------------------------------------- 5. size
def chart_size():
    params = {m: R["size"][m]["parameters"] / 1e6 for m in MODELS}
    macs = {m: R["size"][m]["macs_640_upto_head"][-1] / 1e9 for m in MODELS}
    fig, (a1, a2) = new_fig(w=10.5, h=4.8, ncols=2, left=0.19, right=0.96, top=0.74, wspace=0.18)
    for ax, vals, fmt, xmax, xl, head in ((a1, params, "{:.2f} M", 32, "Parameters (millions)", "Parameters"),
                                          (a2, macs, "{:.2f} G", 48, "Multiply-adds per frame (billions)",
                                           "Multiply-adds, 640 × 640, every head computed")):
        style(ax, grid="x")
        hbars(ax, vals, fmt, xmax)
        ax.set_xlabel(xl, fontsize=10.5, color=INK2, labelpad=8)
        ax.set_title(head, loc="left", fontsize=10.5, color=INK2, pad=10)
    titles(fig, f"YOLOv8m is {params['yolov8m'] / params['yolov8n']:.1f}× the size of YOLOv8n and costs "
                f"{macs['yolov8m'] / macs['yolov8n']:.1f}× the multiply-adds",
           "Counted on each model with hooks on every convolution · anytime_v8_fast with no interrupt costs "
           f"{R['size']['anytime_v8_fast']['macs_640_upto_head'][-1] / 1e9:.3f} G, the same as YOLOv8n",
           x=0.03)
    save(fig, "size_and_compute", OUT)


# --------------------------------------------------------------------------- 6. each class
def chart_per_class():
    classes = list(R["accuracy"]["yolov8m"]["final"]["per_class_ap50"])
    n = len(classes)
    fig, ax = new_fig(h=9.0, left=0.15, right=0.95, top=0.82, bottom=0.07)
    style(ax, grid="x")
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlim(0, 0.85)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks(range(n))
    ax.set_yticklabels(classes, fontsize=10.5)
    ax.set_xlabel("AP50 of the Final head", fontsize=10.5, color=INK2, labelpad=8)
    step = 0.165
    for i, c in enumerate(classes):
        for j, m in enumerate(MODELS):
            v = R["accuracy"][m]["final"]["per_class_ap50"][c]
            y = i + (j - 2) * step
            if v is None:
                if j == 2:
                    ax.text(0.01, i, "not in the test images", fontsize=9, color=MUTED, va="center")
                continue
            ax.barh(y, v, height=0.15, color=COLOR[m], edgecolor=SURFACE, linewidth=0.8, zorder=3)
            label(ax, v, y, f"{v:.2f}", dx_in=0.04, dy_pt=0, ha="left", va="center", size=7.5)
    legend(fig, [Patch(fc=COLOR[m], label=NAME[m]) for m in MODELS], x=0.143, y=0.862, ncol=5)
    best = [c for c in classes if R["accuracy"]["yolov8m"]["final"]["per_class_ap50"][c] is not None and
            all(R["accuracy"]["yolov8m"]["final"]["per_class_ap50"][c] >= R["accuracy"][m]["final"]["per_class_ap50"][c]
                for m in MODELS)]
    titles(fig, f"YOLOv8m is the most accurate for {len(best)} of the {n - 1} classes in the test images",
           f"AP50 of each model's Final head · {N_IMG} unseen test images · GPU, FP32\n{TRAIN_NOTE}", x=0.15)
    save(fig, "per_class_accuracy", OUT)


# --------------------------------------------------------------------------- 7. INT8 loss
def int8_results():
    files = {"train6": os.path.join(ROOT, "runs", "anytime_interrupt", "quantization", "results.json"),
             "anytime_v8": os.path.join(ROOT, "runs", "anytime_v8", "study_v8", "results.json"),
             "anytime_v8_fast": os.path.join(ROOT, "runs", "anytime_v8", "study_fast", "results.json")}
    out = {}
    for m, p in files.items():
        with open(p) as f:
            a = json.load(f)["accuracy"]
        out[m] = {k: (a["cpu_fp32"][k]["mAP50"], a["cpu_int8"][k]["mAP50"]) for k in a["cpu_fp32"]}
    return out


def chart_int8():
    q = int8_results()
    heads = ["early_head_1", "early_head_2", "early_head_3", "final"]
    fig, ax = new_fig(top=0.74)
    style(ax)
    ax.set_xlim(-0.6, len(heads) - 0.4)
    ax.set_ylim(0, 16)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels([HEAD_NAME[k] for k in heads], fontsize=10.5)
    ax.set_ylabel("mAP50 lost in INT8 (% of FP32)", fontsize=10.5, color=INK2, labelpad=8)
    drops = {}
    for i, k in enumerate(heads):
        ms = [m for m in ANYTIME if k in q[m]]
        off = [(j - (len(ms) - 1) / 2) * 0.33 for j in range(len(ms))]
        for j, m in enumerate(ms):
            f32, i8 = q[m][k]
            d = 100 * (1 - i8 / f32)
            drops.setdefault(m, []).append(d)
            vbar(ax, i, d, COLOR[m], off[j])
            label(ax, i, d, f"{d:.1f}%", dx_in=off[j], size=8.5)
    legend(fig, [Patch(fc=COLOR[m], label=NAME[m]) for m in ANYTIME], y=0.835, ncol=3)
    titles(fig, f"INT8 costs train6 {min(drops['train6']):.1f}–{max(drops['train6']):.1f}%, anytime_v8 "
                f"{min(drops['anytime_v8']):.1f}–{max(drops['anytime_v8']):.1f}% and anytime_v8_fast "
                f"{min(drops['anytime_v8_fast']):.1f}–{max(drops['anytime_v8_fast']):.1f}%",
           "ONNX Runtime on the CPU · FP32 against static INT8 (8-bit weights per channel, 8-bit activations), "
           "no retraining\n150 test images · 100 calibration images · each model measured in its own study run")
    save(fig, "int8_accuracy_loss", OUT)
    return q


# --------------------------------------------------------------------------- 8. weight ranges
WEIGHT_CSV = {"train6": os.path.join(HERE, "weight_distribution", "layer_weight_stats.csv"),
              "anytime_v8": os.path.join(HERE, "anytime_v8", "weight_distribution", "layer_weight_stats.csv"),
              "anytime_v8_fast": os.path.join(HERE, "anytime_v8_fast", "weight_distribution", "layer_weight_stats.csv")}


def weight_ranges():
    out = {}
    for m, p in WEIGHT_CSV.items():
        with open(p) as f:
            rows = [r for r in csv.DictReader(f) if "dfl" not in r["layer"]]
        out[m] = [(r["layer"], float(r["folded_abs_max"])) for r in rows]
    return out


def chart_weights():
    w = weight_ranges()
    fig, ax = new_fig(h=4.6, left=0.22, right=0.95, top=0.72)
    style(ax, grid="x")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 40)
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.set_ylim(len(ANYTIME) - 0.4, -0.6)
    ax.set_yticks(range(len(ANYTIME)))
    ax.set_yticklabels([NAME[m] for m in ANYTIME], fontsize=10.5)
    ax.set_xlabel("Largest folded weight of a layer (|w|, log scale)", fontsize=10.5, color=INK2, labelpad=8)
    spread = {}
    for i, m in enumerate(ANYTIME):
        vs = [v for _, v in w[m]]
        ax.scatter(vs, [i] * len(vs), s=26, color=COLOR[m], alpha=0.55, edgecolor="none", zorder=3)
        lo, hi = min(vs), max(vs)
        spread[m] = hi / lo
        ax.text(0.105, i - 0.22, f"{len(vs)} layers · {lo:.3f} to {hi:.2f} ({hi / lo:.0f}×)", fontsize=9, color=INK,
                ha="left", va="bottom")
    titles(fig, f"In every model the folded weight ranges differ by {min(spread.values()):.0f}× to "
                f"{max(spread.values()):.0f}× between layers",
           "One dot per convolution layer, BatchNorm folded in, fixed DFL layers left out\nA fixed-point "
           "accelerator needs at least one scale per layer", x=0.03)
    save(fig, "weight_ranges", OUT)
    return w


if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    chart_time_vs_accuracy()
    chart_accuracy()
    chart_network_latency()
    chart_fps()
    chart_size()
    chart_per_class()
    chart_int8()
    chart_weights()
