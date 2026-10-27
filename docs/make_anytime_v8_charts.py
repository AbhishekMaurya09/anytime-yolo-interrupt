"""Draws the result charts of the anytime model on YOLOv8n (every head on every frame) into docs/anytime_v8/.

The same charts as docs/make_anytime_v8_fast_charts.py, pointed at the model trained with
    train_anytime_v8 --name bdd1500
whose early heads run on every frame. The before/after chart does not apply and is left out.

Run from the workspace root (after scripts/study_anytime_v8.py --eager has written study_v8/results.json):
    .venv/Scripts/python docs/make_anytime_v8_charts.py

Data: runs/anytime_v8/study_v8/results.json, runs/anytime_v8/evaluation.json (accuracy, pipeline and FPS
against YOLOv8n in one run), runs/anytime_v8/bdd1500/results.csv.
"""
import json
import os

import make_anytime_v8_fast_charts as F
from make_result_charts import BLUE, INK, INK2, SURFACE, label, legend, line_key, new_fig, save, style, titles
from matplotlib.lines import Line2D

with open(os.path.join(F.V8, "study_v8", "results.json")) as f:
    F.S = json.load(f)
with open(os.path.join(F.V8, "evaluation.json")) as f:
    F.E = json.load(f)
F.MODEL = "Anytime YOLOv8n"
F.LAZY = False
F.TRAIN_RUN = "bdd1500"
F.OUT = os.path.join(F.HERE, "anytime_v8")
F.RESULTS = os.path.join(F.OUT, "results")
F.STUDY = os.path.join(F.OUT, "parameter_study")
F.CLASSES = list(F.S["accuracy"]["gpu_fp32"]["final"]["per_class_ap50"])
E = F.E


def chart_time_vs_accuracy():
    ts = [E["pipeline"]["anytime_v8"][k]["median_ms"] for k in F.KEYS]
    as_ = [E["accuracy"]["anytime_v8"][k]["mAP50"] for k in F.KEYS]
    yt, ya = E["pipeline"]["yolov8n"]["final"]["median_ms"], E["accuracy"]["yolov8n"]["final"]["mAP50"]
    fig, ax = new_fig(right=0.95, top=0.78)
    style(ax)
    ax.grid(axis="x", color="#e1e0d9", lw=1)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 0.34)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel("Time from image to boxes (ms, median)", fontsize=10.5, color=INK2, labelpad=8)
    ax.set_ylabel("mAP50", fontsize=10.5, color=INK2, labelpad=8)
    ax.plot(ts, as_, color=BLUE, lw=2, zorder=3)
    ax.scatter(ts, as_, s=55, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4)
    for t, a, h in zip(ts, as_, F.HEADS):
        label(ax, t, a, f"{h}  {t:.1f} ms · {a:.3f}", dx_in=0.1, dy_pt=-3, ha="left", va="top", size=8.5)
    ax.scatter([yt], [ya], s=80, marker="D", color=INK2, edgecolor=SURFACE, linewidth=2, zorder=5)
    label(ax, yt, ya, f"YOLOv8n  {yt:.1f} ms · {ya:.3f}", dx_in=-0.1, dy_pt=6, ha="right", size=8.5, color=INK)
    legend(fig, [line_key(BLUE, "Anytime YOLOv8n: every head computed on every frame"),
                 Line2D([0], [0], color="none", marker="D", markersize=7, markerfacecolor=INK2,
                        markeredgecolor=SURFACE, label="YOLOv8n")], ncol=2)
    early = yt - ts[0]
    titles(fig, f"Head 1 answers {early:.1f} ms before YOLOv8n; after {yt:.1f} ms YOLOv8n is ahead",
           f"GTX 1650 · resize + network + NMS, boxes on the CPU · {E['n_images']} test images · "
           "each point: the time that head's answer is ready")
    save(fig, "time_vs_accuracy", F.RESULTS)


if __name__ == "__main__":
    F.chart_per_head()
    chart_time_vs_accuracy()
    F.chart_per_class()
    F.chart_learning()
    F.chart_formats()
    F.chart_workload()
    F.chart_parameter_study()
