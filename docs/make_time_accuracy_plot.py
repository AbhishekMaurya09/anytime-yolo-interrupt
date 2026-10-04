"""Time vs accuracy for the four anytime heads.

Run from the workspace root:
    .venv/Scripts/python docs/make_time_accuracy_plot.py
Writes docs/time_vs_accuracy.{png,svg,pdf}.

Timing is read from runs/anytime_interrupt/latency_train6.json (scripts/benchmark_latency.py,
sequential mode). Accuracy was measured on the same 150 bdd1500 test images with the
train6_bdd70k checkpoint on 2026-10-04 (confidence >= 0.001).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

HEADS = ["Head 1", "Head 2", "Head 3", "Final"]
KEYS = ["early_head_1", "early_head_2", "early_head_3", "final"]
MAP50 = [0.250, 0.322, 0.347, 0.365]
MAP50_95 = [0.120, 0.161, 0.187, 0.199]

with open(os.path.join(ROOT, "runs", "anytime_interrupt", "latency_train6.json")) as f:
    timing = json.load(f)["modes"]["sequential"]
T_MED = [timing[k]["median_ms"] for k in KEYS]
T_P90 = [timing[k]["p90_ms"] for k in KEYS]

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(10, 6.2))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# before Head 1 nothing has been published
ax.axvspan(0, T_MED[0], color=MUTED, alpha=0.07, lw=0)
ax.text(T_MED[0] / 2, 0.215, "no answer\navailable yet", ha="center", va="center", fontsize=10.5, color=MUTED)

for y in (0.1, 0.2, 0.3, 0.4):
    ax.axhline(y, color=GRID, lw=1, zorder=1)
for x in T_MED:
    ax.plot([x, x], [0, 0.43], color=GRID, lw=1, zorder=1)

# 90% of frames have the answer by the end of the whisker
for x, x90, y in zip(T_MED, T_P90, MAP50):
    ax.plot([x, x90], [y, y], color=MUTED, lw=1.2, zorder=2, solid_capstyle="butt")
    ax.plot([x90, x90], [y - 0.006, y + 0.006], color=MUTED, lw=1.2, zorder=2)

for ys, color in ((MAP50, BLUE), (MAP50_95, ORANGE)):
    ax.plot(T_MED, ys, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.scatter(T_MED, ys, s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)

for name, x, y in zip(HEADS, T_MED, MAP50):
    ax.text(x, y + 0.040, name, ha="center", va="bottom", fontsize=11.5, color=INK, weight="semibold")
    ax.text(x, y + 0.016, f"{y:.3f}", ha="center", va="bottom", fontsize=10.5, color=INK2)
for i in (0, 3):  # second series: endpoints only
    ax.text(T_MED[i], MAP50_95[i] - 0.018, f"{MAP50_95[i]:.3f}", ha="center", va="top", fontsize=10.5, color=INK2)

ax.set_xlim(0, 54)
ax.set_ylim(0, 0.45)
ax.set_xticks([0] + T_MED)
ax.set_xticklabels(["0"] + [f"{t:.1f}" for t in T_MED], fontsize=10.5, color=INK2)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
ax.set_yticklabels(["0", "0.1", "0.2", "0.3", "0.4"], fontsize=10.5, color=INK2)
ax.tick_params(length=0, pad=6)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.set_xlabel("Time from image in until the head's answer is ready (ms, median)", fontsize=11, color=INK2, labelpad=10)
ax.set_ylabel("Accuracy (mAP)", fontsize=11, color=INK2, labelpad=10)

fig.text(0.075, 0.955, "Each later head is more accurate, but its answer arrives later",
         fontsize=15, weight="semibold", color=INK, va="center")
fig.text(0.075, 0.910, "train6 model (65,000 training images) · GTX 1650 · 150 unseen BDD100K test images · 640 × 640 input",
         fontsize=10.5, color=INK2, va="center")

handles = [
    Line2D([0], [0], color=BLUE, lw=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
           label="mAP50 (box overlap ≥ 50%)"),
    Line2D([0], [0], color=ORANGE, lw=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
           label="mAP50-95 (stricter overlap)"),
    Line2D([0], [0], color=MUTED, lw=1.2, label="whisker: 90% of frames ready by here"),
]
leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.068, 0.885), ncol=3, frameon=False,
                 fontsize=10.5, handlelength=2.2, columnspacing=1.8)
for text in leg.get_texts():
    text.set_color(INK2)

fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.115)
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(HERE, f"time_vs_accuracy.{ext}"), dpi=200, facecolor=SURFACE)
print("saved to", HERE)
