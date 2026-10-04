"""Draws one anytime detection head (models/detect_head.py, models/postprocess.py, models/loss.py).

Run from the workspace root:
    .venv/Scripts/python docs/make_head_diagram.py
Writes docs/head_architecture.{png,svg,pdf}.

Parameter/MAC counts and latencies were measured on 2026-09-28 (GTX 1650, 640x640,
train2 checkpoint, median over 20 test images).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

NAVY = "#14213D"
TEXT = "#3D4757"
MUTED = "#5E6673"
STAGE = "#F3EFE6"
EDGE = "#B9B3A6"
ORANGE = "#D9731A"
DARK_ORANGE = "#B85C0E"
GREY = "#8A8F99"
BLUE = "#2F6DB5"
CLS = "#DCE7F5"
REG = "#B7CDEB"
BG = "#FBFAF7"

fig, ax = plt.subplots(figsize=(22, 19))
ax.set_xlim(0, 40)
ax.set_ylim(-3.8, 31)
ax.axis("off")
fig.patch.set_facecolor(BG)


def box(x, y, w, h, title, sub="", fill=STAGE, edge=EDGE, tcolor=NAVY, fs=11):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc=fill, ec=edge, lw=1.5, zorder=3))
    if sub:
        ax.text(x, y + 0.2, title, ha="center", va="center", fontsize=fs, weight="bold", color=tcolor, zorder=4)
        ax.text(x, y - 0.3, sub, ha="center", va="center", fontsize=fs - 1.5, color=tcolor, zorder=4,
                family="monospace")
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=fs, weight="bold", color=tcolor, zorder=4)


def arrow(x1, y1, x2, y2, color=ORANGE, lw=1.8, label="", lx=0.0, ly=0.0, fs=9.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=color,
                                 lw=lw, zorder=2))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center", va="center", fontsize=fs,
                color=MUTED, zorder=5, bbox=dict(fc=BG, ec="none", pad=0.8))


def card(x0, y0, x1, y1, title, lines, fill="#FFFFFF", edge=EDGE, mono=False, fs=10.5, title_color=NAVY):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.3",
                                fc=fill, ec=edge, lw=1.5, zorder=3))
    ax.text(x0 + 0.35, y1 - 0.5, title, fontsize=12.5, weight="bold", color=title_color, va="center", zorder=4)
    for i, line in enumerate(lines):
        ax.text(x0 + 0.35, y1 - 1.2 - i * 0.6, line, fontsize=fs, color=TEXT, va="center", zorder=4,
                family="monospace" if mono else None)


# ---------------------------------------------------------------- title
ax.text(0.4, 30.4, "Anytime detection head: one of four (early_head_1 / 2 / 3 / final)", fontsize=20,
        weight="bold", color=NAVY, va="center")
ax.text(0.4, 29.6, "All four heads share this design but each has its own weights. Anchor-free, decoupled "
        "(separate class and box branches). Conv = 3×3 conv + BatchNorm + SiLU.", fontsize=12.5, color=MUTED,
        va="center")

# ---------------------------------------------------------------- per-scale branches
ax.text(0.4, 28.5, "1 · Three ScaleHeads, one per feature map", fontsize=15, weight="bold", color=NAVY,
        va="center")

rows = [
    (25.6, "P3", 64, 80, "148.9 K"),
    (20.9, "P4", 96, 40, "334.0 K"),
    (16.2, "P5", 128, 20, "592.8 K"),
]
for y, name, c, s, params in rows:
    box(2.9, y, 4.6, 1.3, f"{name} feature map", f"{c} × {s} × {s}", fill="#E9EEF5", edge="#B7CDEB")
    box(9.8, y + 1.15, 5.8, 1.1, "cls stem: Conv ×2", f"{c}→{c}→{c} · {s}×{s}")
    box(9.8, y - 1.15, 5.8, 1.1, "reg stem: Conv ×2", f"{c}→{c}→{c} · {s}×{s}")
    arrow(5.2, y + 0.3, 6.9, y + 1.0)
    arrow(5.2, y - 0.3, 6.9, y - 1.0)
    box(17.3, y + 1.15, 5.2, 0.95, "1×1 → class logits", f"10 × {s} × {s}", fill=CLS, fs=10)
    box(17.3, y - 0.55, 5.2, 0.95, "1×1 → box l,t,r,b", f"4 × {s} × {s}", fill=REG, fs=10)
    box(17.3, y - 1.65, 5.2, 0.95, "1×1 → objectness", f"1 × {s} × {s}", fill=REG, fs=10)
    arrow(12.7, y + 1.15, 14.7, y + 1.15)
    arrow(12.7, y - 1.05, 14.7, y - 0.6)
    arrow(12.7, y - 1.25, 14.7, y - 1.6)
    ax.text(21.3, y + 1.5, params, fontsize=12, weight="bold", color=DARK_ORANGE, va="center")
    ax.text(21.3, y + 0.95, "params", fontsize=10, color=MUTED, va="center")

ax.text(0.4, 13.6, "Head total: 1.08 M params (≈99% in the 3×3 stems) · 1.72 GMACs per image · "
        "the 4 heads = 77% of the model's compute",
        fontsize=11, color=TEXT, va="center")
ax.text(0.4, 13.0, "Bias of the objectness and class 1×1 convs starts at −4.6 → every cell begins at ≈1% "
        "confidence (avoids an early loss spike)", fontsize=11, color=TEXT, va="center")

# ---------------------------------------------------------------- decode + postprocess
ax.text(24.2, 28.5, "2 · From raw outputs to a result", fontsize=15, weight="bold", color=NAVY, va="center")
card(24.2, 23.3, 39.6, 27.8, "Decode every cell (all three scales)", [
    "cell centre   cx, cy = (i + 0.5) · stride",
    "distances     l,t,r,b = exp(clamp(raw)) · stride",
    "box           (cx−l, cy−t, cx+r, cy+b)",
    "score         σ(obj) × max σ(class) ; class = argmax",
], mono=True, fs=10)
card(24.2, 18.9, 39.6, 22.6, "Gather 8,400 candidates", [
    "6,400 (P3) + 1,600 (P4) + 400 (P5)",
    "keep score ≥ 0.3   (≥ 0.001 when measuring mAP)",
    "cap at the top 1,000 before NMS",
], mono=True, fs=10)
card(24.2, 15.4, 39.6, 18.2, "Per-class greedy NMS", [
    "overlap IoU > 0.5 with a better box → dropped",
    "keep at most 100 boxes",
], mono=True, fs=10)
card(24.2, 11.9, 39.6, 14.7, "Map back and publish", [
    "undo letterbox → original 1280 × 720 pixels",
    "frozen result → ResultManager (the mailbox)",
], mono=True, fs=10, fill="#E9EEF5", edge="#B7CDEB")
for y_top, y_bot in [(23.3, 22.6), (18.9, 18.2), (15.4, 14.7)]:
    arrow(31.9, y_top, 31.9, y_bot, lw=2)
arrow(22.4, 20.9, 24.1, 20.9, color=BLUE, lw=2.2)

# ---------------------------------------------------------------- training
ax.plot([0.4, 39.6], [11.2, 11.2], color=EDGE, lw=1)
card(0.4, 5.2, 19.6, 10.6, "3 · How each head is trained", [
    "object → one scale by size: < 64 px P3 · < 128 px P4 · else P5",
    "positives: centre cell + neighbours inside the box (≤ 9)",
    "objectness  BCE over all cells, summed ÷ #positives",
    "class       BCE on positive cells (one-hot, 10 classes)",
    "box         CIoU × 5 on positive cells",
    "head weight 0.3 · 0.5 · 0.7 · 1.0  (Head 1 → Final)",
], fs=10.5)
card(20.4, 5.2, 39.6, 10.6, "4 · What one head costs", [
    "parameters     1,075,725  (per scale 149 K · 334 K · 593 K)",
    "compute        1.72 GMACs per 640 × 640 image",
    "GPU time       ≈ 3–6 ms (includes the next neck stage)",
    "post-process   ≈ 2.9 ms  (was ≈ 13.6 ms with loop NMS)",
    "fix            decode on GPU, 1 copy to CPU, NMS in numpy",
], mono=True, fs=10)

# ---------------------------------------------------------------- timeline
ax.text(0.4, 4.35, "5 · Measured timeline of one frame, before and after the post-processing fix "
        "(GTX 1650, trained model, median over test images)", fontsize=13, weight="bold", color=NAVY, va="center")
POST = "#F4D9BD"
before = [
    ("letterbox +\nbackbone", 11.6, "#D5D0C4", NAVY),
    ("H1 + FPN", 5.2, CLS, NAVY), ("post", 13.4, POST, NAVY),
    ("H2 + PAN", 5.9, REG, NAVY), ("post", 13.8, POST, NAVY),
    ("H3 + refine", 5.3, "#7FA6DA", NAVY), ("post", 14.1, POST, NAVY),
    ("Final", 2.8, BLUE, "#F6F4EF"), ("post", 13.6, POST, NAVY),
]
after = [
    ("letterbox +\nbackbone", 11.8, "#D5D0C4", NAVY),
    ("H1 + FPN", 5.3, CLS, NAVY), ("post", 2.9, POST, NAVY),
    ("H2 + PAN", 6.0, REG, NAVY), ("post", 2.8, POST, NAVY),
    ("H3 + refine", 5.1, "#7FA6DA", NAVY), ("post", 2.9, POST, NAVY),
    ("Final", 3.0, BLUE, "#F6F4EF"), ("post", 3.0, POST, NAVY),
]
SCALE_MS = sum(s[1] for s in before)  # both bars share one ms-per-unit scale
x0, width, hb = 0.4, 39.2, 1.1


def timeline(segments, yb, caption):
    ax.text(x0, yb + hb + 0.4, caption, fontsize=11.5, weight="bold", color=TEXT, va="center")
    x = x0
    marks = []
    for label, ms, fill, fg in segments:
        w = width * ms / SCALE_MS
        ax.add_patch(Rectangle((x, yb), w, hb, fc=fill, ec=BG, lw=2, zorder=3))
        ax.text(x + w / 2, yb + hb / 2, f"{label}\n{ms:.1f}" if w > 1.6 else f"{ms:.1f}", ha="center",
                va="center", fontsize=8.5, color=fg, zorder=4)
        x += w
        if label == "post":
            marks.append(x)
    for m, name in zip(marks, ["Head 1", "Head 2", "Head 3", "Final"]):
        ax.plot([m, m], [yb - 0.05, yb - 0.35], color=NAVY, lw=1.5)
        ms_here = (m - x0) / width * SCALE_MS
        ax.text(m, yb - 0.7, f"{name} {ms_here:.0f} ms", ha="center", fontsize=10, color=NAVY, weight="bold")


timeline(before, 2.2, "Before: per-box NMS loop on the GPU (≈ 13.6 ms per head)")
timeline(after, -1.9, "After: vectorised decode + one GPU→CPU copy + numpy NMS (≈ 2.9 ms per head)")
ax.text(0.4, -3.35, "ms per step · each head step also computes the next neck stage · 'post' = decode + filter + "
        "NMS + map back · identical detections before and after (320 cases, 17,990 boxes checked)",
        fontsize=10, color=MUTED, va="center")

out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(out_dir, f"head_architecture.{ext}"), dpi=170, bbox_inches="tight", facecolor=BG)
print("saved to", out_dir)
