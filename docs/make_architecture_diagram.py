"""Draws the GelanAnytimeModel architecture (models/gelan_anytime.py).

Run from the workspace root:
    .venv/Scripts/python docs/make_architecture_diagram.py
Writes docs/model_architecture.{png,svg,pdf}.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY = "#14213D"
TEXT = "#3D4757"
MUTED = "#5E6673"
STAGE = "#F3EFE6"
EDGE = "#B9B3A6"
TAP = "#E9EEF5"
ORANGE = "#D9731A"
GREY = "#8A8F99"
HEADS = ["#DCE7F5", "#B7CDEB", "#7FA6DA", "#2F6DB5"]

fig, ax = plt.subplots(figsize=(22, 19))
ax.set_xlim(0, 40)
ax.set_ylim(0, 35)
ax.axis("off")
fig.patch.set_facecolor("#FBFAF7")


def box(x, y, w, h, title, sub="", fill=STAGE, edge=EDGE, tcolor=NAVY, dashed=False, fs=11.5):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.25",
        fc=fill, ec=edge, lw=1.6, ls="--" if dashed else "-", zorder=3))
    if sub:
        ax.text(x, y + 0.28, title, ha="center", va="center", fontsize=fs, weight="bold", color=tcolor, zorder=4)
        ax.text(x, y - 0.42, sub, ha="center", va="center", fontsize=fs - 1.5, color=tcolor, zorder=4,
                family="monospace")
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=fs, weight="bold", color=tcolor, zorder=4)


def arrow(x1, y1, x2, y2, color=ORANGE, label="", lx=0.0, ly=0.0, style="-|>", lw=2.0, rad=0.0, fs=9.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, zorder=2, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center", va="center", fontsize=fs,
                color=color if color != GREY else MUTED, zorder=5,
                bbox=dict(fc="#FBFAF7", ec="none", pad=1.0))


# ---------------------------------------------------------------- title
ax.text(0.4, 34.5, "GelanAnytimeModel: tiny GELAN-style detector with four anytime heads",
        fontsize=20, weight="bold", color=NAVY, va="center")
ax.text(0.4, 33.7, "Input 640 × 640 · 10 classes · 5.45 M parameters · shapes are channels × height × width "
        "(batch 1)", fontsize=12.5, color=MUTED, va="center")

# ---------------------------------------------------------------- columns
BW, BH = 6.0, 2.1
XS, XB, XF, XP, XR = 3.3, 10.6, 18.0, 25.4, 32.8        # stem, backbone, FPN, PAN, refine
Y3, Y4, Y5 = 27.2, 22.6, 18.0                           # P3, P4, P5 rows

for x, name in [(XS, "Stem"), (XB, "Backbone"), (XF, "FPN · top-down"),
                (XP, "PAN · bottom-up"), (XR, "Final refine")]:
    ax.text(x, 31.9, name, ha="center", fontsize=14, weight="bold", color=NAVY)
for x, p in [(XB, "0.46 M incl. stem"), (XF, "0.13 M"), (XP, "0.41 M"), (XR, "0.14 M")]:
    ax.text(x, 31.2, p, ha="center", fontsize=11, color=MUTED, family="monospace")

# row labels
for y, lab, note in [(Y3, "P3", "stride 8\nsmall objects"), (Y4, "P4", "stride 16\nmedium"),
                     (Y5, "P5", "stride 32\nlarge")]:
    ax.text(39.6, y + 0.3, lab, ha="right", fontsize=14, weight="bold", color=NAVY)
    ax.text(39.6, y - 0.6, note, ha="right", va="top", fontsize=9.5, color=MUTED)

# tap-group shading (what each head reads)
for x in (XB, XF, XP, XR):
    ax.add_patch(FancyBboxPatch((x - BW / 2 - 0.3, Y5 - BH / 2 - 0.35), BW + 0.6, (Y3 - Y5) + BH + 0.7,
                                boxstyle="round,pad=0,rounding_size=0.4", fc=TAP, ec="none", zorder=1))

# ---- stem
box(XS, 30.0, 5.2, 1.5, "Input image", "3 × 640 × 640", fill=NAVY, edge=NAVY, tcolor="#F6F4EF")
box(XS, 27.2, 5.2, 1.7, "Conv 3×3 s2", "16 × 320 × 320")
box(XS, 24.6, 5.2, 1.7, "Conv 3×3 s2", "32 × 160 × 160")
box(XS, 22.0, 5.2, 1.7, "ELAN (n=2)", "32 × 160 × 160")
arrow(XS, 29.25, XS, 28.05)
arrow(XS, 26.35, XS, 25.45)
arrow(XS, 23.75, XS, 22.85)
arrow(XS + 2.6, 22.3, XB - BW / 2, Y3 - 0.4, label="Down 3×3 s2", ly=0.55, lx=-0.4, rad=-0.25)

# ---- backbone
box(XB, Y3, BW, BH, "P3_bb · Down + ELAN", "64 × 80 × 80")
box(XB, Y4, BW, BH, "P4_bb · Down + ELAN", "96 × 40 × 40")
box(XB, Y5, BW, BH, "P5_bb · Down + ELAN + SPPF", "128 × 20 × 20", fs=10.5)
arrow(XB, Y3 - BH / 2, XB, Y4 + BH / 2, label="Down s2  64→96", lx=0.0)
arrow(XB, Y4 - BH / 2, XB, Y5 + BH / 2, label="Down s2  96→128", lx=0.0)

# ---- FPN (top-down)
box(XF, Y5, BW, BH, "P5_td = P5_bb", "128 × 20 × 20", dashed=True)
box(XF, Y4, BW, BH, "P4_td · ELAN", "96 × 40 × 40")
box(XF, Y3, BW, BH, "P3_td · ELAN", "64 × 80 × 80")
arrow(XF, Y5 + BH / 2, XF, Y4 - BH / 2, label="1×1 128→96, Up ×2", lx=0.0)
arrow(XF, Y4 + BH / 2, XF, Y3 - BH / 2, label="1×1 96→64, Up ×2", lx=0.0)
for y, t in [(Y3, "concat"), (Y4, "concat"), (Y5, "identity")]:
    arrow(XB + BW / 2, y, XF - BW / 2, y, color=GREY, label=t, ly=0.35)

# ---- PAN (bottom-up)
box(XP, Y3, BW, BH, "P3_bu · ELAN", "64 × 80 × 80")
box(XP, Y4, BW, BH, "P4_bu · ELAN", "96 × 40 × 40")
box(XP, Y5, BW, BH, "P5_bu · ELAN", "128 × 20 × 20")
arrow(XP, Y3 - BH / 2, XP, Y4 + BH / 2, label="Down s2  64→96", lx=0.0)
arrow(XP, Y4 - BH / 2, XP, Y5 + BH / 2, label="Down s2  96→128", lx=0.0)
for y, t in [(Y3, "self-refine"), (Y4, "concat"), (Y5, "concat")]:
    arrow(XF + BW / 2, y, XP - BW / 2, y, color=GREY, label=t, ly=0.35)

# ---- refine
box(XR, Y3, BW, BH, "P3_f · ELAN", "64 × 80 × 80")
box(XR, Y4, BW, BH, "P4_f · ELAN", "96 × 40 × 40")
box(XR, Y5, BW, BH, "P5_f · ELAN", "128 × 20 × 20")
for y in (Y3, Y4, Y5):
    arrow(XP + BW / 2, y, XR - BW / 2, y, color=GREY)

# ---- heads
HY, HH = 12.6, 3.3
names = [("Head 1", "early_head_1"), ("Head 2", "early_head_2"), ("Head 3", "early_head_3"), ("Final", "final")]
for i, (x, (h, n)) in enumerate(zip((XB, XF, XP, XR), names)):
    fg = "#F6F4EF" if i == 3 else NAVY
    ax.add_patch(FancyBboxPatch((x - BW / 2, HY - HH / 2), BW, HH, boxstyle="round,pad=0.02,rounding_size=0.3",
                                fc=HEADS[i], ec=HEADS[i] if i < 3 else NAVY, lw=1.6, zorder=3))
    ax.text(x, HY + 1.05, f"{h}", ha="center", fontsize=14, weight="bold", color=fg, zorder=4)
    ax.text(x, HY + 0.35, n, ha="center", fontsize=10, color=fg, family="monospace", zorder=4)
    ax.text(x, HY - 0.35, "3 ScaleHeads · 1.08 M", ha="center", fontsize=10, color=fg, zorder=4)
    ax.text(x, HY - 1.05, "8,400 × (10 + 1 + 4)", ha="center", fontsize=10, color=fg,
            family="monospace", zorder=4)
    arrow(x, Y5 - BH / 2 - 0.35, x, HY + HH / 2, color="#2F6DB5", label="reads P3 · P4 · P5", lw=2.2)

# saves + timeline
ax.text(XS, HY + 0.5, "each head's\nresult is saved\nto ResultManager\nthe moment it\nis ready", ha="center",
        va="center", fontsize=10.5, color=TEXT, style="italic")
arrow(XB - BW / 2, 9.9, XR + BW / 2, 9.9, color=NAVY, lw=1.6,
      label="compute order in one forward pass (no early exit) → answers improve over time", ly=-0.45, fs=11)

# ---------------------------------------------------------------- insets
ax.plot([0.4, 39.6], [8.7, 8.7], color=EDGE, lw=1)
ax.text(0.4, 8.2, "Inside the blocks", fontsize=15, weight="bold", color=NAVY, va="center")

SW, SH = 2.6, 0.95


def small(x, y, t, w=SW, fill=STAGE, fs=9.5):
    box(x, y, w, SH, t, fill=fill, fs=fs)


# ELAN
ax.text(0.6, 7.2, "ELAN block (c_in → c_out)", fontsize=12, weight="bold", color=NAVY)
small(1.9, 5.4, "input x")
small(5.2, 6.3, "1×1 conv → y1")
small(5.2, 4.5, "1×1 conv → y2")
small(8.5, 4.5, "3×3 conv ×n")
small(11.3, 5.4, "concat all", w=2.3, fill="#E4DDCF")
small(11.3, 3.3, "1×1 conv → out", w=2.3)
arrow(3.2, 5.6, 3.9, 6.2, lw=1.4)
arrow(3.2, 5.2, 3.9, 4.6, lw=1.4)
arrow(6.5, 4.5, 7.2, 4.5, lw=1.4)
arrow(6.5, 6.3, 10.15, 5.75, color=GREY, lw=1.2)
arrow(9.8, 4.7, 10.15, 5.05, color=GREY, lw=1.2)
arrow(11.3, 4.92, 11.3, 3.78, lw=1.4)
ax.text(6.9, 2.9, "concat = [y1, y2, every 3×3 output]\nn = 1 or 2 per block", ha="center", va="center",
        fontsize=9.5, color=MUTED)

# SPPF
ax.text(13.3, 7.2, "SPPF (end of backbone, 128 → 128)", fontsize=12, weight="bold", color=NAVY)
small(14.8, 5.4, "1×1 conv → 64")
small(18.1, 5.4, "MaxPool 5")
small(21.4, 5.4, "MaxPool 5")
small(24.7, 5.4, "MaxPool 5")
small(19.75, 3.0, "concat 4 × 64 = 256", w=4.4, fill="#E4DDCF")
small(19.75, 1.3, "1×1 conv → 128", w=4.4)
arrow(16.1, 5.4, 16.8, 5.4, lw=1.4)
arrow(19.4, 5.4, 20.1, 5.4, lw=1.4)
arrow(22.7, 5.4, 23.4, 5.4, lw=1.4)
for sx in (14.8, 18.1, 21.4, 24.7):
    arrow(sx, 4.92, 19.75 + (sx - 19.75) * 0.35, 3.48, color=GREY, lw=1.2)
arrow(19.75, 2.52, 19.75, 1.78, lw=1.4)

# ScaleHead
ax.text(27.4, 7.2, "ScaleHead (one per scale, per head)", fontsize=12, weight="bold", color=NAVY)
small(28.6, 4.3, "feature map", w=2.3)
small(32.2, 5.9, "cls stem: 3×3 ×2", w=3.3)
small(32.2, 2.7, "reg stem: 3×3 ×2", w=3.3)
small(37.3, 5.9, "1×1 → 10 class", w=3.6, fill=HEADS[0])
small(37.3, 3.4, "1×1 → 4 box l,t,r,b", w=3.6, fill=HEADS[1])
small(37.3, 1.9, "1×1 → 1 objectness", w=3.6, fill=HEADS[1])
arrow(29.75, 4.6, 30.5, 5.6, lw=1.4)
arrow(29.75, 4.0, 30.5, 3.0, lw=1.4)
arrow(33.85, 5.9, 35.5, 5.9, lw=1.4)
arrow(33.85, 2.9, 35.5, 3.3, lw=1.4)
arrow(33.85, 2.5, 35.5, 2.0, lw=1.4)
ax.text(33.6, 0.55, "score = σ(obj) × max σ(class) · box = cell centre ± exp(ltrb) × stride",
        ha="center", fontsize=9.5, color=MUTED, family="monospace")

out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(out_dir, f"model_architecture.{ext}"), dpi=170, bbox_inches="tight",
                facecolor=fig.get_facecolor())
print("saved to", out_dir)
