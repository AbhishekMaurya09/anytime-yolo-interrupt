"""
visualize_anytime.py
----------------------
- visualize_detection(frame, result): draw one head's boxes over the
  image with class/confidence/head-name/frame-id/latency annotations
  (brief section 17).
- plot_quality_vs_time / plot_interrupt_vs_head: the required research
  plots (brief section 28, plots 1-3). Plot 4 (same image, all heads'
  boxes overlaid/side-by-side) is `visualize_progression`.

Uses matplotlib only (no cv2 dependency) so it runs in this sandbox and
anywhere numpy/matplotlib are available; swap in cv2 if your pipeline
already standardizes on it.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from ..inference.detection_result import AnytimeDetectionResult


def visualize_detection(image: np.ndarray, result: AnytimeDetectionResult, ax=None, save_path: Optional[str] = None):
    """image: HxWx3 array (RGB, 0-255 or 0-1)."""
    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=(8, 8 * image.shape[0] / max(image.shape[1], 1)))

    ax.imshow(image)
    ax.set_title(
        f"Frame: {result.frame_id}   Head: {result.head_name}   "
        f"Latency: {result.elapsed_ms:.2f} ms"
    )
    ax.axis("off")

    for box, score, cls in zip(result.boxes, result.scores, result.classes):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(y1 - 4, 0), f"{cls} {score:.2f}",
            color="white", fontsize=8,
            bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"),
        )

    if save_path and created_ax:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    return ax


def visualize_progression(image: np.ndarray, results: Sequence[AnytimeDetectionResult], save_path: Optional[str] = None):
    """Plot 4: the same image with Head1/Head2/Head3/Final boxes, side by side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        visualize_detection(image, result, ax=ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    return fig


def plot_quality_vs_time(head_names: Sequence[str], times_ms: Sequence[float], map_values: Sequence[float],
                          ylabel: str = "mAP50", save_path: Optional[str] = None):
    """Plot 1 / 2: accuracy (mAP50 or mAP50-95) vs time-to-result."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times_ms, map_values, "o-", color="crimson")
    for name, t, v in zip(head_names, times_ms, map_values):
        ax.annotate(name, (t, v), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Time-to-result (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. time-to-result")
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    return fig


def plot_interrupt_vs_head(interrupt_times_ms: Sequence[float], latest_heads: Sequence[str], save_path: Optional[str] = None):
    """Plot 3: interrupt time vs. which head's result would be returned."""
    unique_heads = list(dict.fromkeys(h for h in latest_heads if h is not None))
    head_to_y = {h: i for i, h in enumerate(unique_heads)}
    ys = [head_to_y.get(h, -1) for h in latest_heads]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(interrupt_times_ms, ys, where="post", color="navy")
    ax.set_yticks(range(len(unique_heads)))
    ax.set_yticklabels(unique_heads)
    ax.set_xlabel("Interrupt time (ms)")
    ax.set_ylabel("Latest available head")
    ax.set_title("Interrupt time vs. latest available head")
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    return fig
