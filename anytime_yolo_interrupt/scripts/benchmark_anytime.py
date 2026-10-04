#!/usr/bin/env python3
"""
benchmark_anytime.py
-----------------------
Runs the core research experiment (brief section 26): a sweep of
interrupt times, recording which head's result is returned, its
latency, and its detection count. Prints a table and saves Plot 3
(interrupt time vs. latest available head).

    python benchmark_anytime.py --interrupt-times 5 10 15 20 25 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anytime_yolo_interrupt.inference import AnytimeDetector, MockAnytimeAdapter, ResultManager
from anytime_yolo_interrupt.evaluation import benchmark_interrupt_times, trials_to_table
from anytime_yolo_interrupt.visualization import plot_interrupt_vs_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interrupt-times", type=float, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--frame-id", type=int, default=100)
    parser.add_argument("--log-dir", type=str, default="runs/anytime_interrupt")
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    rm = ResultManager(log_dir=args.log_dir)
    detector = AnytimeDetector(MockAnytimeAdapter(), rm)
    frame = {"width": 1280, "height": 720}

    trials = benchmark_interrupt_times(detector, rm, frame, args.frame_id, args.interrupt_times)
    print(trials_to_table(trials))

    out_json = os.path.join(args.log_dir, "interrupt_benchmark.json")
    with open(out_json, "w") as f:
        json.dump([t.__dict__ for t in trials], f, indent=2)
    print(f"\nSaved: {out_json}")

    out_png = os.path.join(args.log_dir, "plot3_interrupt_vs_head.png")
    plot_interrupt_vs_head(
        [t.interrupt_time_ms for t in trials],
        [t.latest_head_name for t in trials],
        save_path=out_png,
    )
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
