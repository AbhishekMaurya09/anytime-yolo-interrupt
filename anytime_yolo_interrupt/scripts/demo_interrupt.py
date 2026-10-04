#!/usr/bin/env python3
"""
demo_interrupt.py
-------------------
Complete demonstration (brief section 35 / 36):
  1. Load model (mock adapter here; swap for the real GelanTAnytimeAdapter).
  2. Load one test image.
  3-7. Run one forward pass, generating Head1/2/3/Final results.
  8. Simulate an external interrupt at a configurable time.
  9-10. Fetch and print the latest available result.
  11. Save a visualization.

Usage:
    python demo_interrupt.py --interrupt-after 18
    python demo_interrupt.py --mode sensor --sensor-value 4.2 --threshold 5.0
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from anytime_yolo_interrupt.inference import (
    AnytimeDetector, MockAnytimeAdapter, ResultManager,
    InterruptManager, SensorInterface,
)
from anytime_yolo_interrupt.visualization import visualize_progression


def build_pipeline(log_dir=None):
    result_manager = ResultManager(log_dir=log_dir)
    adapter = MockAnytimeAdapter()
    detector = AnytimeDetector(adapter, result_manager)
    interrupt_manager = InterruptManager(result_manager)
    return detector, result_manager, interrupt_manager


def run_interrupt_demo(interrupt_after_ms: float, frame_id: int = 100, log_dir: str = None):
    detector, result_manager, interrupt_manager = build_pipeline(log_dir)
    frame = {"width": 1280, "height": 720}

    print("=" * 50)
    print("INTERRUPT-DRIVEN ANYTIME YOLO")
    print("=" * 50)
    print(f"\nFrame ID: {frame_id}\n")

    all_results = []

    def inference_worker():
        results = detector.process_frame(frame, frame_id)
        for r in results:
            all_results.append(r)
            tag = "Final" if r.is_final else r.head_name.replace("_", " ").title()
            print(f"[{tag}]")
            print(f"Result available at: {r.elapsed_ms:.2f} ms")
            print(f"Detections: {r.num_detections()}")
            print("Latest result updated.\n")

    infer_thread = threading.Thread(target=inference_worker)
    infer_thread.start()

    time.sleep(interrupt_after_ms / 1000.0)
    interrupt_manager.trigger_interrupt()
    interrupt_time = interrupt_after_ms

    # Read the latest result RIGHT NOW -- this is the actual interrupt
    # response. We only join() afterwards so the demo's console output
    # from any still-running heads finishes printing before we exit;
    # joining here does not change what get_latest_result() returned.
    response = interrupt_manager.get_latest_result()

    print("-" * 50)
    print(f"External interrupt received at: {interrupt_time:.2f} ms")
    print("-" * 50)

    if response.has_result:
        r = response.result
        print("\nLatest completed result:")
        print(f"    Frame ID: {r.frame_id}")
        print(f"    Head: {r.head_name}")
        print(f"    Available at: {r.elapsed_ms:.2f} ms")
        print(f"    Number of objects: {r.num_detections()}")
        print(f"\n{response.message}")
        print("-" * 50)
    else:
        print(f"\n{response.message}")
        print("-" * 50)

    infer_thread.join()  # let the (still-running-in-background) forward pass finish

    if log_dir:
        image = np.zeros((frame["height"], frame["width"], 3), dtype=np.uint8) + 40
        out_path = os.path.join(log_dir, f"frame_{frame_id}_progression.png")
        visualize_progression(image, all_results, save_path=out_path)
        print(f"\nVisualization saved to: {out_path}")

    return response


def run_sensor_demo(sensor_value: float, threshold: float, frame_id: int = 100, log_dir: str = None):
    detector, result_manager, interrupt_manager = build_pipeline(log_dir)
    sensor = SensorInterface(interrupt_manager, threshold=threshold)
    frame = {"width": 1280, "height": 720}

    print(f"Sensor value: {sensor_value}")
    print(f"Threshold: {threshold}")

    def inference_worker():
        detector.process_frame(frame, frame_id)

    infer_thread = threading.Thread(target=inference_worker)
    infer_thread.start()
    time.sleep(0.010)  # let some heads complete before the sensor reading arrives

    event = sensor.sensor_callback(sensor_value)
    print(f"Interrupt: {event.triggered}")

    response = interrupt_manager.get_latest_result() if event.triggered else None  # read now, before join()
    infer_thread.join()

    if event.triggered:
        print("\nFetching latest YOLO result...")
        if response.has_result:
            print(f"\nLatest head: {response.result.head_name}")
        else:
            print(f"\n{response.message}")
    return sensor.last_event


def main():
    parser = argparse.ArgumentParser(description="Interrupt-driven anytime YOLO demo")
    parser.add_argument("--mode", choices=["interrupt", "sensor"], default="interrupt")
    parser.add_argument("--interrupt-after", type=float, default=18.0, help="ms")
    parser.add_argument("--sensor-value", type=float, default=4.2)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--frame-id", type=int, default=100)
    parser.add_argument("--log-dir", type=str, default="runs/anytime_interrupt")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    if args.mode == "interrupt":
        run_interrupt_demo(args.interrupt_after, frame_id=args.frame_id, log_dir=args.log_dir)
    else:
        run_sensor_demo(args.sensor_value, args.threshold, frame_id=args.frame_id, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
