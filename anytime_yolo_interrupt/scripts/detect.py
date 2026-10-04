#!/usr/bin/env python3
"""
detect.py
----------
Unified entry point for the four inference modes required by the brief
(section 23):

    python detect.py --mode normal
    python detect.py --mode progressive
    python detect.py --mode interrupt --interrupt-after-ms 18
    python detect.py --mode sensor --sensor-value 4.2 --threshold 5.0
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anytime_yolo_interrupt.inference import (
    AnytimeDetector, MockAnytimeAdapter, ResultManager, InterruptManager, SensorInterface,
)


def make_detector(log_dir):
    result_manager = ResultManager(log_dir=log_dir)
    detector = AnytimeDetector(MockAnytimeAdapter(), result_manager)
    return detector, result_manager


def mode_normal(args):
    detector, rm = make_detector(args.log_dir)
    frame = {"width": 1280, "height": 720}
    results = detector.process_frame(frame, args.frame_id)
    final = next(r for r in results if r.is_final)
    print(f"Final detection: {final.num_detections()} objects at {final.elapsed_ms:.2f} ms")


def mode_progressive(args):
    detector, rm = make_detector(args.log_dir)
    frame = {"width": 1280, "height": 720}
    results = detector.process_frame(frame, args.frame_id)
    for r in results:
        print(r)


def mode_interrupt(args):
    detector, rm = make_detector(args.log_dir)
    interrupt_manager = InterruptManager(rm)
    frame = {"width": 1280, "height": 720}

    t = threading.Thread(target=lambda: detector.process_frame(frame, args.frame_id))
    t.start()
    time.sleep(args.interrupt_after_ms / 1000.0)
    interrupt_manager.trigger_interrupt()
    response = interrupt_manager.get_latest_result()  # read now, not after join()
    t.join()

    print("Interrupt occurred")
    if response.has_result:
        r = response.result
        print("Latest completed result:")
        print(f"    Frame: {r.frame_id}")
        print(f"    Head: {r.head_name}")
        print(f"    Latency: {r.elapsed_ms:.2f} ms")
        print(f"    Number of detections: {r.num_detections()}")
    else:
        print(response.message)


def mode_sensor(args):
    detector, rm = make_detector(args.log_dir)
    interrupt_manager = InterruptManager(rm)
    sensor = SensorInterface(interrupt_manager, threshold=args.threshold)
    frame = {"width": 1280, "height": 720}

    t = threading.Thread(target=lambda: detector.process_frame(frame, args.frame_id))
    t.start()
    time.sleep(0.010)
    event = sensor.sensor_callback(args.sensor_value)
    response = interrupt_manager.get_latest_result() if event.triggered else None  # read now
    t.join()

    print(f"Sensor value: {args.sensor_value}")
    print(f"Threshold: {args.threshold}")
    print(f"Interrupt: {event.triggered}")
    if event.triggered:
        print("\nFetching latest YOLO result...")
        if response.has_result:
            print(f"\nLatest head: {response.result.head_name}")
        else:
            print(f"\n{response.message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["normal", "progressive", "interrupt", "sensor"], required=True)
    parser.add_argument("--interrupt-after-ms", type=float, default=18.0)
    parser.add_argument("--sensor-value", type=float, default=4.2)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--frame-id", type=int, default=100)
    parser.add_argument("--log-dir", type=str, default="runs/anytime_interrupt")
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    {
        "normal": mode_normal,
        "progressive": mode_progressive,
        "interrupt": mode_interrupt,
        "sensor": mode_sensor,
    }[args.mode](args)


if __name__ == "__main__":
    main()
