"""
interrupt_benchmark.py
------------------------
The central research experiment (brief section 26): for a range of
interrupt times, determine which head's result would actually be
returned, and record its latency/quality/detection-count.

This does NOT re-run the model once per interrupt time. It runs the
frame ONCE, records the real elapsed_ms at which each head's result
became available, and then -- for each requested interrupt time --
answers "which head had already completed by then?" by comparison
against those real, measured timestamps. That is the only correct way
to do this: interrupt time is compared against *measured* per-head
availability, never simulated after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..inference.anytime_detector import AnytimeDetector
from ..inference.detection_result import AnytimeDetectionResult
from ..inference.result_manager import ResultManager


@dataclass
class InterruptTrial:
    frame_id: int
    interrupt_time_ms: float
    latest_head_name: Optional[str]
    latest_head_latency_ms: Optional[float]
    num_detections: Optional[int]
    message: str


def benchmark_interrupt_times(
    detector: AnytimeDetector,
    result_manager: ResultManager,
    frame: dict,
    frame_id: int,
    interrupt_times_ms: Sequence[float],
) -> List[InterruptTrial]:
    # One real forward pass; every head's true elapsed_ms is recorded.
    results: List[AnytimeDetectionResult] = detector.process_frame(frame, frame_id)
    results_sorted = sorted(results, key=lambda r: r.elapsed_ms)

    trials = []
    for t in interrupt_times_ms:
        latest = None
        for r in results_sorted:
            if r.elapsed_ms <= t:
                latest = r
            else:
                break
        if latest is None:
            trials.append(InterruptTrial(
                frame_id=frame_id,
                interrupt_time_ms=t,
                latest_head_name=None,
                latest_head_latency_ms=None,
                num_detections=None,
                message="NO DETECTION AVAILABLE YET",
            ))
        else:
            trials.append(InterruptTrial(
                frame_id=frame_id,
                interrupt_time_ms=t,
                latest_head_name=latest.head_name,
                latest_head_latency_ms=latest.elapsed_ms,
                num_detections=latest.num_detections(),
                message=f"Returning {latest.head_name} detection immediately.",
            ))
    return trials


def trials_to_table(trials: List[InterruptTrial]) -> str:
    lines = [f"{'Interrupt (ms)':>15} | {'Latest head':>14} | {'Latency (ms)':>12} | {'#Objects':>8}"]
    lines.append("-" * len(lines[0]))
    for t in trials:
        head = t.latest_head_name or "-"
        lat = f"{t.latest_head_latency_ms:.2f}" if t.latest_head_latency_ms is not None else "-"
        n = str(t.num_detections) if t.num_detections is not None else "-"
        lines.append(f"{t.interrupt_time_ms:>15.1f} | {head:>14} | {lat:>12} | {n:>8}")
    return "\n".join(lines)
