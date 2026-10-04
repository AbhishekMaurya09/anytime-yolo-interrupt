"""
interrupt_manager.py
---------------------
Clean interrupt abstraction (brief section 9).

InterruptManager does NOT touch the model and does NOT run inference.
It only:
  - tracks an interrupt flag,
  - on interrupt, reads (never recomputes) the latest completed result
    from a ResultManager.

This mirrors the required semantics (section 33):
    Neural network:  "I am still computing."
    Safety system:   "I need the latest available detection NOW."
    Result manager:  "The latest completed result is Head 2."
    Safety system:   "Use Head 2."
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from .detection_result import AnytimeDetectionResult
from .result_manager import ResultManager

NO_DETECTION_AVAILABLE = "NO DETECTION AVAILABLE YET"


@dataclass
class InterruptResponse:
    interrupted_at: float
    frame_id: Optional[int]
    result: Optional[AnytimeDetectionResult]
    message: str

    @property
    def has_result(self) -> bool:
        return self.result is not None


class InterruptManager:
    def __init__(self, result_manager: ResultManager):
        self._result_manager = result_manager
        self._flag = threading.Event()
        self._lock = threading.Lock()
        self._interrupt_count = 0

    # -- control -----------------------------------------------------
    def trigger_interrupt(self) -> None:
        with self._lock:
            self._interrupt_count += 1
        self._flag.set()

    def clear_interrupt(self) -> None:
        self._flag.clear()

    def is_interrupted(self) -> bool:
        return self._flag.is_set()

    @property
    def interrupt_count(self) -> int:
        with self._lock:
            return self._interrupt_count

    # -- the operation that actually matters --------------------------
    def get_latest_result(self) -> InterruptResponse:
        """Fetch the most recently completed detection. Never blocks on
        the inference thread, never re-runs the model."""
        now = time.perf_counter()
        result = self._result_manager.get_latest()
        if result is None:
            return InterruptResponse(
                interrupted_at=now,
                frame_id=None,
                result=None,
                message=NO_DETECTION_AVAILABLE,
            )
        return InterruptResponse(
            interrupted_at=now,
            frame_id=result.frame_id,
            result=result,
            message=(
                f"Latest completed head: {result.head_name} "
                f"(available at {result.elapsed_ms:.2f} ms, "
                f"{result.num_detections()} detections)"
            ),
        )
