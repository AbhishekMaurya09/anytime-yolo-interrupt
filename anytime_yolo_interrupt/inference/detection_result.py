"""
detection_result.py
--------------------
Defines the AnytimeDetectionResult structure: the atomic, immutable
unit of information produced at every detection tap (intermediate
head or final head) during one forward pass of the anytime detector.

Design notes
------------
- Immutable (frozen dataclass): once constructed, a result cannot be
  mutated in place. This is what makes "atomic update" of the shared
  `latest_result` trivial and safe -- we always *replace* the pointer
  to a fully-built object, we never mutate fields of an object another
  thread might be reading. See result_manager.py.
- boxes/scores/classes are stored as plain Python lists of
  already-CPU, already-detached, already-NMS'd values -- never raw
  GPU tensors. This keeps the object cheap to hold, cheap to log to
  JSON, and safe to read from another thread without any CUDA context
  concerns.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class AnytimeDetectionResult:
    frame_id: int
    head_id: int
    head_name: str

    timestamp: float          # wall-clock time.time() when this result was produced
    elapsed_ms: float         # time since the start of this frame's inference

    boxes: List[List[float]]  # [[x1, y1, x2, y2], ...] in ORIGINAL image coordinates
    scores: List[float]
    classes: List[str]        # class name strings (not indices) for readability

    image_width: int
    image_height: int

    is_final: bool = False

    # Optional extras that don't change the contract but are handy.
    class_ids: Optional[List[int]] = None
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def num_detections(self) -> int:
        return len(self.boxes)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @staticmethod
    def from_dict(d: dict) -> "AnytimeDetectionResult":
        return AnytimeDetectionResult(**d)

    @staticmethod
    def empty(frame_id: int, image_width: int, image_height: int) -> "AnytimeDetectionResult":
        """A well-defined, explicit 'no detections yet' placeholder.

        NOT used as a substitute for 'no result available yet' -- the
        InterruptManager distinguishes 'no result at all' (returns None /
        an explicit NO_DETECTION_AVAILABLE marker) from 'a head completed
        but found zero objects' (this, a real empty-but-valid result).
        """
        return AnytimeDetectionResult(
            frame_id=frame_id,
            head_id=-1,
            head_name="none",
            timestamp=time.time(),
            elapsed_ms=0.0,
            boxes=[],
            scores=[],
            classes=[],
            image_width=image_width,
            image_height=image_height,
            is_final=False,
        )

    def __repr__(self) -> str:  # concise, readable console output
        return (
            f"AnytimeDetectionResult(frame={self.frame_id}, head={self.head_name}, "
            f"t={self.elapsed_ms:.2f}ms, n_det={self.num_detections()}, "
            f"final={self.is_final})"
        )
