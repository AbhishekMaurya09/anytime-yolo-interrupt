"""
anytime_detector.py
---------------------
The core "one forward pass, N observation taps, no early return" loop
(brief sections 2, 6, 31, 32).

This module is intentionally decoupled from any specific model
implementation via `AnytimeModelAdapter`. That is the ONE seam you need
to implement to plug in the real GELAN-t-anytime (DKuhse/anytime-yolo)
network:

    class GelanTAnytimeAdapter(AnytimeModelAdapter):
        def run_shared_trunk(self, frame): ...      # backbone + neck, once
        def run_head(self, features, head_index): ...  # one sub-exit's raw preds
        def postprocess(self, raw_pred, head_index, orig_shape): ...  # NMS + coord remap

For this deliverable (no network access to clone the real repo, no
access to your local dataset), a `MockAnytimeAdapter` is provided so
the whole pipeline (result manager, interrupt manager, sensor
interface, logging, evaluation, tests, plotting) is fully runnable and
tested end-to-end right now. Swapping MockAnytimeAdapter for a real
GelanTAnytimeAdapter is a drop-in replacement -- nothing else in this
codebase needs to change.

Why a class-based adapter instead of just calling the real model
directly here: the real repo's forward pass is PyTorch/CUDA-specific
(needs torch.cuda.synchronize() for correct timing, per brief section
11) while everything else in this framework (locking, frame identity,
interrupt handling, JSON logging) is plain Python/threading and should
not need torch installed just to be tested or reused elsewhere.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from .detection_result import AnytimeDetectionResult
from .result_manager import ResultManager

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@dataclass
class HeadSpec:
    """Static description of one detection tap."""
    head_id: int
    head_name: str
    is_final: bool = False


class AnytimeModelAdapter(ABC):
    """Implement this against the real model. See module docstring."""

    heads: Sequence[HeadSpec]

    @abstractmethod
    def run_shared_trunk(self, frame: Any) -> Any:
        """Backbone + neck (or, for GELAN-t-anytime's transposed
        variant, whatever shared computation precedes the first
        sub-exit). Runs exactly once per frame."""

    @abstractmethod
    def run_head(self, trunk_state: Any, head_index: int) -> Any:
        """Execute ONE detection head/tap given the trunk state (and any
        state carried over from prior heads, if the real architecture
        needs it -- see note below). Must NOT stop the network; the
        caller (process_frame) always continues to the next head
        afterwards. Returns raw (pre-NMS) predictions.

        Note: real GELAN-t-anytime heads at different sub-exit tuples
        may consume progressively more of the trunk's computation
        (brief section 4 of the paper: sub-exits are matched across the
        three feature-pyramid resolutions). If your adapter needs to
        carry intermediate feature maps forward between calls, keep
        them inside `trunk_state` (e.g. as a mutable container) rather
        than as extra hidden globals, so nothing here needs to know
        about it.
        """

    @abstractmethod
    def postprocess(
        self, raw_pred: Any, head_index: int, orig_width: int, orig_height: int
    ) -> Tuple[List[List[float]], List[float], List[str]]:
        """NMS + coordinate remap to the original image. Must return
        (boxes, scores, class_name_strings) as plain Python lists -- see
        detection_result.py for why."""


class MockAnytimeAdapter(AnytimeModelAdapter):
    """A deterministic, dependency-free stand-in for the real network so
    the rest of the framework can be built, run and tested without
    torch/CUDA/the actual repo/the actual dataset. Simulates:
      - a small fixed compute cost per head (increasing depth = more
        time, mirroring Table 8/10 in the AnytimeYOLO paper),
      - detection quality that improves at later heads (more boxes,
        higher confidence), mirroring the anytime-quality curves in
        the paper's Fig. 4/5/8/9.
    Replace with GelanTAnytimeAdapter for real results.
    """

    CLASS_NAMES = [
        "pedestrian", "rider", "car", "bus", "truck", "bicycle",
        "motorcycle", "traffic light", "traffic sign", "train",
    ]

    def __init__(self, seed: int = 0):
        import random
        self._rng = random.Random(seed)
        self.heads = [
            HeadSpec(1, "early_head_1"),
            HeadSpec(2, "early_head_2"),
            HeadSpec(3, "early_head_3"),
            HeadSpec(4, "final", is_final=True),
        ]
        # Simulated per-head compute cost in seconds (cumulative timing
        # is measured by the caller, this only models the work done).
        self._head_cost_s = {1: 0.003, 2: 0.004, 3: 0.004, 4: 0.005}

    def run_shared_trunk(self, frame: Any) -> Any:
        time.sleep(0.002)  # simulated backbone/neck cost
        h, w = frame["height"], frame["width"]
        return {"width": w, "height": h}

    def run_head(self, trunk_state: Any, head_index: int) -> Any:
        cost = self._head_cost_s[self.heads[head_index].head_id]
        time.sleep(cost)
        # Quality improves monotonically with head index.
        n_objects = 3 + head_index
        w, h = trunk_state["width"], trunk_state["height"]
        raw = []
        for i in range(n_objects):
            cx = self._rng.uniform(0.1, 0.9) * w
            cy = self._rng.uniform(0.1, 0.9) * h
            bw = self._rng.uniform(0.05, 0.25) * w
            bh = self._rng.uniform(0.05, 0.25) * h
            conf = min(0.99, 0.55 + 0.12 * head_index + self._rng.uniform(-0.03, 0.05))
            cls = self._rng.choice(self.CLASS_NAMES)
            raw.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, conf, cls))
        return raw

    def postprocess(self, raw_pred, head_index, orig_width, orig_height):
        boxes, scores, classes = [], [], []
        for x1, y1, x2, y2, conf, cls in raw_pred:
            boxes.append([
                round(max(0.0, x1), 1), round(max(0.0, y1), 1),
                round(min(orig_width, x2), 1), round(min(orig_height, y2), 1),
            ])
            scores.append(round(conf, 4))
            classes.append(cls)
        return boxes, scores, classes


class AnytimeDetector:
    """Runs one forward pass per frame, saving a result at every tap
    without ever exiting early (brief sections 2, 31, 32).

    pipelined=False: each head's postprocess runs on the calling thread,
    so the next stage waits for it.
    pipelined=True: postprocess + publish run on one background thread,
    in head order, while the calling thread goes straight on to the next
    stage. Results are still published strictly in head order, and
    process_frame still returns only after every head is published.

    elapsed_ms is measured when a result is published, i.e. when an
    interrupt could actually receive it.
    """

    def __init__(self, adapter: AnytimeModelAdapter, result_manager: ResultManager, pipelined: bool = False):
        self.adapter = adapter
        self.result_manager = result_manager
        self.pipelined = pipelined
        # One worker => FIFO, so heads are published in order.
        self._post_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anytime-post") if pipelined else None

    def process_frame(self, frame: dict, frame_id: int) -> List[AnytimeDetectionResult]:
        """
        frame: dict with at least {"width": int, "height": int, ...pixel data...}
        Returns the list of AnytimeDetectionResult produced, in order.
        """
        self.result_manager.start_frame(frame_id)

        self._cuda_sync()
        frame_start = time.perf_counter()

        trunk_state = self.adapter.run_shared_trunk(frame)
        self._cuda_sync()

        pending = []
        for head_index, spec in enumerate(self.adapter.heads):
            raw_pred = self.adapter.run_head(trunk_state, head_index)
            self._cuda_sync()  # correct GPU timing (brief section 11)

            # There is no "if interrupted: return" in this loop (section 32).
            if self._post_pool is not None:
                pending.append(self._post_pool.submit(
                    self._publish, raw_pred, head_index, spec, frame, frame_id, frame_start))
            else:
                pending.append(self._publish(raw_pred, head_index, spec, frame, frame_id, frame_start))

        if self._post_pool is not None:
            return [f.result() for f in pending]  # waits for every head; re-raises worker errors
        return pending

    def _publish(self, raw_pred, head_index, spec, frame, frame_id, frame_start) -> AnytimeDetectionResult:
        boxes, scores, classes = self.adapter.postprocess(
            raw_pred, head_index, frame["width"], frame["height"]
        )
        result = AnytimeDetectionResult(
            frame_id=frame_id,
            head_id=spec.head_id,
            head_name=spec.head_name,
            timestamp=time.time(),
            elapsed_ms=(time.perf_counter() - frame_start) * 1000.0,
            boxes=boxes,
            scores=scores,
            classes=classes,
            image_width=frame["width"],
            image_height=frame["height"],
            is_final=spec.is_final,
        )
        # Atomic publish.
        self.result_manager.update(result)
        return result

    def close(self) -> None:
        if self._post_pool is not None:
            self._post_pool.shutdown(wait=True)

    @staticmethod
    def _cuda_sync() -> None:
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
