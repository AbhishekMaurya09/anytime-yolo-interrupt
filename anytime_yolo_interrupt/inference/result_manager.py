"""
result_manager.py
------------------
Thread-safe storage for the "latest completed detection result" per
frame, plus a full per-frame, per-head history (used for persistent
logging and offline evaluation).

CRITICAL CORRECTNESS PROPERTIES (see project brief sections 8, 14, 15):

1. Atomic update, no partial reads.
   AnytimeDetectionResult is a frozen dataclass (see detection_result.py).
   A writer never mutates fields in place; it always builds a brand new,
   fully-formed object and then swaps a reference under a lock. A reader
   under the same lock therefore always sees either the old, complete
   object or the new, complete object -- never something half-written.

2. Frame identity is explicit.
   `latest_result` is keyed by frame_id. When frame N+1 starts, the
   manager is told explicitly (`start_frame`), so an interrupt landing
   just after a new frame has started can never be served frame N's
   detections while believing it is frame N+1's.

3. No fabrication.
   `get_latest()` returns None (not a fake empty result) if nothing has
   completed yet for the current frame. Callers decide how to report
   that (see interrupt_manager.py: "NO DETECTION AVAILABLE YET").
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import sys
import tempfile
import threading
from collections import defaultdict
from typing import Dict, List, Optional

from .detection_result import AnytimeDetectionResult

_FLUSH = object()  # tells the writer thread to write whatever it is holding


class ResultManager:
    def __init__(self, log_dir: Optional[str] = None):
        self._lock = threading.Lock()

        self._current_frame_id: Optional[int] = None
        self._latest_result: Optional[AnytimeDetectionResult] = None

        # frame_id -> head_name -> AnytimeDetectionResult
        # Kept for the *current* and recently-completed frames so the
        # evaluator / logger can retrieve full per-head history.
        self._history: Dict[int, Dict[str, AnytimeDetectionResult]] = defaultdict(dict)

        self.log_dir = log_dir
        self._write_queue: Optional["queue.Queue"] = None
        if self.log_dir:
            os.makedirs(os.path.join(self.log_dir, "frames"), exist_ok=True)
            # Disk writes run on their own thread and are held back until the
            # frame's final head has been published, so nothing is written
            # while a frame is being processed. Measured on a GTX 1650
            # laptop: writing each result as it was published delayed every
            # later head by about 6 ms, even from a second thread.
            self._write_queue = queue.Queue()
            threading.Thread(target=self._writer_loop, name="result-writer", daemon=True).start()
            atexit.register(self.flush)

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------
    def start_frame(self, frame_id: int) -> None:
        """Must be called exactly once at the start of each frame's
        inference, before any head result for that frame is produced.
        This is what prevents Frame N+1 from ever being served Frame N's
        latest_result (brief section 15)."""
        with self._lock:
            self._current_frame_id = frame_id
            self._latest_result = None  # no completed detection for this frame yet

    # ------------------------------------------------------------------
    # Writer side (called from the inference thread after each head)
    # ------------------------------------------------------------------
    def update(self, result: AnytimeDetectionResult) -> None:
        """Atomically publish a newly completed detection as the latest
        result for its frame. Also stores it in history and (if enabled)
        persists it to disk with atomic file replacement."""
        with self._lock:
            # Guard against a stale/late write from a previous frame's
            # head arriving after a new frame has already started.
            if self._current_frame_id is not None and result.frame_id != self._current_frame_id:
                # Still record it in history (useful for offline analysis),
                # but never let it become "latest" for the wrong frame.
                self._history[result.frame_id][result.head_name] = result
                return

            self._latest_result = result
            self._history[result.frame_id][result.head_name] = result

        # Disk I/O is queued for the writer thread: outside the lock and off
        # the inference thread.
        if self._write_queue is not None:
            self._write_queue.put(result)

    def flush(self) -> None:
        """Write anything still held back and block until it is on disk."""
        if self._write_queue is not None:
            self._write_queue.put(_FLUSH)
            self._write_queue.join()

    def _writer_loop(self) -> None:
        pending: List[AnytimeDetectionResult] = []
        while True:
            item = self._write_queue.get()
            try:
                if item is not _FLUSH:
                    pending.append(item)
                if item is _FLUSH or item.is_final:
                    batch, pending = pending, []
                    for result in batch:
                        self._persist(result)
                    if batch:
                        self._atomic_write_json(os.path.join(self.log_dir, "latest_result.json"),
                                                batch[-1].to_dict())
            except Exception as exc:  # a failed log write must not stop inference
                print(f"[ResultManager] could not save results: {exc}", file=sys.stderr)
            finally:
                self._write_queue.task_done()

    # ------------------------------------------------------------------
    # Reader side (called from the interrupt / safety thread)
    # ------------------------------------------------------------------
    def get_latest(self) -> Optional[AnytimeDetectionResult]:
        with self._lock:
            return self._latest_result  # frozen dataclass -> safe to hand out directly

    def get_frame_result(self, frame_id: int, head_name: Optional[str] = None):
        with self._lock:
            frame_hist = self._history.get(frame_id, {})
            if head_name is not None:
                return frame_hist.get(head_name)
            return dict(frame_hist)

    def get_history(self, frame_id: int) -> List[AnytimeDetectionResult]:
        with self._lock:
            return list(self._history.get(frame_id, {}).values())

    # ------------------------------------------------------------------
    # Persistence (brief section 16): atomic write-temp-then-rename
    # ------------------------------------------------------------------
    def _persist(self, result: AnytimeDetectionResult) -> None:
        frame_dir = os.path.join(self.log_dir, "frames", f"frame_{result.frame_id:06d}")
        os.makedirs(frame_dir, exist_ok=True)

        name = "final.json" if result.is_final else f"{result.head_name}.json"
        target = os.path.join(frame_dir, name)
        self._atomic_write_json(target, result.to_dict())
        # latest_result.json is written once per batch by the writer loop.

    @staticmethod
    def _atomic_write_json(target_path: str, payload: dict) -> None:
        directory = os.path.dirname(target_path)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target_path)  # atomic on POSIX and Windows
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
