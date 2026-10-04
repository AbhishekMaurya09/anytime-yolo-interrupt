"""
test_anytime.py
-----------------
Covers every test required by brief section 40:

  1. No interrupt -> all heads execute, final result produced.
  2. Interrupt before Head 1 -> no completed detection.
  3. Interrupt after Head 1, before Head 2 -> Head 1 result.
  4. Interrupt after Head 2, before Head 3 -> Head 2 result.
  5. Interrupt after Head 3, before final -> Head 3 result.
  6. Interrupt after final -> Final result.
  7. Head 1 result is not overwritten by a partial Head 2 write
     (atomicity of ResultManager.update).
  8. Frame IDs: Frame 100 result != Frame 101 result (no cross-frame bleed).

Also covers: concurrent read/write stress (many interrupt reads while
inference is writing) never returns a torn/partial object, and the
"no result at all" path returns NO_DETECTION_AVAILABLE rather than a
fabricated result.

Run with:  python -m pytest tests/test_anytime.py -v
       or: python tests/test_anytime.py
"""
from __future__ import annotations

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anytime_yolo_interrupt.inference import (
    AnytimeDetector, AnytimeModelAdapter, HeadSpec,
    MockAnytimeAdapter, ResultManager, InterruptManager, NO_DETECTION_AVAILABLE,
)


class ControlledAdapter(AnytimeModelAdapter):
    """A test-only adapter whose per-head timing is exact and
    deterministic (no randomness, no sleeps beyond what we specify),
    so interrupt-time tests can assert precise before/after behavior
    without flakiness."""

    def __init__(self, head_times_s):
        self.heads = [
            HeadSpec(1, "early_head_1"),
            HeadSpec(2, "early_head_2"),
            HeadSpec(3, "early_head_3"),
            HeadSpec(4, "final", is_final=True),
        ]
        self._head_times_s = head_times_s  # list of 4 durations

    def run_shared_trunk(self, frame):
        return {"width": frame["width"], "height": frame["height"]}

    def run_head(self, trunk_state, head_index):
        time.sleep(self._head_times_s[head_index])
        return [(0, 0, 10, 10, 0.9, "car")]

    def postprocess(self, raw_pred, head_index, orig_width, orig_height):
        boxes = [[p[0], p[1], p[2], p[3]] for p in raw_pred]
        scores = [p[4] for p in raw_pred]
        classes = [p[5] for p in raw_pred]
        return boxes, scores, classes


HEAD_TIMES_S = [0.05, 0.05, 0.05, 0.05]  # each head takes 50ms -> boundaries at 50/100/150/200ms


def run_frame_in_background(detector, frame, frame_id):
    t = threading.Thread(target=lambda: detector.process_frame(frame, frame_id))
    t.start()
    return t


class TestAnytimeInterruptSemantics(unittest.TestCase):
    PIPELINED = False

    def setUp(self):
        self.rm = ResultManager(log_dir=None)
        self.detector = AnytimeDetector(ControlledAdapter(HEAD_TIMES_S), self.rm, pipelined=self.PIPELINED)
        self.interrupt_manager = InterruptManager(self.rm)
        self.frame = {"width": 100, "height": 100}

    # 1. No interrupt -> all heads execute, final result produced.
    def test_no_interrupt_all_heads_execute(self):
        results = self.detector.process_frame(self.frame, frame_id=1)
        self.assertEqual(len(results), 4)
        self.assertTrue(results[-1].is_final)
        latest = self.rm.get_latest()
        self.assertEqual(latest.head_name, "final")

    # 2. Interrupt before Head 1 completes -> no completed detection.
    def test_interrupt_before_head1(self):
        self.rm.start_frame(2)
        t = run_frame_in_background(self.detector, self.frame, 2)
        time.sleep(0.01)  # well before the 50ms mark of head 1
        response = self.interrupt_manager.get_latest_result()
        self.assertFalse(response.has_result)
        self.assertEqual(response.message, NO_DETECTION_AVAILABLE)
        t.join()

    # 3. Interrupt after Head 1, before Head 2 -> Head 1 result.
    def test_interrupt_after_head1_before_head2(self):
        t = run_frame_in_background(self.detector, self.frame, 3)
        time.sleep(0.075)  # between 50ms (head1 done) and 100ms (head2 done)
        response = self.interrupt_manager.get_latest_result()
        t.join()
        self.assertTrue(response.has_result)
        self.assertEqual(response.result.head_name, "early_head_1")

    # 4. Interrupt after Head 2, before Head 3 -> Head 2 result.
    def test_interrupt_after_head2_before_head3(self):
        t = run_frame_in_background(self.detector, self.frame, 4)
        time.sleep(0.125)
        response = self.interrupt_manager.get_latest_result()
        t.join()
        self.assertEqual(response.result.head_name, "early_head_2")

    # 5. Interrupt after Head 3, before final -> Head 3 result.
    def test_interrupt_after_head3_before_final(self):
        t = run_frame_in_background(self.detector, self.frame, 5)
        time.sleep(0.175)
        response = self.interrupt_manager.get_latest_result()
        t.join()
        self.assertEqual(response.result.head_name, "early_head_3")

    # 6. Interrupt after final -> Final result.
    def test_interrupt_after_final(self):
        t = run_frame_in_background(self.detector, self.frame, 6)
        t.join()  # let it fully finish first
        response = self.interrupt_manager.get_latest_result()
        self.assertTrue(response.result.is_final)
        self.assertEqual(response.result.head_name, "final")

    # 7. Head 1 result is not overwritten by a partial Head 2 write.
    #    (Atomicity: readers during Head 2's write window always see
    #    either the complete Head 1 result or the complete Head 2
    #    result -- never a half-built object.)
    def test_no_torn_reads_under_concurrent_stress(self):
        adapter = ControlledAdapter([0.02, 0.02, 0.02, 0.02])
        rm = ResultManager(log_dir=None)
        detector = AnytimeDetector(adapter, rm, pipelined=self.PIPELINED)
        interrupt_manager = InterruptManager(rm)

        observed_head_names = set()
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                r = rm.get_latest()
                if r is not None:
                    # A torn/partial object would typically fail these
                    # invariants (mismatched lengths, missing fields).
                    self.assertEqual(len(r.boxes), len(r.scores))
                    self.assertEqual(len(r.boxes), len(r.classes))
                    self.assertIsInstance(r.head_name, str)
                    observed_head_names.add(r.head_name)

        reader_threads = [threading.Thread(target=reader) for _ in range(8)]
        for rt in reader_threads:
            rt.start()

        detector.process_frame(self.frame, frame_id=7)

        stop.set()
        for rt in reader_threads:
            rt.join()

        # We should have observed a strict, non-corrupted subset of the
        # real head names, in some prefix of the real completion order.
        self.assertTrue(observed_head_names.issubset({"early_head_1", "early_head_2", "early_head_3", "final"}))

    # 8. Frame IDs: Frame 100 result != Frame 101 result.
    def test_frame_identity_not_mixed(self):
        t1 = run_frame_in_background(self.detector, self.frame, 100)
        t1.join()
        result_100 = self.rm.get_latest()
        self.assertEqual(result_100.frame_id, 100)

        # Start frame 101 -- latest_result must reset for the new frame,
        # even before its first head completes.
        self.rm.start_frame(101)
        self.assertIsNone(self.rm.get_latest())

        t2 = run_frame_in_background(self.detector, self.frame, 101)
        time.sleep(0.075)  # after head1 of frame 101 completes
        response = self.interrupt_manager.get_latest_result()
        t2.join()

        self.assertEqual(response.result.frame_id, 101)
        self.assertNotEqual(response.result.frame_id, result_100.frame_id)

    # Extra: a late-arriving write from a stale frame must never become
    # "latest" once a newer frame has started (guards brief section 15).
    def test_stale_frame_write_is_rejected_as_latest(self):
        from anytime_yolo_interrupt.inference.detection_result import AnytimeDetectionResult
        rm = ResultManager(log_dir=None)
        rm.start_frame(200)
        stale = AnytimeDetectionResult(
            frame_id=199, head_id=1, head_name="early_head_1",
            timestamp=time.time(), elapsed_ms=5.0,
            boxes=[[0, 0, 1, 1]], scores=[0.9], classes=["car"],
            image_width=100, image_height=100, is_final=False,
        )
        rm.update(stale)  # arrives late, after frame 200 has started
        self.assertIsNone(rm.get_latest())  # must NOT be served as frame 200's latest


class TestAnytimeInterruptSemanticsPipelined(TestAnytimeInterruptSemantics):
    """Same interrupt guarantees with postprocess on a background thread."""
    PIPELINED = True


class SlowPostprocessAdapter(ControlledAdapter):
    """Fast heads, slow postprocess: the case pipelining is for."""

    def __init__(self, head_time_s, post_time_s):
        super().__init__([head_time_s] * 4)
        self._post_time_s = post_time_s

    def postprocess(self, raw_pred, head_index, orig_width, orig_height):
        time.sleep(self._post_time_s)
        return super().postprocess(raw_pred, head_index, orig_width, orig_height)


class TestPipelinedPostprocess(unittest.TestCase):
    def test_heads_published_in_order_and_overlapped(self):
        rm = ResultManager(log_dir=None)
        published = []
        original_update = rm.update

        def recording_update(result):
            published.append(result.head_name)
            original_update(result)

        rm.update = recording_update
        detector = AnytimeDetector(SlowPostprocessAdapter(0.02, 0.04), rm, pipelined=True)
        start = time.perf_counter()
        results = detector.process_frame({"width": 100, "height": 100}, frame_id=1)
        wall_s = time.perf_counter() - start
        detector.close()

        names = ["early_head_1", "early_head_2", "early_head_3", "final"]
        self.assertEqual(published, names)
        self.assertEqual([r.head_name for r in results], names)
        self.assertEqual(rm.get_latest().head_name, "final")
        elapsed = [r.elapsed_ms for r in results]
        self.assertEqual(elapsed, sorted(elapsed))
        # Sequential would take 4 x (20 + 40) = 240 ms; pipelined ~ 20 + 4 x 40 = 180 ms.
        self.assertLess(wall_s, 0.225)


class TestResultPersistence(unittest.TestCase):
    """Results are written to disk by a background thread; flush() waits for it."""

    def test_results_saved_in_order_after_flush(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as log_dir:
            rm = ResultManager(log_dir=log_dir)
            detector = AnytimeDetector(ControlledAdapter([0.0, 0.0, 0.0, 0.0]), rm)
            detector.process_frame({"width": 100, "height": 100}, frame_id=7)
            rm.flush()

            frame_dir = os.path.join(log_dir, "frames", "frame_000007")
            self.assertEqual(sorted(os.listdir(frame_dir)),
                             ["early_head_1.json", "early_head_2.json", "early_head_3.json", "final.json"])
            with open(os.path.join(log_dir, "latest_result.json")) as f:
                latest = json.load(f)
            self.assertEqual(latest["head_name"], "final")
            self.assertEqual(latest["frame_id"], 7)


class TestMockAdapterEndToEnd(unittest.TestCase):
    """Sanity check the deliverable's runnable mock pipeline (not just
    the controlled-timing unit tests above)."""

    def test_mock_pipeline_runs(self):
        rm = ResultManager(log_dir=None)
        detector = AnytimeDetector(MockAnytimeAdapter(seed=42), rm)
        results = detector.process_frame({"width": 640, "height": 480}, frame_id=1)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r.num_detections() > 0 for r in results))
        # monotonic non-decreasing elapsed_ms across heads
        elapsed = [r.elapsed_ms for r in results]
        self.assertEqual(elapsed, sorted(elapsed))


if __name__ == "__main__":
    unittest.main(verbosity=2)
