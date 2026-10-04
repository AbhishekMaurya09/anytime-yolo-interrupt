#!/usr/bin/env python3
"""
interrupt_inference_live.py
------------------------------
Interactive inference loop you asked for:

  - Loads a trained checkpoint.
  - Steps through a folder of images one at a time.
  - For each image, runs the real forward pass (all 4 taps, never
    exiting early) on one long-lived background thread.
  - The MAIN thread blocks on `input()` waiting for you to press Enter.
  - The instant you press Enter, an interrupt is triggered and the
    LATEST completed head's result is fetched immediately (no wait,
    no recompute -- see inference/interrupt_manager.py).
  - If a ground-truth label file exists for that image, the fetched
    head's detections are scored against it (precision/recall/mean IoU
    for that one image), and a cumulative per-head mAP50 (computed only
    over the frames that particular head has actually been fetched for
    so far) is updated and logged.
  - Every press appends one line to `accuracy_log.jsonl` in --log-dir,
    plus a running `summary.json` with per-head cumulative accuracy.

Usage:
    python interrupt_inference_live.py \
        --checkpoint runs/anytime_interrupt/train1/last.pt \
        --data-root "C:\\Users\\abhis\\Downloads\\archive (2)" \
        --split test \
        --data-yaml "C:\\Users\\abhis\\Downloads\\archive (2)\\data.yaml" \
        --device cuda --log-dir runs/anytime_interrupt/live1

Press Ctrl+C at any time to stop early -- partial results are saved
and reported as partial (never mislabeled as a full test-set metric,
brief section 25).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from anytime_yolo_interrupt.inference import ResultManager, InterruptManager, AnytimeDetector, NO_DETECTION_AVAILABLE
from anytime_yolo_interrupt.data import load_data_yaml, read_yolo_labels, find_image_label_pairs
from anytime_yolo_interrupt.evaluation.metrics import evaluate_predictions, ImagePrediction, ImageGroundTruth


def load_image(path: str) -> np.ndarray:
    try:
        import cv2
        img = cv2.imread(path)
        return cv2_bgr_to_rgb(img)
    except ImportError:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))


def cv2_bgr_to_rgb(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def match_single_image(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thres=0.5):
    """Greedy IoU matching for ONE image -> (precision, recall, mean_iou_of_matches)."""
    if len(gt_boxes) == 0:
        return (1.0 if len(pred_boxes) == 0 else 0.0), 1.0, 0.0

    matched_gt = [False] * len(gt_boxes)
    tp = 0
    ious = []
    for pb, pc in zip(pred_boxes, pred_classes):
        best_iou, best_j = 0.0, -1
        for j, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
            if matched_gt[j] or gc != pc:
                continue
            iou = _iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thres:
            matched_gt[best_j] = True
            tp += 1
            ious.append(best_iou)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return precision, recall, mean_iou


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--data-yaml", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.3)
    parser.add_argument("--log-dir", type=str, default="runs/anytime_interrupt/live")
    parser.add_argument("--auto-interrupt-after-ms", type=float, default=None,
                         help="Optional: auto-press-Enter after N ms (for unattended/scripted runs). "
                              "Omit to require an actual keypress each frame.")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # Imported here (not at module top) so this script's --help works
    # even without torch installed; the adapter genuinely needs torch.
    from anytime_yolo_interrupt.inference.gelan_adapter import GelanAnytimeAdapter

    data_cfg = load_data_yaml(args.data_yaml)
    class_names = list(data_cfg["names"].values()) if isinstance(data_cfg["names"], dict) else list(data_cfg["names"])

    adapter = GelanAnytimeAdapter(
        checkpoint_path=args.checkpoint, class_names=class_names,
        device=args.device, img_size=args.img_size, conf_thres=args.conf_thres,
    )

    result_manager = ResultManager(log_dir=args.log_dir)
    detector = AnytimeDetector(adapter, result_manager)
    interrupt_manager = InterruptManager(result_manager)

    images_dir = os.path.join(args.data_root, args.split, "images")
    labels_dir = os.path.join(args.data_root, args.split, "labels")
    pairs = find_image_label_pairs(images_dir, labels_dir)
    if not pairs:
        print(f"No images found under {images_dir}")
        return

    # Per-head accumulated predictions/GTs, ONLY over frames where that
    # head was actually the one fetched -- this is genuinely "the
    # accuracy of what you'd have gotten, had you interrupted here".
    per_head_preds = {name: [] for name in ["early_head_1", "early_head_2", "early_head_3", "final"]}
    per_head_gts = {name: [] for name in ["early_head_1", "early_head_2", "early_head_3", "final"]}
    per_head_press_count = {name: 0 for name in per_head_preds}

    log_path = os.path.join(args.log_dir, "accuracy_log.jsonl")

    # ONE long-lived inference thread, fed through a queue. Starting a new
    # thread for every frame costs ~20 ms each time (the GPU libraries set up
    # per-thread state again), which would make every head look 20 ms late.
    jobs: "queue.Queue" = queue.Queue()
    frame_done = threading.Event()
    worker_errors = []

    def inference_worker():
        while True:
            job = jobs.get()
            if job is None:
                return
            try:
                job()
            except Exception as exc:  # reported on the main thread
                worker_errors.append(exc)
            finally:
                frame_done.set()

    worker = threading.Thread(target=inference_worker, name="anytime-inference", daemon=True)
    worker.start()

    def run_on_worker(job, wait: bool):
        frame_done.clear()
        jobs.put(job)
        if wait:
            frame_done.wait()

    # Warm-up: the first pass on a new thread is slow. Uses a throwaway
    # ResultManager so nothing from it is logged or served.
    warm_frame = adapter.prepare_frame(load_image(pairs[0][0]))
    warm_detector = AnytimeDetector(adapter, ResultManager(log_dir=None))
    run_on_worker(lambda: warm_detector.process_frame(warm_frame, 0), wait=True)

    print(f"Loaded {len(pairs)} images from {images_dir}.")
    print("Press ENTER at any time during a frame's inference to interrupt it "
          "and fetch the latest completed head's result. Ctrl+C to stop early.\n")

    try:
        for frame_id, (img_path, label_path) in enumerate(pairs):
            image = load_image(img_path)
            frame = adapter.prepare_frame(image)

            run_on_worker(lambda f=frame, i=frame_id: detector.process_frame(f, i), wait=False)

            if args.auto_interrupt_after_ms is not None:
                time.sleep(args.auto_interrupt_after_ms / 1000.0)
            else:
                input(f"[frame {frame_id}] {os.path.basename(img_path)} -- press ENTER to interrupt... ")

            interrupt_manager.trigger_interrupt()
            response = interrupt_manager.get_latest_result()  # read NOW, before join()

            log_entry = {"frame_id": frame_id, "image": img_path}

            if not response.has_result:
                print(f"  -> {NO_DETECTION_AVAILABLE}")
                log_entry.update({"head_name": None, "message": NO_DETECTION_AVAILABLE})
            else:
                r = response.result
                print(f"  -> head={r.head_name}  latency={r.elapsed_ms:.2f}ms  "
                      f"detections={r.num_detections()}")
                log_entry.update({
                    "head_name": r.head_name, "elapsed_ms": r.elapsed_ms,
                    "num_detections": r.num_detections(),
                })

                gt_raw = read_yolo_labels(label_path, frame["width"], frame["height"])
                if gt_raw.shape[0] > 0 or os.path.exists(label_path):
                    gt_boxes = gt_raw[:, 1:5].tolist()
                    gt_classes = [class_names[int(c)] for c in gt_raw[:, 0]]

                    precision, recall, mean_iou = match_single_image(
                        r.boxes, r.classes, gt_boxes, gt_classes,
                    )
                    print(f"     this-frame @ {r.head_name}: precision={precision:.2f} "
                          f"recall={recall:.2f} mean_IoU={mean_iou:.2f}")
                    log_entry.update({
                        "single_image_precision": precision,
                        "single_image_recall": recall,
                        "single_image_mean_iou": mean_iou,
                    })

                    per_head_preds[r.head_name].append(ImagePrediction(
                        boxes=np.array(r.boxes, dtype=float).reshape(-1, 4),
                        scores=np.array(r.scores, dtype=float),
                        classes=np.array(r.classes, dtype=object),
                    ))
                    per_head_gts[r.head_name].append(ImageGroundTruth(
                        boxes=np.array(gt_boxes, dtype=float).reshape(-1, 4),
                        classes=np.array(gt_classes, dtype=object),
                    ))
                    per_head_press_count[r.head_name] += 1

                    cum = evaluate_predictions(
                        per_head_preds[r.head_name], per_head_gts[r.head_name],
                        class_names, n_images_expected=len(pairs),
                    )
                    print(f"     cumulative mAP50 @ {r.head_name} so far "
                          f"({per_head_press_count[r.head_name]} presses): {cum.map50:.3f}")
                    log_entry["cumulative_mAP50_this_head"] = cum.map50
                    log_entry["cumulative_presses_this_head"] = per_head_press_count[r.head_name]

            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            frame_done.wait()  # let this frame's full forward pass finish before moving on
            if worker_errors:
                raise worker_errors[0]

    except (KeyboardInterrupt, EOFError) as e:
        # EOFError: stdin closed (e.g. Enter presses piped in ran out).
        how = "Ctrl+C" if isinstance(e, KeyboardInterrupt) else "end of input"
        print(f"\nStopped early ({how}).")
        frame_done.wait()
    finally:
        jobs.put(None)  # stop the inference thread

    # Final honest summary -- explicit about partial coverage per head.
    summary = {}
    for name in per_head_preds:
        n = per_head_press_count[name]
        if n == 0:
            summary[name] = {"presses": 0, "mAP50": None, "note": "never fetched via interrupt in this session"}
            continue
        result = evaluate_predictions(per_head_preds[name], per_head_gts[name], class_names, n_images_expected=len(pairs))
        summary[name] = {
            "presses": n, "mAP50": result.map50, "mAP50_95": result.map50_95,
            "precision": result.precision, "recall": result.recall,
            "note": f"based on the {n} frame(s) where this head was actually fetched "
                    f"(out of {len(pairs)} total images) -- NOT a full test-set metric.",
        }

    with open(os.path.join(args.log_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SESSION SUMMARY (accuracy at each head, over frames actually fetched)")
    print("=" * 60)
    for name, s in summary.items():
        if s["mAP50"] is None:
            print(f"  {name:>14}: never fetched")
        else:
            print(f"  {name:>14}: presses={s['presses']:>3}  mAP50={s['mAP50']:.3f}  "
                  f"P={s['precision']:.2f}  R={s['recall']:.2f}")
    print(f"\nFull log: {log_path}")
    print(f"Summary:  {os.path.join(args.log_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
