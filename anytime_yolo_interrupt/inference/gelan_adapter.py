"""
inference/gelan_adapter.py
-----------------------------
The REAL AnytimeModelAdapter (see anytime_detector.py for the interface
this implements), wired to the from-scratch GelanAnytimeModel in
models/gelan_anytime.py. This replaces MockAnytimeAdapter for actual
inference/logging once you have a trained checkpoint.

Each call to run_head() does genuinely more computation than the last
(backbone -> +topdown -> +bottomup -> +final-refine), so the elapsed_ms
recorded by AnytimeDetector for each tap reflects real, incremental
work -- not a single monolithic forward pass measured four times.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from ..models.gelan_anytime import GelanAnytimeModel, build_model, STRIDES
from ..models.postprocess import postprocess_head
from ..data.yolo_dataset import letterbox, unletterbox_boxes
from .anytime_detector import AnytimeModelAdapter, HeadSpec


class GelanAnytimeAdapter(AnytimeModelAdapter):
    def __init__(
        self,
        checkpoint_path: str,
        class_names: List[str],
        device: str = "cuda",
        img_size: int = 640,
        conf_thres: float = 0.3,
        iou_thres: float = 0.5,
        half: bool = False,
    ):
        """half=True runs the network in 16-bit floats (GPU only). The raw
        head outputs are converted back to 32-bit before decoding, because
        box coordinates up to 640 px lose sub-pixel precision in 16 bits."""
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.half = half and self.device.type == "cuda"
        self.class_names = class_names
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.strides = STRIDES

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = build_model(len(class_names)).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        if self.half:
            self.model.half()

        self.heads = [
            HeadSpec(1, "early_head_1"),
            HeadSpec(2, "early_head_2"),
            HeadSpec(3, "early_head_3"),
            HeadSpec(4, "final", is_final=True),
        ]

    # ------------------------------------------------------------------
    def run_shared_trunk(self, frame: dict):
        """frame: {"image": HxWx3 uint8 RGB array, "width": W, "height": H}.
        Letterboxes, runs the backbone ONCE, and stashes state that later
        run_head() calls will extend (see module docstring)."""
        image = frame["image"]
        canvas, info = letterbox(image, self.img_size)
        img_tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.half:
            img_tensor = img_tensor.half()

        with torch.no_grad():
            p3_bb, p4_bb, p5_bb = self.model.forward_backbone(img_tensor)

        letterbox_info = {
            "scale": info.scale, "pad_x": info.pad_x, "pad_y": info.pad_y,
            "orig_width": info.orig_width, "orig_height": info.orig_height,
        }
        # AnytimeDetector's postprocess() call doesn't thread trunk_state
        # through -- stash this here (run_shared_trunk always runs first,
        # exactly once per frame, before any postprocess() call) so
        # postprocess() below can map boxes back to original coordinates.
        self._current_letterbox_info = letterbox_info

        return {
            "letterbox_info": letterbox_info,
            "p3_bb": p3_bb, "p4_bb": p4_bb, "p5_bb": p5_bb,
        }

    def run_head(self, trunk_state: dict, head_index: int):
        """head_index 0..3. Does NOT stop the network -- always advances
        trunk_state so the NEXT call has more-processed features ready,
        exactly mirroring GelanAnytimeModel.forward()'s chain."""
        with torch.no_grad():
            if head_index == 0:
                feats = (trunk_state["p3_bb"], trunk_state["p4_bb"], trunk_state["p5_bb"])
                raw = self.model.head1(list(feats))
                p3_td, p4_td, p5_td = self.model.forward_topdown(*feats)
                trunk_state["p3_td"], trunk_state["p4_td"], trunk_state["p5_td"] = p3_td, p4_td, p5_td

            elif head_index == 1:
                feats = (trunk_state["p3_td"], trunk_state["p4_td"], trunk_state["p5_td"])
                raw = self.model.head2(list(feats))
                p3_bu, p4_bu, p5_bu = self.model.forward_bottomup(*feats)
                trunk_state["p3_bu"], trunk_state["p4_bu"], trunk_state["p5_bu"] = p3_bu, p4_bu, p5_bu

            elif head_index == 2:
                feats = (trunk_state["p3_bu"], trunk_state["p4_bu"], trunk_state["p5_bu"])
                raw = self.model.head3(list(feats))
                p3_f, p4_f, p5_f = self.model.forward_final_refine(*feats)
                trunk_state["p3_f"], trunk_state["p4_f"], trunk_state["p5_f"] = p3_f, p4_f, p5_f

            elif head_index == 3:
                feats = (trunk_state["p3_f"], trunk_state["p4_f"], trunk_state["p5_f"])
                raw = self.model.head_final(list(feats))

            else:
                raise ValueError(f"Unknown head_index {head_index}")

        if self.half:
            raw = [tuple(t.float() for t in scale) for scale in raw]
        return raw  # list of (cls_logits, obj_logit, reg_raw) per scale

    def postprocess(self, raw_pred, head_index: int, orig_width: int, orig_height: int):
        dets = postprocess_head(raw_pred, self.strides, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        det = dets[0]  # batch size 1 for live/interactive inference

        boxes = det["boxes"].cpu().numpy()
        scores = det["scores"].cpu().numpy().tolist()
        cls_ids = det["cls_ids"].cpu().numpy()

        # NOTE: this needs the SAME letterbox info run_shared_trunk computed.
        # AnytimeDetector doesn't thread that through postprocess()'s
        # signature, so GelanAnytimeAdapter stashes it on `self` right
        # before each process_frame() call -- see run_on_frame() below,
        # which is the entry point live inference scripts should use
        # instead of calling process_frame() directly.
        info = self._current_letterbox_info
        boxes_orig = unletterbox_boxes(boxes, info) if boxes.shape[0] else boxes
        classes = [self.class_names[c] for c in cls_ids]

        return boxes_orig.tolist(), scores, classes

    # ------------------------------------------------------------------
    def prepare_frame(self, image: np.ndarray) -> dict:
        """Convenience: build the `frame` dict AnytimeDetector.process_frame
        expects from a raw HxWx3 RGB array, and remember this frame's
        letterbox info for postprocess()'s coordinate remap."""
        h, w = image.shape[:2]
        # run_shared_trunk will letterbox again internally; we only need
        # width/height here up front so AnytimeDetectionResult's
        # image_width/image_height fields are correct immediately.
        return {"image": image, "width": w, "height": h}
