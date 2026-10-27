"""
models/anytime_v8.py
----------------------
Anytime detector built on a pretrained YOLOv8n (Ultralytics).

The YOLOv8n backbone, neck and detection head are kept exactly as they are,
so they can start from the COCO-pretrained weights. Two light extra heads
tap the network earlier, so one forward pass produces three answers of
increasing quality, and the pass never exits early:

    layer  4, 6, 9   (backbone P3, P4, P5)        -> Head 1   (LiteDetect, new)
    layer 15, 12, 9  (after the top-down FPN)     -> Head 2   (LiteDetect, new)
    layer 15, 18, 21 (after the bottom-up PAN)    -> Final    (YOLOv8n's own Detect)

All three heads have YOLOv8's output format (DFL box distributions + class
scores, anchor-free, strides 8/16/32), so they use the same decoding, NMS and
loss (task-aligned assignment, CIoU + DFL + BCE). A LiteDetect head has one
3x3 convolution per branch (box and class) instead of two, at `width`
channels, so it costs far less than the final head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import Conv, DWConv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

HEAD_NAMES = ["early_head_1", "early_head_2", "final"]
TAPS = [(4, 6, 9), (15, 12, 9)]       # layer indices feeding Head 1 and Head 2, ordered P3, P4, P5
HEAD_WEIGHTS = (0.3, 0.6, 1.0)        # loss weight of Head 1, Head 2, Final


class LiteDetect(Detect):
    """YOLOv8 Detect head with one 3x3 conv per branch, for the early anytime taps.

    With depthwise, the 3x3 conv is split into a depthwise 3x3 and a 1x1 conv, which
    needs about 5x fewer multiply-adds at width 32."""

    def __init__(self, nc: int, ch: tuple[int, ...], width: int = 32, depthwise: bool = False):
        super().__init__(nc=nc, ch=ch)

        def branch(x, out):
            first = nn.Sequential(DWConv(x, x, 3), Conv(x, width, 1)) if depthwise else Conv(x, width, 3)
            return nn.Sequential(first, nn.Conv2d(width, out, 1))

        self.cv2 = nn.ModuleList(branch(x, 4 * self.reg_max) for x in ch)
        self.cv3 = nn.ModuleList(branch(x, self.nc) for x in ch)


class AnytimeV8(DetectionModel):
    """YOLOv8 DetectionModel plus two early LiteDetect heads.

    forward() in training returns a list of three raw prediction dicts (one per
    head) for the loss. In eval mode it returns the final head's output exactly
    like YOLOv8, so Ultralytics' validator works unchanged; forward_anytime()
    returns the decoded output of all three heads.
    """

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True, head_width=32, depthwise=False,
                 head_weights=HEAD_WEIGHTS):
        self._anytime = False   # plain YOLOv8 behaviour while the parent builds strides
        self.return_all = False
        self.head_weights = head_weights
        super().__init__(cfg, ch=ch, nc=nc, verbose=False)
        final = self.model[-1]
        with torch.no_grad():   # channel count of every tapped layer
            feats = self._run_layers(torch.zeros(1, ch, 64, 64))[1]
        self.early = nn.ModuleList(
            LiteDetect(final.nc, tuple(feats[i].shape[1] for i in taps), head_width, depthwise) for taps in TAPS)
        for head in self.early:
            head.stride = final.stride.clone()
            head.bias_init()
        self._anytime = True
        if verbose:
            self.info()

    def _run_layers(self, x):
        """Run the YOLOv8 layers in order; return the final Detect output and every saved feature map."""
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save or m.i in {i for t in TAPS for i in t} else None)
        return x, y

    def _predict_once(self, x, profile=False, embed=None):
        if not self._anytime or profile or embed:
            return super()._predict_once(x, profile, embed)
        out, y = self._run_layers(x)
        if self.training:
            return [head([y[i] for i in taps]) for head, taps in zip(self.early, TAPS)] + [out]
        if self.return_all:
            return [head([y[i] for i in taps])[0] for head, taps in zip(self.early, TAPS)] + [out[0]]
        return out

    def forward_anytime(self, x):
        """Decoded predictions (B, 4 + nc, anchors) of Head 1, Head 2 and Final, in that order."""
        self.return_all = True
        try:
            return self._predict_once(x)
        finally:
            self.return_all = False

    def init_criterion(self):
        return AnytimeV8Loss(self)


class AnytimeV8Loss:
    """Weighted sum of the YOLOv8 detection loss over the three heads."""

    def __init__(self, model):
        self.base = v8DetectionLoss(model)
        self.weights = getattr(model, "head_weights", HEAD_WEIGHTS)

    def __call__(self, preds, batch):
        if not isinstance(preds, list):          # validation: the final head only, as in YOLOv8
            return self.base(preds, batch)
        total, items = 0.0, None
        for w, p in zip(self.weights, preds):
            if w == 0:   # e.g. the final head when the YOLOv8n part is frozen
                continue
            loss, parts = self.base(p, batch)
            total = total + w * loss
            items = {k: w * v for k, v in parts.items()} if items is None else \
                {k: items[k] + w * v for k, v in parts.items()}
        return total, items


class AnytimeV8Trainer(DetectionTrainer):
    """Ultralytics' detection trainer, building AnytimeV8 instead of a plain DetectionModel.

    anytime_kwargs are passed to AnytimeV8 (head_width, depthwise, head_weights)."""

    def __init__(self, *args, anytime_kwargs=None, **kwargs):
        self.anytime_kwargs = anytime_kwargs or {}
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(
            AnytimeV8(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose, **self.anytime_kwargs))
        if weights:
            model.load(weights)
        return model
