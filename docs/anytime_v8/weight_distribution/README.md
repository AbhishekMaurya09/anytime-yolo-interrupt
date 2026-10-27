# Weight value distribution of every layer (anytime YOLOv8n)

Two CSV files describe the weight values of all **90 convolution layers**
of the anytime YOLOv8n model (`runs/anytime_v8/bdd1500/weights/best.pt`),
**3,659,712 weights** in total. Every number is computed directly from the
checkpoint by `docs/make_weight_distribution.py --model anytime_v8`.

The columns are the same as for train6; see
[`docs/weight_distribution/README.md`](../../weight_distribution/README.md)
for the meaning of each column of `layer_weight_stats.csv` and
`layer_weight_histogram.csv`.

## The layers

Layers are listed in **execution order**: backbone, Head 1, FPN, Head 2,
PAN, Final head. This model computes every head on every frame.

| Stage (`stage` column) | Layers | Weights | What it is |
|---|---|---|---|
| `backbone` | 27 | 1,267,632 | YOLOv8n layers 0 to 9 |
| `head1` | 13 | 265,168 | Early head 1 (`early.0`), one 3×3 convolution per branch |
| `fpn` | 8 | 184,320 | YOLOv8n layers 10 to 15 |
| `head2` | 13 | 265,168 | Early head 2 (`early.1`), one 3×3 convolution per branch |
| `pan` | 10 | 798,720 | YOLOv8n layers 16 to 21 |
| `head_final` | 19 | 878,704 | YOLOv8n's Detect head (layer 22) |

- **69 layers** have a BatchNorm after them. The other 21 are the 18
  output convolutions of the three heads and the 3 DFL layers.
- **Each head has a DFL layer** (`*.dfl.conv`, 16 weights). It is a 1×1
  convolution whose fixed weights are 0, 1, …, 15. It turns the 16 box-edge
  scores into one distance, and it is never trained. That is why the
  largest weight of every head stage shows as 15.0.
- **The early heads are full 3×3 convolutions**, not depthwise. That is why
  each one has 265,168 weights, against 43,856 in the fast model.

## Two versions of each layer

| Version | What it is |
|---|---|
| `trained` | The weights as stored in the checkpoint |
| `folded` | BatchNorm folded in: `w' = w × γ / √(var + ε)`. This is what an accelerator or the INT8 model multiplies with |

Ultralytics stores checkpoints in FP16, so the `trained` values are FP16
numbers. The statistics are computed from them in double precision.

## What the data shows

- **The whole network was trained**, starting from COCO-pretrained YOLOv8n,
  together with the two early heads. So unlike the fast model, the YOLOv8n
  layers here are not the fine-tuned YOLOv8n.
- **Folding changes the ranges a lot.** The first layer, `model.0.conv`,
  has trained weights between −0.515 and 0.483, but folded weights
  between −23.54 and 18.48.
- **The early heads have small folded weights.** Their 12 layers with
  BatchNorm reach a folded magnitude of 0.218 to 0.654. The Final head's
  12 such layers reach 0.612 to 11.19.
- **The ranges differ greatly between layers.** Leaving out the 3 fixed
  DFL layers, the largest folded weight per layer goes from 0.211
  (`model.22.cv3.0.2`) to 23.54 (`model.0.conv`), about 111×. A
  fixed-point accelerator therefore needs at least one scale per layer.

## Histogram pictures (`histograms/`)

Drawn from `layer_weight_histogram.csv` by `docs/make_weight_histograms.py`.

| File | Contents |
|---|---|
| `histograms/overview_trained_page_1.png` … `_4.png` | 24 layers per page, weights as trained |
| `histograms/overview_folded_page_1.png` … `_4.png` | 24 layers per page, BatchNorm folded in |
| `histograms/all_layers.pdf` | All 8 overview pages in one file |
| `histograms/layers/<order>_<layer>.png` | One picture per layer (90 files): trained and folded side by side |

Each panel uses that layer's own 64 bins, so the x-axes differ between
panels. The dark vertical line marks 0.

## How the files were checked

- `layer_weight_stats.csv` has 90 rows. `layer_weight_histogram.csv` has
  11,520 rows (90 layers × 2 versions × 64 bins).
- The model has 3,666,058 parameters in total. The CSVs count only
  convolution weights, so BatchNorm parameters and biases are not
  included.

## Regenerate

```bash
.venv\Scripts\python docs/make_weight_distribution.py --model anytime_v8 --checkpoint runs/anytime_v8/bdd1500/weights/best.pt --out docs/anytime_v8/weight_distribution
```
```bash
.venv\Scripts\python docs/make_weight_histograms.py --src docs/anytime_v8/weight_distribution --model-name "anytime YOLOv8n"
```
