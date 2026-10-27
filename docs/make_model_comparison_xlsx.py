"""Writes docs/model_comparison/model_comparison.xlsx: every value behind the comparison charts.

Blue = measured value copied at full precision from the source file named on each sheet.
Black = Excel formula on the blue values.

    .venv/Scripts/python docs/make_model_comparison_xlsx.py
"""
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

from make_model_comparison_charts import (ANYTIME, HEAD_NAME, MODELS, NAME, N_IMG, OUT, R, int8_results,
                                          weight_ranges)
from make_parameter_study_xlsx import BLACK, BLUE, BOLD, BOX, HEAD_FILL, NOTE, TITLE, note, put

FIRST = 5
SRC = "Source: runs/model_comparison/results.json"
ROWS = [(m, k) for m in MODELS for k in R["heads"][m]]   # one row per answer a model can give


def sheet(wb, name, title, source, headers, widths):
    ws = wb.create_sheet(name)
    ws["A1"] = title
    ws["A1"].font = TITLE
    ws["A2"] = source
    ws["A2"].font = NOTE
    for c, (text, w) in enumerate(zip(headers, widths), start=1):
        cell = ws.cell(row=4, column=c, value=text)
        cell.font = BOLD
        cell.fill = HEAD_FILL
        cell.border = BOX
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[4].height = 48
    ws.freeze_panes = "C5"
    return ws


def model_head(ws, r, m, k):
    for c, v in ((1, NAME[m]), (2, HEAD_NAME[k])):
        cell = ws.cell(row=r, column=c, value=v)
        cell.font = BOLD
        cell.border = BOX


def row_of(m, k):
    return FIRST + ROWS.index((m, k))


def accuracy_sheet(wb):
    ws = sheet(wb, "Accuracy", f"Accuracy of every head on {N_IMG} unseen test images (GPU, FP32)",
               SRC + " accuracy.<model>.<head>",
               ["Model", "Head", "mAP50", "mAP50-95", "Precision", "Recall", "Share of the model's Final mAP50",
                "mAP50 − YOLOv8n"], [26, 10, 12, 12, 12, 12, 16, 14])
    y = row_of("yolov8n", "final")
    for m, k in ROWS:
        r = row_of(m, k)
        model_head(ws, r, m, k)
        a = R["accuracy"][m][k]
        for c, key in ((3, "mAP50"), (4, "mAP50_95"), (5, "precision"), (6, "recall")):
            put(ws, r, c, a[key], "0.0000", True)
        put(ws, r, 7, f"=C{r}/C{row_of(m, 'final')}", "0.0%", False)
        put(ws, r, 8, f"=C{r}-C{y}", "+0.0000;-0.0000;0.0000", False)
    n = FIRST + len(ROWS) + 1
    note(ws, n, "train6 was trained on 65,000 images; the other four models on the same 1,200 bdd1500 images.")
    note(ws, n + 1, "Metric: this project's code, confidence ≥ 0.001, NMS IoU 0.7, up to 300 boxes.")


def time_sheet(wb):
    ws = sheet(wb, "Time", "Time until each head's answer (GTX 1650, median and 90th percentile, ms)",
               SRC + " pipeline.<model>.<head>, network_only_640.<model>.<head>",
               ["Model", "Head", "Pipeline median", "Pipeline 90th pct", "Pipeline − YOLOv8n",
                "Network only, 640 × 640, median", "Network only, 90th pct", "Network − YOLOv8n"],
               [26, 10, 13, 13, 13, 15, 13, 13])
    y = row_of("yolov8n", "final")
    for m, k in ROWS:
        r = row_of(m, k)
        model_head(ws, r, m, k)
        put(ws, r, 3, R["pipeline"][m][k]["median_ms"], "0.00", True)
        put(ws, r, 4, R["pipeline"][m][k]["p90_ms"], "0.00", True)
        put(ws, r, 5, f"=C{r}-C{y}", "+0.00;-0.00;0.00", False)
        put(ws, r, 6, R["network_only_640"][m][k]["median_ms"], "0.00", True)
        put(ws, r, 7, R["network_only_640"][m][k]["p90_ms"], "0.00", True)
        put(ws, r, 8, f"=F{r}-F{y}", "+0.00;-0.00;0.00", False)
    n = FIRST + len(ROWS) + 1
    note(ws, n, "Pipeline: image in to boxes on the CPU (resize, network, NMS); models interleaved frame by frame. "
                "train6 runs at 640 × 640 through its own code; the others at 640 × 384 (letterboxed 1280 × 720).")
    note(ws, n + 1, "Network only: the same 640 × 640 input for every model, up to that head.")
    note(ws, n + 2, "anytime_v8_fast computes an early head only when an interrupt asks for it; anytime_v8 and train6 "
                    "compute every head on every frame.")


def fps_sheet(wb):
    ws = sheet(wb, "FPS", "Frames per second with no interrupt (GTX 1650, image in to boxes)",
               SRC + " fps.<model>", ["Model", "", "Repeat 1", "Repeat 2", "Repeat 3", "Median", "÷ YOLOv8n"],
               [26, 2, 12, 12, 12, 12, 12])
    y = FIRST + MODELS.index("yolov8n")
    for i, m in enumerate(MODELS):
        r = FIRST + i
        ws.cell(row=r, column=1, value=NAME[m]).font = BOLD
        for c, v in enumerate(R["fps"][m]["per_repeat"], start=3):
            put(ws, r, c, v, "0.0", True)
        put(ws, r, 6, R["fps"][m]["median"], "0.0", True)
        put(ws, r, 7, f"=F{r}/F{y}", "0.00\"×\"", False)
    note(ws, FIRST + len(MODELS) + 1, f"{N_IMG} test images back to back, 3 repeats, median.")


def size_sheet(wb):
    ws = sheet(wb, "Size", "Parameters and multiply-adds up to each head (640 × 640 input)",
               SRC + " size.<model>", ["Model", "Head", "Parameters (whole model)", "Multiply-adds up to this head",
                                       "Billions", "÷ YOLOv8n"], [26, 10, 16, 18, 12, 12])
    y = row_of("yolov8n", "final")
    for m, k in ROWS:
        r = row_of(m, k)
        model_head(ws, r, m, k)
        put(ws, r, 3, R["size"][m]["parameters"], "#,##0", True)
        put(ws, r, 4, R["size"][m]["macs_640_upto_head"][R["heads"][m].index(k)], "#,##0", True)
        put(ws, r, 5, f"=D{r}/1E9", "0.000", False)
        put(ws, r, 6, f"=D{r}/D{y}", "0.00\"×\"", False)
    note(ws, FIRST + len(ROWS) + 1, "Counted with hooks on every convolution. For train6 and anytime_v8 every earlier "
                                    "head is included; for anytime_v8_fast only the layers that head needs.")


def per_class_sheet(wb):
    classes = list(R["accuracy"]["yolov8m"]["final"]["per_class_ap50"])
    ws = sheet(wb, "Per-class AP50", f"AP50 of each model's Final head for each class ({N_IMG} test images)",
               SRC + " accuracy.<model>.final.per_class_ap50",
               ["Class", ""] + [NAME[m] for m in MODELS] + ["Best"], [16, 2, 16, 14, 16, 12, 12, 12])
    for i, c in enumerate(classes):
        r = FIRST + i
        ws.cell(row=r, column=1, value=c).font = BOLD
        for j, m in enumerate(MODELS):
            v = R["accuracy"][m]["final"]["per_class_ap50"][c]
            put(ws, r, 3 + j, v if v is not None else "not in test images", "0.0000", v is not None)
        if R["accuracy"]["yolov8m"]["final"]["per_class_ap50"][c] is not None:
            put(ws, r, 8, f"=INDEX($C$4:$G$4,MATCH(MAX(C{r}:G{r}),C{r}:G{r},0))", "@", False)


def int8_sheet(wb):
    q = int8_results()
    ws = sheet(wb, "INT8", "mAP50 in FP32 and INT8, CPU (ONNX Runtime), three anytime models",
               "Sources: runs/anytime_interrupt/quantization/results.json, runs/anytime_v8/study_v8/results.json, "
               "runs/anytime_v8/study_fast/results.json (accuracy.cpu_fp32 / cpu_int8)",
               ["Model", "Head", "CPU FP32", "CPU INT8", "FP32 − INT8", "INT8 loss (% of FP32)"],
               [26, 10, 12, 12, 12, 14])
    r = FIRST
    for m in ANYTIME:
        for k, (f32, i8) in q[m].items():
            model_head(ws, r, m, k)
            put(ws, r, 3, f32, "0.0000", True)
            put(ws, r, 4, i8, "0.0000", True)
            put(ws, r, 5, f"=C{r}-D{r}", "0.0000", False)
            put(ws, r, 6, f"=E{r}/C{r}", "0.0%", False)
            r += 1
    note(ws, r + 1, "Static QDQ: 8-bit weights per output channel, 8-bit activations, MinMax on 100 calibration "
                    "images; 150 test images; no retraining. YOLOv8n and YOLOv8m were not quantised.")


def weight_sheet(wb):
    w = weight_ranges()
    ws = sheet(wb, "Weight ranges", "Largest folded weight per layer (BatchNorm folded in, DFL layers left out)",
               "Sources: layer_weight_stats.csv (folded_abs_max) in docs/weight_distribution, "
               "docs/anytime_v8/weight_distribution, docs/anytime_v8_fast/weight_distribution",
               ["Model", "", "Layers", "Smallest", "Largest", "Largest ÷ smallest", "Smallest layer", "Largest layer"],
               [26, 2, 10, 12, 12, 14, 26, 26])
    for i, m in enumerate(ANYTIME):
        r = FIRST + i
        ws.cell(row=r, column=1, value=NAME[m]).font = BOLD
        lo = min(w[m], key=lambda t: t[1])
        hi = max(w[m], key=lambda t: t[1])
        put(ws, r, 3, len(w[m]), "0", True)
        put(ws, r, 4, lo[1], "0.0000", True)
        put(ws, r, 5, hi[1], "0.0000", True)
        put(ws, r, 6, f"=E{r}/D{r}", "0\"×\"", False)
        put(ws, r, 7, lo[0], "@", True)
        put(ws, r, 8, hi[0], "@", True)


def readme_sheet(wb):
    ws = wb.active
    ws.title = "Read me"
    rows = [
        ("Five detectors compared: measured results", TITLE),
        ("train6 (GELAN, 4 heads), anytime_v8, anytime_v8_fast, YOLOv8n, YOLOv8m — measured in one run on the same "
         f"GPU (NVIDIA GTX 1650 laptop) and the same {N_IMG} bdd1500 test images.", BLACK),
        ("Checkpoints: " + "; ".join(f"{m}: {R['checkpoints'][m]}" for m in MODELS), BLACK),
        ("train6 was trained on 65,000 images (bdd70k); the other four on the same 1,200 bdd1500 images.", BLACK),
        ("", BLACK),
        ("Sheets", BOLD),
        ("Accuracy: mAP50, mAP50-95, precision and recall of every head", BLACK),
        ("Time: image-to-boxes time and network-only time up to every head", BLACK),
        ("FPS: frames per second with no interrupt", BLACK),
        ("Size: parameters and multiply-adds up to every head", BLACK),
        ("Per-class AP50: each model's Final head for each class", BLACK),
        ("INT8: FP32 against INT8 accuracy of the three anytime models", BLACK),
        ("Weight ranges: spread of the largest folded weight between layers", BLACK),
        ("", BLACK),
        ("Colours", BOLD),
        ("Blue text = measured value, copied at full precision from the source file named at the top of each sheet.",
         BLUE),
        ("Black text = Excel formula calculated from the blue values.", BLACK),
        ("", BLACK),
        (f"All timing on mains power (results.json on_battery = {str(R['on_battery']).lower()}).", BLACK),
        ("Regenerate: .venv/Scripts/python docs/make_model_comparison_xlsx.py "
         "(charts: docs/make_model_comparison_charts.py).", NOTE),
    ]
    for r, (text, font) in enumerate(rows, start=1):
        ws.cell(row=r, column=1, value=text).font = font
    ws.column_dimensions["A"].width = 120


def main():
    wb = Workbook()
    readme_sheet(wb)
    accuracy_sheet(wb)
    time_sheet(wb)
    fps_sheet(wb)
    size_sheet(wb)
    per_class_sheet(wb)
    int8_sheet(wb)
    weight_sheet(wb)
    wb.calculation.fullCalcOnLoad = True
    path = os.path.join(OUT, "model_comparison.xlsx")
    wb.save(path)
    print("saved", path)


if __name__ == "__main__":
    main()
