"""Writes docs/anytime_v8_fast/parameter_study/parameter_study.xlsx: the measured values behind the
charts of the fast anytime model on YOLOv8n, one sheet per output, with differences and ratios as formulas.

Run from the workspace root:
    .venv/Scripts/python docs/make_anytime_v8_fast_xlsx.py
    .venv/Scripts/python docs/make_anytime_v8_fast_xlsx.py --v8     (anytime_v8, every head on every frame)

Measured values are copied at full precision from runs/anytime_v8/study_fast/results.json and
runs/anytime_v8/evaluation_fast.json (blue text); every derived number is an Excel formula (black text).
"""
import json
import os
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from make_parameter_study_xlsx import BLACK, BLUE, BOLD, BOX, HEAD_FILL, NOTE, TITLE, note, put  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

V8 = os.path.join(ROOT, "runs", "anytime_v8")
LAZY = "--v8" not in sys.argv   # fast model: early heads only on interrupt; --v8: every head on every frame
STUDY_NAME = "study_fast" if LAZY else "study_v8"
EVAL_NAME = "evaluation_fast.json" if LAZY else "evaluation.json"
MODEL_NAME = "fast anytime YOLOv8n" if LAZY else "anytime YOLOv8n"
WHEN = "on interrupt" if LAZY else "every frame"
STUDY_DIR = os.path.join(V8, STUDY_NAME)
OUT = os.path.join(HERE, "anytime_v8_fast" if LAZY else "anytime_v8", "parameter_study", "parameter_study.xlsx")
HEADS = ["Head 1", "Head 2", "Final"]
KEYS = ["early_head_1", "early_head_2", "final"]
FIRST = 5
LAST = FIRST + len(HEADS) - 1

with open(os.path.join(STUDY_DIR, "results.json")) as f:
    S = json.load(f)
with open(os.path.join(V8, EVAL_NAME)) as f:
    E = json.load(f)
with open(os.path.join(V8, "evaluation.json")) as f:
    P = json.load(f)


def sheet(wb, name, title, source, headers, widths, rows=HEADS):
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
    ws.freeze_panes = "B5"
    for r, h in enumerate(rows, start=FIRST):
        cell = ws.cell(row=r, column=1, value=h)
        cell.font = BOLD
        cell.border = BOX
    return ws


def accuracy_sheet(wb, name, metric, label):
    ws = sheet(wb, name, f"Output: {label} on {S['n_images']} unseen test images ({MODEL_NAME})",
               f"Source: runs/anytime_v8/{STUDY_NAME}/results.json, accuracy.<version>.<head>." + metric
               + f"; YOLOv8n from runs/anytime_v8/{EVAL_NAME}",
               ["Head", "GPU FP32", "GPU FP16", "CPU FP32", "CPU INT8", "FP16 − FP32 (GPU)", "FP32 − INT8 (CPU)",
                "INT8 loss (% of FP32)", "Share of Final head, FP32 (GPU)", "YOLOv8n, GPU FP32",
                "Share of YOLOv8n"],
               [10, 12, 12, 12, 12, 14, 14, 14, 16, 14, 12])
    y = E["accuracy"]["yolov8n"]["final"][metric]
    for i, k in enumerate(KEYS):
        r = FIRST + i
        for col, tag in ((2, "gpu_fp32"), (3, "gpu_fp16"), (4, "cpu_fp32"), (5, "cpu_int8")):
            put(ws, r, col, S["accuracy"][tag][k][metric], "0.0000", True)
        put(ws, r, 6, f"=C{r}-B{r}", "+0.0000;-0.0000;0.0000", False)
        put(ws, r, 7, f"=D{r}-E{r}", "0.0000", False)
        put(ws, r, 8, f"=IF(D{r}=0,\"\",G{r}/D{r})", "0.0%", False)
        put(ws, r, 9, f"=IF($B${LAST}=0,\"\",B{r}/$B${LAST})", "0.0%", False)
        put(ws, r, 10, y, "0.0000", True)
        put(ws, r, 11, f"=IF(J{r}=0,\"\",B{r}/J{r})", "0.0%", False)
    note(ws, LAST + 2, "FP16 was run only on the GPU and INT8 only on the CPU (INT8 on this GPU needs TensorRT, "
                       "not installed). INT8: ONNX Runtime static QDQ quantisation, per-channel 8-bit weights, "
                       "8-bit activations, MinMax calibration on 100 images.")
    note(ws, LAST + 3, "The Final head is the frozen fine-tuned YOLOv8n, so in FP32 it equals YOLOv8n exactly." if LAZY
         else "YOLOv8n is the separately fine-tuned YOLOv8n, measured in the same evaluation run.")
    note(ws, LAST + 4, "Blue = measured value copied from the source file at full precision. Black = formula.")


def per_class_sheet(wb):
    classes = list(S["accuracy"]["gpu_fp32"]["final"]["per_class_ap50"])
    ws = sheet(wb, "Per-class AP50", "AP50 of each class and head (GPU FP32 and CPU INT8)",
               f"Source: runs/anytime_v8/{STUDY_NAME}/results.json, accuracy.<version>.<head>.per_class_ap50; "
               "empty = class not in the test images",
               ["Class", "Head 1 FP32", "Head 2 FP32", "Final FP32", "Head 1 INT8", "Head 2 INT8", "Final INT8",
                "Final: FP32 − INT8"], [16, 12, 12, 12, 12, 12, 12, 14], rows=classes)
    for i, c in enumerate(classes):
        r = FIRST + i
        for j, (tag, k) in enumerate([("gpu_fp32", k) for k in KEYS] + [("cpu_int8", k) for k in KEYS]):
            v = S["accuracy"][tag][k]["per_class_ap50"][c]
            if v is not None:
                put(ws, r, 2 + j, v, "0.0000", True)
        if S["accuracy"]["gpu_fp32"]["final"]["per_class_ap50"][c] is not None:
            put(ws, r, 8, f"=D{r}-G{r}", "0.0000", False)
    note(ws, FIRST + len(classes) + 1, "Blue = measured value. Black = formula.")


def latency_sheet(wb):
    if not LAZY:
        return latency_sheet_eager(wb)
    ws = sheet(wb, "Latency", "Output: time until the answer, when an interrupt asks for that head (mains power)",
               f"Source: runs/anytime_v8/{STUDY_NAME}/results.json latency.<version>.<head>; earlier model from "
               "runs/anytime_v8/evaluation.json pipeline.anytime_v8",
               ["Head", "GPU FP32 median (ms)", "GPU FP32 90th pct (ms)", "GPU FP16 median (ms)",
                "GPU FP16 90th pct (ms)", "GPU FP16 ÷ FP32", "CPU FP32 (ms)", "CPU INT8 (ms)", "CPU FP32 ÷ INT8",
                "Earlier model, GPU FP32 (ms)", "Saved vs earlier (ms)"],
               [10, 13, 13, 13, 13, 12, 12, 12, 12, 14, 14])
    for i, k in enumerate(KEYS):
        r = FIRST + i
        put(ws, r, 2, S["latency"]["gpu_fp32"][k]["median_ms"], "0.00", True)
        put(ws, r, 3, S["latency"]["gpu_fp32"][k]["p90_ms"], "0.00", True)
        put(ws, r, 4, S["latency"]["gpu_fp16"][k]["median_ms"], "0.00", True)
        put(ws, r, 5, S["latency"]["gpu_fp16"][k]["p90_ms"], "0.00", True)
        put(ws, r, 6, f"=IF(B{r}=0,\"\",D{r}/B{r})", "0.00\"×\"", False)
        put(ws, r, 7, S["latency"]["cpu_fp32"][k]["median_ms"], "0.00", True)
        put(ws, r, 8, S["latency"]["cpu_int8"][k]["median_ms"], "0.00", True)
        put(ws, r, 9, f"=IF(H{r}=0,\"\",G{r}/H{r})", "0.00\"×\"", False)
        put(ws, r, 10, P["pipeline"]["anytime_v8"][k]["median_ms"], "0.00", True)
        put(ws, r, 11, f"=J{r}-B{r}", "+0.00;-0.00;0.00", False)
    r = LAST + 2
    for c, (text, v, fmt) in enumerate((("Frames per second, GPU FP32 (no interrupt)", S["fps"]["gpu_fp32"]["median"], "0.0"),
                                        ("Frames per second, GPU FP16 (no interrupt)", S["fps"]["gpu_fp16"]["median"], "0.0"),
                                        ("Frames per second, earlier model", P["fps"]["anytime_v8"]["median"], "0.0"),
                                        ("YOLOv8n pipeline (ms), evaluation_fast.json",
                                         E["pipeline"]["yolov8n"]["final"]["median_ms"], "0.00"),
                                        ("Interrupt response, Head 1 (ms), evaluation_fast.json",
                                         E["lazy"]["interrupt_response_ms"]["early_head_1"]["median_ms"], "0.00"),
                                        ("Interrupt response, Head 2 (ms), evaluation_fast.json",
                                         E["lazy"]["interrupt_response_ms"]["early_head_2"]["median_ms"], "0.00"))):
        ws.cell(row=r + c, column=1, value=text).font = BOLD
        put(ws, r + c, 2, v, fmt, True)
    ws.column_dimensions["A"].width = 46
    n = r + 7
    note(ws, n, "GPU (GTX 1650): image in to boxes on the CPU (resize, network, NMS), first 60 test images, two "
                "untimed warm-up passes; only the layers the asked head needs run (early heads only on interrupt).")
    note(ws, n + 1, "CPU: network only (ONNX Runtime) on a graph cut to the layers that head needs, 60 test images, "
                    "15 warm-up runs, median of 3 interleaved repeats. Compare formats within one device only.")
    note(ws, n + 2, "Blue = measured value copied from the source file at full precision. Black = formula.")


def latency_sheet_eager(wb):
    ws = sheet(wb, "Latency", "Output: time until each head's answer, every head computed in order (mains power)",
               f"Source: runs/anytime_v8/{STUDY_NAME}/results.json latency.<version>.<head>; YOLOv8n and the "
               f"150-image pipeline from runs/anytime_v8/{EVAL_NAME}",
               ["Head", "GPU FP32 median (ms)", "GPU FP32 90th pct (ms)", "GPU FP16 median (ms)",
                "GPU FP16 90th pct (ms)", "GPU FP16 ÷ FP32", "CPU FP32 (ms)", "CPU INT8 (ms)", "CPU FP32 ÷ INT8",
                "GPU FP32, 150 images (ms)", "YOLOv8n, 150 images (ms)", "Earlier than YOLOv8n (ms)"],
               [10, 13, 13, 13, 13, 12, 12, 12, 12, 14, 14, 14])
    for i, k in enumerate(KEYS):
        r = FIRST + i
        put(ws, r, 2, S["latency"]["gpu_fp32"][k]["median_ms"], "0.00", True)
        put(ws, r, 3, S["latency"]["gpu_fp32"][k]["p90_ms"], "0.00", True)
        put(ws, r, 4, S["latency"]["gpu_fp16"][k]["median_ms"], "0.00", True)
        put(ws, r, 5, S["latency"]["gpu_fp16"][k]["p90_ms"], "0.00", True)
        put(ws, r, 6, f"=IF(B{r}=0,\"\",D{r}/B{r})", "0.00\"×\"", False)
        put(ws, r, 7, S["latency"]["cpu_fp32"][k]["median_ms"], "0.00", True)
        put(ws, r, 8, S["latency"]["cpu_int8"][k]["median_ms"], "0.00", True)
        put(ws, r, 9, f"=IF(H{r}=0,\"\",G{r}/H{r})", "0.00\"×\"", False)
        put(ws, r, 10, E["pipeline"]["anytime_v8"][k]["median_ms"], "0.00", True)
        put(ws, r, 11, E["pipeline"]["yolov8n"]["final"]["median_ms"], "0.00", True)
        put(ws, r, 12, f"=K{r}-J{r}", "+0.00;-0.00;0.00", False)
    r = LAST + 2
    for c, (text, v) in enumerate((("Frames per second, GPU FP32 (60 images)", S["fps"]["gpu_fp32"]["median"]),
                                   ("Frames per second, GPU FP16 (60 images)", S["fps"]["gpu_fp16"]["median"]),
                                   ("Frames per second, GPU FP32 (150 images)", E["fps"]["anytime_v8"]["median"]),
                                   ("Frames per second, YOLOv8n (150 images)", E["fps"]["yolov8n"]["median"]))):
        ws.cell(row=r + c, column=1, value=text).font = BOLD
        put(ws, r + c, 2, v, "0.0", True)
    ws.column_dimensions["A"].width = 46
    n = r + 5
    note(ws, n, "GPU (GTX 1650): image in to boxes on the CPU (resize, network, NMS); each head's time is when its "
                "boxes are ready inside one full pass, so earlier heads and their NMS are included.")
    note(ws, n + 1, "CPU: network only (ONNX Runtime) on the graph cut after that head (earlier heads included), "
                    "60 test images, 15 warm-up runs, median of 3 interleaved repeats. Compare within one device only.")
    note(ws, n + 2, "Blue = measured value copied from the source file at full precision. Black = formula.")


def size_sheet(wb):
    ws = sheet(wb, "File size", "Output: model file size for each head (ONNX file with exactly the layers it needs)",
               f"Source: runs/anytime_v8/{STUDY_NAME}/results.json size_mb (os.path.getsize of "
               f"runs/anytime_v8/{STUDY_NAME}/<format>_{'head<k>_only' if LAZY else 'upto_head<k>'}.onnx), "
               "in MB = 1,000,000 bytes",
               ["Head", "FP32 (MB)", "FP16 (MB)", "INT8 (MB)", "FP32 ÷ FP16", "FP32 ÷ INT8"],
               [22, 11, 11, 11, 12, 12], rows=HEADS + ["Whole model (all heads)"])
    for i in range(len(HEADS) + 1):
        r = FIRST + i
        for col, fmt in ((2, "fp32"), (3, "fp16"), (4, "int8")):
            v = S["size_mb"][fmt]["graph_for_head"][i] if i < len(HEADS) else S["size_mb"][fmt]["whole_model"]
            put(ws, r, col, v, "0.000", True)
        put(ws, r, 5, f"=IF(C{r}=0,\"\",B{r}/C{r})", "0.00\"×\"", False)
        put(ws, r, 6, f"=IF(D{r}=0,\"\",B{r}/D{r})", "0.00\"×\"", False)
    r = FIRST + len(HEADS) + 2
    for c, (text, v) in enumerate((("PyTorch state_dict, FP32 (MB)", S["size_mb"]["pytorch_state_dict"]["fp32"]),
                                   ("PyTorch state_dict, FP16 (MB)", S["size_mb"]["pytorch_state_dict"]["fp16"]),
                                   ("Checkpoint file last.pt (MB)", S["size_mb"]["checkpoint_file"]))):
        ws.cell(row=r + c, column=1, value=text).font = BOLD
        put(ws, r + c, 2, v, "0.000", True)
    note(ws, r + 4, "The Final head's file has no early heads (they run only on interrupt), so it is the "
                    "YOLOv8n network alone. File size does not depend on the hardware." if LAZY else
         "Each file is the model cut after that head, so earlier heads are included (they always run first). "
         "File size does not depend on the hardware.")
    note(ws, r + 5, "Blue = measured value. Black = formula.")


def compute_sheet(wb):
    c = S["compute"]
    stages = [("backbone", "Backbone (layers 0-9)"), ("early_head_1", f"Head 1 ({WHEN})"),
              ("fpn", "FPN (layers 10-15)"), ("early_head_2", f"Head 2 ({WHEN})"),
              ("pan", "PAN (layers 16-21)"), ("final_head", "Final head (layer 22)")]
    ws = sheet(wb, "Multiply-adds", "Output: multiply-adds and parameters of each stage, one 640 × 640 image",
               f"Source: runs/anytime_v8/{STUDY_NAME}/results.json compute (forward hooks on every convolution)",
               ["Stage", "Multiply-adds", "Billions", "Parameters",
                "Share of a frame with no interrupt" if LAZY else "Share of a frame"],
               [26, 18, 12, 14, 18], rows=[n for _, n in stages])
    rows = {}
    for i, (k, _) in enumerate(stages):
        r = FIRST + i
        rows[k] = r
        put(ws, r, 2, c["macs_per_stage"][k], "#,##0", True)
        put(ws, r, 3, f"=B{r}/1000000000", "0.000", False)
        put(ws, r, 4, c["parameters_per_stage"][k], "#,##0", True)
    frame = "+".join(f"B{rows[k]}" for k in ("backbone", "fpn", "pan", "final_head")
                     + (() if LAZY else ("early_head_1", "early_head_2")))
    for k, r in rows.items():
        put(ws, r, 5, f"=B{r}/({frame})", "0.0%", False)
    r = FIRST + len(stages) + 1
    b, f_, p_, h1, h2, fin = (f"B{rows[k]}" for k in ("backbone", "fpn", "pan", "early_head_1", "early_head_2",
                                                       "final_head"))
    totals = ((("Up to Head 1 (backbone + Head 1)", f"={b}+{h1}"),
               ("Up to Head 2 (backbone + FPN + Head 2)", f"={b}+{f_}+{h2}"),
               ("Final, no interrupt (= YOLOv8n)", f"={b}+{f_}+{p_}+{fin}"),
               ("Final, every head computed", f"={b}+{f_}+{p_}+{fin}+{h1}+{h2}")) if LAZY else
              (("Up to Head 1 (backbone + Head 1)", f"={b}+{h1}"),
               ("Up to Head 2 (backbone + Head 1 + FPN + Head 2)", f"={b}+{h1}+{f_}+{h2}"),
               ("Final, every head computed (how this model runs)", f"={b}+{f_}+{p_}+{fin}+{h1}+{h2}"),
               ("Final without the early heads", f"={b}+{f_}+{p_}+{fin}")))
    for j, (text, formula) in enumerate(totals):
        ws.cell(row=r + j, column=1, value=text).font = BOLD
        put(ws, r + j, 2, formula, "#,##0", False)
        put(ws, r + j, 3, f"=B{r + j}/1000000000", "0.000", False)
    ws.cell(row=r + 5, column=1, value="Parameters, total").font = BOLD
    put(ws, r + 5, 4, c["parameters_total"], "#,##0", True)
    note(ws, r + 7, "Counted for one 640 × 640 image. The same for FP32, FP16 and INT8 and on GPU and CPU.")
    note(ws, r + 8, "Blue = measured (counted) value. Black = formula.")


def readme_sheet(wb):
    ws = wb.active
    ws.title = "Read me"
    rows = [
        (f"{MODEL_NAME[0].upper() + MODEL_NAME[1:]}: measured results", TITLE),
        ("Model: runs/anytime_v8/bdd1500_fast/weights/last.pt (frozen fine-tuned YOLOv8n + two depthwise early "
         "heads, early heads only on interrupt). No retraining for FP16 or INT8." if LAZY else
         "Model: runs/anytime_v8/bdd1500/weights/best.pt (YOLOv8n trained together with two early heads, every "
         "head on every frame). No retraining for FP16 or INT8.", BLACK),
        ("", BLACK),
        ("Inputs", BOLD),
        ("Head that answers: Head 1, Head 2, Final", BLACK),
        ("Number format: FP32 (as trained), FP16, INT8", BLACK),
        ("Hardware: GPU (NVIDIA GTX 1650 laptop) and CPU (same laptop)", BLACK),
        ("FP16 was run only on the GPU and INT8 only on the CPU; INT8 on this GPU needs TensorRT, which is not installed.", BLACK),
        ("", BLACK),
        ("Sheets", BOLD),
        ("mAP50, mAP50-95: accuracy of each head in each format, compared with YOLOv8n", BLACK),
        ("Per-class AP50: each class and head, FP32 and INT8", BLACK),
        ("Latency: time until the answer when an interrupt asks for that head; frames per second" if LAZY else
         "Latency: time until each head's answer in one full pass; frames per second", BLACK),
        ("File size: ONNX file with exactly the layers each head needs", BLACK),
        ("Multiply-adds: computation and parameters of every stage", BLACK),
        ("", BLACK),
        ("How the values were measured", BOLD),
        (f"Accuracy: first {S['n_images']} images of the bdd1500 test split, never used in training; input "
         f"{S['input_shape'][2]} × {S['input_shape'][3]} (letterboxed 1280 × 720 frames).", BLACK),
        ("INT8: ONNX Runtime static QDQ quantisation, per-channel 8-bit weights, 8-bit activations, MinMax "
         "calibration on 100 images.", BLACK),
        ("GPU timing: first 60 test images, image in to boxes on the CPU, two untimed warm-up passes, median and 90th percentile.", BLACK),
        ("CPU timing: first 60 test images, network only, 15 warm-up runs, 3 interleaved repeats, median of the 3 medians.", BLACK),
        (f"All timing on mains power (results.json on_battery = {str(S['on_battery']).lower()}).", BLACK),
        ("", BLACK),
        ("Colours", BOLD),
        ("Blue text = measured value, copied at full precision from the source file named at the top of each sheet.", BLUE),
        ("Black text = Excel formula calculated from the blue values.", BLACK),
        ("", BLACK),
        ("Regenerate: .venv/Scripts/python docs/make_anytime_v8_fast_xlsx.py" + ("" if LAZY else " --v8")
         + " (charts: docs/make_anytime_v8" + ("_fast" if LAZY else "") + "_charts.py).", NOTE),
    ]
    for r, (text, font) in enumerate(rows, start=1):
        ws.cell(row=r, column=1, value=text).font = font
    ws.column_dimensions["A"].width = 120


def main():
    wb = Workbook()
    readme_sheet(wb)
    accuracy_sheet(wb, "mAP50", "mAP50", "mAP50")
    accuracy_sheet(wb, "mAP50-95", "mAP50_95", "mAP50-95")
    per_class_sheet(wb)
    latency_sheet(wb)
    size_sheet(wb)
    compute_sheet(wb)
    wb.calculation.fullCalcOnLoad = True
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wb.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
