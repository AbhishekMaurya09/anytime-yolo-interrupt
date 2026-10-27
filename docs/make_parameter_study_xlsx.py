"""Writes docs/parameter_study/parameter_study.xlsx: the measured values behind the five
parameter-study graphs, one sheet per output, with differences and ratios as formulas.

Run from the workspace root:
    .venv/Scripts/python docs/make_parameter_study_xlsx.py

Measured values are copied at full precision from the files under runs/anytime_interrupt/
(blue text); every derived number is an Excel formula (black text).
"""
import json
import os

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs", "anytime_interrupt")
QDIR = os.path.join(RUNS, "quantization")
OUT = os.path.join(HERE, "parameter_study", "parameter_study.xlsx")

HEADS = ["Head 1", "Head 2", "Head 3", "Final"]
KEYS = ["early_head_1", "early_head_2", "early_head_3", "final"]

FONT = "Arial"
BLUE = Font(name=FONT, color="0000FF")
BLACK = Font(name=FONT, color="000000")
BOLD = Font(name=FONT, bold=True)
TITLE = Font(name=FONT, bold=True, size=13)
NOTE = Font(name=FONT, italic=True, color="595959", size=9)
HEAD_FILL = PatternFill("solid", fgColor="DCE7F5")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
FIRST = 5  # first data row (Head 1); Final is FIRST + 3

with open(os.path.join(QDIR, "results.json")) as f:
    Q = json.load(f)
with open(os.path.join(QDIR, "cpu_latency_repeat.json")) as f:
    CPU = json.load(f)
with open(os.path.join(RUNS, "model_sizes.json")) as f:
    SIZES = json.load(f)


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
    ws.freeze_panes = "B5"
    for r, h in enumerate(HEADS, start=FIRST):
        cell = ws.cell(row=r, column=1, value=h)
        cell.font = BOLD
        cell.border = BOX
    return ws


def put(ws, row, col, value, fmt, measured):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = BLUE if measured else BLACK
    cell.number_format = fmt
    cell.border = BOX
    return cell


def note(ws, row, text):
    ws.cell(row=row, column=1, value=text).font = NOTE


def accuracy_sheet(wb, name, metric, label):
    ws = sheet(wb, name, f"Output: {label} on 150 unseen test images (train6)",
               "Source: runs/anytime_interrupt/quantization/results.json, accuracy.<version>.<head>." + metric,
               ["Head", "GPU FP32", "GPU FP16", "CPU FP32", "CPU INT8", "FP16 − FP32 (GPU)", "FP32 − INT8 (CPU)",
                "INT8 loss (% of FP32)", "Share of Final head, FP32 (GPU)"],
               [10, 12, 12, 12, 12, 14, 14, 14, 16])
    for i, k in enumerate(KEYS):
        r = FIRST + i
        for col, tag in ((2, "gpu_fp32"), (3, "gpu_fp16"), (4, "cpu_fp32"), (5, "cpu_int8")):
            put(ws, r, col, Q["accuracy"][tag][k][metric], "0.0000", True)
        put(ws, r, 6, f"=C{r}-B{r}", "+0.0000;-0.0000;0.0000", False)
        put(ws, r, 7, f"=D{r}-E{r}", "0.0000", False)
        put(ws, r, 8, f"=IF(D{r}=0,\"\",G{r}/D{r})", "0.0%", False)
        put(ws, r, 9, f"=IF($B${FIRST + 3}=0,\"\",B{r}/$B${FIRST + 3})", "0.0%", False)
    note(ws, FIRST + 5, "FP16 was run only on the GPU and INT8 only on the CPU (INT8 on this GPU needs TensorRT, "
                        "not installed). INT8: ONNX Runtime static quantisation, per-channel 8-bit weights, "
                        "8-bit activations, MinMax calibration on 100 images.")
    note(ws, FIRST + 6, "Blue = measured value copied from the source file at full precision. Black = formula.")


def latency_sheet(wb):
    ws = sheet(wb, "Latency", "Output: time until each head's answer is ready (train6, laptop on mains power)",
               "Sources: GPU from results.json latency.<version>.<head>; CPU from cpu_latency_repeat.json "
               "(median of 3 interleaved repeats); single CPU run from results.json latency.cpu_<version>",
               ["Head", "GPU FP32 median (ms)", "GPU FP32 90th pct (ms)", "GPU FP16 median (ms)",
                "GPU FP16 90th pct (ms)", "GPU FP16 ÷ FP32 (median)", "CPU FP32 (ms), 3 repeats",
                "CPU INT8 (ms), 3 repeats", "CPU FP32 ÷ INT8", "CPU FP32 (ms), single run",
                "CPU INT8 (ms), single run"],
               [10, 13, 13, 13, 13, 13, 14, 14, 12, 14, 14])
    for i, k in enumerate(KEYS):
        r = FIRST + i
        put(ws, r, 2, Q["latency"]["gpu_fp32"][k]["median_ms"], "0.00", True)
        put(ws, r, 3, Q["latency"]["gpu_fp32"][k]["p90_ms"], "0.00", True)
        put(ws, r, 4, Q["latency"]["gpu_fp16"][k]["median_ms"], "0.00", True)
        put(ws, r, 5, Q["latency"]["gpu_fp16"][k]["p90_ms"], "0.00", True)
        put(ws, r, 6, f"=IF(B{r}=0,\"\",D{r}/B{r})", "0.00\"×\"", False)
        put(ws, r, 7, CPU["fp32"][i], "0.00", True)
        put(ws, r, 8, CPU["int8"][i], "0.00", True)
        put(ws, r, 9, f"=IF(H{r}=0,\"\",G{r}/H{r})", "0.00\"×\"", False)
        put(ws, r, 10, Q["latency"]["cpu_fp32"][k]["median_ms"], "0.00", True)
        put(ws, r, 11, Q["latency"]["cpu_int8"][k]["median_ms"], "0.00", True)
    note(ws, FIRST + 5, "GPU (GTX 1650): whole pipeline (resize + network + post-processing), first 60 test images, "
                        "two untimed warm-up passes. CPU: network only (ONNX Runtime), first 60 test images, "
                        "15 warm-up runs.")
    note(ws, FIRST + 6, "The graph uses the 3-repeat CPU columns. The single-run columns are from the same script "
                        "run; single repeats varied by about ±15%. GPU and CPU columns measure different things: "
                        "compare formats within one device only.")
    note(ws, FIRST + 7, "Blue = measured value copied from the source file at full precision. Black = formula.")


def size_sheet(wb):
    ws = sheet(wb, "File size", "Output: model file size up to each head (ONNX file cut after that head)",
               "Source: size on disk of runs/anytime_interrupt/quantization/{cpu_fp32,fp16,cpu_int8}_upto_head<k>.onnx "
               "(measured with os.path.getsize; same files as runs/anytime_interrupt/model_sizes.json)",
               ["Head", "FP32 (bytes)", "FP16 (bytes)", "INT8 (bytes)", "FP32 (MB)", "FP16 (MB)", "INT8 (MB)",
                "FP32 ÷ FP16", "FP32 ÷ INT8"],
               [16, 14, 14, 14, 11, 11, 11, 12, 12])
    files = [("cpu_fp32_upto_head{}.onnx", 2), ("fp16_upto_head{}.onnx", 3), ("cpu_int8_upto_head{}.onnx", 4)]
    for i in range(4):
        r = FIRST + i
        for pattern, col in files:
            put(ws, r, col, os.path.getsize(os.path.join(QDIR, pattern.format(i + 1))), "#,##0", True)
        for col, src in ((5, "B"), (6, "C"), (7, "D")):
            put(ws, r, col, f"={src}{r}/1000000", "0.00", False)
        put(ws, r, 8, f"=IF(C{r}=0,\"\",B{r}/C{r})", "0.00\"×\"", False)
        put(ws, r, 9, f"=IF(D{r}=0,\"\",B{r}/D{r})", "0.00\"×\"", False)
    r = FIRST + 4
    ws.cell(row=r, column=1, value="Whole model file").font = BOLD
    ws.cell(row=r, column=1).border = BOX
    for name, col in (("model_fp32.onnx", 2), ("model_fp16.onnx", 3), ("model_int8.onnx", 4)):
        put(ws, r, col, os.path.getsize(os.path.join(QDIR, name)), "#,##0", True)
    for col, src in ((5, "B"), (6, "C"), (7, "D")):
        put(ws, r, col, f"={src}{r}/1000000", "0.00", False)
    put(ws, r, 8, f"=IF(C{r}=0,\"\",B{r}/C{r})", "0.00\"×\"", False)
    put(ws, r, 9, f"=IF(D{r}=0,\"\",B{r}/D{r})", "0.00\"×\"", False)
    ws.cell(row=r, column=1).comment = Comment(
        "model_fp32.onnx, model_fp16.onnx, model_int8.onnx: the full exported models. They are a few kB smaller "
        "than the 'Final' cut files, because cutting a graph rewrites its description.", "parameter study")
    note(ws, FIRST + 6, "1 MB = 1,000,000 bytes. Earlier heads are included in each file, because in an anytime "
                        "pass they always run first. File size does not depend on the hardware.")
    note(ws, FIRST + 7, "Blue = measured value. Black = formula.")


def compute_sheet(wb):
    ws = sheet(wb, "Multiply-adds", "Output: multiply-adds up to each head, one 640 × 640 image",
               "Source: runs/anytime_interrupt/model_sizes.json, macs_upto_head (counted on the model with "
               "forward hooks on all 151 convolutions)",
               ["Head", "Multiply-adds up to this head", "Added by this step", "Share of full pass",
                "Up to this head (billions)"],
               [10, 20, 18, 12, 16])
    for i, v in enumerate(SIZES["macs_upto_head"]):
        r = FIRST + i
        put(ws, r, 2, v, "#,##0", True)
        put(ws, r, 3, f"=B{r}" if i == 0 else f"=B{r}-B{r - 1}", "#,##0", False)
        put(ws, r, 4, f"=IF($B${FIRST + 3}=0,\"\",B{r}/$B${FIRST + 3})", "0.0%", False)
        put(ws, r, 5, f"=B{r}/1000000000", "0.000", False)
    note(ws, FIRST + 5, "Earlier heads are included, because in an anytime pass they always run first. The count "
                        "is the same for FP32, FP16 and INT8 and on GPU and CPU. It matches "
                        "docs/accelerator_workload.csv exactly.")
    note(ws, FIRST + 6, "Blue = measured (counted) value. Black = formula.")


def readme_sheet(wb):
    ws = wb.active
    ws.title = "Read me"
    rows = [
        ("Parameter study: three inputs, five outputs", TITLE),
        ("Model: runs/anytime_interrupt/train6_bdd70k/last.pt. No retraining for FP16 or INT8.", BLACK),
        ("", BLACK),
        ("Inputs", BOLD),
        ("Head that answers: Head 1, Head 2, Head 3, Final", BLACK),
        ("Number format: FP32 (as trained), FP16, INT8", BLACK),
        ("Hardware: GPU (NVIDIA GTX 1650 laptop) and CPU (same laptop)", BLACK),
        ("FP16 was run only on the GPU and INT8 only on the CPU; INT8 on this GPU needs TensorRT, which is not installed.", BLACK),
        ("", BLACK),
        ("Outputs (one sheet each)", BOLD),
        ("mAP50: accuracy, box overlap at least 50%", BLACK),
        ("mAP50-95: stricter accuracy, overlap thresholds 50% to 95%", BLACK),
        ("Latency: time from the image arriving until each head's answer is ready", BLACK),
        ("File size: size on disk of the ONNX model cut after each head", BLACK),
        ("Multiply-adds: computation up to each head, counted on the model", BLACK),
        ("", BLACK),
        ("How the values were measured", BOLD),
        ("Accuracy: first 150 images of the bdd1500 test split, never used in training.", BLACK),
        ("INT8: ONNX Runtime static quantisation, per-channel 8-bit weights, 8-bit activations, MinMax calibration on 100 images.", BLACK),
        ("GPU timing: first 60 test images, whole pipeline, two untimed warm-up passes, median and 90th percentile.", BLACK),
        ("CPU timing: first 60 test images, network only, 15 warm-up runs, 3 interleaved repeats, median of the 3 medians.", BLACK),
        ("All timing on mains power (results.json on_battery = false).", BLACK),
        ("", BLACK),
        ("Colours", BOLD),
        ("Blue text = measured value, copied at full precision from the source file named at the top of each sheet.", BLUE),
        ("Black text = Excel formula calculated from the blue values.", BLACK),
        ("", BLACK),
        ("Regenerate: .venv/Scripts/python docs/make_parameter_study_xlsx.py (graphs: docs/make_result_charts.py).", NOTE),
    ]
    for r, (text, font) in enumerate(rows, start=1):
        ws.cell(row=r, column=1, value=text).font = font
    ws.column_dimensions["A"].width = 120


def main():
    wb = Workbook()
    readme_sheet(wb)
    accuracy_sheet(wb, "mAP50", "mAP50", "mAP50")
    accuracy_sheet(wb, "mAP50-95", "mAP50_95", "mAP50-95")
    latency_sheet(wb)
    size_sheet(wb)
    compute_sheet(wb)
    wb.calculation.fullCalcOnLoad = True   # Excel computes every formula when the file is opened
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wb.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
