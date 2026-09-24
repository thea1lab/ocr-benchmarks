# rapidocr

[RapidOCR](https://github.com/RapidAI/RapidOCR) on one image or on the shared corpus.
The wheel ships the small PP-OCRv6 detection and recognition models and runs them
with ONNX Runtime on CPU.

> Part of [`ocr-benchmarks`](../README.md). Pages live in [`../corpus`](../corpus/README.md).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python ocr.py ../corpus/pages/receipt.png
```

The text is written next to the image as `.md`, unless you pass a second path.

From the repo root, `python run.py rapidocr` times every shared page and
refreshes the dashboard.
