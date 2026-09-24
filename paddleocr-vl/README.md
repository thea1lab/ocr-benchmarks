# paddleocr-vl

[PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6), 0.9B,
Apache-2.0. The prompt is the official `OCR:`.

Paddle's published page score uses their layout pipeline and then this model.
This runner is the single-prompt transformers path.

> Part of [`ocr-benchmarks`](../README.md). Pages live in [`../corpus`](../corpus/README.md).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python ocr.py ../corpus/pages/table.png
```

From the repo root: `python run.py paddleocr-vl`.
