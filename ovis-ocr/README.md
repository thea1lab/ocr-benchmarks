# ovis-ocr

[OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2), a 0.8B end-to-end page parser.
The prompt is the model's Markdown prompt, with thinking turned off. Decoding is
greedy, capped at 2048 new tokens.

> Part of [`ocr-benchmarks`](../README.md). Pages live in [`../corpus`](../corpus/README.md).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The PyTorch wheel resolved here is the CUDA 13 build (`torch==2.14.0`). A GPU is
optional; without one the script stays on CPU.

## Run

```bash
python ocr.py ../corpus/pages/clean-letter.png
```

First run downloads the weights. From the repo root, `python run.py ovis-ocr`
times the shared pages.
