# teleocr

[TeleOCR](https://huggingface.co/StarDoc-AI/TeleOCR), 1.2B, Apache-2.0.
The prompt is the model card's text prompt: "Please output the text content from
the image." The table and formula prompts are not used.

> Part of [`ocr-benchmarks`](../README.md). Pages live in [`../corpus`](../corpus/README.md).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python ocr.py ../images/chocolate.jpeg
```

From the repo root: `python run.py teleocr`.
