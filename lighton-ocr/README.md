# lighton-ocr

[LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B), an end-to-end
1B model. Input is the image alone, matching the model card. Decoding is greedy,
capped at 2048 new tokens.

> Part of [`ocr-benchmarks`](../README.md). Pages live in [`../corpus`](../corpus/README.md).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python ocr.py ../corpus/pages/clean-letter.png
```

From the repo root: `python run.py lighton-ocr`.
