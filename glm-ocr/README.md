# glm-ocr

[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), 0.9B, MIT license.
This runner sends the official `Text Recognition:` prompt to the checkpoint.

The number on OmniDocBench uses Z.ai's SDK, which puts PP-DocLayout-V3 in front
of this model. That layout step is not part of this run.

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

From the repo root: `python run.py glm-ocr`.
