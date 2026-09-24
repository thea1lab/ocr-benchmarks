# ocr-benchmarks

Local OCR models, each in its own subproject, scored on the same five pages.
The pages and the numbers are in this repo. Virtualenvs and model weights are not.

## Layout

```
ocr-benchmarks/
├── images/                  # phone photo of a package
├── corpus/                  # drawn pages, transcripts, manifest
├── bench/                   # model list and the shared loader
├── run.py                   # runs the models and refreshes the results
├── RESULTS.md               # the comparison, rendered on GitHub
├── dashboard/index.html     # the same comparison as a page
├── rapidocr/                # PP-OCRv6 small, ONNX, CPU
├── ovis-ocr/                # OvisOCR2, 0.8B
├── paddleocr-vl/            # PaddleOCR-VL-1.6, 0.9B
├── glm-ocr/                 # GLM-OCR, 0.9B
├── lighton-ocr/             # LightOnOCR-2-1B, 1B
└── teleocr/                 # TeleOCR, 1.2B
```

Each subproject has its own `README.md`, `requirements.txt`, and `.venv/`.
Don't share a venv. Torch, transformers, and CUDA pins collide.

## Results

Run on an i9-12900H, 31 GB RAM, RTX 3060 Laptop (6 GB). About 2 GB of that
GPU was already in use by the desktop. Full per-page numbers, missed fields,
and output excerpts are in [`RESULTS.md`](RESULTS.md). The same table is in
[`dashboard/index.html`](dashboard/index.html).

| Model | Median s/page | Load s | Character accuracy | Fields |
| --- | ---: | ---: | ---: | ---: |
| [RapidOCR](rapidocr/) | 1.0 | 0.2 | 84% | 25/26 |
| [OvisOCR2](ovis-ocr/) | 34.7 | 40.1 | 95% | 25/26 |
| [PaddleOCR-VL-1.6](paddleocr-vl/) | 31.2 | 12.9 | 93% | 22/26 |
| [GLM-OCR](glm-ocr/) | 2.2 | 56.8 | 100% | 26/26 |
| [LightOnOCR-2-1B](lighton-ocr/) | 2.5 | 7.4 | 100% | 26/26 |
| [TeleOCR](teleocr/) | 2.0 | 4.9 | 65% | 18/26 |

Character accuracy is the transcript match on the drawn pages. Markup is
ignored, and so is a tail a model repeats after a good read. Fields are the
26 facts in [`corpus/manifest.json`](corpus/manifest.json): an order number,
a total, a footnote, a table cell. Seconds per page are warm extraction,
after the weights have loaded.

GLM-OCR and LightOnOCR-2-1B are the only two that got every fact. LightOn
is the faster of those two once it is loaded. RapidOCR is the one-second
CPU option. It reads a single column and a receipt, and it reads the two
columns on the same line.

The pages are a clean letter, a two-column page with a footnote, a table
plus an equation, a tilted receipt, and the package photo. The two hard
pages are also `corpus/paper.pdf`.

```bash
python run.py                  # every model
python run.py lighton-ocr      # one model
```

Each model needs its own `.venv` first. See that subproject's README.

## Adding a model

```bash
mkdir my-new-ocr && cd my-new-ocr
python3.12 -m venv .venv && source .venv/bin/activate
pip install ...
pip freeze > requirements.txt   # pin the packages you installed
```

Follow [`lighton-ocr/`](lighton-ocr/) for a vision-language model, or
[`rapidocr/`](rapidocr/) for a small ONNX engine. Add the model to
[`bench/models.json`](bench/models.json), run `python run.py my-new-ocr`,
and the results files update from that run.
