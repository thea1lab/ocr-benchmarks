# ocr-benchmarks

Grab-bag of OCR / document-parsing models, each in its own subproject with
its own dependencies. Test images are shared at the repo root.

## Layout

```
ocr-benchmarks/
├── images/                  # shared photo input
├── corpus/                  # drawn pages, transcripts, manifest
├── bench/                   # model list and the shared loader
├── dashboard/index.html     # generated comparison
├── rapidocr/                # PP-OCRv6 small via ONNX
├── ovis-ocr/                # OvisOCR2
├── paddleocr-vl/            # PaddleOCR-VL-1.6
├── glm-ocr/                 # GLM-OCR
├── lighton-ocr/             # LightOnOCR-2-1B
└── teleocr/                 # TeleOCR
```

Each subproject has its own `README.md`, `requirements.txt`, and `.venv/`.
Don't share a venv across subprojects — torch / transformers / CUDA
pinnings collide.

## Benchmark

Shared pages live in [`corpus/`](corpus/manifest.json). Every model sees the same pixels.
`python run.py` writes [`RESULTS.md`](RESULTS.md) and
[`dashboard/index.html`](dashboard/index.html).

The set is five pages: a clean letter, a two-column page with a footnote,
a table plus an equation, a tilted receipt, and a phone photo of a package.
Character accuracy is the transcript match. Fields are the facts that have
to show up (an order number, a total, a footnote). Seconds per page are
warm extraction, after weights are loaded.

## Subprojects

| Project                                  | Model                                                                               | Status   |
| ---------------------------------------- | ----------------------------------------------------------------------------------- | -------- |
| [`rapidocr`](rapidocr/)                  | [RapidOCR](https://github.com/RapidAI/RapidOCR) (PP-OCRv6 small, ONNX)             | Testing  |
| [`ovis-ocr`](ovis-ocr/)                  | [ATH-MaaS/OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) (0.8B)                | Testing  |
| [`paddleocr-vl`](paddleocr-vl/)          | [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) (0.9B)    | Testing  |
| [`glm-ocr`](glm-ocr/)                    | [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (0.9B)                    | Testing  |
| [`lighton-ocr`](lighton-ocr/)            | [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) (1B)           | Testing  |
| [`teleocr`](teleocr/)                    | [StarDoc-AI/TeleOCR](https://huggingface.co/StarDoc-AI/TeleOCR) (1.2B)              | Testing  |

## Adding a new model

```bash
mkdir my-new-ocr && cd my-new-ocr
python3 -m venv .venv && source .venv/bin/activate
# pip install ...; pip freeze > requirements.txt
```

Add a row to the table above and link to the new subdir's README.

## Measured runs

Individual benchmark tables live inside each subproject's README. See each
project for machine, dtype, tuning knobs, and timing.
