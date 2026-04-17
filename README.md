# ocr-benchmarks

Grab-bag of OCR / document-parsing models, each in its own subproject with
its own dependencies. Test images are shared at the repo root.

## Layout

```
ocr-benchmarks/
├── images/                  # shared test inputs
├── qianfan-ocr-test/        # baidu/Qianfan-OCR
│   ├── ocr.py
│   ├── requirements.txt
│   └── README.md
└── ...                      # more subprojects as added
```

Each subproject has its own `README.md`, `requirements.txt`, and `.venv/`.
Don't share a venv across subprojects — torch / transformers / CUDA
pinnings collide.

## Subprojects

| Project                                  | Model                                                                 | Status   |
| ---------------------------------------- | --------------------------------------------------------------------- | -------- |
| [`qianfan-ocr-test`](qianfan-ocr-test/)  | [baidu/Qianfan-OCR](https://huggingface.co/baidu/Qianfan-OCR) (4B VL) | Testing  |

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
