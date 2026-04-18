# dots-mocr-test

Minimal script that runs [rednote-hilab/dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr)
on a single image and writes the extracted text.

> Part of [`ocr-benchmarks`](../README.md). Test images live at the repo
> root in `../images/`; each subproject keeps its own venv and deps.

## Requirements

- Python 3.10+
- ~6 GB free disk for the model weights (3B params, bf16)
- A GPU is strongly recommended. CPU works but is very slow.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU users (CUDA):** install the CUDA build of torch instead of the
> default wheel, e.g.
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
> For the fastest path, also install `flash-attn==2.8.0.post2`; the script
> automatically uses it on CUDA and falls back to `sdpa` elsewhere.

## Running

```bash
python ocr.py ../images/chocolate.jpeg
```

Output is written next to the input with a `.md` extension (e.g.
`../images/chocolate.md`). Pass a second argument to override:

```bash
python ocr.py ../images/invoice.jpg out/invoice.md
```

On first run the script downloads the model into `./weights/DotsMOCR/`
(~6 GB). The HF cache path contains periods (`dots.mocr`), which breaks
`trust_remote_code`'s dynamic module import — so we snapshot-download to a
period-free local dir, following the model card's workaround.

## Tuning

Edit constants at the top of `ocr.py`:

- `PROMPT` — defaults to plain-text OCR (`"Extract the text content from
  this image."`). Swap for `PROMPT_LAYOUT` to get a full layout JSON with
  bboxes, categories, and per-cell Markdown/LaTeX/HTML content.
- `MAX_NEW_TOKENS` (default 24000) — cap on generation length. Biggest
  latency knob on MPS/CPU. Drop for short images, raise if long tables get
  truncated.

## Performance & limitations

dots.mocr is a 3B Qwen3-VL-based OCR model. On its own benchmarks
([olmOCR-bench 83.9, OmniDocBench v1.5 TextEdit 0.031][card]) it beats
PaddleOCR-VL-1.5, HunyuanOCR, and dots.ocr among similarly sized models,
and trails only Gemini 3 Pro overall.

- **CUDA (NVIDIA GPU)** — target platform. Use bf16 + flash-attn 2.
- **MPS (Apple Silicon)** — works but slow. bf16 has gaps on MPS, so the
  script uses fp16 + sdpa. Expect minutes per page.
- **CPU** — works, very slow. Not recommended for anything dense.

### Measured runs

| Machine | Backend | dtype | max_new_tokens | Image | Extraction time |
| ------- | ------- | ----- | -------------- | ----- | --------------- |
| _(todo)_ |         |       |                |       |                 |

[card]: https://huggingface.co/rednote-hilab/dots.mocr
