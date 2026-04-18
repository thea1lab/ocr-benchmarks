# qianfan-ocr-test

Minimal script that runs [baidu/Qianfan-OCR](https://huggingface.co/baidu/Qianfan-OCR)
on a single image and writes the extracted text as Markdown.

> Part of [`ocr-benchmarks`](../README.md). Test images live at the repo
> root in `../images/`; each subproject keeps its own venv and deps.

## Requirements

- Python 3.10+
- ~10 GB free disk for the model weights (downloaded on first run)
- A GPU is strongly recommended. CPU-only works but is very slow.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

Upgrade pip and install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU users (CUDA):** install the CUDA build of torch from
> https://pytorch.org/get-started/locally/ instead of the default wheel,
> e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`.

## Running the script

Make sure the virtual environment is active (you'll see `(.venv)` in your
prompt). If not, re-run:

```bash
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

Then run the script with the path to an image from the shared `images/`
folder at the repo root:

```bash
python ocr.py ../images/chocolate.jpeg
```

The Markdown output is written next to the input file — e.g.
`../images/chocolate.jpeg` produces `../images/chocolate.md`. Pass a second
argument to choose a different output path:

```bash
python ocr.py ../images/invoice.jpg out/invoice.md
```

On the first run the model (~5 B params, bf16) is downloaded from Hugging Face
into `~/.cache/huggingface/` — expect a few minutes depending on bandwidth.
Subsequent runs load from cache and are much faster.

## Tuning

Edit the constants at the top of `ocr.py`:

- `MAX_TILES` (default 8) — upper bound on how many 448×448 tiles the image
  is split into. Higher keeps more detail for dense pages but adds 256 visual
  tokens of prefill per tile. The model card's reference is 12; drop lower
  on MPS/CPU, raise for dense multi-column pages on CUDA.
- `MAX_NEW_TOKENS` (default 4096) — generation length cap. Biggest latency
  knob on MPS. Raise if long tables get truncated.
- `IMAGE_SIZE` — tile side length. Leave at 448 (model requirement).

## Performance & limitations

Qianfan-OCR is designed for GPU serving — the [paper][qianfan-paper] reports
~1 page/sec with W8A8 quantization on an **A100**. What to expect on other
hardware:

- **CUDA (NVIDIA GPU)** — target platform, full bf16 support, fast.
- **MPS (Apple Silicon)** — works but is **much slower**. PyTorch's MPS
  backend [adapts CUDA ops to Metal][mlx-vs-mps] and many paths are
  unoptimized; a [2026 comparative study][mlx-benchmark] puts PyTorch MPS
  at ~7–9 tok/s vs. MLX at ~230 tok/s on the same Mac. `bfloat16` also
  has gaps on MPS ([CPU/CUDA only per HF docs][hf-bf16]; some ops silently
  fall back to CPU), so the script uses `float16` there. Expect several
  minutes per page.
- **CPU only** — works but very slow (many minutes per page). Not recommended.

### Measured runs

| Machine                  | Backend | dtype | MAX_TILES | max_new_tokens | Image            | Extraction time |
| ------------------------ | ------- | ----- | --------- | -------------- | ---------------- | --------------- |
| MacBook Pro (M4 Pro, 48 GB) | MPS     | fp16  | 8         | 4096           | `chocolate.jpeg` | 90.15 s         |

**Quality caveat:** on the MPS run above, output precision was **not good
enough** for the test image. Likely causes, all tradeoffs we made for speed
on Mac:

- `dtype=float16` — the model was trained in bf16; fp16's narrower dynamic
  range can degrade VLM outputs. bf16 is what the model card uses but is
  unreliable on MPS (see above).
- `MAX_TILES=8` vs. the model card's 12 — less detail preserved per tile,
  which hurts dense layouts and small text.
- `MAX_NEW_TOKENS=4096` — can truncate long tables mid-row.
- Simplified prompt — the original `"Parse this document to Markdown.
  Convert all text and consider empty cells in the tables."` gives the
  model explicit guidance about empty table cells; the short form doesn't.

If quality matters more than latency, revert these (use bf16, `MAX_TILES=12`,
`MAX_NEW_TOKENS=16384`, full prompt) — ideally on a CUDA box, since the
full-precision config will be painfully slow on MPS.

If you need real throughput on a Mac, the practical options are:
1. Rent a cloud GPU (Colab, Runpod, Lambda, vast.ai — ~$0.30–2/hr).
2. Use a different OCR model with a native MLX port — e.g.
   [vllm-mlx][vllm-mlx] reports up to 525 tok/s for text models on M4 Max.

Qianfan-OCR has no official MLX implementation, so porting it isn't trivial.

[qianfan-paper]: https://arxiv.org/html/2603.13398v1
[mlx-vs-mps]: https://medium.com/@koypish/mps-or-mlx-for-domestic-ai-the-answer-will-surprise-you-df4b111de8a0
[mlx-benchmark]: https://arxiv.org/pdf/2511.05502
[hf-bf16]: https://huggingface.co/docs/transformers/main/perf_train_special
[vllm-mlx]: https://arxiv.org/html/2601.19139v2
