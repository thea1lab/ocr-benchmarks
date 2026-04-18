"""Simple dots.mocr runner: load an image, emit extracted text.

dots.mocr is a 3B multimodal OCR model (rednote-hilab) that handles layout
detection, text recognition, tables, formulas, and structured graphics.
Unlike Qianfan-OCR's fixed 448x448 tiling, dots.mocr builds on a Qwen3-VL
encoder that accepts native-resolution images directly — the processor
handles all patching internally.
"""

import os
import sys
import time
from pathlib import Path

# dots.mocr's modeling code reads LOCAL_RANK from env even when not running
# under torchrun. Set it before transformers imports or load will KeyError.
os.environ.setdefault("LOCAL_RANK", "0")

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

# Hugging Face repo id.
MODEL_REPO = "rednote-hilab/dots.mocr"

# The HF cache stores weights under a dir name containing periods
# (`models--rednote-hilab--dots.mocr`), which breaks trust_remote_code's
# dynamic module import because Python module names can't contain periods.
# The model card's workaround: snapshot_download into a period-free local dir
# and load from that path instead.
MODEL_DIR = Path(__file__).parent / "weights" / "DotsMOCR"

# Default: plain-text OCR. Produces clean readable text with no structure.
PROMPT = "Extract the text content from this image."

# Alternative: full layout JSON with bboxes, categories, and per-cell content
# formatted as Markdown / LaTeX (formulas) / HTML (tables). Swap PROMPT to this
# if you want structured output instead of plain text.
PROMPT_LAYOUT = (
    "Please output the layout information from the PDF image, including each "
    "layout element's bbox, its category, and the corresponding text content "
    "within the bbox. Bbox format: [x1, y1, x2, y2]. Layout Categories: "
    "['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', "
    "'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. "
    "Text Extraction & Formatting Rules: Picture - omit text field; "
    "Formula - format as LaTeX; Table - format as HTML; All Others - format "
    "as Markdown. Constraints: output text must be original from image with "
    "no translation; all layout elements sorted by human reading order. "
    "Final Output: entire output must be a single JSON object."
)

# Upper bound on generated tokens. The official demo uses 24000 so long
# multi-page layouts don't truncate; biggest latency knob on MPS/CPU.
MAX_NEW_TOKENS = 24000


def ensure_weights():
    """Download the model to MODEL_DIR if not already present."""
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        return
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_REPO} to {MODEL_DIR} (~6 GB, first run only)...", flush=True)
    snapshot_download(repo_id=MODEL_REPO, local_dir=str(MODEL_DIR))


def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_path> [output.md]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"Error: image not found: {image_path}")
        sys.exit(1)

    output_path = sys.argv[2] if len(sys.argv) > 2 else image_path.with_suffix(".md")

    ensure_weights()

    # Pick dtype + attention backend per accelerator. flash-attn is CUDA-only;
    # MPS/CPU fall back to sdpa. bf16 is unreliable on MPS (many ops silently
    # drop to CPU), so use fp16 there even though the model was trained in bf16.
    if torch.cuda.is_available():
        accel = f"CUDA ({torch.cuda.get_device_name(0)})"
        dtype = torch.bfloat16
        attn = "flash_attention_2"
    elif torch.backends.mps.is_available():
        accel = "MPS (Apple Silicon)"
        dtype = torch.float16
        attn = "sdpa"
    else:
        accel = "CPU only — this will be slow"
        dtype = torch.float32
        attn = "sdpa"
    print(f"Accelerator: {accel} (dtype={dtype}, attn={attn})", flush=True)
    print("Loading model...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        attn_implementation=attn,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    devices = {str(p.device) for p in model.parameters()}
    print(f"Model loaded on: {', '.join(sorted(devices))}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print("Extracting text...", flush=True)
    start = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    # Strip the prompt tokens — generate() returns prompt + completion concatenated.
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    output = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    elapsed = time.perf_counter() - start

    Path(output_path).write_text(output, encoding="utf-8")
    print(f"Extraction took {elapsed:.2f}s")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
