"""Simple Qianfan-OCR runner: load an image, emit Markdown.

Qianfan-OCR uses an "AnyResolution" vision encoder: instead of squishing a
high-res document into one 448x448 crop (which would destroy small text), it
tiles the image into a grid of 448x448 patches and encodes each one. The
helpers below implement that tiling.
"""

import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from PIL import Image

# Standard ImageNet normalization. The ViT was trained expecting inputs scaled
# to these per-channel mean/std, so every tile must be normalized with them.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Hugging Face repo id. AutoModel.from_pretrained will download + cache it.
MODEL_PATH = "baidu/Qianfan-OCR"

# Side length (px) of a single tile fed to the ViT. The model card specifies
# 448x448 tiles producing 256 visual tokens each — don't change this.
IMAGE_SIZE = 448

# Maximum number of 448x448 tiles we're willing to split the image into.
# The tiler picks a grid (e.g. 3x4 = 12 tiles) whose aspect ratio best matches
# the source image, capped by this number. More tiles => more detail preserved
# for dense pages, but also more visual tokens (256 per tile) and more VRAM /
# latency. Model card's reference uses 12; we default to 8 because on MPS the
# prefill cost per tile is steep. Bump up for very dense pages.
MAX_TILES = 8

# Cap on generated tokens. The model card uses no explicit cap (defaults to
# 1024); we allow 4096 so long tables aren't truncated but generation stays
# bounded — this is by far the biggest latency knob on MPS.
MAX_NEW_TOKENS = 4096


def build_transform(input_size):
    """Preprocessing pipeline applied to each tile before it hits the ViT:
    force RGB, resize to input_size x input_size, to tensor, normalize."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Pick the (cols, rows) grid whose aspect ratio is closest to the image's.

    target_ratios is the set of candidate grids (e.g. (1,1), (2,1), (3,4)...)
    capped by MAX_TILES. On a tie between two grids, prefer the larger grid
    when the source image is big enough to "fill" it (the area heuristic) —
    this keeps more detail for high-res pages instead of downsampling them
    into a smaller grid.
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """Split `image` into a grid of `image_size`-sided tiles.

    Args:
        min_num / max_num: bounds on total tile count (cols * rows).
        image_size: tile side length in pixels.
        use_thumbnail: if multiple tiles are produced, also append a single
            downscaled full-image tile. This gives the model a "global view"
            alongside the detailed tiles, which helps with layout/reading order.

    Returns a list of PIL tiles in reading order (left-to-right, top-to-bottom).
    """
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    # Enumerate every (cols, rows) grid whose total tile count is in range.
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )

    ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_w, orig_h, image_size)

    # Resize the source so it exactly fills the chosen grid, then slice it up.
    target_w, target_h = image_size * ratio[0], image_size * ratio[1]
    resized = image.resize((target_w, target_h))
    tiles = []
    cols = target_w // image_size
    for i in range(ratio[0] * ratio[1]):
        box = ((i % cols) * image_size, (i // cols) * image_size,
               ((i % cols) + 1) * image_size, ((i // cols) + 1) * image_size)
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))

    return tiles


def load_image(path):
    """Open an image file and return a stacked tensor of preprocessed tiles,
    shape (num_tiles, 3, IMAGE_SIZE, IMAGE_SIZE), ready for model.chat()."""
    image = Image.open(path).convert("RGB")
    transform = build_transform(IMAGE_SIZE)
    tiles = dynamic_preprocess(image, image_size=IMAGE_SIZE, use_thumbnail=True, max_num=MAX_TILES)
    return torch.stack([transform(t) for t in tiles])


def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_path> [output.md]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    # Validate the input path *before* touching the model — loading weights
    # downloads ~10 GB on first run, so we don't want a typo to waste that.
    if not image_path.is_file():
        print(f"Error: image not found: {image_path}")
        sys.exit(1)

    # Default output: same path as input but with a .md extension.
    output_path = sys.argv[2] if len(sys.argv) > 2 else image_path.with_suffix(".md")

    # Pick dtype per backend: bf16 is only well-supported on CUDA; on MPS
    # many bf16 ops silently fall back to CPU which tanks performance, so
    # prefer fp16 there. CPU stays on fp32 to avoid slow emulated kernels.
    if torch.cuda.is_available():
        accel = f"CUDA ({torch.cuda.get_device_name(0)})"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        accel = "MPS (Apple Silicon)"
        dtype = torch.float16
    else:
        accel = "CPU only — this will be slow"
        dtype = torch.float32
    print(f"Accelerator: {accel} (dtype={dtype})", flush=True)
    print("Loading model (first run downloads ~10 GB)...", flush=True)

    # device_map="auto" lets accelerate place layers on GPU/MPS/CPU as available.
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    devices = {str(p.device) for p in model.parameters()}
    print(f"Model loaded on: {', '.join(sorted(devices))}")

    # Match the model's dtype/device, otherwise the forward pass will error.
    pixel_values = load_image(image_path).to(dtype).to(model.device)

    print("Extracting text...", flush=True)
    start = time.perf_counter()
    with torch.no_grad():
        response = model.chat(
            tokenizer,
            pixel_values=pixel_values,
            question="Parse this document to Markdown.",
            generation_config={"max_new_tokens": MAX_NEW_TOKENS},
        )
    elapsed = time.perf_counter() - start

    Path(output_path).write_text(response, encoding="utf-8")
    print(f"Extraction took {elapsed:.2f}s")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
