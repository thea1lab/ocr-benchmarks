"""Shared single-image and multi-page loop for the vision-language runners.

Each subproject keeps its own model id and prompt. This file only handles
loading, timing, and the CLI those runners share.
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor

MAX_NEW_TOKENS = 2048


def jobs_from_argv(argv):
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: python ocr.py <image> [output.md]", file=sys.stderr)
        print("       python ocr.py --suite jobs.tsv", file=sys.stderr)
        sys.exit(1 if not argv else 0)
    if argv[0] == "--suite":
        if len(argv) != 2:
            print("Usage: python ocr.py --suite jobs.tsv", file=sys.stderr)
            sys.exit(1)
        jobs = []
        for line in Path(argv[1]).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image, output = line.split("\t")
            jobs.append((Path(image), Path(output)))
        return jobs
    image = Path(argv[0])
    output = Path(argv[1]) if len(argv) > 1 else image.with_suffix(".md")
    return [(image, output)]


def from_pretrained(cls, model_id, trust, dtype, config=None, key_mapping=None):
    kwargs = {"trust_remote_code": trust}
    if config is not None:
        kwargs["config"] = config
    if key_mapping is not None:
        kwargs["key_mapping"] = key_mapping
    try:
        return cls.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        kwargs.pop("key_mapping", None)
        return cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


def paddle_config(model_id, trust):
    """The 1.6 checkpoint omits vision rope settings that transformers 5.17 requires."""
    if "PaddleOCR" not in model_id:
        return None
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    vision = config.vision_config
    try:
        params = vision.rope_parameters
    except AttributeError:
        params = None
    if not params:
        vision.__dict__["rope_parameters"] = {"rope_type": "axial", "rope_theta": 10000.0}
    return config


def load_model(model_id, trust, dtype, model_class=None, key_mapping=None):
    config = paddle_config(model_id, trust)
    if model_class:
        cls = getattr(__import__("transformers", fromlist=[model_class]), model_class)
        return from_pretrained(cls, model_id, trust, dtype, config, key_mapping)
    try:
        return from_pretrained(AutoModelForImageTextToText, model_id, trust, dtype, config, key_mapping)
    except (ValueError, OSError, KeyError):
        return from_pretrained(AutoModel, model_id, trust, dtype, config, key_mapping)


def move_to_device(batch, device, dtype):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def prepare_inputs(processor, image, prompt, style, template_kwargs):
    image = image.convert("RGB")
    if style == "teleocr":
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        chat = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(text=[chat], images=[image], padding=True, return_tensors="pt")

    content = [{"type": "image", "image": image}]
    if prompt:
        content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    kwargs.update(template_kwargs)
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def generate(model, processor, image_path, prompt, style, template_kwargs, device, dtype):
    from PIL import Image

    image = Image.open(image_path)
    inputs = prepare_inputs(processor, image, prompt, style, template_kwargs)
    inputs = move_to_device(inputs, device, dtype)
    inputs.pop("mm_token_type_ids", None)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
    input_len = inputs["input_ids"].shape[1]
    if output_ids.shape[1] > input_len:
        trimmed = output_ids[0, input_len:]
    else:
        trimmed = output_ids[0]
    text = processor.decode(trimmed, skip_special_tokens=True)
    return text.strip(), int(trimmed.shape[0])


def oom(exc):
    return "out of memory" in str(exc).lower()


def main(model_id, prompt, trust_remote_code=False, style="chat", template_kwargs=None, model_class=None, key_mapping=None):
    template_kwargs = dict(template_kwargs or {})
    jobs = jobs_from_argv(sys.argv[1:])
    for image, _output in jobs:
        if not image.is_file():
            print(f"Error: image not found: {image}", file=sys.stderr)
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"DEVICE {device}", flush=True)
    print(f"Loading {model_id}...", flush=True)
    started = time.perf_counter()
    try:
        model = load_model(model_id, trust_remote_code, dtype, model_class, key_mapping).eval()
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except RuntimeError as exc:
        if oom(exc):
            print("ERROR model does not fit in GPU memory", file=sys.stderr)
            sys.exit(2)
        raise
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"LOADED {time.perf_counter() - started:.2f}", flush=True)

    for image, output in jobs:
        print(f"Extracting {image.name}...", flush=True)
        started = time.perf_counter()
        try:
            text, n_tokens = generate(
                model, processor, image, prompt, style, template_kwargs, device, dtype
            )
            if device == "cuda":
                torch.cuda.synchronize()
        except RuntimeError as exc:
            if oom(exc):
                print(f"ERROR out of memory on {image.name}", file=sys.stderr)
                sys.exit(2)
            raise
        elapsed = time.perf_counter() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"PAGE {output.stem} {elapsed:.2f} {n_tokens}", flush=True)
        print(f"Wrote {output}", flush=True)

    if device == "cuda":
        print(f"VRAM_MIB {torch.cuda.max_memory_allocated() / (1024 * 1024):.0f}", flush=True)
    else:
        print("VRAM_MIB 0", flush=True)
