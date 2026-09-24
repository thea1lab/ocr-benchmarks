"""RapidOCR runner. PP-OCRv6 small models, ONNX Runtime, CPU."""

import sys
import time
from pathlib import Path

from rapidocr import RapidOCR


def jobs_from_argv(argv):
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: python ocr.py <image> [output.md]", file=sys.stderr)
        print("       python ocr.py --suite jobs.tsv", file=sys.stderr)
        sys.exit(1 if not argv else 0)
    if argv[0] == "--suite":
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


def reading_order(rows):
    items = []
    for box, text in rows:
        if not str(text).strip():
            continue
        try:
            ys = [point[1] for point in box]
            xs = [point[0] for point in box]
            items.append((sum(ys) / len(ys), min(xs), max(ys) - min(ys), str(text)))
        except (TypeError, IndexError, ValueError, ZeroDivisionError):
            items.append((0, 0, 0, str(text)))
    if not items:
        return ""
    items.sort(key=lambda item: item[0])
    heights = sorted(item[2] for item in items if item[2] > 0) or [16]
    tolerance = max(8, heights[len(heights) // 2] * 0.6)
    lines = [[items[0]]]
    for item in items[1:]:
        if abs(item[0] - lines[-1][0][0]) <= tolerance:
            lines[-1].append(item)
        else:
            lines.append([item])
    ordered = []
    for line in lines:
        line.sort(key=lambda item: item[1])
        ordered.extend(item[3] for item in line)
    return "\n".join(ordered)


def to_text(result):
    if result is None:
        return ""
    txts = getattr(result, "txts", None)
    boxes = getattr(result, "boxes", None)
    if txts:
        if boxes is None:
            return "\n".join(str(text) for text in txts)
        return reading_order(list(zip(boxes, txts)))

    data = result
    if isinstance(result, tuple) and result and isinstance(result[0], (list, tuple)):
        data = result[0]
    rows = []
    if isinstance(data, (list, tuple)):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
                rows.append((item[0], item[1]))
    return reading_order(rows)


def main():
    jobs = jobs_from_argv(sys.argv[1:])
    for image, _output in jobs:
        if not image.is_file():
            print(f"Error: image not found: {image}", file=sys.stderr)
            sys.exit(1)

    print("DEVICE cpu", flush=True)
    print("Loading RapidOCR...", flush=True)
    started = time.perf_counter()
    engine = RapidOCR()
    print(f"LOADED {time.perf_counter() - started:.2f}", flush=True)

    for image, output in jobs:
        print(f"Extracting {image.name}...", flush=True)
        started = time.perf_counter()
        text = to_text(engine(str(image)))
        elapsed = time.perf_counter() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"PAGE {output.stem} {elapsed:.2f} 0", flush=True)
        print(f"Wrote {output}", flush=True)

    print("VRAM_MIB 0", flush=True)


if __name__ == "__main__":
    main()
