"""PaddleOCR-VL-1.6 runner. 0.9B model with the official OCR prompt.

The published page-level scores use the PaddleOCR layout pipeline. The
transformers path used here is the single-prompt model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.vlm_runner import main

if __name__ == "__main__":
    main("PaddlePaddle/PaddleOCR-VL-1.6", "OCR:", trust_remote_code=False)
