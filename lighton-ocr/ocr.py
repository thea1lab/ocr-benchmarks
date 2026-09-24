"""LightOnOCR-2-1B runner. Image-only input, as in the model card."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.vlm_runner import main

if __name__ == "__main__":
    main(
        "lightonai/LightOnOCR-2-1B",
        None,
        model_class="LightOnOcrForConditionalGeneration",
    )
